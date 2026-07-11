"""Brain-driven Navigation Drone Environment.

Self-contained environment that integrates:
  1. Pretrained PPO navigator policy (frozen)
  2. Brain state machine (lawnmower coverage + YOLO search)
  3. Perception module (YOLO person detection)
  4. AE depth encoder + Low-level flight controller (inherited from AEPPODroneEnv)

The Brain module generates high-level waypoints, the frozen PPO policy navigates
to them, and the LLC converts velocity commands to motor forces. The external
caller simply calls env.step() — all intelligence is internal.

Architecture:
    Camera RGB/Depth → Perception (YOLO) → Brain State Machine → Waypoint
    → Frozen PPO(obs) → High-level actions → LLC → Motor forces → Physics
"""
from __future__ import annotations

import os
import sys
import math

import numpy as np
import torch
import gymnasium as gym

from .ae_ppo_drone_env import AEPPODroneEnv
from .brain_nav_drone_env_cfg import BrainNavDroneEnvCfg
from first_drone.models.perception import PerceptionModule
from first_drone.models.brain import BrainModule

from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
import isaaclab.sim as sim_utils
from isaaclab.sensors import TiledCamera, ContactSensor
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.math import subtract_frame_transforms, quat_from_euler_xyz, wrap_to_pi

from isaaclab.markers import CUBOID_MARKER_CFG  # isort: skip

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../../../../")
)


def resolve_navigator_checkpoint(path: str | None = None) -> str:
    """Resolve PPO navigator checkpoint path (file, run dir, or auto-discovery)."""
    nav_base = os.path.join(_REPO_ROOT, "logs", "rsl_rl", "navigation_drone_direct")
    tried: list[str] = []

    def _try(candidate: str | None) -> str | None:
        if not candidate:
            return None
        candidate = os.path.abspath(candidate.strip().strip("`\"'"))
        tried.append(candidate)
        if os.path.isfile(candidate):
            return candidate
        if os.path.isdir(candidate):
            for name in ("model_1450.pt", "model_latest.pt"):
                file_path = os.path.join(candidate, name)
                tried.append(file_path)
                if os.path.isfile(file_path):
                    return file_path
            import glob

            models = sorted(
                glob.glob(os.path.join(candidate, "model_*.pt")),
                key=os.path.getmtime,
                reverse=True,
            )
            for file_path in models:
                tried.append(file_path)
                return file_path
        return None

    resolved = _try(path)
    if resolved:
        return resolved

    for subdir in ("bestmodel", "29-06_23-04", "eyal_best"):
        resolved = _try(os.path.join(nav_base, subdir, "model_1450.pt"))
        if resolved:
            return resolved

    unique_tried = list(dict.fromkeys(tried))
    lines = "\n".join(f"  • {p}" for p in unique_tried[:12])
    raise FileNotFoundError(
        "\n[BrainNavEnv] Navigator checkpoint not found.\n"
        f"  Requested: {path or '(auto)'}\n"
        "  Tried:\n"
        f"{lines}\n"
        "  Place model_1450.pt under logs/rsl_rl/navigation_drone_direct/bestmodel/\n"
        "  or pass --navigator_checkpoint <path_to_model.pt>.\n"
    )


class BrainNavDroneEnv(AEPPODroneEnv):
    """Self-contained Brain + PPO navigation environment.

    Subclasses AEPPODroneEnv to inherit AE + LLC physics. Brain play uses the
    Multilevel_AE_PPO map (no LiDAR, poles-only obstacles, goal-only debug vis).
    """

    cfg: BrainNavDroneEnvCfg

    def __init__(self, cfg: BrainNavDroneEnvCfg, render_mode: str | None = None, **kwargs):
        self._ensure_ae_checkpoint_file(cfg)
        # Initialize full AEPPODroneEnv scene (drone, camera, AE, LLC, room, terrain, etc.)
        super().__init__(cfg, render_mode, **kwargs)

        # Mark as brain-play mode so AEPPODroneEnv doesn't reset on goal-reached
        self.is_brain_play = True
        self._closing = False
        self._mission_complete = False
        self.spawned_targets_local = []
        self.dynamic_spawn_active = False
        self._dynamic_spawn_names: list[str] = []
        self._dynamic_spawn_prims = []
        self._require_ae_checkpoint()

        if getattr(self, "_room_bounds_local", None) is not None:
            print(f"[BrainNavEnv] Spawn bounds from USD room: {self._room_bounds_local}\n")

        self._sync_dynamic_obstacle_registry()
        self._sync_map_geometry_from_usd()
        self._setup_rescue_persons()
        self._verify_worker_person()

        # ---------- Load Frozen PPO Navigator Policy ----------
        self._load_navigator_policy()

        # ---------- Initialize Perception Module ----------
        self._perception = PerceptionModule(
            use_mock=self.cfg.use_mock_perception,
            person_conf_threshold=self.cfg.yolo_person_conf_threshold,
            noted_conf_threshold=getattr(self.cfg, "yolo_noted_conf_threshold", 0.70),
            min_bbox_area_frac=self.cfg.yolo_min_bbox_area_frac,
            min_bbox_height_frac=getattr(self.cfg, "yolo_min_bbox_height_frac", 0.10),
            min_person_aspect=getattr(self.cfg, "yolo_min_person_aspect", 0.55),
            noted_confirm_frames=getattr(self.cfg, "yolo_noted_confirm_frames", 2),
            yolo_camera_upscale=int(getattr(self.cfg, "yolo_camera_upscale", 2)),
            yolo_imgsz=int(getattr(self.cfg, "yolo_imgsz", 1280)),
            yolo_sharpen=bool(getattr(self.cfg, "yolo_sharpen", True)),
            camera_focal_length=float(self.cfg.tiled_camera.spawn.focal_length),
            camera_horizontal_aperture=float(self.cfg.tiled_camera.spawn.horizontal_aperture),
            rescue_person_slots=self._build_rescue_person_log_slots(),
            person_match_radius=float(getattr(self.cfg, "brain_person_match_radius", 5.0)),
            yolo_clahe=getattr(self.cfg, "yolo_clahe", False),
            show_opencv=bool(getattr(self.cfg, "yolo_show_opencv", True)),
        )
        print(
            f"\n[BrainNavEnv] Perception initialized (use_mock={self.cfg.use_mock_perception}, "
            f"person_conf>={self.cfg.yolo_person_conf_threshold:.0%}, "
            f"opencv={'on' if getattr(self.cfg, 'yolo_show_opencv', True) else 'off'})\n"
        )

        # ---------- Initialize Brain Module ----------
        self._brain = BrainModule(
            _BrainEnvAdapter(self),
            step_size=self.cfg.brain_step_size,
            safety_margin=self.cfg.brain_safety_margin,
        )
        print(f"\n[BrainNavEnv] Brain initialized (step_size={self.cfg.brain_step_size}m, "
              f"margin={self.cfg.brain_safety_margin}m, "
              f"sequential={getattr(self.cfg, 'brain_use_sequential_spawns', False)})\n")

        # ---------- Internal State Buffers ----------
        self._timestep = 0
        self._last_obstacle_segment = -1
        self._last_obstacle_scan_mode = False
        self._last_person_found = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._last_person_world_xyz = torch.zeros((self.num_envs, 3), device=self.device)
        self._segment_crash_counts: dict[int, int] = {}
        self._final_rescue_person_prim = None
        self._final_center_person_prim = None
        self._room3_rescue_person_prim = None
        self._steps_since_last_scan = 100
        self._was_scanning = False

        self.set_debug_vis(self.cfg.debug_vis)

        if getattr(self.cfg, "brain_disable_episode_timeout", True):
            play_len = 3600.0
            self._update_episode_length(play_len)
            print(f"[BrainNavEnv] Episode timeout disabled for play (length={play_len}s).\n")

        self._apply_brain_spawn_and_goal(self._robot._ALL_INDICES, mission_snapshot=None)
        self._setup_rescue_persons()
        self._randomize_obstacles(self._robot._ALL_INDICES)

    @staticmethod
    def _ensure_ae_checkpoint_file(cfg: BrainNavDroneEnvCfg) -> None:
        """Restore 64-dim AE from Git LFS (Multilevel_AE_PPO) when missing locally."""
        ae_path = os.path.abspath(getattr(cfg, "ae_checkpoint_path", "") or "")
        if not ae_path or os.path.isfile(ae_path):
            return
        if int(getattr(cfg, "ae_latent_dim", 32)) != 64:
            return

        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../../"))
        rel_path = os.path.relpath(ae_path, repo_root).replace("\\", "/")
        os.makedirs(os.path.dirname(ae_path), exist_ok=True)

        import subprocess

        for cmd in (
            ["git", "lfs", "pull", "--include", rel_path],
            ["git", "checkout", "Multilevel_AE_PPO", "--", rel_path],
            ["git", "lfs", "pull", "--include", rel_path],
        ):
            try:
                subprocess.run(cmd, cwd=repo_root, check=False, capture_output=True, text=True)
            except Exception:
                pass
            if os.path.isfile(ae_path) and os.path.getsize(ae_path) > 1_000_000:
                print(f"[BrainNavEnv] Restored AE checkpoint from Git LFS: {ae_path}\n")
                return

    def _require_ae_checkpoint(self) -> None:
        """Fail fast if the 64-dim AE from Multilevel_Train (bdo85ahx) is not loaded."""
        latent = int(getattr(self.cfg, "ae_latent_dim", 32))
        ae_path = os.path.abspath(getattr(self.cfg, "ae_checkpoint_path", "") or "")
        if not ae_path or not os.path.isfile(ae_path):
            raise FileNotFoundError(
                "\n[BrainNavEnv] FATAL: AE checkpoint missing — navigator cannot work without it.\n"
                f"  Expected: {ae_path}\n"
                "  model_1450.pt (WandB Multilevel_Train run bdo85ahx) was trained with:\n"
                "    • 64-dim AE at logs/ae_1_7_latent_64/ae_final.pt\n"
                "  Restore it from the Multilevel_AE_PPO branch (Git LFS):\n"
                "    git lfs pull --include=logs/ae_1_7_latent_64/ae_final.pt\n"
                "    git checkout Multilevel_AE_PPO -- logs/ae_1_7_latent_64/ae_final.pt\n"
                "  Or pass --ae_checkpoint <path> in brain_nav_play.py.\n"
            )
        fc_w = getattr(self.ae, "fc_z", None)
        if fc_w is not None and hasattr(fc_w, "weight"):
            loaded_latent = int(fc_w.weight.shape[0])
            if loaded_latent != latent:
                raise RuntimeError(
                    f"[BrainNavEnv] AE latent mismatch: cfg.ae_latent_dim={latent}, "
                    f"checkpoint produces {loaded_latent}-dim latents ({ae_path})."
                )
        print(f"[BrainNavEnv] AE verified: {latent}-dim latent from {ae_path}\n")

    def _sync_dynamic_obstacle_registry(self) -> None:
        """Register poles for parent helpers (Multilevel: poles only)."""
        self._room3_obstacles = getattr(self, "_room3_obstacles", [])
        self._corr1_obstacles = getattr(self, "_corr1_obstacles", [])
        self._corr2_obstacles = getattr(self, "_corr2_obstacles", [])
        poles = getattr(self, "_poles", [])
        self._pillars = list(poles)
        if self._pillars:
            default_shape = {"type": "cylinder", "radius": 0.05, "half_z": 1.25}
            self._obstacle_shapes = [default_shape] * len(self._pillars)
            self._obstacle_collision_radii = [0.15] * len(self._pillars)

    def _setup_scene(self):
        """Multilevel_AE_PPO scene: roof map, 21 poles, camera, contact — no LiDAR."""
        self._robot = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self._robot
        self._pillars = []
        self._room3_obstacles = []
        self._corr1_obstacles = []
        self._corr2_obstacles = []

        room_usd_path = os.path.abspath(self.cfg.room_usd_path)
        if not os.path.isfile(room_usd_path):
            raise FileNotFoundError(f"Room USD not found: {room_usd_path}")
        print(f"\n[MAP] Loading room USD (Multilevel layout): {room_usd_path}\n")

        room_cfg = sim_utils.UsdFileCfg(usd_path=room_usd_path)
        room_cfg.func("/World/envs/env_0/Room", room_cfg)

        # --- Poles (same layout as Multilevel_AE_PPO) ---
        self._poles = []
        for i in range(self.cfg.num_poles):
            pole_cfg = RigidObjectCfg(
                prim_path=f"/World/envs/env_.*/Pole_{i}",
                spawn=self.cfg.pole_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -100.0)),
            )
            pole = RigidObject(pole_cfg)
            self.scene.rigid_objects[f"pole_{i}"] = pole
            self._poles.append(pole)

        # --- Room 3 / Room 4 obstacles (Multilevel play — randomized per mission segment) ---
        self._room3_obstacles = []
        obstacle_types = [
            ("wall", self.cfg.wall_spawn, self.cfg.num_room3_walls),
            ("cone", self.cfg.cone_spawn, self.cfg.num_room3_cones),
            ("big_gate", self.cfg.big_gate_spawn, self.cfg.num_room3_big_gates),
            ("small_gate", self.cfg.small_gate_spawn, self.cfg.num_room3_small_gates),
            ("poles_triangle", self.cfg.poles_triangle_spawn, self.cfg.num_room3_poles_triangles),
        ]
        for name, spawn_cfg, count in obstacle_types:
            for i in range(count):
                obj_cfg = RigidObjectCfg(
                    prim_path=f"/World/envs/env_.*/Room3_{name}_{i}",
                    spawn=spawn_cfg,
                    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -100.0)),
                )
                obj = RigidObject(obj_cfg)
                self.scene.rigid_objects[f"room3_{name}_{i}"] = obj
                self._room3_obstacles.append(obj)

        self._corr1_obstacles = []
        for i in range(self.cfg.num_room4_corr1):
            corr_cfg = RigidObjectCfg(
                prim_path=f"/World/envs/env_.*/Room4_corr1_{i}",
                spawn=self.cfg.corr1_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -100.0)),
            )
            obj = RigidObject(corr_cfg)
            self.scene.rigid_objects[f"room4_corr1_{i}"] = obj
            self._corr1_obstacles.append(obj)

        self._corr2_obstacles = []
        for i in range(self.cfg.num_room4_corr2):
            corr_cfg = RigidObjectCfg(
                prim_path=f"/World/envs/env_.*/Room4_corr2_{i}",
                spawn=self.cfg.corr2_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -100.0)),
            )
            obj = RigidObject(corr_cfg)
            self.scene.rigid_objects[f"room4_corr2_{i}"] = obj
            self._corr2_obstacles.append(obj)

        try:
            import omni.usd
            from pxr import Usd, UsdGeom, Gf

            stage = omni.usd.get_context().get_stage()
            if stage:
                colors_dict = {
                    "wall": (0.15, 0.35, 0.95),
                    "cone": (0.95, 0.80, 0.05),
                    "big_gate": (0.60, 0.10, 0.90),
                    "small_gate": (0.05, 0.80, 0.85),
                    "poles_triangle": (0.10, 0.80, 0.25),
                }

                def _set_prim_color(prim_path, color_rgb):
                    prim = stage.GetPrimAtPath(prim_path)
                    if not prim.IsValid():
                        return
                    color_vec = Gf.Vec3f(*color_rgb)
                    for p in Usd.PrimRange(prim):
                        if p.IsA(UsdGeom.Gprim):
                            gprim = UsdGeom.Gprim(p)
                            gprim.CreateDisplayColorAttr().Set([color_vec])

                for name, _, count in obstacle_types:
                    for i in range(count):
                        _set_prim_color(f"/World/envs/env_0/Room3_{name}_{i}", colors_dict[name])
                print("[BrainNavEnv] Colored Room 3 obstacles.\n")
        except Exception as e:
            print(f"[BrainNavEnv] Could not color Room 3 obstacles: {e}\n")

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self._tiled_camera = TiledCamera(self.cfg.tiled_camera)
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor

        self._view_camera = TiledCamera(self.cfg.view_camera)

        self._view_left_camera:  "TiledCamera | None" = None
        self._view_right_camera: "TiledCamera | None" = None
        try:
            self._view_left_camera  = TiledCamera(self.cfg.view_left_camera)
            self._view_right_camera = TiledCamera(self.cfg.view_right_camera)
            print("[BrainNavEnv] Dashboard view cameras created (rear/left/right TiledCamera).")
        except Exception as _view_exc:
            print(f"[BrainNavEnv] Side view cameras not created: {_view_exc}")
            self._view_left_camera = None
            self._view_right_camera = None

        # No LiDAR in Multilevel training — policy uses camera + state only
        self._lidar = None
        self._last_lidar_scan = None

        from pxr import UsdGeom
        collision_prim = self.sim.stage.GetPrimAtPath("/World/envs/env_0/Drone/body/body_collision")
        if collision_prim.IsValid():
            UsdGeom.Imageable(collision_prim).MakeInvisible()

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        self.scene.sensors["tiled_camera"] = self._tiled_camera

        # Register the rear/behind drone camera so the dashboard can read its RGB.
        if self._view_camera is not None:
            self.scene.sensors["view_camera"] = self._view_camera

        # Register body-mounted left/right dashboard view cameras.
        if self._view_left_camera is not None:
            self.scene.sensors["view_left_camera"] = self._view_left_camera
        if self._view_right_camera is not None:
            self.scene.sensors["view_right_camera"] = self._view_right_camera

        # Behind-drone chase camera for the viewport (Multilevel_AE_PPO)
        try:
            import omni.usd
            import omni.kit.viewport.utility
            from pxr import UsdGeom

            stage = omni.usd.get_context().get_stage()
            if stage:
                cam_path = "/World/envs/env_0/Drone/body/Camera_View"
                cam_prim = stage.GetPrimAtPath(cam_path)
                if cam_prim.IsValid():
                    cam = UsdGeom.Camera(cam_prim)
                    cam.GetVerticalApertureAttr().Set(15.2908)
                    cam.GetHorizontalApertureAttr().Set(20.955)
                    cam.GetFocalLengthAttr().Set(15.5)
                    cam.GetFocusDistanceAttr().Set(400.0)
                    viewport_window = omni.kit.viewport.utility.get_active_viewport_window()
                    if viewport_window is not None:
                        viewport_window.set_active_camera(cam_path)
                        print(f"[BrainNavEnv] Viewport camera set to chase view: {cam_path}")
        except Exception as e:
            print(f"[BrainNavEnv] Could not set chase viewport camera (headless?): {e}")

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        self._cache_room_spawn_bounds()
        self._sync_dynamic_obstacle_registry()

    def _build_sequential_spawn_sequence(self) -> tuple[tuple, list[str]]:
        """Build scan/nav waypoints: rooms 1–4, then corr1 → corr2 → Worker final room."""
        zones = getattr(self, "_map_zones", None) or {}
        room_keys = sorted(
            (k for k in zones if k.startswith("room_")),
            key=lambda k: int(k.split("_", 1)[1]),
        )
        if len(room_keys) < 2:
            centers = getattr(self, "_room_segment_centers", None) or []
            if len(centers) < 2:
                seq = [tuple(p) for p in getattr(self.cfg, "brain_spawn_sequence", ())]
                labels = [
                    "finish (Worker)" if i == len(seq) - 1 else f"waypoint_{i + 1}"
                    for i in range(len(seq))
                ]
                return tuple(seq), labels
            seq = [tuple(c) for c in centers]
            labels = [f"room_{i + 1}" for i in range(len(centers))]
            finish = list(seq[-1])
            segments = getattr(self, "_room_segments", {})
            if segments:
                last_idx = max(segments.keys())
                _, _, ly0, ly1 = segments[last_idx]["bounds"]
                finish[1] = ly0 + 0.35
                finish[2] = 1.0
            seq.append(tuple(finish))
            labels.append("finish")
            return tuple(seq), labels

        seq: list[tuple] = []
        labels: list[str] = []
        for key in room_keys:
            seq.append(zones[key]["center"])
            labels.append(key.replace("_", " "))

        # After room 4: corr1 entrance → corr2 junction → final room center (never Worker GPS).
        corr1 = tuple(getattr(self.cfg, "brain_room4_corr1_waypoint", (0.0, -17.0, 1.0)))
        corr2 = tuple(getattr(self.cfg, "brain_room4_corr2_waypoint", (0.0, -20.5, 1.0)))
        if "corridor" in zones:
            lx0, lx1, ly0, ly1 = zones["corridor"]["bounds"]
            # Entrance from room 4 = north edge of corridor zone (higher Y)
            corr1 = (0.5 * (lx0 + lx1), max(ly0, ly1) - 0.35, 1.0)
        if "side_coridors" in zones:
            lx0, lx1, ly0, ly1 = zones["side_coridors"]["bounds"]
            # Junction where corr1 meets corr2 (north edge of side corridor zone)
            corr2 = (0.5 * (lx0 + lx1), max(ly0, ly1) - 0.35, 1.0)
        seq.append(corr1)
        labels.append("corr1 entrance")
        seq.append(corr2)
        labels.append("corr2 junction")

        # Final room — generic patrol point only; brain finds humans via YOLO, not map GPS.
        finish = getattr(self.cfg, "brain_final_room_waypoint", (-5.0, -21.0, 1.0))
        if getattr(self.cfg, "brain_use_worker_gps_for_nav", False):
            worker_pt = getattr(self, "_finish_point_local", None)
            if worker_pt is not None:
                finish = worker_pt
        elif "side_coridors" in zones:
            lx0, lx1, ly0, ly1 = zones["side_coridors"]["bounds"]
            finish = (min(lx0, lx1) + 0.35, min(ly0, ly1) + 0.35, 1.0)
        elif room_keys:
            lx0, lx1, ly0, ly1 = zones[room_keys[-1]]["bounds"]
            finish = (0.5 * (lx0 + lx1), 0.5 * (ly0 + ly1), 1.0)
        seq.append(tuple(finish))
        labels.append("final room")

        return tuple(seq), labels

    def _build_multilevel_hybrid_sequence(self) -> tuple[tuple, list[str]]:
        """Rooms 1–4 from Multilevel cfg; corr1/corr2/final room from map geometry."""
        cfg_seq = getattr(self.cfg, "brain_spawn_sequence", ())
        cfg_labels = list(getattr(self.cfg, "brain_spawn_labels", ()))
        room_pts = [tuple(p) for p in cfg_seq[:4]]
        room_labels = cfg_labels[:4] if len(cfg_labels) >= 4 else [
            f"room {i + 1} entrance" for i in range(len(room_pts))
        ]

        # Fixed Multilevel coords — corridor pass at corr2 junction (-20.5), then final room.
        corr1 = tuple(getattr(self.cfg, "brain_room4_corr1_waypoint", (0.0, -20.5, 1.0)))
        corr2 = tuple(getattr(self.cfg, "brain_room4_corr2_waypoint", (0.0, -20.5, 1.0)))
        final = tuple(getattr(self.cfg, "brain_final_room_waypoint", (-6.0, -21.5, 1.0)))

        if getattr(self.cfg, "brain_use_usd_corridor_waypoints", False):
            zones = getattr(self, "_map_zones", None) or {}
            if "corridor" in zones:
                lx0, lx1, ly0, ly1 = zones["corridor"]["bounds"]
                corr1 = (0.5 * (lx0 + lx1), max(ly0, ly1) - 0.35, 1.0)
            if "side_coridors" in zones:
                lx0, lx1, ly0, ly1 = zones["side_coridors"]["bounds"]
                fx = min(lx0, lx1) + 0.5
                fy = min(ly0, ly1) + 0.5
                candidate = (fx, fy, 1.0)
                if math.hypot(candidate[0] - corr2[0], candidate[1] - corr2[1]) >= 2.0:
                    final = candidate

        if getattr(self.cfg, "brain_single_corridor_to_final", True):
            seq = tuple(room_pts + [corr1, final])
            labels = room_labels + ["corridor pass", "final room"]
        else:
            seq = tuple(room_pts + [corr1, corr2, final])
            labels = room_labels + ["corr1 entrance", "corr2 junction", "final room"]
        return seq, labels

    def _get_usd_edit_layer(self):
        from pxr import Usd

        stage = self.sim.stage
        edit_layer = stage.GetSessionLayer()
        if edit_layer is None or edit_layer.empty:
            edit_layer = stage.GetRootLayer()
        return edit_layer

    def _build_rescue_person_log_slots(self) -> list[dict]:
        """Fixed YOLO log slots — one row per physical rescue person."""
        room3 = getattr(self.cfg, "brain_room3_person_local", (0.0, -10.0, 0.0))
        final_a = getattr(self.cfg, "brain_final_person_local", (-4.0, -20.0, 0.0))
        return [
            {"id": "room3", "xyz": tuple(room3), "label": "Room 3 person"},
            {"id": "final_a", "xyz": tuple(final_a), "label": "Final room"},
        ]

    def _local_to_world_vec3d(self, local_xyz: tuple[float, float, float]):
        from pxr import Gf

        origin = self._terrain.env_origins[0].cpu().numpy()
        px, py, pz = float(local_xyz[0]), float(local_xyz[1]), float(local_xyz[2])
        return Gf.Vec3d(origin[0] + px, origin[1] + py, origin[2] + pz)

    def _set_prim_visibility(self, prim, *, visible: bool) -> None:
        from pxr import Usd, UsdGeom

        stage = self.sim.stage
        with Usd.EditContext(stage, self._get_usd_edit_layer()):
            over = stage.OverridePrim(prim.GetPath())
            imageable = UsdGeom.Imageable(over)
            if imageable:
                if visible:
                    imageable.MakeVisible()
                else:
                    imageable.MakeInvisible()

    def _resolve_rescue_person_usd(self) -> str:
        """Path to the Isaac Sim default character USD (F_Business_02 with clothes/textures)."""
        custom = (getattr(self.cfg, "brain_rescue_person_usd", "") or "").strip()
        if custom:
            if os.path.isabs(custom) or "://" in custom:
                return custom
            return os.path.abspath(custom)
        from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
        nucleus_dir = ISAAC_NUCLEUS_DIR
        if not nucleus_dir or nucleus_dir.startswith("None/"):
            import carb
            settings = carb.settings.get_settings()
            asset_root = settings.get("/persistent/isaac/asset_root/default")
            if not asset_root:
                asset_root = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1"
            nucleus_dir = f"{asset_root}/Isaac"

        char = getattr(self.cfg, "brain_rescue_person_character", "F_Business_02")
        return f"{nucleus_dir}/People/Characters/{char}/{char}.usd"

    def _ensure_person_textures(self, root_prim) -> None:
        """Keep rescue persons visible; only override materials when explicitly requested."""
        from pxr import Usd, UsdGeom

        if root_prim is None or not root_prim.IsValid():
            return

        for prim in Usd.PrimRange(root_prim):
            img = UsdGeom.Imageable(prim)
            if img:
                img.MakeVisible()

        if not getattr(self.cfg, "brain_person_override_textures", False):
            return

        import isaaclab.sim as sim_utils
        from isaaclab.sim.utils import bind_visual_material
        import omni.usd

        mesh_colors: dict[str, tuple[float, float, float]] = {
            "skin": (0.76, 0.60, 0.46),
            "head": (0.76, 0.60, 0.46),
            "face": (0.76, 0.60, 0.46),
            "body": (0.76, 0.60, 0.46),
            "hand": (0.76, 0.60, 0.46),
            "arm": (0.76, 0.60, 0.46),
            "leg": (0.76, 0.60, 0.46),
            "shirt": (0.15, 0.35, 0.65),  # contrasting dark blue shirt to separate head/arms silhouette
            "vest": (0.22, 0.20, 0.28),
            "jeans": (0.24, 0.36, 0.58),
            "pants": (0.24, 0.36, 0.58),
            "shoes": (0.14, 0.12, 0.10),
            "tennisshoes": (0.14, 0.12, 0.10),
            "hair": (0.12, 0.08, 0.06),
            "default": (0.76, 0.60, 0.46),  # default to skin color to prevent untextured white heads
        }

        def _color_key(mesh_name: str) -> str:
            n = mesh_name.lower()
            for key in mesh_colors:
                if key != "default" and key in n:
                    return key
            return "default"

        stage = omni.usd.get_context().get_stage()
        mat_root = "/World/envs/env_0/Room/RescuePersons/Materials"
        bound = 0

        try:
            for prim in Usd.PrimRange(root_prim):
                if not prim.IsA(UsdGeom.Mesh):
                    continue

                ckey = _color_key(prim.GetName())
                rgb = mesh_colors[ckey]
                print(f"[Override Debug] Mesh Name: '{prim.GetName()}' -> Color Key: '{ckey}' (RGB: {rgb})")
                
                # Unbind existing remote materials to prevent async S3 downloads from overwriting our local overrides
                from pxr import UsdShade
                UsdShade.MaterialBindingAPI(prim).UnbindAllBindings()
                
                mat_path = f"{mat_root}/{ckey}"
                if not stage.GetPrimAtPath(mat_path).IsValid():
                    mat_cfg = sim_utils.PreviewSurfaceCfg(
                        diffuse_color=rgb,
                        roughness=0.72,
                        metallic=0.0,
                    )
                    mat_cfg.func(mat_path, mat_cfg)

                bind_visual_material(
                    str(prim.GetPath()),
                    mat_path,
                    stage=stage,
                    stronger_than_descendants=True,
                )
                bound += 1

            print(
                f"[BrainNavEnv] Applied fallback materials to {bound} meshes under "
                f"{root_prim.GetPath()}\n"
            )
        except Exception as exc:
            print(f"[BrainNavEnv] Could not apply person textures: {exc}\n")

    def _person_wrapper_scale(self) -> tuple[float, float, float]:
        target = float(getattr(self.cfg, "brain_person_scale", 0.35))
        native = float(getattr(self.cfg, "brain_person_asset_native_scale", 1.0))
        height_mul = float(getattr(self.cfg, "brain_person_height_scale", 1.0))
        sx = sz = target / max(native, 1e-6)
        sy = sx * height_mul
        return float(sx), float(sy), float(sz)

    def _get_static_person_wrapper_scale(self) -> tuple[float, float, float]:
        """Read the exact wrapper scale from the Room 3 static person (source of truth)."""
        from pxr import UsdGeom

        template = getattr(self, "_room3_rescue_person_prim", None)
        if template is None or not template.IsValid():
            return self._person_wrapper_scale()
        for op in UsdGeom.Xformable(template).GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                v = op.Get()
                return float(v[0]), float(v[1]), float(v[2])
        return self._person_wrapper_scale()

    def _refresh_static_person_target_height(self) -> float | None:
        """Re-measure Room 3 person height once USD meshes are loaded."""
        template = getattr(self, "_room3_rescue_person_prim", None)
        if template is None or not template.IsValid():
            return None
        h = self._person_bbox_height(template)
        if h and h > 0.3:
            self._static_person_target_height = h
            return h
        return getattr(self, "_static_person_target_height", None)

    def _align_person_scale_to_static_template(self, wrapper_prim) -> bool:
        """Resize a spawned person until its world height matches Room 3 / Final static persons."""
        from pxr import Gf, UsdGeom

        template = getattr(self, "_room3_rescue_person_prim", None)
        if template is None or not template.IsValid() or wrapper_prim is None:
            return False
        if not wrapper_prim.IsValid():
            return False

        target_h = self._refresh_static_person_target_height()
        if target_h is None or target_h < 0.1:
            target_h = getattr(self, "_static_person_target_height", None)
        if not target_h or target_h < 0.1:
            return False

        current_h = self._person_bbox_height(wrapper_prim)
        if not current_h or current_h < 0.1:
            return False
        if abs(current_h - target_h) < 0.08:
            return True

        ratio = target_h / current_h
        xform = UsdGeom.Xformable(wrapper_prim)
        scaled = False
        for op in xform.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                v = op.Get()
                op.Set(
                    Gf.Vec3d(
                        float(v[0]) * ratio,
                        float(v[1]) * ratio,
                        float(v[2]) * ratio,
                    )
                )
                scaled = True
                break
        if not scaled:
            sx, sy, sz = self._get_static_person_wrapper_scale()
            xform.AddScaleOp().Set(Gf.Vec3d(sx * ratio, sy * ratio, sz * ratio))

        new_h = self._person_bbox_height(wrapper_prim)
        print(
            f"[BrainNavEnv] Aligned {wrapper_prim.GetName()} height "
            f"{current_h:.2f}m -> {new_h:.2f}m (target {target_h:.2f}m)\n"
        )
        return abs(new_h - target_h) < 0.08

    def _process_pending_person_scale_fixes(self) -> None:
        """Retry scale alignment after async USD character payloads finish loading."""
        pending = getattr(self, "_pending_person_scale_fix", None)
        if not pending:
            return
        still_pending = []
        for prim in pending:
            if prim is None or not prim.IsValid():
                continue
            if not self._align_person_scale_to_static_template(prim):
                still_pending.append(prim)
        self._pending_person_scale_fix = still_pending

    def _queue_person_scale_fix(self, wrapper_prim) -> None:
        if wrapper_prim is None or not wrapper_prim.IsValid():
            return
        pending = getattr(self, "_pending_person_scale_fix", None)
        if pending is None:
            pending = []
        if wrapper_prim not in pending:
            pending.append(wrapper_prim)
        self._pending_person_scale_fix = pending

    def _person_spawn_local_z(self) -> float:
        """Floor height for rescue persons (same as static room-3 / final placements)."""
        return float(getattr(self.cfg, "brain_person_floor_z", 0.0))

    def _is_valid_dynamic_spawn_xy(self, x: float, y: float, margin: float = 0.25) -> bool:
        """True when (x,y) is on the navigable floor inside the map, clear of obstacles."""
        bounds = getattr(self, "_room_bounds_local", None)
        if bounds is None:
            bounds = getattr(self.cfg, "map_bounds", None)
        if bounds is not None and len(bounds) == 4:
            min_x, max_x, min_y, max_y = bounds
            if (
                x < float(min_x) + margin
                or x > float(max_x) - margin
                or y < float(min_y) + margin
                or y > float(max_y) - margin
            ):
                return False
        if not hasattr(self, "_is_on_navigable_floor"):
            return True
        tx = torch.tensor([x], device=self.device, dtype=torch.float32)
        ty = torch.tensor([y], device=self.device, dtype=torch.float32)
        if not bool(self._is_on_navigable_floor(tx, ty, margin=margin).item()):
            return False
        if hasattr(self, "_is_inside_map_obstacle"):
            if bool(self._is_inside_map_obstacle(tx, ty, margin=margin).item()):
                return False
        return True

    def _sample_dynamic_spawn_positions(
        self, count: int, *, min_drone_dist: float = 2.0
    ) -> list[tuple[float, float, float]]:
        """Pick random floor positions inside the walkable map (no scene changes)."""
        import random as _rng

        floor_z = self._person_spawn_local_z()
        margin = 0.25
        placed: list[tuple[float, float, float]] = []
        seen_xy: set[tuple[int, int]] = set()

        d_pos = self._robot.data.root_pos_w[0] - self._terrain.env_origins[0]
        dx, dy = float(d_pos[0].item()), float(d_pos[1].item())

        def _try_place(x: float, y: float) -> bool:
            key = (int(round(x * 10)), int(round(y * 10)))
            if key in seen_xy:
                return False
            if not self._is_valid_dynamic_spawn_xy(x, y, margin=margin):
                return False
            if math.hypot(x - dx, y - dy) < min_drone_dist:
                return False
            for px, py, _ in placed:
                if math.hypot(x - px, y - py) < 2.0:
                    return False
            seen_xy.add(key)
            placed.append((x, y, floor_z))
            return True

        cells = getattr(self, "_interior_walkable_spawn_cells", None)
        if cells is None or cells.shape[0] == 0:
            cells = getattr(self, "_walkable_spawn_cells", None)
        n_cells = int(cells.shape[0]) if cells is not None else 0
        if n_cells > 0:
            idxs = list(range(n_cells))
            _rng.shuffle(idxs)
            for idx in idxs:
                if len(placed) >= count:
                    break
                _try_place(float(cells[idx, 0].item()), float(cells[idx, 1].item()))

        if len(placed) < count:
            zones = [
                (-2.05,  2.05,  -2.05,  2.05),
                (-2.05,  2.05,  -8.05, -2.00),
                (-4.05,  4.05, -16.05, -7.95),
                (-8.55, -4.45, -23.05, -17.95),
                (-4.50,  0.55, -22.05, -16.00),
            ]
            attempts = 0
            max_attempts = max(500, count * 400)
            while len(placed) < count and attempts < max_attempts:
                attempts += 1
                zone = _rng.choice(zones)
                x = _rng.uniform(zone[0] + margin, zone[1] - margin)
                y = _rng.uniform(zone[2] + margin, zone[3] - margin)
                _try_place(x, y)

        if len(placed) < count:
            print(
                f"[BrainNavEnv] Spawn sampling: {len(placed)}/{count} valid positions "
                f"(walkable_cells={n_cells})."
            )
        return placed

    def _disable_physics_under_prim(self, root_prim) -> None:
        """Visual-only persons: strip rigid-body/collision to avoid PhysX attach errors."""
        try:
            from pxr import Usd, UsdPhysics

            for prim in Usd.PrimRange(root_prim):
                if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    rb = UsdPhysics.RigidBodyAPI(prim)
                    if rb.GetRigidBodyEnabledAttr().IsValid():
                        rb.GetRigidBodyEnabledAttr().Set(False)
                if prim.HasAPI(UsdPhysics.CollisionAPI):
                    col = UsdPhysics.CollisionAPI(prim)
                    if col.GetCollisionEnabledAttr().IsValid():
                        col.GetCollisionEnabledAttr().Set(False)
        except Exception as exc:
            print(f"[BrainNavEnv] Could not disable person physics: {exc}")

    def _spawn_rescue_person_wrapper(
        self,
        name: str,
        local_xyz: tuple[float, float, float],
        *,
        yaw_deg: float = 90.0,
    ):
        """Spawn Isaac Sim F_Business_02 (default clothed character) at uniform scale."""
        from pxr import Gf, Usd, UsdGeom

        stage = self.sim.stage
        person_usd = self._resolve_rescue_person_usd()
        ref_path = (getattr(self.cfg, "brain_rescue_person_usd_ref", "") or "").strip()
        scope = getattr(self.cfg, "brain_rescue_person_scope", "RescuePersons")
        parent_path = f"/World/envs/env_0/Room/{scope}"
        wrapper_path = f"{parent_path}/{name}"
        char_path = f"{wrapper_path}/Character"

        sx, sy, sz = self._get_static_person_wrapper_scale()
        world_pos = self._local_to_world_vec3d(local_xyz)

        edit_layer = self._get_usd_edit_layer()
        with Usd.EditContext(stage, edit_layer):
            if not stage.GetPrimAtPath(parent_path).IsValid():
                stage.DefinePrim(parent_path, "Xform")
            if stage.GetPrimAtPath(wrapper_path).IsValid():
                stage.RemovePrim(wrapper_path)

            wrapper = stage.DefinePrim(wrapper_path, "Xform")
            xform = UsdGeom.Xformable(wrapper)
            xform.ClearXformOpOrder()
            xform.AddTranslateOp().Set(world_pos)
            xform.AddRotateXYZOp().Set(Gf.Vec3f(0.0, 0.0, float(yaw_deg)))
            xform.AddScaleOp().Set(Gf.Vec3d(sx, sy, sz))

            char = stage.DefinePrim(char_path, "Xform")
            if ref_path:
                char.GetReferences().AddReference(person_usd, ref_path)
            else:
                char.GetReferences().AddReference(person_usd)
            UsdGeom.Imageable(wrapper).MakeVisible()

        wrapper_prim = stage.GetPrimAtPath(wrapper_path)
        if not wrapper_prim.IsValid():
            raise RuntimeError(f"Failed to spawn rescue person wrapper at {wrapper_path}")
        self._ensure_person_textures(wrapper_prim)
        self._disable_physics_under_prim(wrapper_prim)
        print(
            f"[BrainNavEnv] Spawned {name} from {person_usd} "
            f"scale=({sx:.3f},{sy:.3f},{sz:.3f})\n"
        )
        return wrapper_prim

    def _clone_rescue_person_from_template(
        self,
        new_name: str,
        local_xyz: tuple[float, float, float],
        *,
        template_name: str | None = None,
        yaw_deg: float = 90.0,
    ):
        """Clone an existing default rescue person so scale/materials match exactly."""
        from pxr import Gf, Sdf, Usd, UsdGeom

        template_name = template_name or getattr(
            self.cfg, "brain_room3_person_name", "RescuePerson_Room3"
        )
        stage = self.sim.stage
        scope = getattr(self.cfg, "brain_rescue_person_scope", "RescuePersons")
        parent_path = f"/World/envs/env_0/Room/{scope}"
        template_path = f"{parent_path}/{template_name}"
        wrapper_path = f"{parent_path}/{new_name}"

        template_prim = getattr(self, "_room3_rescue_person_prim", None)
        if template_prim is None or not template_prim.IsValid():
            template_prim = stage.GetPrimAtPath(template_path)
        if template_prim is None or not template_prim.IsValid():
            return self._spawn_rescue_person_wrapper(
                new_name, local_xyz, yaw_deg=yaw_deg
            )

        edit_layer = self._get_usd_edit_layer()
        world_pos = self._local_to_world_vec3d(local_xyz)

        with Usd.EditContext(stage, edit_layer):
            if stage.GetPrimAtPath(wrapper_path).IsValid():
                stage.RemovePrim(wrapper_path)
            Sdf.CopySpec(
                edit_layer,
                Sdf.Path(template_path),
                edit_layer,
                Sdf.Path(wrapper_path),
            )

            wrapper = stage.GetPrimAtPath(wrapper_path)
            if not wrapper.IsValid():
                raise RuntimeError(f"Failed to clone rescue person to {wrapper_path}")

            xform = UsdGeom.Xformable(wrapper)
            for op in xform.GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                    op.Set(world_pos)
                elif op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ:
                    op.Set(Gf.Vec3f(0.0, 0.0, float(yaw_deg)))
            UsdGeom.Imageable(wrapper).MakeVisible()

        wrapper_prim = stage.GetPrimAtPath(wrapper_path)
        self._ensure_person_textures(wrapper_prim)
        self._disable_physics_under_prim(wrapper_prim)
        sx, sy, sz = self._person_wrapper_scale()
        h = self._person_bbox_height(wrapper_prim)
        print(
            f"[BrainNavEnv] Cloned {new_name} from {template_name} "
            f"scale=({sx:.3f},{sy:.3f},{sz:.3f}) at "
            f"({local_xyz[0]:.2f}, {local_xyz[1]:.2f}, {local_xyz[2]:.2f})"
            f"{f', height={h:.2f}m' if h else ''}\n"
        )
        return wrapper_prim

    def _hide_map_default_person(self) -> None:
        """Hide the map's embedded F_Business_02 (wrong default position/scale)."""
        from pxr import Usd, UsdGeom

        stage = self.sim.stage
        scope = getattr(self.cfg, "brain_rescue_person_scope", "RescuePersons")
        for path in (
            "/World/envs/env_0/Room/F_Business_02",
            "/World/envs/env_0/Room/RescuePerson_Final",
            "/World/envs/env_0/Room/RescuePerson_Final_Center",
        ):
            prim = stage.GetPrimAtPath(path)
            if prim.IsValid():
                self._set_prim_visibility(prim, visible=False)

        room = stage.GetPrimAtPath("/World/envs/env_0/Room")
        if room.IsValid():
            for prim in Usd.PrimRange(room):
                path = prim.GetPath().pathString
                name = prim.GetName().lower()
                if f"/{scope}/" in path or path.endswith(f"/{scope}"):
                    continue
                if "f_business" in name or "female_adult_business" in name:
                    self._set_prim_visibility(prim, visible=False)
                elif name == "f_business_02" and prim.IsA(UsdGeom.Xformable):
                    self._set_prim_visibility(prim, visible=False)

    def _person_bbox_height(self, prim) -> float | None:
        from pxr import UsdGeom

        if prim is None or not prim.IsValid():
            return None
        cache = UsdGeom.BBoxCache(0.0, ["default"])
        box = cache.ComputeWorldBound(prim).GetRange()
        size = box.GetMax() - box.GetMin()
        return float(size[2])

    def _setup_rescue_persons(self) -> None:
        """Three persons via wrapper Xforms: room 3, final room A, final room B."""
        if not getattr(self.cfg, "spawn_person", True):
            return
        try:
            self._hide_map_default_person()

            room3_name = getattr(self.cfg, "brain_room3_person_name", "RescuePerson_Room3")
            final_name = getattr(self.cfg, "brain_final_person_name", "RescuePerson_Final")
            center_name = getattr(
                self.cfg, "brain_final_person_center_name", "RescuePerson_Final_Center"
            )
            sx, sy, sz = self._person_wrapper_scale()

            room3 = getattr(self.cfg, "brain_room3_person_local", (0.0, -10.0, 0.0))
            room3_prim = self._spawn_rescue_person_wrapper(
                room3_name, tuple(room3), yaw_deg=90.0
            )
            self._room3_rescue_person_prim = room3_prim
            h3 = self._person_bbox_height(room3_prim)
            self._static_person_target_height = h3
            print(
                f"[BrainNavEnv] Room 3 person ({room3_prim.GetPath()}) "
                f"scale=({sx:.2f},{sy:.2f},{sz:.2f}) at "
                f"({room3[0]:.2f}, {room3[1]:.2f}, {room3[2]:.2f})"
                f"{f', height={h3:.2f}m' if h3 else ''}.\n"
            )

            final_local = getattr(self.cfg, "brain_final_person_local", (-4.0, -20.0, 0.0))
            final_prim = self._spawn_rescue_person_wrapper(
                final_name, tuple(final_local), yaw_deg=-90.0
            )
            self._final_rescue_person_prim = final_prim
            px, py, pz = float(final_local[0]), float(final_local[1]), float(final_local[2])
            self._finish_point_local = (px, py, 1.0)
            hf = self._person_bbox_height(final_prim)
            print(
                f"[BrainNavEnv] Final-room person A ({final_prim.GetPath()}) "
                f"scale=({sx:.2f},{sy:.2f},{sz:.2f}) at "
                f"({px:.2f}, {py:.2f}, {pz:.2f})"
                f"{f', height={hf:.2f}m' if hf else ''}.\n"
            )

            self._final_center_person_prim = None

            if hasattr(self, "_perception") and self._perception is not None:
                self._perception._rescue_person_slots = self._build_rescue_person_log_slots()
        except Exception as exc:
            print(f"[BrainNavEnv] Could not set up rescue persons: {exc}\n")
            import traceback
            traceback.print_exc()

    def _relocate_final_rescue_person(self) -> None:
        """Legacy alias — keeps room-3 + final-room persons in sync."""
        self._setup_rescue_persons()

    def _hold_scan_in_place(self, lock_local: np.ndarray) -> None:
        """Hold XYZ during 360° scan and apply a nose-down pitch so the camera
        sees people on the floor instead of pointing at the horizon."""
        import math

        env_ids = self._robot._ALL_INDICES
        origin = self._terrain.env_origins[env_ids]
        quat_w = self._robot.data.root_quat_w[env_ids]  # (N, 4) as (w,x,y,z)

        # Extract current yaw (rotation around Z in world frame)
        w, x, y, z = quat_w[:, 0], quat_w[:, 1], quat_w[:, 2], quat_w[:, 3]
        yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))  # (N,)

        # NOTE: do NOT sync self._target_yaw here — that kills the yaw error
        # the LLC needs to actually spin the drone. Yaw sync is done ONCE on
        # SCAN exit (see step()) to prevent snap.

        # Smooth pitch transition: ramp up from 0 to max_pitch at the start of scan, and ramp down to 0 at the end
        yaw_accum = float(getattr(self._brain, "scan_yaw_accumulated", 0.0))
        max_pitch_deg = float(getattr(self.cfg, "brain_scan_pitch_deg", 22.0))
        ramp_yaw = 0.5  # radians to ramp up/down (~28.6 degrees of rotation)

        if yaw_accum < ramp_yaw:
            factor = yaw_accum / ramp_yaw
        elif yaw_accum > 2.0 * math.pi - ramp_yaw:
            factor = max(0.0, (2.0 * math.pi - yaw_accum) / ramp_yaw)
        else:
            factor = 1.0

        p_rad = (max_pitch_deg * factor) * math.pi / 180.0
        cp2 = float(math.cos(p_rad / 2.0))
        sp2 = float(math.sin(p_rad / 2.0))
        cy2 = torch.cos(yaw / 2.0)
        sy2 = torch.sin(yaw / 2.0)

        new_w = cp2 * cy2
        new_x = -sp2 * sy2
        new_y = sp2 * cy2
        new_z = cp2 * sy2
        new_quat = torch.stack([new_w, new_x, new_y, new_z], dim=1)

        # Use actual current drone position to avoid teleporting/locking spatially, but lock Z to the entry height to prevent dropping/climbing
        pos = self._robot.data.root_pos_w[env_ids].clone()
        if not hasattr(self, "_scan_lock_z") or self._scan_lock_z is None:
            self._scan_lock_z = pos[:, 2].clone()
        pos[:, 2] = self._scan_lock_z

        pose = torch.cat([pos, new_quat], dim=1)
        self._robot.write_root_pose_to_sim(pose, env_ids)
        # Note: do NOT write zero velocity here to preserve physical flight and smooth transitions

    def _snap_drone_to_local(self, local_xyz: np.ndarray) -> None:
        """Snap drone to an exact SLAM checkpoint (env-local frame)."""
        env_ids = self._robot._ALL_INDICES
        state = self._robot.data.default_root_state[env_ids].clone()
        origin = self._terrain.env_origins[env_ids]
        state[:, 0] = float(local_xyz[0]) + origin[:, 0]
        state[:, 1] = float(local_xyz[1]) + origin[:, 1]
        state[:, 2] = float(local_xyz[2] if len(local_xyz) > 2 else 1.0) + origin[:, 2]
        state[:, 7:] = 0.0
        self._robot.write_root_pose_to_sim(state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(state[:, 7:], env_ids)

    def _verify_worker_person(self) -> None:
        """Confirm both rescue persons exist after placement."""
        if not getattr(self.cfg, "spawn_person", True):
            return
        try:
            room_prim = self.sim.stage.GetPrimAtPath("/World/envs/env_0/Room")
            if not room_prim.IsValid():
                print("[BrainNavEnv] WARNING: Room prim missing — cannot verify rescue persons.\n")
                return

            room3 = getattr(self, "_room3_rescue_person_prim", None)
            final = getattr(self, "_final_rescue_person_prim", None)
            if room3 is None:
                room3 = self._find_rescue_person_source_prim("/World/envs/env_0/Room")
            if final is None:
                final_name = getattr(self.cfg, "brain_final_person_name", "RescuePerson_Final").lower()
                from pxr import Usd

                for prim in Usd.PrimRange(room_prim):
                    if prim.GetName().lower() == final_name:
                        final = prim
                        break

            if room3 and room3.IsValid():
                r3 = getattr(self.cfg, "brain_room3_person_local", (0.0, -10.0, 0.0))
                h = self._person_bbox_height(room3)
                print(
                    f"[BrainNavEnv] Room 3 rescue person: {room3.GetPath()} "
                    f"target ({r3[0]:.2f}, {r3[1]:.2f}, {r3[2]:.2f})"
                    f"{f', height={h:.2f}m' if h else ''}.\n"
                )
            else:
                print("[BrainNavEnv] WARNING: Room 3 person prim not found.\n")

            if final and final.IsValid():
                finish = getattr(self, "_finish_point_local", None) or getattr(
                    self.cfg, "brain_final_person_local", (-4.0, -20.0, 0.0)
                )
                hf = self._person_bbox_height(final)
                print(
                    f"[BrainNavEnv] Final-room person A: {final.GetPath()} "
                    f"target ({finish[0]:.2f}, {finish[1]:.2f}, {finish[2]:.2f})"
                    f"{f', height={hf:.2f}m' if hf else ''}.\n"
                )
            else:
                print(
                    "[BrainNavEnv] WARNING: Final-room person A missing — "
                    "check _setup_rescue_persons / RescuePerson_Final reference.\n"
                )

            pass
        except Exception as exc:
            print(f"[BrainNavEnv] Could not verify rescue persons: {exc}\n")

    def _sync_map_geometry_from_usd(self) -> None:
        """Apply measured final_flat.usd dimensions to config and Brain spawn sequence."""
        raw = getattr(self, "_map_bounds_raw_local", None)
        if raw is not None:
            self.cfg.map_bounds = tuple(raw)
            print(
                f"[BrainNavEnv] map_bounds synced from USD: "
                f"X=[{raw[0]:.2f}, {raw[1]:.2f}] Y=[{raw[2]:.2f}, {raw[3]:.2f}]\n"
            )

        if not getattr(self.cfg, "brain_auto_room_spawns", True):
            seq, labels = self._build_multilevel_hybrid_sequence()
            self.cfg.brain_spawn_sequence = seq
            self._brain_spawn_labels = labels
            print("[BrainNavEnv] Multilevel + USD corridor spawn sequence:")
            for label, pt in zip(self._brain_spawn_labels, seq):
                print(f"  • {label}: ({pt[0]:.2f}, {pt[1]:.2f}, {pt[2]:.2f})")
            print()
            return

        seq, labels = self._build_sequential_spawn_sequence()
        if len(seq) < 2:
            print("[BrainNavEnv] WARNING: Could not parse map zones from USD — using cfg spawn sequence.\n")
            seq = tuple(getattr(self.cfg, "brain_spawn_sequence", ()))
            labels = [
                "finish (Worker)" if i == len(seq) - 1 else f"waypoint_{i + 1}"
                for i in range(len(seq))
            ]

        self.cfg.brain_spawn_sequence = seq
        self._brain_spawn_labels = labels
        print("[BrainNavEnv] Sequential spawn sequence from USD map zones:")
        for label, pt in zip(labels, seq):
            print(f"  • {label}: ({pt[0]:.2f}, {pt[1]:.2f}, {pt[2]:.2f})")
        print()

    def _capture_brain_mission(self):
        if not hasattr(self, "_brain"):
            return None
        # For SLAM brain, skip mission capture if people have been found (no need to re-rescue)
        found = getattr(self._brain, "found_person", False) or len(getattr(self._brain, "rescued_people", [])) > 0
        if found:
            return None
        if not getattr(self.cfg, "brain_preserve_mission_on_crash", True):
            return None
        snap = self._brain.capture_mission_snapshot()
        if snap is not None and getattr(self.cfg, "brain_crash_respawn_in_place", True):
            pos = self._robot.data.root_pos_w[0] - self._terrain.env_origins[0]
            snap["crash_local_xyz"] = (
                float(pos[0].item()),
                float(pos[1].item()),
                max(1.0, float(pos[2].item())),
            )
        return snap

    def _get_spawn1_local(self) -> tuple[float, float, float]:
        """Always return spawn1 from the configured sequence (env-local)."""
        seq = getattr(self.cfg, "brain_spawn_sequence", None)
        if seq and len(seq) > 0:
            return float(seq[0][0]), float(seq[0][1]), float(seq[0][2])
        return 0.0, 0.0, 1.0

    def _get_current_segment_spawn_local(self) -> tuple[float, float, float]:
        """Return the spawn point for the current SLAM segment (env-local)."""
        seq = getattr(self.cfg, "brain_spawn_sequence", None)
        if seq and hasattr(self, "_brain"):
            idx = min(max(int(self._brain.segment_idx), 0), len(seq) - 1)
            return float(seq[idx][0]), float(seq[idx][1]), float(seq[idx][2])
        return self._get_spawn1_local()

    def _is_local_xy_walkable(self, x: float, y: float) -> bool:
        if not hasattr(self, "_is_on_navigable_floor"):
            return True
        tx = torch.tensor([x], device=self.device, dtype=torch.float32)
        ty = torch.tensor([y], device=self.device, dtype=torch.float32)
        if not bool(self._is_on_navigable_floor(tx, ty).item()):
            return False
        if hasattr(self, "_is_inside_map_obstacle"):
            if bool(self._is_inside_map_obstacle(tx, ty).item()):
                return False
        return True

    def _find_rescue_person_source_prim(self, root_path: str = "/World/envs/env_0"):
        """Return the room-3 rescue wrapper prim (not the hidden map default)."""
        from pxr import Usd, UsdGeom

        room3_name = getattr(self.cfg, "brain_room3_person_name", "RescuePerson_Room3").lower()
        scope = getattr(self.cfg, "brain_rescue_person_scope", "RescuePersons").lower()
        root = self.sim.stage.GetPrimAtPath(root_path)
        if not root.IsValid():
            return None
        for prim in Usd.PrimRange(root):
            path = prim.GetPath().pathString.lower()
            name = prim.GetName().lower()
            if scope in path and name == room3_name and prim.IsA(UsdGeom.Xformable):
                return prim
            if name in ("rescueperson_room3", "rescueperson_final", "rescueperson_final_center"):
                if prim.IsA(UsdGeom.Xformable):
                    return prim
        return self._find_rescue_person_prim(root_path)

    def _find_rescue_person_prim(self, root_path: str = "/World/envs/env_0"):
        """Locate the human character prim in the loaded map USD (e.g. F_Business_02)."""
        from pxr import Usd, UsdGeom

        keywords = (
            "worker", "character", "person", "human", "reallusion", "cc_base",
            "business", "female_adult", "male_adult", "female", "male",
        )
        skip = ("looks", "shader", "material", "skin", "hair", "cloth", "eye", "teeth", "tongue")
        root = self.sim.stage.GetPrimAtPath(root_path)
        if not root.IsValid():
            return None
        candidates = []
        for prim in Usd.PrimRange(root):
            name = prim.GetName().lower()
            path = prim.GetPath().pathString.lower()
            if any(s in name for s in skip):
                continue
            if not any(k in name or k in path for k in keywords):
                continue
            if not prim.IsA(UsdGeom.Xformable):
                continue
            candidates.append(prim)
        if not candidates:
            return None
        for prim in candidates:
            n = prim.GetName().lower()
            if "business" in n or n == "worker":
                return prim
        return min(candidates, key=lambda p: len(p.GetPath().pathString.split("/")))

    def _get_safe_checkpoint_local(self, segment_idx: int) -> tuple[float, float, float]:
        seq = getattr(self.cfg, "brain_spawn_sequence", ())
        if seq and 0 <= segment_idx < len(seq):
            p = seq[segment_idx]
            return float(p[0]), float(p[1]), float(p[2] if len(p) > 2 else 1.0)
        return self._get_spawn1_local()

    def _get_crash_respawn_local(self) -> tuple[float, float, float]:
        """Crash respawn XY — use room-4 entrance during corridor/final transit (matches training spawn)."""
        seg = int(self._brain.segment_idx) if hasattr(self, "_brain") else 0
        seq = getattr(self.cfg, "brain_spawn_sequence", ())
        if (
            getattr(self.cfg, "brain_corridor_crash_respawn_at_room4", True)
            and seq
            and seg >= 3
            and len(seq) > 3
        ):
            cp = seq[3]
            return float(cp[0]), float(cp[1]), float(cp[2] if len(cp) > 2 else 1.0)
        return self._get_current_segment_spawn_local()

    def _sample_brain_spawn_xyz(
        self,
        env_count: int,
        crash_local: tuple[float, float, float] | None = None,
        force_checkpoint: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Spawn at crash position, safe checkpoint, or segment entrance."""
        device = self.device
        use_in_place = (
            not force_checkpoint
            and crash_local is not None
            and getattr(self.cfg, "brain_crash_respawn_in_place", True)
            and hasattr(self, "_brain")
            and self._brain.state == "GOTO_WAYPOINT"
            and self._is_local_xy_walkable(float(crash_local[0]), float(crash_local[1]))
        )
        if use_in_place:
            sx, sy, sz = float(crash_local[0]), float(crash_local[1]), float(crash_local[2])
            spawn_x = torch.full((env_count,), sx, device=device)
            spawn_y = torch.full((env_count,), sy, device=device)
            spawn_z = torch.full((env_count,), sz, device=device)
            return spawn_x, spawn_y, spawn_z
        if getattr(self.cfg, "brain_use_sequential_spawns", False):
            if getattr(self.cfg, "brain_preserve_mission_on_crash", True) and hasattr(self, "_brain"):
                sx, sy, sz = self._get_crash_respawn_local()
            else:
                sx, sy, sz = self._get_spawn1_local()
            spawn_x = torch.full((env_count,), sx, device=device)
            spawn_y = torch.full((env_count,), sy, device=device)
            spawn_z = torch.full((env_count,), sz, device=device)
            return spawn_x, spawn_y, spawn_z
        sx, sy = self._sample_navigable_spawn_xy(env_count)
        spawn_z = torch.full((env_count,), 1.0, device=device)
        return sx, sy, spawn_z

    def _apply_brain_spawn_and_goal(self, env_ids, mission_snapshot=None):
        """Place drone on navigable floor and set PPO target from the Brain SLAM state."""
        from isaaclab.utils.math import euler_xyz_from_quat

        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        env_count = env_ids.shape[0]

        crash_local = None
        force_checkpoint = False
        if mission_snapshot is not None:
            crash_local = mission_snapshot.get("crash_local_xyz")
            seg = int(mission_snapshot.get("segment_idx", 0))
            crash_n = self._segment_crash_counts.get(seg, 0)
            max_in_place = int(getattr(self.cfg, "brain_crash_max_in_place", 2))
            if crash_n >= max_in_place:
                force_checkpoint = True
                sx, sy, sz = self._get_safe_checkpoint_local(seg)
                crash_local = (sx, sy, sz)

        if mission_snapshot is None:
            if hasattr(self, "_brain") and getattr(self.cfg, "brain_use_sequential_spawns", False):
                self._brain.reset_mission_from_start()

        spawn_x, spawn_y, spawn_z = self._sample_brain_spawn_xyz(
            env_count, crash_local=crash_local, force_checkpoint=force_checkpoint
        )

        default_root_state = self._robot.data.default_root_state[env_ids].clone()
        default_root_state[:, 0] = spawn_x + self._terrain.env_origins[env_ids, 0]
        default_root_state[:, 1] = spawn_y + self._terrain.env_origins[env_ids, 1]
        default_root_state[:, 2] = spawn_z
        default_root_state[:, 7:] = 0.0

        if hasattr(self, "_brain"):
            if mission_snapshot is not None:
                self._brain.restore_mission_snapshot(mission_snapshot)
                seg = int(self._brain.segment_idx)
                self._segment_crash_counts[seg] = self._segment_crash_counts.get(seg, 0) + 1
                skip_after = int(getattr(self.cfg, "brain_corridor_crash_skip_after", 3))
                corridor_from = int(getattr(self.cfg, "brain_no_snap_from_segment", 4))
                if (
                    self._segment_crash_counts[seg] >= skip_after
                    and seg >= corridor_from
                    and self._brain.force_skip_to_next_checkpoint()
                ):
                    self._segment_crash_counts[seg] = 0
                elif self._segment_crash_counts[seg] >= int(
                    getattr(self.cfg, "brain_crash_max_in_place", 2)
                ):
                    self._brain.resync_nav_target_from_sequence()
                self._brain.prepare_crash_respawn()
            elif not hasattr(self, "_brain_spawn_initialized"):
                self._brain_spawn_initialized = True

        goal_local = np.zeros((env_count, 3), dtype=np.float32)
        for i in range(env_count):
            if hasattr(self, "_brain"):
                gl = self._brain.get_brain_goal_local(
                    drone_local_xy=(float(spawn_x[i].item()), float(spawn_y[i].item()))
                )
                goal_local[i] = gl
            else:
                goal_local[i] = (float(spawn_x[i].item()), float(spawn_y[i].item()), 1.0)

        self._desired_pos_w[env_ids, 0] = (
            torch.tensor(goal_local[:, 0], device=self.device) + self._terrain.env_origins[env_ids, 0]
        )
        self._desired_pos_w[env_ids, 1] = (
            torch.tensor(goal_local[:, 1], device=self.device) + self._terrain.env_origins[env_ids, 1]
        )
        self._desired_pos_w[env_ids, 2] = torch.tensor(goal_local[:, 2], device=self.device)

        # Fixed orientation from cf2x / Multilevel training (NOT facing goal at spawn)
        default_root_state[:, 3] = 0.7071
        default_root_state[:, 4] = 0.0
        default_root_state[:, 5] = 0.0
        default_root_state[:, 6] = -0.7071
        _, _, spawn_yaw = euler_xyz_from_quat(default_root_state[:, 3:7])
        self._target_yaw[env_ids] = spawn_yaw

        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)

        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        self._prev_dist_to_goal[env_ids] = torch.linalg.norm(
            self._desired_pos_w[env_ids] - default_root_state[:, :3], dim=1
        )

        sx, sy = float(spawn_x[0].item()), float(spawn_y[0].item())
        sz = float(spawn_z[0].item())
        gx, gy = float(goal_local[0, 0]), float(goal_local[0, 1])
        state = getattr(self._brain, "state", "?") if hasattr(self, "_brain") else "?"
        seg = getattr(self._brain, "segment_idx", "?") if hasattr(self, "_brain") else "?"
        seg_label = self._brain.get_segment_label(int(seg)) if hasattr(self, "_brain") and isinstance(seg, int) else seg
        print(
            f"[BrainNavEnv] Respawned at ({sx:.2f}, {sy:.2f}, {sz:.2f}) | "
            f"nav target ({gx:.2f}, {gy:.2f}) | state={state} | {seg_label}"
            f"{' (in-place crash recovery)' if crash_local and not force_checkpoint else ''}"
            f"{' (safe checkpoint warp)' if force_checkpoint else ''}\n"
        )

    def _reset_idx(self, env_ids: torch.Tensor | None = None):
        """Crash/episode reset: respawn and preserve SLAM progress when configured."""
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        mission = self._capture_brain_mission()
        if (
            getattr(self.cfg, "brain_use_sequential_spawns", False)
            and hasattr(self, "_brain")
            and not getattr(self.cfg, "brain_preserve_mission_on_crash", True)
        ):
            self._brain.reset_mission_from_start()
            mission = None

        super()._reset_idx(env_ids)
        self._apply_brain_spawn_and_goal(env_ids, mission_snapshot=mission)
        self._stuck_step_count = 0
        if env_ids.shape[0] > 0:
            self._prev_drone_pos_xy = self._robot.data.root_pos_w[0, :2].clone()
            self._randomize_obstacles(env_ids)
        else:
            self._prev_drone_pos_xy = None

    def _get_mission_obstacle_level(self) -> int:
        """Map SLAM segment index to Multilevel level for obstacle placement (0–3)."""
        if not hasattr(self, "_brain"):
            return 0
        return min(max(int(self._brain.segment_idx), 0), 3)

    def _randomize_obstacles(self, env_ids: torch.Tensor):
        """Randomize poles + room-3/room-4 obstacles based on current mission segment.

        In real SLAM patrol the whole house is populated ONCE and then left static:
        re-shuffling obstacles per room (or hiding them during a scan) left phantom
        walls in the occupancy map at the old positions and made obstacles flicker.
        """
        real_slam = getattr(self.cfg, "brain_real_slam_mode", False)
        if real_slam and getattr(self, "_real_slam_obstacles_placed", False):
            # House already placed — keep it fixed so the SLAM map stays consistent.
            return

        num_resets = env_ids.shape[0]
        env_origins = self._terrain.env_origins[env_ids]
        mission_level = self._get_mission_obstacle_level()
        levels = torch.full((num_resets,), mission_level, dtype=torch.long, device=self.device)
        hide_for_scan = (
            not real_slam
            and getattr(self.cfg, "brain_hide_obstacles_during_scan", True)
            and hasattr(self, "_brain")
            and self._brain.state == "SCAN"
        )

        # ── Randomize Poles Positions ────────────────────────────────
        for i, pole in enumerate(self._poles):
            state = pole.data.default_root_state[env_ids].clone()
            
            pole_x = torch.zeros(num_resets, device=self.device)
            pole_y = torch.zeros(num_resets, device=self.device)
            pole_z = torch.full((num_resets,), -100.0, device=self.device)
            
            rand_x = torch.zeros(num_resets, device=self.device).uniform_(-1.7, 1.7)
            
            if i < 6:
                # Level 1 (y=0)
                pole_z[:] = 1.0
                pole_x[:] = rand_x[:]
                pole_y[:] = 0.0
            elif i < 11:
                # Level 2 row 1 (y=-4.0)
                pole_z[:] = 1.0
                pole_x[:] = rand_x[:]
                pole_y[:] = -4.0
            elif i < 16:
                # Level 2 row 2 (y=-5.5)
                pole_z[:] = 1.0
                pole_x[:] = rand_x[:]
                pole_y[:] = -5.5
            elif i < 21:
                # Level 2 row 3 (y=-7.0)
                pole_z[:] = 1.0
                pole_x[:] = rand_x[:]
                pole_y[:] = -7.0
            
            state[:, 0] = pole_x + env_origins[:, 0]
            state[:, 1] = pole_y + env_origins[:, 1]
            state[:, 2] = pole_z + env_origins[:, 2]
            state[:, 3] = 1.0
            state[:, 4:7] = 0.0
            state[:, 7:] = 0.0
            pole.write_root_pose_to_sim(state[:, :7], env_ids)
            pole.write_root_velocity_to_sim(state[:, 7:], env_ids)

        # ── Randomize Room 3 Obstacles ────────────────────────────────
        # 12 grid cells (4 rows of Y, 3 columns of X) - Compressed to center (X in [-1.5, 1.5], Y in [-14.0, -11.0])
        grid_x = torch.tensor([-1.5, 0.0, 1.5], device=self.device)
        grid_y = torch.tensor([-11.0, -12.0, -13.0, -14.0], device=self.device)
        gy, gx = torch.meshgrid(grid_y, grid_x, indexing='ij')
        grid_positions = torch.stack([gx.flatten(), gy.flatten()], dim=-1)  # (12, 2)

        # Never spawn dynamic obstacles on top of a rescue person (breaks YOLO silhouette).
        person_xy = torch.tensor(
            [
                getattr(self.cfg, "brain_room3_person_local", (0.0, -10.0, 0.0))[:2],
                getattr(self.cfg, "brain_final_person_local", (-4.0, -20.0, 0.0))[:2],
                getattr(self.cfg, "brain_final_person_center_local", (-6.0, -21.5, 0.0))[:2],
            ],
            device=self.device,
            dtype=torch.float32,
        )
        keep_cells = torch.ones(grid_positions.shape[0], dtype=torch.bool, device=self.device)
        excl_r = float(getattr(self.cfg, "brain_person_obstacle_exclusion_m", 2.5))
        for i in range(grid_positions.shape[0]):
            dists = torch.linalg.norm(grid_positions[i] - person_xy, dim=1)
            if torch.any(dists < excl_r):
                keep_cells[i] = False
        grid_positions = grid_positions[keep_cells]

        if real_slam:
            # Static house: room-3 props only (room-4 corridor props appear when exploring).
            is_level3 = torch.ones(num_resets, dtype=torch.bool, device=self.device)
            is_level4 = torch.zeros(num_resets, dtype=torch.bool, device=self.device)
            # Fixed, readable layout — four props near the room corners, away from the person.
            slam_room3_slots = torch.tensor(
                [[-1.2, -12.0], [1.2, -12.0], [-1.2, -13.5], [1.2, -13.5]],
                device=self.device,
                dtype=torch.float32,
            )
        else:
            is_level3 = levels == 2
            is_level4 = levels == 3
            slam_room3_slots = None
        if hide_for_scan:
            is_level3 = torch.zeros_like(is_level3)
            is_level4 = torch.zeros_like(is_level4)

        num_level3_resets = torch.count_nonzero(is_level3).item()
        slam_room3_cap = int(getattr(self.cfg, "brain_slam_room3_max_obstacles", 4)) if real_slam else len(self._room3_obstacles)

        if num_level3_resets > 0 and not real_slam:
            n_grid = grid_positions.shape[0]
            n_room3 = len(self._room3_obstacles)
            perms = torch.stack([torch.randperm(n_grid, device=self.device) for _ in range(num_level3_resets)])
            import math
            rand_yaws = torch.zeros(num_level3_resets, n_room3, device=self.device).uniform_(0, 2 * math.pi)
            
        for j, obstacle in enumerate(self._room3_obstacles):
            state = obstacle.data.default_root_state[env_ids].clone()
            
            # Default: hide it under the ground
            obs_x = torch.zeros(num_resets, device=self.device)
            obs_y = torch.zeros(num_resets, device=self.device)
            obs_z = torch.full((num_resets,), -100.0, device=self.device)
            obs_qw = torch.ones(num_resets, device=self.device)
            obs_qz = torch.zeros(num_resets, device=self.device)
            
            if real_slam and num_level3_resets > 0 and j < slam_room3_cap and slam_room3_slots is not None:
                slot = slam_room3_slots[min(j, slam_room3_slots.shape[0] - 1)]
                obs_x[is_level3] = slot[0]
                obs_y[is_level3] = slot[1]
                obs_z[is_level3] = 0.0
            elif num_level3_resets > 0 and j < grid_positions.shape[0]:
                # Get the assigned cell index for this obstacle in each Level 3 env
                assigned_cell_indices = perms[:, j]  # (num_level3_resets,)
                assigned_positions = grid_positions[assigned_cell_indices]  # (num_level3_resets, 2)
                
                # Add small random noise (±0.3m in X and Y)
                noise_x = torch.zeros(num_level3_resets, device=self.device).uniform_(-0.3, 0.3)
                noise_y = torch.zeros(num_level3_resets, device=self.device).uniform_(-0.3, 0.3)
                
                obs_x[is_level3] = assigned_positions[:, 0] + noise_x
                obs_y[is_level3] = assigned_positions[:, 1] + noise_y
                obs_z[is_level3] = 0.0  # Lowered by 1m (was 1.0)
                
                # Apply random yaw
                yaw = rand_yaws[:, j]
                obs_qw[is_level3] = torch.cos(yaw / 2.0)
                obs_qz[is_level3] = torch.sin(yaw / 2.0)
                
            state[:, 0] = obs_x + env_origins[:, 0]
            state[:, 1] = obs_y + env_origins[:, 1]
            state[:, 2] = obs_z + env_origins[:, 2]
            state[:, 3] = obs_qw
            state[:, 4] = 0.0
            state[:, 5] = 0.0
            state[:, 6] = obs_qz
            state[:, 7:] = 0.0
            
            obstacle.write_root_pose_to_sim(state[:, :7], env_ids)
            obstacle.write_root_velocity_to_sim(state[:, 7:], env_ids)

        # ── Randomize Room 4 Obstacles ────────────────────────────────
        num_level4_resets = torch.count_nonzero(is_level4).item()

        if num_level4_resets > 0:
            n_c1 = max(len(self._corr1_obstacles), 1)
            n_c2 = max(len(self._corr2_obstacles), 1)
            n_slots = max(n_c1, n_c2, 3)
            # Generate random permutations of corridor slots for each env
            perms_c1 = torch.stack([torch.randperm(n_slots, device=self.device) for _ in range(num_level4_resets)])
            perms_c2 = torch.stack([torch.randperm(n_slots, device=self.device) for _ in range(num_level4_resets)])
            
            perms_h1 = torch.stack([torch.randperm(n_slots, device=self.device) for _ in range(num_level4_resets)])
            perms_h2 = torch.stack([torch.randperm(n_slots, device=self.device) for _ in range(num_level4_resets)])

            y_positions_c1 = torch.linspace(-17.2, -19.45, n_slots, device=self.device)
            x_positions_c2 = torch.linspace(-3.8, -0.65, n_slots, device=self.device)
            z_positions = torch.linspace(0.4, 1.6, n_slots, device=self.device)

        # 1. Room 4.1 Corridor Obstacles (corr1)
        for j, obstacle in enumerate(self._corr1_obstacles):
            state = obstacle.data.default_root_state[env_ids].clone()
            
            obs_x = torch.zeros(num_resets, device=self.device)
            obs_y = torch.zeros(num_resets, device=self.device)
            obs_z = torch.full((num_resets,), -100.0, device=self.device)
            
            if num_level4_resets > 0:
                # All 5 obstacles are active
                assigned_y = y_positions_c1[perms_c1[:, j]]
                noise_y = torch.zeros(num_level4_resets, device=self.device).uniform_(-0.05, 0.05)
                
                assigned_z = z_positions[perms_h1[:, j]]
                noise_z = torch.zeros(num_level4_resets, device=self.device).uniform_(-0.05, 0.05)
                
                obs_x[is_level4] = 0.0
                obs_y[is_level4] = assigned_y + noise_y
                obs_z[is_level4] = (assigned_z + noise_z).clamp(0.4, 1.6)

            state[:, 0] = obs_x + env_origins[:, 0]
            state[:, 1] = obs_y + env_origins[:, 1]
            state[:, 2] = obs_z + env_origins[:, 2]
            state[:, 3] = 1.0
            state[:, 4:7] = 0.0
            state[:, 7:] = 0.0
            obstacle.write_root_pose_to_sim(state[:, :7], env_ids)
            obstacle.write_root_velocity_to_sim(state[:, 7:], env_ids)

        # 2. Room 4.2 Corridor Obstacles (corr2)
        for j, obstacle in enumerate(self._corr2_obstacles):
            state = obstacle.data.default_root_state[env_ids].clone()
            
            obs_x = torch.zeros(num_resets, device=self.device)
            obs_y = torch.zeros(num_resets, device=self.device)
            obs_z = torch.full((num_resets,), -100.0, device=self.device)
            
            if num_level4_resets > 0:
                # All 5 obstacles are active
                assigned_x = x_positions_c2[perms_c2[:, j]]
                noise_x = torch.zeros(num_level4_resets, device=self.device).uniform_(-0.05, 0.05)
                
                assigned_z = z_positions[perms_h2[:, j]]
                noise_z = torch.zeros(num_level4_resets, device=self.device).uniform_(-0.05, 0.05)
                
                obs_x[is_level4] = assigned_x + noise_x
                obs_y[is_level4] = -20.5
                obs_z[is_level4] = (assigned_z + noise_z).clamp(0.4, 1.6)

            state[:, 0] = obs_x + env_origins[:, 0]
            state[:, 1] = obs_y + env_origins[:, 1]
            state[:, 2] = obs_z + env_origins[:, 2]
            state[:, 3] = 1.0
            state[:, 4:7] = 0.0
            state[:, 7:] = 0.0
            obstacle.write_root_pose_to_sim(state[:, :7], env_ids)
            obstacle.write_root_velocity_to_sim(state[:, 7:], env_ids)

        if real_slam:
            # Mark the house as placed so future resets/segment changes leave it static.
            self._real_slam_obstacles_placed = True

    def _preprocess_depth(self) -> torch.Tensor:
        """Normalize depth for AE; downsample 512×288 brain-play camera to 72×128."""
        depth = super()._preprocess_depth()
        ae_h, ae_w = 72, 128
        if depth.shape[-2] != ae_h or depth.shape[-1] != ae_w:
            depth = torch.nn.functional.interpolate(
                depth, size=(ae_h, ae_w), mode="bilinear", align_corners=False
            )
        # Parent stores full-res depth; keep AE/dashboard buffer at 72×128.
        self._last_depth_processed = depth
        return depth

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Multilevel-style termination: contact + height bounds only (no LiDAR)."""
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        if getattr(self.cfg, "brain_disable_episode_timeout", True):
            time_out = torch.zeros_like(time_out)

        contact_forces = self._contact_sensor.data.net_forces_w
        threshold = getattr(self.cfg, "contact_force_threshold", 0.0001)
        hit_obstacle = contact_forces.norm(dim=-1).max(dim=-1).values > threshold
        # Suppress false crashes during and shortly after 360° SCAN
        # (position-hold writes cause brief residual contact forces on exit)
        in_scan = hasattr(self, "_brain") and self._brain.state == "SCAN"
        post_scan_grace = getattr(self, "_steps_since_last_scan", 100) < 10
        if in_scan or post_scan_grace:
            hit_obstacle = torch.zeros_like(hit_obstacle)

        relative_z = self._robot.data.root_pos_w[:, 2] - self._terrain.env_origins[:, 2]
        hit_bounds = (relative_z < 0.1) | (relative_z > 1.9)
        died = hit_obstacle | hit_bounds

        if self.num_envs > 0:
            if hit_bounds[0].item() and relative_z[0].item() < 0.1:
                self._env0_crash_reason = "Floor Bounds Collision"
            elif hit_bounds[0].item():
                self._env0_crash_reason = "Ceiling Bounds Collision"
            elif hit_obstacle[0].item():
                self._env0_crash_reason = "Obstacle Collision"
            else:
                self._env0_crash_reason = None

        self._last_died = died
        if getattr(self.cfg, "brain_reset_on_crash", True):
            terminated = died
        else:
            terminated = torch.zeros_like(died)
        return terminated, time_out

    def _get_observations(self) -> dict:
        """Build 77-dim flat observation vector for model_1450 (WandB bdo85ahx).

        Layout (Multilevel_Train / model_1450.pt actor mlp.0: [128, 77]):
          z_img(64) + desired_pos_b(3) + target_dist(1) + lin_vel(3) + ang_vel(3) + gravity(3)
        """
        depth = self._preprocess_depth()
        z_img = self.ae.encode_detached(depth)  # (B, 64)

        desired_pos_b, _ = subtract_frame_transforms(
            self._robot.data.root_pos_w, self._robot.data.root_quat_w, self._desired_pos_w
        )
        target_dist = torch.linalg.norm(
            self._desired_pos_w - self._robot.data.root_pos_w, dim=1, keepdim=True
        )  # (B, 1)

        obs = torch.cat(
            [
                z_img,
                desired_pos_b,
                target_dist,
                self._robot.data.root_lin_vel_b,
                self._robot.data.root_ang_vel_b,
                self._robot.data.projected_gravity_b,
            ],
            dim=-1,
        )  # 64 + 3 + 1 + 3 + 3 + 3 = 77

        if self._is_play_script() and self.cfg.debug_vis:
            self._update_dashboard(depth)

        return {"policy": obs}

    def _update_dashboard(self, depth):
        """Override to prevent dashboard plotting from expecting LiDAR data."""
        pass

    def _verify_navigator_policy(self) -> None:
        """Fail fast if checkpoint input size does not match the 77-dim observation layout."""
        expected = int(self.cfg.observation_space)
        obs_dict = self._get_observations()
        policy_obs = obs_dict["policy"]
        actual = int(policy_obs.shape[-1])
        if actual != expected:
            raise RuntimeError(
                f"[BrainNavEnv] Observation size mismatch: built {actual}-dim tensor, "
                f"cfg.observation_space={expected}. Check ae_latent_dim and _get_observations()."
            )

        with torch.inference_mode():
            if self._navigator_policy_expects_dict:
                actions = self._navigator_policy(obs_dict)
            else:
                actions = self._navigator_policy(policy_obs)
        if actions.shape[-1] != self.cfg.action_space:
            raise RuntimeError(
                f"[BrainNavEnv] Navigator action size mismatch: got {actions.shape[-1]}, "
                f"expected {self.cfg.action_space}."
            )
        print(
            f"[BrainNavEnv] Navigator verified: obs={actual}-dim, actions={actions.shape[-1]}-dim\n"
        )

    def _load_navigator_policy(self):
        """Load the pretrained PPO navigator policy as a frozen inference module."""
        checkpoint_path = resolve_navigator_checkpoint(self.cfg.navigator_checkpoint_path)
        self.cfg.navigator_checkpoint_path = checkpoint_path

        checkpoint_name = os.path.basename(checkpoint_path)
        is_rsl_checkpoint = checkpoint_name.startswith("model_") and checkpoint_name.endswith(".pt")

        # Try JIT only for exported/policy.pt — not for RSL-RL model_*.pt checkpoints.
        jit_error_msg = ""
        if not is_rsl_checkpoint:
            try:
                policy_dir = os.path.dirname(checkpoint_path)
                jit_policy_path = os.path.join(policy_dir, "exported", "policy.pt")

                if os.path.exists(jit_policy_path):
                    print(f"\n[BrainNavEnv] Loading JIT navigator policy: {jit_policy_path}\n")
                    self._navigator_policy = torch.jit.load(jit_policy_path, map_location=self.device)
                    self._navigator_policy.eval()
                    self._navigator_policy_expects_dict = False
                    self._verify_navigator_policy()
                    return

                if checkpoint_path.endswith("policy.pt"):
                    print(f"\n[BrainNavEnv] Loading JIT navigator policy: {checkpoint_path}\n")
                    self._navigator_policy = torch.jit.load(checkpoint_path, map_location=self.device)
                    self._navigator_policy.eval()
                    self._navigator_policy_expects_dict = False
                    self._verify_navigator_policy()
                    return

                raise FileNotFoundError(f"No exported JIT policy found at: {jit_policy_path}")

            except Exception as jit_err:
                jit_error_msg = str(jit_err)
                print(f"\n[BrainNavEnv] JIT loading failed ({jit_err}). Trying RSL-RL Runner...\n")

        # Load RSL-RL checkpoint (model_1450.pt)
        try:
            from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
            from rsl_rl.runners import OnPolicyRunner

            agent_cfg = load_cfg_from_registry("Brain-Nav-Drone-Direct-v0", "rsl_rl_cfg_entry_point")
            agent_cfg.device = self.sim.cfg.device

            agent_dict = agent_cfg.to_dict()
            for model_key in ["actor", "critic"]:
                if model_key in agent_dict:
                    agent_dict[model_key].pop("stochastic", None)
                    agent_dict[model_key].pop("init_noise_std", None)
                    agent_dict[model_key].pop("noise_std_type", None)
                    agent_dict[model_key].pop("state_dependent_std", None)

            temp_gym_env = _RslRlCompatWrapper(self)
            runner = OnPolicyRunner(temp_gym_env, agent_dict, log_dir=None, device=agent_cfg.device)
            runner.load(checkpoint_path)
            self._navigator_policy = runner.get_inference_policy(device=self.device)
            self._navigator_policy_expects_dict = True
            # Expose actor for dashboard policy-saliency (play_saliency.py parity)
            self._navigator_actor = runner.alg.actor
            self._navigator_actor.eval()
            print(f"\n[BrainNavEnv] Loaded navigator via RSL-RL Runner: {checkpoint_path}\n")
            self._verify_navigator_policy()

        except Exception as runner_err:
            raise RuntimeError(
                f"\n[BrainNavEnv] Failed to load navigator policy from: {checkpoint_path}\n"
                f"  Expected 77-dim obs (model_1450 / Multilevel_Train bdo85ahx).\n"
                f"  JIT error: {jit_error_msg}\n"
                f"  Runner error: {runner_err}\n"
            )

    def step(self, action):
        """Override step to run the internal Brain → PPO pipeline.

        The external `action` argument is ignored — the Brain module
        determines all high-level commands, and the frozen PPO policy
        generates the low-level navigation actions.
        """
        # Dynamic material override binding (runs once USD references are fully loaded in background)
        if not getattr(self, "_person_materials_bound", False) and getattr(self.cfg, "brain_person_override_textures", False):
            from pxr import Usd, UsdGeom
            bound_count = 0
            if hasattr(self, "_room3_rescue_person_prim") and self._room3_rescue_person_prim is not None:
                self._ensure_person_textures(self._room3_rescue_person_prim)
                for prim in Usd.PrimRange(self._room3_rescue_person_prim):
                    if prim.IsA(UsdGeom.Mesh):
                        bound_count += 1
            if hasattr(self, "_final_rescue_person_prim") and self._final_rescue_person_prim is not None:
                self._ensure_person_textures(self._final_rescue_person_prim)
                for prim in Usd.PrimRange(self._final_rescue_person_prim):
                    if prim.IsA(UsdGeom.Mesh):
                        bound_count += 1
            if bound_count > 0:
                self._person_materials_bound = True
                self._refresh_static_person_target_height()
                print(f"\n[BrainNavEnv] Async USD references loaded! Bound {bound_count} meshes successfully with custom textures.\n")

        self._process_pending_person_scale_fixes()

        if hasattr(self, "_brain") and self._brain.state == "SCAN":
            self._steps_since_last_scan = 0
        else:
            self._steps_since_last_scan = getattr(self, "_steps_since_last_scan", 100) + 1
        with torch.inference_mode():
            # 1. Grab camera outputs and drone state
            rgb_image = self._tiled_camera.data.output["rgb"].clone()
            depth_image = self._tiled_camera.data.output["depth"].clone()
            drone_pos = self._robot.data.root_pos_w.clone()
            drone_quat = self._robot.data.root_quat_w.clone()
            # Replace infinity values in depth
            depth_image[depth_image == float("inf")] = 10.0

            # Side view cameras for YOLO
            rgb_left = None
            depth_left = None
            if self._view_left_camera is not None and self._view_left_camera.data.output is not None:
                rgb_left = self._view_left_camera.data.output.get("rgb")
                depth_left = self._view_left_camera.data.output.get("depth")
                if rgb_left is not None:
                    rgb_left = rgb_left.clone()
                if depth_left is not None:
                    depth_left = depth_left.clone()
                    depth_left[depth_left == float("inf")] = 10.0

            rgb_right = None
            depth_right = None
            if self._view_right_camera is not None and self._view_right_camera.data.output is not None:
                rgb_right = self._view_right_camera.data.output.get("rgb")
                depth_right = self._view_right_camera.data.output.get("depth")
                if rgb_right is not None:
                    rgb_right = rgb_right.clone()
                if depth_right is not None:
                    depth_right = depth_right.clone()
                    depth_right[depth_right == float("inf")] = 10.0

            # 2. Run Perception (YOLO + de-projection) — always run during SCAN spin so we don't miss a person
            run_yolo = (self._timestep % max(1, self.cfg.brain_yolo_interval)) == 0
            if run_yolo:
                defer_seg = int(getattr(self.cfg, "brain_rescue_min_segment", 3))
                if getattr(self.cfg, "brain_real_slam_mode", False):
                    rescue_armed = True
                else:
                    rescue_armed = (
                        not getattr(self.cfg, "brain_use_sequential_spawns", False)
                        or self._brain.segment_idx >= defer_seg
                    )
                person_found, person_world_xyz = self._perception.process_camera_data(
                    rgb_image,
                    depth_image,
                    drone_pos,
                    drone_quat,
                    rescue_armed=rescue_armed,
                    scan_label=(
                        self._brain.get_segment_label()
                        if self._brain.state == "SCAN"
                        else None
                    ),
                    rgb_left=None if getattr(self.cfg, "yolo_front_camera_only", False) else rgb_left,
                    depth_left=None if getattr(self.cfg, "yolo_front_camera_only", False) else depth_left,
                    rgb_right=None if getattr(self.cfg, "yolo_front_camera_only", False) else rgb_right,
                    depth_right=None if getattr(self.cfg, "yolo_front_camera_only", False) else depth_right,
                )
                self._last_person_found = person_found
                self._last_person_world_xyz = person_world_xyz
                if getattr(self._perception, "person_ever_detected", False):
                    self._brain.person_noted_anywhere = True
                if person_found.any():
                    p = person_world_xyz[0].cpu().numpy()
                    print(
                        f"[BrainNavEnv] YOLO person detected at world "
                        f"({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f}) | brain state={self._brain.state}"
                    )
            else:
                person_found = self._last_person_found
                person_world_xyz = self._last_person_world_xyz

            # 3. Update Brain State Machine
            desired_pos_w, target_yaw = self._brain.update(
                person_found, person_world_xyz, drone_pos, drone_quat
            )

            snap_pos = getattr(self._brain, "_pending_scan_snap_pos", None)
            if snap_pos is not None:
                self._brain._pending_scan_snap_pos = None
                no_snap_from = int(getattr(self.cfg, "brain_no_snap_from_segment", 4))
                if (
                    getattr(self.cfg, "brain_snap_drone_on_scan", True)
                    and int(self._brain.segment_idx) < no_snap_from
                ):
                    self._snap_drone_to_local(snap_pos)
                    drone_pos = self._robot.data.root_pos_w.clone()

            # In real SLAM the house is placed once and stays static (no per-segment
            # reshuffle, no hide-during-scan) so the occupancy map never picks up
            # phantom walls from obstacles that teleported or vanished.
            if not getattr(self.cfg, "brain_real_slam_mode", False):
                seg = int(self._brain.segment_idx)
                if seg != self._last_obstacle_segment:
                    self._last_obstacle_segment = seg
                    self._randomize_obstacles(self._robot._ALL_INDICES)
                    print(
                        f"[BrainNavEnv] Obstacles re-randomized for segment {seg} "
                        f"({self._brain.get_segment_label(seg)}, level={self._get_mission_obstacle_level()}).\n"
                    )

                scan_mode = self._brain.state == "SCAN"
                if scan_mode != self._last_obstacle_scan_mode:
                    self._last_obstacle_scan_mode = scan_mode
                    self._randomize_obstacles(self._robot._ALL_INDICES)
                    if scan_mode:
                        print("[BrainNavEnv] Dynamic obstacles hidden during 360 SCAN (clear YOLO view).\n")
                    else:
                        print("[BrainNavEnv] Dynamic obstacles restored for navigation.\n")

            slam_rescued = getattr(self._brain, "rescued_people", None)
            slam_found_anyone = (slam_rescued is not None and len(slam_rescued) > 0)
            legacy_found = getattr(self._brain, "found_person", False)

            if (
                self._brain.state == "COMPLETE"
                and (slam_found_anyone or legacy_found)
                and not self._mission_complete
            ):
                self._mission_complete = True
                if slam_rescued and len(slam_rescued) > 0:
                    positions = "; ".join(f"X:{p[0]:.2f} Y:{p[1]:.2f}" for p in slam_rescued)
                    print(
                        f"\n[BrainNavEnv] MISSION COMPLETE — SLAM patrol finished. {len(slam_rescued)} person(s) found.\n"
                        f"  Locations: {positions}\n"
                    )
                else:
                    target = self._brain.target_person_pos
                    print(
                        "\n[BrainNavEnv] MISSION COMPLETE — high-confidence person reached.\n"
                        f"  Rescue coordinates (local): X:{target[0]:.2f} Y:{target[1]:.2f} Z:{target[2]:.2f}\n"
                    )
            elif (
                self._brain.state == "COMPLETE"
                and getattr(self._brain, "mission_finished", False)
                and not (slam_found_anyone or legacy_found)
                and not self._mission_complete
            ):
                self._mission_complete = True
                finish = self.cfg.brain_spawn_sequence[-1]
                noted = getattr(self._perception, "person_ever_detected", False) or getattr(
                    self._brain, "person_noted_anywhere", False
                )
                if noted:
                    print(
                        "\n[BrainNavEnv] MISSION COMPLETE — full patrol finished "
                        "(person was detected but not rescued).\n"
                        f"  Final checkpoint (local): X:{finish[0]:.2f} Y:{finish[1]:.2f} Z:{finish[2]:.2f}\n"
                    )
                else:
                    print(
                        "\n[BrainNavEnv] MISSION COMPLETE — full patrol finished (no person found).\n"
                        f"  Final checkpoint (local): X:{finish[0]:.2f} Y:{finish[1]:.2f} Z:{finish[2]:.2f}\n"
                    )

            # 4. Set navigation goal — yaw is integrated by PPO actions (Multilevel training)
            self._desired_pos_w[:, :] = desired_pos_w

            # 5. Re-evaluate observations for the navigator policy with the new targets
            obs_dict = self._get_observations()

            # 6. Action determination (bypassing PPO during high-level Brain states)
            if self._brain.state == "SCAN":
                scan_decel_steps = 15
                steps_in_scan = int(getattr(self, "_scan_step_count", 0))
                if steps_in_scan == 0:
                    # Capture the last PPO velocity and entry Z when entering SCAN
                    self._scan_entry_actions = getattr(self, "_previous_actions", torch.zeros((self.num_envs, 4), device=self.device)).clone()
                    self._scan_entry_z = drone_pos[:, 2].clone()
                    self._scan_hover_pos = None
                self._scan_step_count = steps_in_scan + 1

                yaw_rate = float(getattr(self.cfg, "brain_scan_yaw_rate", 0.05))

                # Active Z correction to prevent climbing/sinking from the very first frame of SCAN
                z_err = self._scan_entry_z - drone_pos[:, 2]
                z_action = (z_err * 2.0).clamp(-0.5, 0.5)

                if steps_in_scan < scan_decel_steps:
                    # Gradually reduce XY velocity to zero
                    blend = 1.0 - (steps_in_scan / scan_decel_steps)  # 1.0 → 0.0
                    entry_vel = getattr(self, "_scan_entry_actions", torch.zeros((self.num_envs, 4), device=self.device))
                    ppo_actions = torch.zeros((self.num_envs, 4), device=self.device)
                    ppo_actions[:, :2] = entry_vel[:, :2] * blend
                    # Ramp yaw spin up
                    ppo_actions[:, 3] = yaw_rate * (1.0 - blend)
                    # Maintain locked Z altitude
                    ppo_actions[:, 2] = z_action
                else:
                    # Capture hover position once on deceleration end
                    if getattr(self, "_scan_hover_pos", None) is None:
                        self._scan_hover_pos = drone_pos.clone()

                    from isaaclab.utils.math import quat_apply_inverse
                    ppo_actions = torch.zeros((self.num_envs, 4), device=self.device)
                    ppo_actions[:, 3] = yaw_rate

                    # Compute XY and Z errors relative to the hover position
                    pos_err = self._scan_hover_pos - drone_pos
                    # Rotate position error into drone's body frame for LLC velocity commands
                    pos_err_b = quat_apply_inverse(drone_quat, pos_err[:, :3])

                    # P-controller for holding hover position
                    ppo_actions[:, 0] = (pos_err_b[:, 0] * 1.5).clamp(-0.4, 0.4)
                    ppo_actions[:, 1] = (pos_err_b[:, 1] * 1.5).clamp(-0.4, 0.4)
                    ppo_actions[:, 2] = (pos_err_b[:, 2] * 2.0).clamp(-0.5, 0.5)
            elif self._brain.state == "COMPLETE":
                self._scan_step_count = 0
                # Hover in place at final target coordinates
                ppo_actions = torch.zeros((self.num_envs, 4), device=self.device)
            elif self._brain.state == "APPROACH_TARGET":
                self._scan_step_count = 0
                # Rescue: use PPO navigator to fly toward detected person (not spin)
                policy_obs = obs_dict if self._navigator_policy_expects_dict else obs_dict["policy"]
                ppo_actions = self._navigator_policy(policy_obs)
            else:
                self._scan_step_count = 0
                # If we are in EXPLORE state but have no active path, hover/hold in place safely (zeros)
                # to prevent querying the PPO policy with target_dist=0 (which is out-of-distribution
                # and causes the policy to output incorrect forward commands into walls).
                is_explore = (self._brain.state == "EXPLORE")
                has_path = (getattr(self._brain, "astar_path_world", None) is not None and len(self._brain.astar_path_world) >= 2)
                if is_explore and not has_path:
                    ppo_actions = torch.zeros((self.num_envs, 4), device=self.device)
                else:
                    policy_obs = obs_dict if self._navigator_policy_expects_dict else obs_dict["policy"]
                    ppo_actions = self._navigator_policy(policy_obs)

        # Sync target_yaw at boundary transitions (entering and exiting SCAN) BEFORE stepping the simulator
        is_scanning = hasattr(self, "_brain") and self._brain.state == "SCAN"
        was_scanning = getattr(self, "_was_scanning", False)
        if is_scanning != was_scanning:
            qw = self._robot.data.root_quat_w
            w, x, y, z = qw[:, 0], qw[:, 1], qw[:, 2], qw[:, 3]
            self._target_yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        self._was_scanning = is_scanning

        if not is_scanning:
            self._scan_lock_z = None
            ppo_actions = ppo_actions.clone()
            
            if getattr(self.cfg, "brain_force_yaw_to_target", True):
                if isinstance(target_yaw, torch.Tensor):
                    self._target_yaw = target_yaw.clone().to(self.device)
                else:
                    self._target_yaw = torch.tensor(target_yaw, device=self.device, dtype=torch.float32).repeat(self.num_envs)
                ppo_actions[:, 3] = 0.0

            # Proportional height-holding controller to prevent vertical drift
            # Target height is specified in self._desired_pos_w[:, 2] (typically 1.0m)
            z_err = self._desired_pos_w[:, 2] - self._robot.data.root_pos_w[:, 2]
            ppo_actions[:, 2] = torch.clamp(1.8 * z_err, -0.45, 0.45)

        # 7. Step the parent environment with the PPO-generated actions
        obs, rewards, terminated, truncated, info = super().step(ppo_actions)

        # Detect stuck drone (wedged against wall, no XY movement) and force reset
        pos_now = self._robot.data.root_pos_w[0]
        if self._prev_drone_pos_xy is None:
            self._prev_drone_pos_xy = pos_now[:2].clone()
            self._stuck_step_count = 0
        else:
            moved = torch.norm(pos_now[:2] - self._prev_drone_pos_xy).item()
            if moved < 0.03 and self._brain.state in ("GOTO_WAYPOINT", "APPROACH_TARGET", "EXPLORE"):
                self._stuck_step_count += 1
            else:
                self._stuck_step_count = 0
            self._prev_drone_pos_xy = pos_now[:2].clone()

        if self._stuck_step_count > 300:
            allow_skip = getattr(self.cfg, "brain_allow_stuck_arrival_skip", False)
            d_local = self._robot.data.root_pos_w[0] - self._terrain.env_origins[0]
            if allow_skip and self._brain.try_complete_goto_arrival(d_local.cpu().numpy()):
                self._stuck_step_count = 0
                print("[BrainNavEnv] Near-target stuck — skipped to next checkpoint (debug only).\n")
            elif getattr(self.cfg, "brain_stuck_respawn", False):
                print(
                    "[BrainNavEnv] Drone stuck — respawning at current room entrance "
                    "(resume GOTO, no 360 SCAN until checkpoint reached)."
                )
                self._reset_idx(torch.tensor([0], device=self.device))
            else:
                self._stuck_step_count = 0
                if self._timestep % 200 == 0:
                    print(
                        "[BrainNavEnv] Drone moving slowly toward target — "
                        "continuing GOTO (no respawn).\n"
                    )

        # Log crash without reset when brain_reset_on_crash=False
        if not getattr(self.cfg, "brain_reset_on_crash", True):
            last_died = getattr(self, "_last_died", None)
            if last_died is not None and last_died.any() and not getattr(self, "_logged_crash_no_reset", False):
                self._logged_crash_no_reset = True
                print("[BrainNavEnv] Drone crashed — sim kept running (brain_reset_on_crash=False). Press Ctrl+C to exit.")

        # 8. Reset handling (_reset_idx preserves SLAM mission and respawns on map floor)
        dones = terminated | truncated
        if dones.any():
            last_died = getattr(self, "_last_died", None)
            for env_id in range(self.num_envs):
                if not dones[env_id].item():
                    continue
                if last_died is not None and last_died[env_id].item():
                    reason = getattr(self, "_env0_crash_reason", "unknown")
                    if getattr(self.cfg, "brain_use_sequential_spawns", False):
                        print(
                            f"[BrainNavEnv] Crash ({reason}) — respawned at current segment, "
                            f"SLAM progress preserved (state={self._brain.state}, seg={self._brain.segment_idx})."
                        )
                    else:
                        wp = getattr(self._brain, "current_wp_idx", 0)
                        print(
                            f"[BrainNavEnv] Crash — respawned on map floor, "
                            f"continuing SLAM (state={self._brain.state}, wp={wp})."
                        )
                else:
                    print(f"[BrainNavEnv] Environment {env_id} reset (timeout).")

        # 9. Periodic status logging
        if self._timestep % 100 == 0:
            d_pos = drone_pos[0]
            g_pos = desired_pos_w[0]
            dist = torch.norm(d_pos - g_pos).item()
            yolo_conf = getattr(self._perception, "last_best_person_conf", 0.0)
            yolo_note = (
                f"YOLO best person={yolo_conf:.0%} (need {self.cfg.yolo_person_conf_threshold:.0%})"
            )
            seg = getattr(self._brain, "segment_idx", 0)
            seg_label = self._brain.get_segment_label(seg) if hasattr(self._brain, "get_segment_label") else f"seg {seg}"
            visited, total = self._brain.coverage_stats() if hasattr(self._brain, "coverage_stats") else (0, 0)
            cov_note = f"coverage={visited}/{total}" if total > 0 else ""
            scan_note = " [360 SCAN]" if self._brain.state == "SCAN" else ""
            print(
                f"[BrainNavEnv Step {self._timestep}] State: {self._brain.state}{scan_note} | {seg_label} | "
                f"Drone: ({d_pos[0].item():.2f}, {d_pos[1].item():.2f}, {d_pos[2].item():.2f}) | "
                f"Target: ({g_pos[0].item():.2f}, {g_pos[1].item():.2f}, {g_pos[2].item():.2f}) | "
                f"Dist: {dist:.2f}m | {yolo_note} {cov_note}"
            )

        self._timestep += 1
        return obs, rewards, terminated, truncated, info

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Goal marker only — hide ae_ppo LiDAR / pillar / ceiling debug views."""
        for attr in (
            "ceiling_visualizer",
            "drone_tracker_visualizer",
            "pillar_zone_visualizer_list",
        ):
            if hasattr(self, attr):
                viz = getattr(self, attr)
                if isinstance(viz, list):
                    for item in viz:
                        item.set_visibility(False)
                else:
                    viz.set_visibility(False)
        if hasattr(self, "_draw") and self._draw is not None:
            self._draw.clear_lines()

        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                size = self.cfg.goal_radius * 1.5
                marker_cfg.markers["cuboid"].size = (size, size, size)
                marker_cfg.prim_path = "/Visuals/Command/goal_position"
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
            self.goal_pos_visualizer.set_visibility(True)
        elif hasattr(self, "goal_pos_visualizer"):
            self.goal_pos_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        """Draw SLAM/nav target only (no LiDAR rays or pillar zones)."""
        if not hasattr(self, "goal_pos_visualizer"):
            return
        goal_pos = self._desired_pos_w.clone()
        drone_pos = self._robot.data.root_pos_w[:, :3]
        dist_to_drone = torch.norm(goal_pos - drone_pos, dim=-1)
        too_close = dist_to_drone < 0.65
        goal_pos[too_close, 2] -= 10.0
        self.goal_pos_visualizer.visualize(goal_pos)

    def set_debug_vis(self, debug_vis: bool) -> None:
        """Brain play: use step()-time debug draw only (no post-update subscription).

        The post-update weakref callback can fire during Omniverse shutdown and crash
        omni.syntheticdata / omni.graph.core plugins.
        """
        if debug_vis:
            if hasattr(self, "_debug_vis_handle") and self._debug_vis_handle is not None:
                self._debug_vis_handle.unsubscribe()
                self._debug_vis_handle = None
            self._set_debug_vis_impl(True)
        else:
            if hasattr(self, "_debug_vis_handle") and self._debug_vis_handle is not None:
                self._debug_vis_handle.unsubscribe()
                self._debug_vis_handle = None
            self._set_debug_vis_impl(False)

    def close(self):
        """Disable debug draw and OpenCV windows before Isaac Sim tears down."""
        self._closing = True
        try:
            self.set_debug_vis(False)
        except Exception:
            pass
        try:
            import cv2
            # Do NOT destroy windows on close so OpenCV remains visible
            # cv2.destroyAllWindows()
        except Exception:
            pass
        super().close()

    # ------------------------------------------------------------------
    #  Dynamic target spawning (dashboard-triggered, operator-only map hints)
    # ------------------------------------------------------------------
    def _hide_static_rescue_persons_for_dynamic_spawn(self) -> None:
        """Hide default room-3 / final-room persons when operator spawns new targets."""
        for attr in (
            "_room3_rescue_person_prim",
            "_final_rescue_person_prim",
            "_final_center_person_prim",
        ):
            prim = getattr(self, attr, None)
            if prim is not None and prim.IsValid():
                self._set_prim_visibility(prim, visible=False)

    def _restore_static_rescue_persons(self) -> None:
        """Show default static persons again after a failed dynamic spawn."""
        for attr in (
            "_room3_rescue_person_prim",
            "_final_rescue_person_prim",
            "_final_center_person_prim",
        ):
            prim = getattr(self, attr, None)
            if prim is not None and prim.IsValid():
                self._set_prim_visibility(prim, visible=True)
        self.spawned_targets_local = []
        self.dynamic_spawn_active = False

    def _clear_dynamic_spawned_persons(self) -> None:
        """Remove ALL dashboard-spawned person wrappers from the stage."""
        from pxr import Usd

        stage = self.sim.stage
        scope = getattr(self.cfg, "brain_rescue_person_scope", "RescuePersons")
        parent_path = f"/World/envs/env_0/Room/{scope}"
        to_remove: list[str] = []
        parent = stage.GetPrimAtPath(parent_path)
        if parent.IsValid():
            for child in parent.GetChildren():
                if child.GetName().startswith("DynamicSpawn_"):
                    to_remove.append(str(child.GetPath()))
        for name in getattr(self, "_dynamic_spawn_names", []):
            path = f"{parent_path}/{name}"
            if path not in to_remove:
                to_remove.append(path)
        with Usd.EditContext(stage, self._get_usd_edit_layer()):
            for path in to_remove:
                prim = stage.GetPrimAtPath(path)
                if prim.IsValid():
                    stage.RemovePrim(path)
        self._dynamic_spawn_names = []
        self._dynamic_spawn_prims = []
        self.spawned_targets_local = []
        self.dynamic_spawn_active = False
        brain = getattr(self, "_brain", None)
        if brain is not None:
            brain.rescued_people = []
            if hasattr(brain, "rescued_people_conf"):
                brain.rescued_people_conf = []

    def count_spawned_targets_detected(self) -> tuple[int, int]:
        """Return (detected_count, total_spawned) for YOLO confirmations >= rescue threshold."""
        import numpy as np

        spawned = getattr(self, "spawned_targets_local", []) or []
        if not spawned:
            return 0, 0
        brain = getattr(self, "_brain", None)
        rescued = getattr(brain, "rescued_people", []) or [] if brain else []
        rescued_conf = getattr(brain, "rescued_people_conf", []) or [] if brain else []
        thresh = float(getattr(self.cfg, "yolo_person_conf_threshold", 0.70))
        origin = self._terrain.env_origins[0].cpu().numpy()
        detected = 0
        for tgt in spawned:
            tw = np.array(
                [float(tgt[0]) + float(origin[0]), float(tgt[1]) + float(origin[1])],
                dtype=np.float64,
            )
            for idx, rp in enumerate(rescued):
                conf = float(rescued_conf[idx]) if idx < len(rescued_conf) else 0.0
                if conf < thresh:
                    continue
                rw = np.asarray(rp[:2], dtype=np.float64)
                if float(np.linalg.norm(rw - tw)) < 1.5:
                    detected += 1
                    break
        return detected, len(spawned)

    def spawn_random_targets(self, count: int = 2):
        """Spawn *count* F_Business_02 persons at random safe map positions.

        Operator-only: positions are stored for the dashboard (cyan/pink stars).
        The drone brain is NOT given these coordinates — YOLO must find them.
        Replaces static room-3 / final persons when triggered.
        """
        count = max(1, min(15, int(count)))
        placed = self._sample_dynamic_spawn_positions(count)
        if not placed:
            print(
                "[BrainNavEnv] Dynamic spawn aborted — no valid floor positions found. "
                "Static persons left unchanged."
            )
            return

        # Hide full-scale map-embedded F_Business_02 (can overlap spawn sites).
        self._hide_map_default_person()
        self._hide_static_rescue_persons_for_dynamic_spawn()
        self._clear_dynamic_spawned_persons()
        self._refresh_static_person_target_height()
        target_h = getattr(self, "_static_person_target_height", None)

        default_yaw = 90.0  # same facing as RescuePerson_Room3
        names: list[str] = []
        prims = []
        confirmed: list[tuple[float, float, float]] = []
        for i, local_xyz in enumerate(placed):
            name = f"DynamicSpawn_{i:02d}"
            try:
                prim = self._spawn_rescue_person_wrapper(
                    name, local_xyz, yaw_deg=default_yaw
                )
                if not self._align_person_scale_to_static_template(prim):
                    self._queue_person_scale_fix(prim)
                names.append(name)
                prims.append(prim)
                confirmed.append(local_xyz)
            except Exception as exc:
                print(f"[BrainNavEnv] Failed to spawn {name}: {exc}")

        if not confirmed:
            print(
                "[BrainNavEnv] Dynamic spawn failed during USD placement — "
                "restoring static persons."
            )
            self._restore_static_rescue_persons()
            return

        self._dynamic_spawn_names = names
        self._dynamic_spawn_prims = prims
        self.spawned_targets_local = confirmed
        self.dynamic_spawn_active = True

        # Drone gets no GPS hints to spawn positions — operator map markers only.
        if hasattr(self, "_perception") and self._perception is not None:
            self._perception._rescue_person_slots = []

        th_str = f"{target_h:.2f}m" if target_h else "unknown"
        print(
            f"[BrainNavEnv] Dynamic spawn: {len(confirmed)}/{count} persons placed "
            f"(operator map only, drone blind, target_height={th_str}): "
            f"{[(round(p[0], 2), round(p[1], 2)) for p in confirmed]}"
        )



class _BrainEnvAdapter:
    """Lightweight adapter so BrainModule can access env internals.

    BrainModule expects an object with `.unwrapped` returning the
    DirectRLEnv instance. When Brain is used inside the env itself,
    we provide this thin wrapper.
    """

    def __init__(self, env: BrainNavDroneEnv):
        self._env = env

    @property
    def unwrapped(self):
        return self._env

    @property
    def device(self):
        return self._env.device


class _RslRlCompatWrapper:
    """Minimal compatibility wrapper for RSL-RL OnPolicyRunner initialization.

    OnPolicyRunner needs an env-like object with specific attributes
    during construction. This wrapper provides just enough interface
    to load a checkpoint without a full RslRlVecEnvWrapper.
    """

    def __init__(self, env: BrainNavDroneEnv):
        self._env = env
        self.cfg = env.cfg
        self.num_envs = env.num_envs
        self.device = env.device
        self.num_obs = env.cfg.observation_space
        self.num_actions = env.cfg.action_space
        self.max_episode_length = env.max_episode_length
        self.num_privileged_obs = None
        self.obs_dict = {"policy": torch.zeros(env.num_envs, env.cfg.observation_space, device=env.device)}

    @property
    def unwrapped(self):
        return self._env

    def get_observations(self):
        return self.obs_dict

    def reset(self):
        return self.obs_dict, {}
