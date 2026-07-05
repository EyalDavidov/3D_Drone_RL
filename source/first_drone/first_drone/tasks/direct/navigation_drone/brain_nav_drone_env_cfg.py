"""Configuration for the Brain-driven Navigation Drone environment.

This environment embeds a pretrained PPO navigator + Brain state machine internally.
It requires an explicit checkpoint path to the trained navigator policy.
"""

import os

# Get the repository root directory
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../../"))

import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.sensors import TiledCameraCfg, ContactSensorCfg, CameraCfg
from .ae_ppo_drone_env_cfg import AEPPODroneEnvCfg
from isaaclab.scene import InteractiveSceneCfg

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlMLPModelCfg, RslRlPpoAlgorithmCfg


@configclass
class BrainNavDroneEnvCfg(AEPPODroneEnvCfg):
    """Config for the self-contained Brain navigation environment.

    Inherits all physical setup (drone, camera, AE, LLC, terrain, obstacles)
    from AEPPODroneEnvCfg and adds Brain-specific parameters.
    """

    # ---------- Scene: Single environment for Brain-controlled mission ----------
    # replicate_physics=False avoids cloning/replicating envs (prevents "new stage opened" resets in play).
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1, env_spacing=6.0, replicate_physics=False
    )

    # ---------- Map (Multilevel_AE_PPO / model_1450 training layout) ----------
    map_scale: float = 1.0
    map_bounds: tuple = (-8.55, 4.05, -23.05, 2.05)
    map_obstacles: tuple = ()

    room_usd_path: str = os.path.join(_REPO_ROOT, "assets", "final_roof_flat.usd")
    use_direct_room_spawn: bool = True
    room_spawn_scale: tuple = (1.0, 1.0, 1.0)
    room_spawn_translation: tuple = (0.0, 0.0, 0.0)
    room_spawn_orientation: tuple = (1.0, 0.0, 0.0, 0.0)

    # YOLO camera — 512×288 native; depth still downsampled to 72×128 for model_1450
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Drone/body/Camera",
        height=288,
        width=512,
        data_types=["depth", "rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0,
            focus_distance=6.0,
            f_stop=0.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 1.0e5),
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.01, 0.0, 0.015), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"
        ),
    )

    # Chase / behind-drone viewport camera (Multilevel_AE_PPO)
    view_camera: CameraCfg = CameraCfg(
        prim_path="/World/envs/env_0/Drone/body/Camera_View",
        update_period=0.0,
        height=720,
        width=1280,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=15.5,
            focus_distance=400.0,
            horizontal_aperture=33.0,
            clipping_range=(0.1, 100000.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(-0.15, 0.0, 0.05), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"
        ),
    )

    # --- Dashboard follow cameras (play-mode only, not used by policy) ---
    # World-anchored so they are NOT double-transformed by the drone body.
    # Their world pose is driven every step via set_world_poses_from_view()
    # so they always AIM AT the drone (see BrainNavDroneEnv.update_follow_cameras).
    # The offset here is only the spawn placeholder; runtime poses override it.

    # Chase camera — trails the drone from behind + above, looking at it.
    chase_camera: CameraCfg = CameraCfg(
        prim_path="/World/DashCam_Chase",
        update_period=0.0,
        height=180,
        width=320,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=14.0,
            focus_distance=400.0,
            horizontal_aperture=33.0,
            clipping_range=(0.05, 100000.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 2.0), rot=(1.0, 0.0, 0.0, 0.0), convention="ros"
        ),
    )

    # Left-side camera — offset to the drone's left, looking at it.
    left_camera: CameraCfg = CameraCfg(
        prim_path="/World/DashCam_Left",
        update_period=0.0,
        height=180,
        width=320,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=14.0,
            focus_distance=400.0,
            horizontal_aperture=33.0,
            clipping_range=(0.05, 100000.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 2.0), rot=(1.0, 0.0, 0.0, 0.0), convention="ros"
        ),
    )

    # Top-down camera — hovers above the drone, looking straight down at it.
    top_camera: CameraCfg = CameraCfg(
        prim_path="/World/DashCam_Top",
        update_period=0.0,
        height=180,
        width=320,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=12.0,
            focus_distance=400.0,
            horizontal_aperture=33.0,
            clipping_range=(0.05, 100000.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 5.0), rot=(1.0, 0.0, 0.0, 0.0), convention="ros"
        ),
    )

    # ---------- Terrain: no visible default Isaac Lab ground plane ----------
    # final_flat.usd provides floor/walls/collision. Keep terrain only for env_origins bookkeeping.
    terrain: TerrainImporterCfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.0, 0.0, 0.0),
            opacity=0.0,
        ),
    )

    # ---------- Pretrained Navigator Policy ----------
    # Auto-resolved by resolve_navigator_checkpoint() — searches bestmodel, 29-06_23-04, eyal_best
    navigator_checkpoint_path: str = os.path.join(
        _REPO_ROOT, "logs", "rsl_rl", "navigation_drone_direct", "bestmodel", "model_1450.pt"
    )

    # ---------- Perception ----------
    use_mock_perception: bool = False  # False = real YOLO detection; True = mock (no detection)
    yolo_person_conf_threshold: float = 0.50
    yolo_min_bbox_area_frac: float = 0.0001
    yolo_min_bbox_height_frac: float = 0.02
    yolo_min_person_aspect: float = 0.30
    yolo_noted_confirm_frames: int = 2
    yolo_camera_upscale: int = 3   # 512×288 → 1536×864 before YOLO (makes distant humans larger/sharper)
    yolo_imgsz: int = 1280         # match upscaled feed so YOLO keeps full detail
    yolo_sharpen: bool = True
    yolo_clahe: bool = False       # colorized models don't need color-distorting CLAHE contrast boost
    yolo_show_opencv: bool = True  # False when web dashboard renders YOLO/SLAM natively

    # ---------- Walkable floor (R-shaped map) ----------
    walkable_grid_resolution: float = 0.4   # Meters per occupancy cell when parsing floor meshes from USD
    walkable_min_cells: int = 80            # Reject grid if smaller — fall back to room AABB
    walkable_min_area_m2: float = 20.0      # Reject grid if footprint area is too small

    # ---------- Brain Module Parameters ----------
    brain_step_size: float = 10.0       # Lawnmower corridor spacing in meters
    brain_safety_margin: float = 0.7    # Wall clearance for waypoint generation in meters
    brain_yolo_interval: int = 1        # Run YOLO every N steps (1 = every step, needed during SCAN spin)
    brain_scan_yaw_rate: float = 0.05   # SCAN spin action (slower = better YOLO frames)
    brain_scan_pin_position: bool = True  # hold XYZ during 360 so the drone does not drift upward
    brain_scan_pitch_deg: float = 22.0   # nose-down pitch during SCAN (positive = nose down in Isaac Lab x-fwd y-left z-up)
    brain_scan_vertical_comp: float = -0.12  # downward action to prevent climb during spin
    brain_scan_forward_tilt: float = 0.06   # smaller forward action to spin more in-place

    # In play mode, reset the drone after a crash so it can continue the mission.
    brain_reset_on_crash: bool = True
    brain_disable_episode_timeout: bool = True  # Brain play runs until rescue or manual exit
    brain_preserve_mission_on_crash: bool = True  # Sequential: respawn at current segment, keep SLAM progress

    # ---------- Sequential SLAM mission (Multilevel level spawns) ----------
    brain_use_sequential_spawns: bool = True
    # Use Multilevel training entrances — NOT USD zone centers (those are mid-room / near humans).
    brain_auto_room_spawns: bool = False
    # Multilevel room entrances (fixed); corridor + final room filled from USD at runtime.
    brain_spawn_sequence: tuple = (
        (0.0, 1.5, 1.0),      # Level 1 entrance
        (0.0, -2.5, 1.0),     # Level 2 entrance
        (0.0, -8.5, 1.0),     # Level 3 entrance
        (0.0, -16.5, 1.0),    # Level 4 entrance
    )
    brain_spawn_labels: tuple = (
        "room 1 entrance",
        "room 2 entrance",
        "room 3 entrance",
        "room 4 entrance",
    )
    # Multilevel training coords only — USD corridor zone overlaps room 4 and breaks nav.
    brain_use_usd_corridor_waypoints: bool = False
    brain_room4_corr1_waypoint: tuple = (0.0, -20.5, 1.0)  # corr2 junction — deep corridor (training path)
    brain_room4_corr2_waypoint: tuple = (0.0, -20.5, 1.0)
    brain_single_corridor_to_final: bool = True  # corr2 junction -> final in one pass
    brain_final_room_waypoint: tuple = (-6.0, -21.5, 1.0)
    brain_final_person_local: tuple | None = (-6.0, -21.5, 0.0)  # final room center (middle of the room)
    brain_final_person_center_local: tuple = (-6.0, -21.5, 0.0)  # second person — final room center
    brain_final_person_center_name: str = "RescuePerson_Final_Center"
    brain_room3_person_name: str = "RescuePerson_Room3"
    brain_person_scale: float = 0.35  # ~half human height (uniform, same as before)
    brain_person_asset_native_scale: float = 0.7  # F_Business_02 root scale in nucleus USD
    brain_person_height_scale: float = 1.0
    brain_room3_person_local: tuple = (0.0, -10.0, 0.0)  # room 3 center, visible from entrance scan
    # Empty → Isaac nucleus default (F_Business_02 with clothes/textures from Omniverse CDN)
    brain_rescue_person_usd: str = ""
    brain_rescue_person_character: str = "F_Business_02"
    brain_rescue_person_usd_ref: str = ""  # empty = reference full character USD (keep materials)
    brain_person_override_textures: bool = True  # override textures with local colors to prevent black models
    brain_rescue_person_scope: str = "RescuePersons"
    brain_person_match_radius: float = 5.0  # YOLO log slot match radius (m)
    brain_person_obstacle_exclusion_m: float = 2.5  # keep dynamic obstacles off rescue persons
    brain_final_person_name: str = "RescuePerson_Final"
    brain_snap_drone_on_scan: bool = False  # never teleport mid-route; spin where the drone actually arrives
    brain_no_snap_from_segment: int = 4
    brain_skip_scan_segment_indices: tuple = (4,)  # corridor pass-through only
    brain_corridor_arrival_radius: float = 0.85
    brain_corridor_crash_respawn_at_room4: bool = True  # corridor crashes respawn at room-4 entrance
    brain_final_room_arrival_radius: float = 1.2
    brain_scan_arrival_radius: float = 1.0  # must fly this close before 360 (not 2.5m early)
    brain_scan_arrival_dwell_steps: int = 8  # must hold position briefly — avoids glitch triggers
    brain_corridor_crash_skip_after: int = 3
    brain_crash_respawn_in_place: bool = True
    brain_crash_max_in_place: int = 2  # then warp to safe segment checkpoint (breaks wall loops)
    brain_stuck_respawn: bool = False
    brain_use_worker_gps_for_nav: bool = False
    brain_allow_stuck_arrival_skip: bool = False
    yolo_noted_conf_threshold: float = 0.35
    brain_spawn_arrival_radius: float = 2.5  # unused for SCAN; kept for legacy stuck-skip helper
    brain_coverage_mark_radius: float = 2.0   # meters to mark occupancy cells as visited after scan
    # Defer APPROACH/rescue until final room (segment 5); earlier rooms log detections only
    brain_rescue_min_segment: int = 5
    brain_real_slam_mode: bool = False  # True in RealSlamDroneEnv — rescue anywhere, no sequential defer

    # ---------- Curriculum: Fixed at Level 5 (fully trained policy) ----------
    initial_curriculum_level: int = 5
    # model_1450 (WandB bdo85ahx / Multilevel_Train): 64 AE latent + 13 state = 77-dim
    observation_space: int = 77
    ae_latent_dim: int = 64
    ae_checkpoint_path: str = os.path.join(_REPO_ROOT, "logs", "ae_1_7_latent_64", "ae_final.pt")
    depth_max: float = 5.0  # must match Multilevel_Train AE preprocessing
    is_brain_play: bool = True
    yaw_rate_limit: float = 0.5
    goal_radius: float = 0.20

    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Drone/body",
        history_length=1,
    )
    contact_force_threshold: float = 0.0001  # Match Multilevel_Train (bdo85ahx); SCAN ignores contact in _get_dones

    # ---------- Episode ----------
    episode_length_s: float = 3600.0    # Long run for full search missions (timeout disabled in play)
    debug_vis: bool = False
    show_ae_images: bool = False

    # ---------- Person spawning ----------
    spawn_person: bool = True

    # ---------- Random Obstacles (Dynamic Obstacle Spawning) ----------
    num_pillars: int = 0  # Unused legacy pillars
    
    num_poles: int = 21
    pole_spawn: sim_utils.UsdFileCfg = sim_utils.UsdFileCfg(
        usd_path=os.path.join(_REPO_ROOT, "assets", "1_Pole.usd"),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
    )

    # Room 3/4 dynamic obstacles — sparse in play; no corr obstacles (block narrow paths)
    num_room3_walls: int = 4
    num_room3_cones: int = 3
    num_room3_big_gates: int = 1
    num_room3_small_gates: int = 2
    num_room3_poles_triangles: int = 2

    wall_spawn: sim_utils.UsdFileCfg = sim_utils.UsdFileCfg(
        usd_path=os.path.join(_REPO_ROOT, "assets", "obstacles", "wall.usd"),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
    )
    cone_spawn: sim_utils.UsdFileCfg = sim_utils.UsdFileCfg(
        usd_path=os.path.join(_REPO_ROOT, "assets", "obstacles", "cone.usd"),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
    )
    big_gate_spawn: sim_utils.UsdFileCfg = sim_utils.UsdFileCfg(
        usd_path=os.path.join(_REPO_ROOT, "assets", "obstacles", "big_gate.usd"),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
    )
    small_gate_spawn: sim_utils.UsdFileCfg = sim_utils.UsdFileCfg(
        usd_path=os.path.join(_REPO_ROOT, "assets", "obstacles", "small_gate.usd"),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
    )
    poles_triangle_spawn: sim_utils.UsdFileCfg = sim_utils.UsdFileCfg(
        usd_path=os.path.join(_REPO_ROOT, "assets", "obstacles", "poles_triangle.usd"),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
    )
    corr1_spawn: sim_utils.UsdFileCfg = sim_utils.UsdFileCfg(
        usd_path=os.path.join(_REPO_ROOT, "assets", "obstacles", "corr1.usd"),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
    )
    corr2_spawn: sim_utils.UsdFileCfg = sim_utils.UsdFileCfg(
        usd_path=os.path.join(_REPO_ROOT, "assets", "obstacles", "corr2.usd"),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
    )
    num_room4_corr1: int = 0
    num_room4_corr2: int = 0
    brain_hide_obstacles_during_scan: bool = True  # Clear sightlines for 360° YOLO in every room


@configclass
class BrainPPOCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 50
    experiment_name = "navigation_drone_direct"
    logger = "wandb"
    wandb_project = "Multilevel_Train"

    # Actor network
    actor = RslRlMLPModelCfg(
        hidden_dims=[128, 64, 32],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            init_std=1.0,
        ),
    )

    # Critic network
    critic = RslRlMLPModelCfg(
        hidden_dims=[128, 64, 32],
        activation="elu",
        obs_normalization=False,
    )

    resume = False
    load_run = ""
    load_checkpoint = ""

    # Map observation dict keys to actor/critic
    obs_groups = {
        "actor": ["policy"],
        "critic": ["policy"],
    }

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.5e-4,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
