"""PPO+AE Drone Environment.

Depth (128×72) → AE Encoder → z_img (32-dim)
z_img + target_rel_body + target_dist + lin_vel + ang_vel + gravity + prev_actions → PPO (49-dim)

The environment returns a flat 49-dim observation vector (NOT raw images).
The AE is owned by the environment, trained offline via the training script scripts/train_ae.py,
and its detached latent is fed to PPO.
"""
from __future__ import annotations

import gymnasium as gym
import torch
import torch.nn.functional as F

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.sensors import TiledCamera
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import (
    subtract_frame_transforms,
    wrap_to_pi,
    quat_from_euler_xyz,
    euler_xyz_from_quat,
    quat_rotate,
    quat_rotate_inverse,
)

from isaaclab.markers import CUBOID_MARKER_CFG, VisualizationMarkersCfg  # isort: skip

from .ae_ppo_drone_env_cfg import AEPPODroneEnvCfg
from first_drone.models.ae import AE
from isaaclab.sensors import MultiMeshRayCaster, MultiMeshRayCasterCfg
from isaaclab.sensors.ray_caster.patterns.patterns_cfg import LidarPatternCfg


class AEPPODroneEnv(DirectRLEnv):
    """PPO+AE drone navigation environment."""

    cfg: AEPPODroneEnvCfg

    def __init__(self, cfg: AEPPODroneEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # ----- Action / wrench buffers -----
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)

        # ----- Goal position (world frame) -----
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)

        # ----- Previous distance to goal (for progress reward) -----
        self._prev_dist_to_goal = torch.zeros(self.num_envs, device=self.device)

        # ----- Physical constants -----
        self._body_id = self._robot.find_bodies("body")[0]
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        # ----- AE (owned by env, trained offline) -----
        self.ae = AE(latent_dim=self.cfg.ae_latent_dim).to(self.device)
        if hasattr(self.cfg, "ae_checkpoint_path") and self.cfg.ae_checkpoint_path is not None:
            import os
            if os.path.exists(self.cfg.ae_checkpoint_path):
                self.ae.load_state_dict(torch.load(self.cfg.ae_checkpoint_path, map_location=self.device))
                print(f"\n[INFO] AE model loaded successfully from {self.cfg.ae_checkpoint_path}\n")
            else:
                print(f"\n[WARNING] AE checkpoint not found at {self.cfg.ae_checkpoint_path}\n")
        self.ae.eval()

        # ----- Low-level flight controller -----
        self.llc = torch.jit.load(self.cfg.llc_checkpoint_path, map_location=self.device)
        self.llc.eval()
        for param in self.llc.parameters():
            param.requires_grad = False

        # ----- High-level navigator buffers -----
        self._desired_vel_b = torch.zeros(self.num_envs, 3, device=self.device)
        self._target_yaw = torch.zeros(self.num_envs, device=self.device)
        self._previous_actions = torch.zeros(self.num_envs, 4, device=self.device)

        # ----- Depth image buffer -----
        self._last_depth_processed = None
        self._last_lidar_scan = None

        # ----- Episode reward logging -----
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "progress", "goal", "collision",
            ]
        }

        # ----- Curriculum State -----
        import sys
        is_play_script = any("play.py" in arg or "play_saliency.py" in arg for arg in sys.argv)
        default_level = 5 if is_play_script else 1
        self.curriculum_level = default_level if is_play_script else getattr(self.cfg, "initial_curriculum_level", default_level)
        self.running_goal_rate = 0.0

        # Scale map obstacles according to map_scale config
        map_scale = getattr(self.cfg, "map_scale", 1.0)
        if map_scale != 1.0:
            scaled_obstacles = []
            for obs in self.cfg.map_obstacles:
                min_x, max_x, min_y, max_y = obs
                scaled_obstacles.append((min_x * map_scale, max_x * map_scale, min_y * map_scale, max_y * map_scale))
            self.cfg.map_obstacles = tuple(scaled_obstacles)
            print(f"\n[MAP] Scaled static obstacles by map_scale = {map_scale}\n")

        self.curriculum_distances = {
            1: (1.5, 3.0),
            2: (3.0, 5.0),
            3: (5.0, 7.0),
            4: (7.0, 9.0),
            5: (9.0, 12.0)
        }

        # AE visualization state
        self._ae_vis_step = 0

        # Debug visualization
        self.set_debug_vis(self.cfg.debug_vis)

        # Force legacy LiDAR mask on for all curriculum runs to ensure correct PPO training
        self.use_legacy_lidar_mask = False
        print("\n[CURRICULUM] Legacy LiDAR masking disabled for all runs to match successful Red run PPO training.\n")

        # Dynamically set initial episode length based on curriculum level
        self._update_episode_length(self._get_episode_length_for_level(self.curriculum_level))

        # Track curriculum level for each individual environment to prevent lagging-episode statistics corruption
        self.env_curriculum_level = torch.ones(self.num_envs, dtype=torch.long, device=self.device) * self.curriculum_level

    def _get_episode_length_for_level(self, level: int) -> float:
        if level >= 5:
            return 20.0  # Capped target distance 12.0m needs max 20 seconds to complete
        elif level == 4:
            return 18.0
        elif level == 3:
            return 15.0
        else:
            return 15.0

    def _update_episode_length(self, length_s: float):
        self.cfg.episode_length_s = length_s
        print(f"\n[CURRICULUM] Episode length dynamically updated to {length_s}s (max_episode_length steps = {self.max_episode_length})\n")






        # ----- Env 0 100-step logging buffers -----
        self._env0_step_counter = 0
        self._env0_accumulated_rewards = {}
        self._env0_died_count = 0.0
        self._env0_timeout_count = 0.0
        self._env0_episode_count_window = 0
        self._env0_episode_lengths = []
        self._env0_final_distances = []
        
        self._env0_log_values = {}
        for key in [
            "progress", "goal", "time", "heading", "vel_align", "ang_vel",
            "yaw_rate", "forward_speed", "action", "action_rate", "sideslip",
            "proximity", "speed_proximity", "stuck", "collision", "tilt", "z_deviation"
        ]:
            self._env0_log_values["Env0_Reward/" + key] = 0.0
            self._env0_accumulated_rewards[key] = 0.0
        self._env0_log_values["Env0_Termination/died"] = 0.0
        self._env0_log_values["Env0_Termination/time_out"] = 0.0
        self._env0_log_values["Env0_Metrics/final_distance_to_goal"] = 0.0
        self._env0_log_values["Env0_Metrics/episode_length"] = 0.0
        self._env0_log_values["Env0_Metrics/collision_rate"] = 0.0

    # ------------------------------------------------------------------
    # Scene setup
    # ------------------------------------------------------------------
    def _setup_scene(self):
        """Create drone, room (empty), dynamic pillars, terrain, camera, and lighting."""
        self._robot = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self._robot

        map_scale = getattr(self.cfg, "map_scale", 1.0)
        # Scale local Y (which becomes parent Z height) by 0.5 * map_scale to compress obstacle heights, and local X/Z by map_scale.
        room_cfg = sim_utils.UsdFileCfg(
            usd_path=self.cfg.room_usd_path,
            scale=(0.01 * map_scale, 0.01 * map_scale * 0.5, 0.01 * map_scale),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        )
        room_cfg.func(
            "/World/envs/env_0/Room",
            room_cfg,
            translation=(25.0 * map_scale, 25.0 * map_scale, -0.9937 * map_scale * 0.5 + 0.01),
            orientation=(0.7071, 0.7071, 0.0, 0.0),
        )

        # Apply CollisionAPI and PhysxCollisionAPI to all meshes inside the room model so they act as physics colliders
        from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema
        room_prim = self.sim.stage.GetPrimAtPath("/World/envs/env_0/Room")
        if room_prim.IsValid():
            for prim in Usd.PrimRange(room_prim):
                if prim.IsA(UsdGeom.Mesh):
                    if not prim.HasAPI(UsdPhysics.CollisionAPI):
                        UsdPhysics.CollisionAPI.Apply(prim)
                    if not prim.HasAPI(PhysxSchema.PhysxCollisionAPI):
                        PhysxSchema.PhysxCollisionAPI.Apply(prim)
                    if not prim.HasAPI(PhysxSchema.PhysxTriangleMeshCollisionAPI):
                        PhysxSchema.PhysxTriangleMeshCollisionAPI.Apply(prim)

        # --- Dynamic obstacles (diverse shapes, defined here to avoid @configclass serialization) ---
        obstacle_spawns = [
            # 0: Thin cylinder (original pillar)
            sim_utils.CylinderCfg(
                radius=0.05, height=2.5,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.15, 0.15)),
            ),
            # 1: Wide cuboid (wall segment)
            sim_utils.CuboidCfg(
                size=(0.15, 0.4, 2.5),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.2, 0.2)),
            ),
            # 2: Thick cylinder (tree trunk)
            sim_utils.CylinderCfg(
                radius=0.15, height=2.5,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.25, 0.1)),
            ),
            # 3: Tall narrow cuboid (pole-like)
            sim_utils.CuboidCfg(
                size=(0.08, 0.08, 2.5),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.6)),
            ),
            # 4: Large sphere
            sim_utils.SphereCfg(
                radius=0.25,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.5, 0.1)),
            ),
            # 5: Flat wide cuboid (barrier/fence)
            sim_utils.CuboidCfg(
                size=(0.05, 0.8, 1.5),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.1)),
            ),
        ]
        # Define shape details for accurate perimeter collision check
        # Scale drone collision offset dynamically based on config's pillar collision radius
        self._drone_collision_offset = self.cfg.pillar_collision_radius - 0.05
        self._obstacle_shapes = [
            # 0: Thin cylinder (original pillar) - radius 0.05, height 2.5 (half-height 1.25)
            {"type": "cylinder", "radius": 0.05, "half_z": 1.25},
            # 1: Wide cuboid (wall segment) - size (0.15, 0.4, 2.5), half-extents (0.075, 0.20, 1.25)
            {"type": "box", "half_x": 0.075, "half_y": 0.20, "half_z": 1.25, "size_x": 0.15, "size_y": 0.40, "height": 2.5},
            # 2: Thick cylinder (tree trunk) - radius 0.15, height 2.5 (half-height 1.25)
            {"type": "cylinder", "radius": 0.15, "half_z": 1.25},
            # 3: Tall narrow cuboid (pole-like) - size (0.08, 0.08, 2.5), half-extents (0.04, 0.04, 1.25)
            {"type": "box", "half_x": 0.04, "half_y": 0.04, "half_z": 1.25, "size_x": 0.08, "size_y": 0.08, "height": 2.5},
            # 4: Large sphere - radius 0.25
            {"type": "sphere", "radius": 0.25},
            # 5: Flat wide cuboid (barrier/fence) - size (0.05, 0.8, 1.5), half-extents (0.025, 0.40, 0.75)
            {"type": "box", "half_x": 0.025, "half_y": 0.40, "half_z": 0.75, "size_x": 0.05, "size_y": 0.80, "height": 1.5},
        ]
        # Keep _obstacle_collision_radii for compatibility (e.g. fallback defaults)
        self._obstacle_collision_radii = [0.15, 0.31, 0.25, 0.16, 0.35, 0.50]

        zone_centers = [(lo + hi) / 2.0 for lo, hi in self.cfg.pillar_x_zones]
        self._pillars = []
        for i in range(self.cfg.num_pillars):
            spawn_cfg = obstacle_spawns[i]
            pillar_cfg = RigidObjectCfg(
                prim_path=f"/World/envs/env_.*/Obstacle_{i}",
                spawn=spawn_cfg,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(zone_centers[i], 0.0, self.cfg.pillar_z)
                ),
            )
            pillar = RigidObject(pillar_cfg)
            self.scene.rigid_objects[f"pillar_{i}"] = pillar
            self._pillars.append(pillar)

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self._tiled_camera = TiledCamera(self.cfg.tiled_camera)

        from isaaclab.sensors import ContactSensor, ContactSensorCfg
        contact_sensor_cfg = ContactSensorCfg(
            prim_path="/World/envs/env_.*/Drone/body",
            history_length=1,
            track_pose=False,
        )
        self._contact_sensor = ContactSensor(contact_sensor_cfg)
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        # Hide the drone collision geometry mesh in the viewport so it doesn't render as a grey box
        from pxr import UsdGeom
        collision_prim = self.sim.stage.GetPrimAtPath("/World/envs/env_0/Drone/body/body_collision")
        if collision_prim.IsValid():
            UsdGeom.Imageable(collision_prim).MakeInvisible()

        # Initialize physical LiDAR (MultiMeshRayCaster) for 360 obstacle detection
        lidar_pattern = LidarPatternCfg(
            channels=1,
            vertical_fov_range=(0.0, 0.0),
            horizontal_fov_range=(-180.0, 180.0),
            horizontal_res=360.0 / 24.0
        )

        mesh_paths = [
            MultiMeshRayCasterCfg.RaycastTargetCfg(
                prim_expr="/World/envs/env_.*/Room",
                track_mesh_transforms=False  # Room is static and doesn't move
            )
        ]
        if self.cfg.num_pillars > 0:
            mesh_paths.append(
                MultiMeshRayCasterCfg.RaycastTargetCfg(
                    prim_expr="/World/envs/env_.*/Obstacle_.*",
                    track_mesh_transforms=True  # Obstacles relocate on reset
                )
            )

        lidar_cfg = MultiMeshRayCasterCfg(
            prim_path="/World/envs/env_.*/Drone/body",
            mesh_prim_paths=mesh_paths,
            pattern_cfg=lidar_pattern,
            max_distance=10.0,
            ray_alignment="base"
        )

        self._lidar = MultiMeshRayCaster(lidar_cfg)
        self.scene.sensors["lidar"] = self._lidar

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        self.scene.sensors["tiled_camera"] = self._tiled_camera

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ------------------------------------------------------------------
    # Depth preprocessing
    # ------------------------------------------------------------------
    def _preprocess_depth(self) -> torch.Tensor:
        """Get, clamp, and normalize depth to [0, 1]."""
        # Raw depth from camera: (B, H, W, 1)
        raw = self._tiled_camera.data.output["depth"].clone()
        # Replace inf with max depth
        raw[raw == float("inf")] = self.cfg.depth_max
        raw[raw != raw] = self.cfg.depth_max  # handle NaN
        # Clamp and normalize
        raw = raw.clamp(0.0, self.cfg.depth_max) / self.cfg.depth_max
        # Permute to (B, 1, H, W) — channels first
        depth = raw.permute(0, 3, 1, 2)
        self._last_depth_processed = depth
        return depth

    # ------------------------------------------------------------------
    # Physics step
    # ------------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor):
        """Convert high-level navigation actions to low-level motor commands."""
        self._previous_actions = self._actions.clone()
        self._actions = actions.clone().clamp(-1.0, 1.0)

        # --- PHASE 2: 6-DOF Release (Agile Navigation & Dodging) ---
        self._desired_vel_b[:, 0] = self._actions[:, 0] * self.cfg.vel_limit[0]
        self._desired_vel_b[:, 1] = self._actions[:, 1] * self.cfg.vel_limit[1]
        self._desired_vel_b[:, 2] = self._actions[:, 2] * self.cfg.vel_limit[2]
        self._target_yaw = wrap_to_pi(self._target_yaw + self._actions[:, 3] * self.cfg.yaw_rate_limit)

        # Prepare low-level controller observation
        lin_vel_b = self._robot.data.root_lin_vel_b
        ang_vel_b = self._robot.data.root_ang_vel_b
        projected_gravity_b = self._robot.data.projected_gravity_b
        current_yaw = self._get_drone_yaw()
        yaw_err = wrap_to_pi(self._target_yaw - current_yaw)

        ll_obs = torch.cat(
            [self._desired_vel_b, yaw_err.unsqueeze(-1), lin_vel_b, ang_vel_b, projected_gravity_b],
            dim=-1,
        )

        # Query the frozen flight controller
        with torch.no_grad():
            ll_actions = self.llc(ll_obs)
            ll_actions = ll_actions.clamp(-1.0, 1.0)

        self._thrust[:, 0, 2] = self.cfg.thrust_to_weight * self._robot_weight * (ll_actions[:, 0] + 1.0) / 2.0
        self._moment[:, 0, :] = self.cfg.moment_scale * ll_actions[:, 1:]

    def _apply_action(self):
        """Apply wrench to the drone body."""
        self._robot.permanent_wrench_composer.set_forces_and_torques(
            body_ids=self._body_id, forces=self._thrust, torques=self._moment
        )

    def _get_drone_yaw(self) -> torch.Tensor:
        _, _, yaw = euler_xyz_from_quat(self._robot.data.root_quat_w)
        return yaw

    def step(self, actions) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        # Call the parent step (handles simulation, rewards, and resets)
        obs, rewards, terminated, truncated, info = super().step(actions)
        
        # 1. Increment env0 step counter
        self._env0_step_counter += 1
        
        # 2. Accumulate step rewards for env0
        if hasattr(self, "_env0_last_step_rewards"):
            for key, val in self._env0_last_step_rewards.items():
                if key in self._env0_accumulated_rewards:
                    self._env0_accumulated_rewards[key] += val
        
        # 3. Track metrics if env 0 reset in this step
        if terminated[0].item() or truncated[0].item():
            reached_goal = (torch.linalg.norm(self._desired_pos_w[0] - self._robot.data.root_pos_w[0]) < self.cfg.goal_radius).item()
            if reached_goal:
                self._env0_last_episode_status = "GOAL REACHED!"
            elif truncated[0].item():
                self._env0_last_episode_status = "TIMEOUT"
            else:
                reason = getattr(self, "_env0_crash_reason", "Unknown Impact")
                self._env0_last_episode_status = f"CRASHED ({reason})"
                
            died = terminated[0].item() and not reached_goal
            self._env0_died_count += float(died)
            self._env0_timeout_count += float(truncated[0].item())
            self._env0_episode_count_window += 1
            if hasattr(self, "_env0_last_episode_length"):
                self._env0_episode_lengths.append(self._env0_last_episode_length)
            if hasattr(self, "_env0_last_final_dist"):
                self._env0_final_distances.append(self._env0_last_final_dist)
        
        # 4. Check if 100 steps completed for env0
        if self._env0_step_counter >= 100:
            # Update log values
            for key in self._env0_accumulated_rewards.keys():
                self._env0_log_values["Env0_Reward/" + key] = self._env0_accumulated_rewards[key] / 100.0
                self._env0_accumulated_rewards[key] = 0.0
            
            self._env0_log_values["Env0_Termination/died"] = self._env0_died_count
            self._env0_log_values["Env0_Termination/time_out"] = self._env0_timeout_count
            
            # Collision rate: crashes / episodes in this 100-step window
            if self._env0_episode_count_window > 0:
                self._env0_log_values["Env0_Metrics/collision_rate"] = self._env0_died_count / self._env0_episode_count_window
            else:
                self._env0_log_values["Env0_Metrics/collision_rate"] = 0.0
            
            self._env0_died_count = 0.0
            self._env0_timeout_count = 0.0
            self._env0_episode_count_window = 0
            
            if len(self._env0_episode_lengths) > 0:
                self._env0_log_values["Env0_Metrics/episode_length"] = sum(self._env0_episode_lengths) / len(self._env0_episode_lengths)
                self._env0_episode_lengths = []
            if len(self._env0_final_distances) > 0:
                self._env0_log_values["Env0_Metrics/final_distance_to_goal"] = sum(self._env0_final_distances) / len(self._env0_final_distances)
                self._env0_final_distances = []
                
            self._env0_step_counter = 0

        # 5. Ensure "log" is in self.extras, and update it with our Env0 values
        if "log" not in self.extras:
            self.extras["log"] = {}
        self.extras["log"].update(self._env0_log_values)
        
        # We also need to log total_steps on every single step so it is always present for step alignment!
        self.extras["log"]["Metrics/total_steps"] = float(self.common_step_counter)
        
        # Manually trigger debug visualization callback on each programmatic step
        if self.cfg.debug_vis:
            self._debug_vis_callback(None)
            
        return obs, rewards, terminated, truncated, info

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------
    def _compute_lidar_scan(self) -> torch.Tensor:
        """Compute 2D LiDAR range scan in the body frame using physical MultiMeshRayCaster."""
        hit_positions = self._lidar.data.ray_hits_w  # (num_envs, 24, 3)
        sensor_pos = self._lidar.data.pos_w.unsqueeze(1)  # (num_envs, 1, 3)
        
        # Calculate Euclidean distances
        distances = torch.norm(hit_positions - sensor_pos, dim=-1)  # (num_envs, 24)
        
        # Filter out hits that are on the floor or ceiling (Z world coordinates)
        # Floor is at Z=0.0, ceiling is at Z = 2.5 * map_scale. When drone tilts, rays hit the floor/ceiling.
        map_scale = getattr(self.cfg, "map_scale", 1.0)
        hit_z = hit_positions[..., 2]  # (num_envs, 24)
        is_floor_or_ceiling = (hit_z < 0.15) | (hit_z > (2.5 * map_scale - 0.10))
        distances[is_floor_or_ceiling] = 10.0
        
        # Clamp to [0.1, 10.0] meters. Non-hits return float('inf') which clamps to 10.0.
        self._last_lidar_scan = distances.clamp(min=0.1, max=10.0)
        return self._last_lidar_scan


    def _get_observations(self) -> dict:
        """Build 73-dim flat observation vector.

        Pipeline:
          1. Preprocess depth → (B, 1, 72, 128) normalized
          2. AE encode (detached) → z_img (B, 32)
          3. Compute state features → (B, 13)
          4. Concatenate with previous actions and 2D LiDAR scan → (B, 73) flat policy observation
        """
        # Step 1: preprocess depth
        depth = self._preprocess_depth()

        # Step 2: AE encode (no gradients for RL)
        z_img = self.ae.encode_detached(depth)  # (B, 32)
        

        # Step 3: state features
        desired_pos_b, _ = subtract_frame_transforms(
            self._robot.data.root_pos_w, self._robot.data.root_quat_w, self._desired_pos_w
        )
        target_dist = torch.linalg.norm(
            self._desired_pos_w - self._robot.data.root_pos_w, dim=1, keepdim=True
        )  # (B, 1)

        # Normalize target relative position and distance for stable network learning
        target_dir_b = desired_pos_b / (target_dist + 1e-6)
        target_dist_scaled = torch.tanh(target_dist / 10.0)

        # Use the already computed 2D LiDAR range scan and normalize to [0, 1]
        if self._last_lidar_scan is None:
            self._compute_lidar_scan()
            
        # Determine LiDAR visibility based on curriculum level
        # KEY DESIGN: Front rays (camera FOV) are ALWAYS limited, even at Level 1-2.
        # This forces the policy to use the camera from day one, preventing LiDAR-only strategies.
        lidar_obs_max_range = 10.0
        lidar_scan = self._last_lidar_scan.clone()
        front_indices = [10, 11, 12, 13, 14]  # Rays within the camera's ~47° FOV
        
        # Build per-ray limits based on curriculum level
        limits = torch.ones((self.num_envs, 24), device=self.device) * lidar_obs_max_range  # Default: full range
        
        if self.curriculum_level <= 2:
            # Levels 1-2: Sides/back = full 10m, Front = 2m only (force camera for forward navigation)
            limits[:, front_indices] = 2.0
        elif self.curriculum_level == 3:
            # Level 3: All rays limited to 0.8m
            limits[:, :] = 0.8
        elif self.curriculum_level == 4:
            # Level 4: All rays limited to 0.5m (emergency-only)
            limits[:, :] = 0.5
        else:
            # Level 5: Front = blind (0.0m), sides = last-resort (0.15m)
            limits[:, :] = 0.15
            limits[:, front_indices] = 0.0
        
        # Apply masking: any ray reading beyond its limit becomes "max range" (no obstacle detected)
        blind_mask = lidar_scan > limits
        lidar_scan[blind_mask] = lidar_obs_max_range
            
        # Store the masked scan for 3D debug visualization (so viewport matches what the policy sees)
        self._last_masked_lidar_scan = lidar_scan.clone()
        
        # Debug print (once) to verify masking is active
        if not getattr(self, "_masking_verified", False):
            import sys
            is_play = any("play.py" in arg or "play_saliency.py" in arg for arg in sys.argv)
            if is_play:
                raw = self._last_lidar_scan[0]
                masked = lidar_scan[0]  # Still in meters (not yet normalized)
                print(f"\n[DEBUG MASKING] Curriculum Level: {self.curriculum_level}")
                print(f"[DEBUG MASKING] Per-ray limits (env 0): {[f'{v:.2f}' for v in limits[0].tolist()]}")
                print(f"[DEBUG MASKING] Raw LiDAR (env 0):      {[f'{v:.2f}' for v in raw.tolist()]}")
                print(f"[DEBUG MASKING] Masked LiDAR (env 0):   {[f'{v:.2f}' for v in masked.tolist()]}")
                self._masking_verified = True
        
        lidar_scan = lidar_scan.clamp(max=lidar_obs_max_range) / lidar_obs_max_range

        # Step 4: concatenate all
        obs = torch.cat(
            [
                z_img,                                # (B, 32) — AE latent
                target_dir_b,                         # (B, 3)  — target unit direction in body frame
                target_dist_scaled,                   # (B, 1)  — scaled target distance
                self._robot.data.root_lin_vel_b,      # (B, 3)  — linear velocity
                self._robot.data.root_ang_vel_b,      # (B, 3)  — angular velocity
                self._robot.data.projected_gravity_b,  # (B, 3)  — orientation summary
                self._actions,                        # (B, 4)  — previous actions
                lidar_scan,                           # (B, 24) — normalized 2D LiDAR range scan
            ],
            dim=-1,
        )  # Total: 32 + 3 + 1 + 3 + 3 + 3 + 4 + 24 = 73

        import sys
        is_play_script = any("play.py" in arg or "play_saliency.py" in arg for arg in sys.argv)
        if (is_play_script and self.cfg.debug_vis) or getattr(self.cfg, "show_ae_images", False):
            self._update_dashboard(depth)

        return {"policy": obs}

    def _update_dashboard(self, depth: torch.Tensor) -> None:
        """Render a unified, premium navigation dashboard in a single OpenCV window."""
        if cv2 is None or np is None:
            return

        # 1. Compute AE reconstruction
        with torch.no_grad():
            z = self.ae.encode(depth)
            recon = self.ae.decode(z)

        depth_img = depth[0, 0].detach().cpu().numpy()
        recon_img = recon[0, 0].detach().cpu().numpy()

        depth_vis = np.uint8(np.clip(depth_img * 255.0, 0, 255))
        recon_vis = np.uint8(np.clip(recon_img * 255.0, 0, 255))

        # Resize AE images to 256x144 for clarity
        ae_w, ae_h = 256, 144
        depth_resized = cv2.resize(depth_vis, (ae_w, ae_h), interpolation=cv2.INTER_NEAREST)
        recon_resized = cv2.resize(recon_vis, (ae_w, ae_h), interpolation=cv2.INTER_NEAREST)
        
        # Convert to BGR
        depth_bgr = cv2.cvtColor(depth_resized, cv2.COLOR_GRAY2BGR)
        recon_bgr = cv2.cvtColor(recon_resized, cv2.COLOR_GRAY2BGR)

        # 2. Render LiDAR 2D polar plot (300x300)
        lidar_size = 300
        lidar_img = np.zeros((lidar_size, lidar_size, 3), dtype=np.uint8)
        center = (lidar_size // 2, lidar_size // 2)

        # Draw range grid circles
        scale = 15.0  # 15 pixels per meter
        cv2.circle(lidar_img, center, int(1.5 * scale), (40, 40, 40), 1)
        cv2.circle(lidar_img, center, int(5.0 * scale), (70, 70, 70), 1)
        cv2.putText(lidar_img, "1.5m", (center[0] + 5, center[1] - int(1.5 * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 80, 80), 1)
        cv2.putText(lidar_img, "5.0m", (center[0] + 5, center[1] - int(5.0 * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1)

        num_rays = 24
        ray_angles_b = torch.linspace(-torch.pi, torch.pi, num_rays + 1, device=self.device)[:-1]

        # Determine active threshold
        threshold = 10.0
        if self.curriculum_level == 1 or self.curriculum_level == 2:
            threshold = 10.0
        elif self.curriculum_level == 3:
            threshold = 0.8
        elif self.curriculum_level == 4:
            threshold = 0.5
        else:
            threshold = 0.0

        # Calculate masked scan for env 0 to show on the dashboard
        masked_scan = self._last_lidar_scan[0].clone()
        if threshold < 10.0:
            if getattr(self, "use_legacy_lidar_mask", False):
                front_indices = [0, 1, 2, 22, 23]
                front_rays = masked_scan[front_indices]
                blind_mask = front_rays > threshold
                front_rays[blind_mask] = 10.0
                masked_scan[front_indices] = front_rays
            else:
                if self.curriculum_level == 5:
                    limits = torch.ones(24, device=self.device) * 0.5
                    front_indices = [10, 11, 12, 13, 14]
                    limits[front_indices] = 0.0
                else:
                    limits = torch.ones(24, device=self.device) * threshold
                blind_mask = masked_scan > limits
                masked_scan[blind_mask] = 10.0

        # Draw the 24 rays
        import math
        for i in range(num_rays):
            physical_dist = self._last_lidar_scan[0, i].item()
            dist = masked_scan[i].item()
            angle = ray_angles_b[i].item()

            dx_b = math.cos(angle)
            dy_b = math.sin(angle)

            pt_end = (
                int(center[0] - dy_b * dist * scale),
                int(center[1] - dx_b * dist * scale)
            )

            is_masked_now = (physical_dist < 9.9) and (dist > 9.9)

            # Determine line color
            if is_masked_now:
                color = (0, 165, 255)  # Orange (Masked obstacle!)
            elif dist < 0.5:
                color = (0, 0, 255)  # Red (Very close obstacle!)
            elif i in [10, 11, 12, 13, 14]:
                color = (0, 255, 255)  # Yellow (Front ray cone)
            else:
                color = (0, 255, 0)  # Green (Normal active ray)

            cv2.line(lidar_img, center, pt_end, color, 1)
            cv2.circle(lidar_img, pt_end, 3, color, -1)

            if i % 2 == 0:
                text_pos = (
                    int(center[0] - dy_b * (dist + 0.3) * scale),
                    int(center[1] - dx_b * (dist + 0.3) * scale)
                )
                cv2.putText(lidar_img, str(i), text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)

        # Draw drone icon in center
        cv2.circle(lidar_img, center, 6, (255, 0, 0), -1)
        cv2.line(lidar_img, center, (center[0], center[1] - 12), (255, 255, 255), 2)

        # 3. Create the Main Dashboard layout (height=520, width=900)
        dash = np.zeros((520, 900, 3), dtype=np.uint8)

        # Place LiDAR scan on the left (centered vertically)
        dash[110:410, 0:300] = lidar_img

        # Place AE images side by side
        dash[0:144, 340:596] = depth_bgr
        dash[0:144, 596:852] = recon_bgr

        # Draw labels above AE images
        cv2.putText(dash, "AE INPUT DEPTH", (350, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(dash, "AE RECONSTRUCTION", (606, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        # 4. Fill text statistics on the right bottom panel
        dist_to_goal = torch.linalg.norm(self._desired_pos_w[0] - self._robot.data.root_pos_w[0]).item()
        
        # Calculate total reward and breakdown for Env 0
        ep_rewards = {key: self._episode_sums[key][0].item() for key in self._episode_sums}
        total_reward = sum(ep_rewards.values())

        # Determine current status
        status = "FLYING"
        status_color = (0, 255, 0)  # Green
        
        # If reset occurred in the last step
        if hasattr(self, "_env0_last_episode_status") and self._env0_last_episode_status:
            last_status = self._env0_last_episode_status
            if "SUCCESS" in last_status:
                last_color = (0, 255, 0)
            elif "TIMEOUT" in last_status:
                last_color = (0, 165, 255)
            else:
                last_color = (0, 0, 255)
        else:
            last_status = "N/A"
            last_color = (150, 150, 150)

        # Draw dividing lines
        cv2.line(dash, (320, 0), (320, 520), (50, 50, 50), 1)
        cv2.line(dash, (320, 150), (900, 150), (50, 50, 50), 1)

        # Write text values (Right Bottom Panel)
        y_start = 175
        dy_text = 20
        
        # Header Info
        cv2.putText(dash, "NAVIGATION STATUS:", (330, y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(dash, status, (530, y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2, cv2.LINE_AA)
        
        cv2.putText(dash, "LAST RESULT:", (330, y_start + dy_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(dash, last_status, (530, y_start + dy_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5, last_color, 1, cv2.LINE_AA)

        cv2.putText(dash, f"Curriculum Level: {self.curriculum_level}", (330, y_start + 2*dy_text), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(dash, f"Goal Distance: {dist_to_goal:.2f} m", (330, y_start + 3*dy_text), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(dash, f"Running Goal Rate: {self.running_goal_rate:.2%}", (330, y_start + 4*dy_text), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(dash, f"Total Episode Reward: {total_reward:.2f}", (330, y_start + 5*dy_text), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

        # Rewards Breakdown Column
        cv2.putText(dash, "REWARDS BREAKDOWN:", (330, y_start + 7*dy_text), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 100, 100), 1, cv2.LINE_AA)
        
        # Sort and print reward items in two columns to save space
        items = list(ep_rewards.items())
        half = (len(items) + 1) // 2
        for j, (k, val) in enumerate(items):
            col = 0 if j < half else 1
            row = j if j < half else j - half
            x_pos = 330 if col == 0 else 610
            y_pos = y_start + 8*dy_text + row*18
            cv2.putText(dash, f"{k}: {val:.2f}", (x_pos, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1, cv2.LINE_AA)

        # Show unified Dashboard
        cv2.imshow("Drone Navigation Dashboard (Env 0)", dash)
        cv2.waitKey(1)

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------
    def _get_rewards(self) -> torch.Tensor:
        """Compute reward — Phase 2: 6-DOF with heading lock."""
        curr_dist = torch.linalg.norm(
            self._desired_pos_w - self._robot.data.root_pos_w, dim=1
        )

        # 1. Progress reward
        progress = self._prev_dist_to_goal - curr_dist
        self._prev_dist_to_goal = curr_dist.clone()

        # 2. Goal reached
        reached_goal = (curr_dist < self.cfg.goal_radius).float()

        # 3. Time penalty
        time_penalty = torch.ones(self.num_envs, device=self.device)

        # 4. Heading alignment
        dx = self._desired_pos_w[:, 0] - self._robot.data.root_pos_w[:, 0]
        dy = self._desired_pos_w[:, 1] - self._robot.data.root_pos_w[:, 1]
        target_yaw = torch.atan2(dy, dx)
        _, _, current_yaw = euler_xyz_from_quat(self._robot.data.root_quat_w)
        heading_error = wrap_to_pi(target_yaw - current_yaw)
        heading_alignment = torch.cos(heading_error)


        # 5. Angular velocity penalty (roll + pitch stability)
        ang_vel_sq = torch.sum(self._robot.data.root_ang_vel_b ** 2, dim=1)

        # 6. Yaw rate penalty — punishes unnecessary turning (action[3])
        yaw_action_sq = self._actions[:, 3] ** 2

        # 7. Action magnitude penalty (smooth commands)
        action_sq = torch.sum(self._actions ** 2, dim=1)

        # 8. Sideslip penalty
        lateral_vel = self._robot.data.root_lin_vel_b[:, 1]
        sideslip_sq = lateral_vel ** 2

        # 9. Forward speed bonus — reward forward body velocity toward goal
        # Decay the bonus near the goal to prevent circling/reward-looping behavior (decay starts at 1.5m down to goal_radius)
        decay = torch.clamp((curr_dist - self.cfg.goal_radius) / (1.5 - self.cfg.goal_radius), min=0.0, max=1.0)
        forward_vel = self._robot.data.root_lin_vel_b[:, 0].clamp(min=0.0)
        forward_speed_bonus = forward_vel * heading_alignment.clamp(min=0.0) * decay

        # Velocity alignment (decayed near goal ONLY when moving towards it to prevent orbiting,
        # but fully penalized when moving away from the goal to prevent overshooting/looping)
        vel_w = self._robot.data.root_lin_vel_w
        to_goal_w = self._desired_pos_w - self._robot.data.root_pos_w
        speed = torch.linalg.norm(vel_w, dim=1)
        dot = torch.sum(vel_w * to_goal_w, dim=1)
        vel_align_denom = speed * curr_dist + 1e-6
        cos_sim = dot / vel_align_denom
        vel_align_max_speed = getattr(self.cfg, "vel_align_max_speed", self.cfg.vel_limit[0])
        speed_factor = (speed / vel_align_max_speed).clamp(0.0, 1.0)

        # Symmetric velocity alignment decay near goal to prevent shaking/overshooting oscillations
        velocity_alignment = cos_sim * speed_factor * decay

        # Proximity penalty: calculated from the physical LiDAR range scan (matching Red Run exactly)
        if self._last_lidar_scan is None:
            self._compute_lidar_scan()
        min_lidar_dist, _ = torch.min(self._last_lidar_scan, dim=1)
        proximity_radius = getattr(self.cfg, "pillar_proximity_radius", 1.2)
        proximity_penalty = ((proximity_radius - min_lidar_dist) / (proximity_radius + 1e-6)).clamp(min=0.0)
        speed_proximity_penalty = speed * proximity_penalty

        # Collision
        died_from_crash = (self.reset_terminated.float() - reached_goal).clamp(min=0.0)

        # Stuck penalty
        stuck_mask = (
            (progress.abs() < 1e-4) & (speed < 0.05) & (heading_alignment > 0.9)
        )
        stuck_penalty = stuck_mask.float() * -0.2

        # Action rate penalty — smooths out commands, reduces shaking
        action_rate_sq = torch.sum((self._actions - self._previous_actions) ** 2, dim=1)

        # Tilt penalty (encourage drone to stay level and avoid aggressive camera tilting)
        projected_gravity_b = self._robot.data.projected_gravity_b
        tilt_penalty = 1.0 - projected_gravity_b[:, 2].abs()

        # Z height deviation penalty — Soft zone from config with linear penalty outside
        z_pos = self._robot.data.root_pos_w[:, 2]
        z_low = getattr(self.cfg, "z_low", 0.7)
        z_high = getattr(self.cfg, "z_high", 1.5)
        z_deviation = torch.zeros_like(z_pos)
        z_deviation = torch.where(z_pos < z_low, z_low - z_pos, z_deviation)
        z_deviation = torch.where(z_pos > z_high, z_pos - z_high, z_deviation)

        rewards = {
            "progress": self.cfg.w_progress * progress,
            "goal": self.cfg.w_goal * reached_goal,
            "time": self.cfg.w_time * time_penalty,
            "heading": self.cfg.w_heading * heading_alignment,  # Removed avoidance_scale to restore smooth gradients
            "vel_align": getattr(self.cfg, "w_vel_align", 0.5) * velocity_alignment,  # Removed avoidance_scale
            "ang_vel": self.cfg.w_ang_vel * ang_vel_sq,
            "yaw_rate": getattr(self.cfg, "w_yaw_rate", -0.1) * yaw_action_sq,
            "forward_speed": getattr(self.cfg, "w_forward_speed", 0.3) * forward_speed_bonus,  # Removed avoidance_scale
            "action": self.cfg.w_action * action_sq,
            "action_rate": getattr(self.cfg, "w_action_rate", -0.02) * action_rate_sq,
            "sideslip": self.cfg.w_sideslip * sideslip_sq,
            "proximity": getattr(self.cfg, "w_proximity", 1.5) * (-proximity_penalty),  # Linear penalty matching Red Run
            "speed_proximity": getattr(self.cfg, "w_speed_proximity", -4.0) * speed_proximity_penalty,
            "stuck": stuck_penalty,
            "collision": self.cfg.collision_penalty * died_from_crash,
            "tilt": getattr(self.cfg, "w_tilt", -0.1) * tilt_penalty,
            "z_deviation": getattr(self.cfg, "w_z_deviation", -0.5) * z_deviation,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        # Clamp step reward to prevent PPO value function explosion.
        # min=-400 matches the collision_penalty so crashes don't dominate advantage estimates.
        reward = torch.clamp(reward, min=-400.0, max=600.0)

        # Store individual step rewards for env 0 to be accumulated in step()
        self._env0_last_step_rewards = {
            key: rewards[key][0].item() for key in rewards.keys()
        }

        # Accumulate for logging
        for key, value in rewards.items():
            if key not in self._episode_sums:
                self._episode_sums[key] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            self._episode_sums[key] += value

        return reward

    def _is_inside_map_obstacle(self, x: torch.Tensor, y: torch.Tensor, margin: float | None = None) -> torch.Tensor:
        """Check if (x, y) local positions are inside any static map obstacle.

        Args:
            x: Local X coordinates, shape (N,).
            y: Local Y coordinates, shape (N,).
            margin: Safety margin around obstacles. If None, uses spawn_obstacle_margin.

        Returns:
            Boolean tensor of shape (N,), True if inside any obstacle.
        """
        if margin is None:
            margin = getattr(self.cfg, "spawn_obstacle_margin", 0.5)
        inside = torch.zeros_like(x, dtype=torch.bool)
        for obs in self.cfg.map_obstacles:
            min_x, max_x, min_y, max_y = obs
            inside = inside | (
                (x >= min_x - margin) & (x <= max_x + margin)
                & (y >= min_y - margin) & (y <= max_y + margin)
            )
        return inside

    def _compute_inside_obstacle_penalty(self) -> torch.Tensor:
        """Per-step penalty: 1.0 if drone is inside/touching any map obstacle, 0.0 otherwise.

        Uses a tight margin (0.05m) so the penalty fires when the drone is
        genuinely overlapping an obstacle, not just near it.
        """
        pos_local = self._robot.data.root_pos_w[:, :3] - self._terrain.env_origins
        inside = self._is_inside_map_obstacle(pos_local[:, 0], pos_local[:, 1], margin=0.05)
        # Also check dynamic pillars
        if len(self._pillars) > 0:
            obstacle_dists = self._compute_obstacle_distances()
            for i in range(len(self._pillars)):
                inside = inside | (obstacle_dists[:, i] < self._drone_collision_offset)
        return inside.float()

    def _compute_obstacle_distances(self) -> torch.Tensor:
        """Compute the 3D Euclidean distance from the drone center to each obstacle boundary.

        Returns:
            torch.Tensor: Shape (num_envs, num_pillars) with distances to each obstacle's boundary.
        """
        if len(self._pillars) == 0:
            return torch.zeros(self.num_envs, 0, device=self.device)

        drone_pos = self._robot.data.root_pos_w[:, :3]  # (num_envs, 3)
        distances = []

        for i, pillar in enumerate(self._pillars):
            cluster_pos = pillar.data.root_pos_w[:, :3]  # (num_envs, 3)
            # Relative vector in world frame
            rel_w = drone_pos - cluster_pos  # (num_envs, 3)
            # Rotate relative vector into the local frame of the pillar
            rel_l = quat_rotate_inverse(pillar.data.root_quat_w, rel_w)  # (num_envs, 3)

            dx = rel_l[:, 0]
            dy = rel_l[:, 1]
            dz = rel_l[:, 2]

            shape = self._obstacle_shapes[i]
            if shape["type"] == "sphere":
                r = shape["radius"]
                dist_to_center = torch.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
                dist = torch.clamp(dist_to_center - r, min=0.0)
            elif shape["type"] == "cylinder":
                r = shape["radius"]
                hz = shape["half_z"]
                dist_xy = torch.clamp(torch.sqrt(dx ** 2 + dy ** 2) - r, min=0.0)
                dist_z = torch.clamp(torch.abs(dz) - hz, min=0.0)
                dist = torch.sqrt(dist_xy ** 2 + dist_z ** 2)
            elif shape["type"] == "box":
                hx = shape["half_x"]
                hy = shape["half_y"]
                hz = shape["half_z"]
                dist_x = torch.clamp(torch.abs(dx) - hx, min=0.0)
                dist_y = torch.clamp(torch.abs(dy) - hy, min=0.0)
                dist_z = torch.clamp(torch.abs(dz) - hz, min=0.0)
                dist = torch.sqrt(dist_x ** 2 + dist_y ** 2 + dist_z ** 2)
            else:
                raise ValueError(f"Unknown obstacle shape type: {shape['type']}")

            distances.append(dist)

        return torch.stack(distances, dim=1)  # (num_envs, num_pillars)

    def _compute_map_obstacle_distances(self) -> torch.Tensor:
        """Compute the 2D distance from the drone center to each static map obstacle.

        Returns:
            torch.Tensor: Shape (num_envs, num_obstacles) with 2D distances to each obstacle's boundary.
        """
        # Drone local position: shape (num_envs, 2)
        pos_local = self._robot.data.root_pos_w[:, :2] - self._terrain.env_origins[:, :2]
        drone_x = pos_local[:, 0].unsqueeze(1)  # (num_envs, 1)
        drone_y = pos_local[:, 1].unsqueeze(1)  # (num_envs, 1)

        # Convert map_obstacles to tensors on the correct device if not already done
        if not hasattr(self, "_map_obs_tensor"):
            obs_list = list(self.cfg.map_obstacles)
            self._map_obs_tensor = torch.tensor(obs_list, device=self.device, dtype=torch.float)  # (num_obstacles, 4)

        # (num_obstacles, 4) -> split into min_x, max_x, min_y, max_y
        min_x = self._map_obs_tensor[:, 0].unsqueeze(0)  # (1, num_obstacles)
        max_x = self._map_obs_tensor[:, 1].unsqueeze(0)  # (1, num_obstacles)
        min_y = self._map_obs_tensor[:, 2].unsqueeze(0)  # (1, num_obstacles)
        max_y = self._map_obs_tensor[:, 3].unsqueeze(0)  # (1, num_obstacles)

        # Compute signed distance along each axis
        dist_x = torch.clamp(min_x - drone_x, min=0.0) + torch.clamp(drone_x - max_x, min=0.0)  # (num_envs, num_obstacles)
        dist_y = torch.clamp(min_y - drone_y, min=0.0) + torch.clamp(drone_y - max_y, min=0.0)  # (num_envs, num_obstacles)

        # 2D Euclidean distance to the bounding box
        dist_2d = torch.sqrt(dist_x ** 2 + dist_y ** 2)  # (num_envs, num_obstacles)
        return dist_2d

    # ------------------------------------------------------------------
    # Termination
    # ------------------------------------------------------------------
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Terminate on floor/ceiling/wall collision or timeout."""
        self._last_lidar_scan = None  # Reset cached scan to force fresh sensor acquisition for this step
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        pos_local = self._robot.data.root_pos_w[:, :3] - self._terrain.env_origins

        map_scale = getattr(self.cfg, "map_scale", 1.0)
        hit_floor = pos_local[:, 2] < 0.1
        hit_ceiling = pos_local[:, 2] > 2.5 * map_scale
        hit_floor_or_ceiling = hit_floor | hit_ceiling
        # Arena map bounds scaled by map_scale: original physical wall meshes start at X/Y = ±24.0.
        map_scale = getattr(self.cfg, "map_scale", 1.0)
        wall_limit = 23.85 * map_scale
        hit_wall = (
            (pos_local[:, 0] > wall_limit) | (pos_local[:, 0] < -wall_limit)
            | (pos_local[:, 1] > wall_limit) | (pos_local[:, 1] < -wall_limit)
        )

        # Fixed collision radius for stable training and strict centering
        current_radius = self.cfg.pillar_collision_radius

        # 1. Check contact sensor for physical collision with meshes (forces > 0.1 N)
        contact_force = torch.linalg.norm(self._contact_sensor.data.net_forces_w[:, 0, :], dim=-1)
        hit_obstacle = contact_force > 0.1  # 0.1 Newton force threshold (high sensitivity)

        # Debug: print contact forces for env 0 when a collision is detected (only in play mode)
        import sys
        import os
        is_play = "play" in os.path.basename(sys.argv[0])
        if is_play:
            if hit_obstacle[0].item():
                print(f"[COLLISION] Env 0: contact_force={contact_force[0].item():.2f} N, "
                      f"pos=({pos_local[0,0].item():.2f}, {pos_local[0,1].item():.2f}, {pos_local[0,2].item():.2f})")
            # Also log total collision count across all envs each step (only if any collisions)
            total_collisions = hit_obstacle.sum().item()
            if total_collisions > 0:
                print(f"[COLLISION] Step {self.common_step_counter}: {int(total_collisions)}/{self.num_envs} envs hit obstacle")

        # 2. Check physical LiDAR distance: crash if drone is closer than collision radius to any obstacle/wall
        if self._last_lidar_scan is None:
            self._compute_lidar_scan()
        min_lidar_dist, _ = torch.min(self._last_lidar_scan, dim=1)
        hit_physical_obstacle = min_lidar_dist < current_radius

        # 3. Check analytical boxes (failsafe for LiDAR blind spots / clipping)
        hit_box_obstacle = self._is_inside_map_obstacle(
            pos_local[:, 0], pos_local[:, 1], margin=current_radius
        )

        # 4. Check analytical dynamic pillars (failsafe for dynamic obstacles)
        hit_dynamic_pillar = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if len(self._pillars) > 0:
            obstacle_dists = self._compute_obstacle_distances()
            for i in range(len(self._pillars)):
                hit_dynamic_pillar = hit_dynamic_pillar | (obstacle_dists[:, i] < current_radius)

        distance_to_goal = torch.linalg.norm(self._desired_pos_w - self._robot.data.root_pos_w, dim=1)
        reached_goal = distance_to_goal < self.cfg.goal_radius

        # Capture crash reason for Env 0
        if self.num_envs > 0:
            if hit_floor[0].item():
                self._env0_crash_reason = "Floor Bounds Collision"
            elif hit_ceiling[0].item():
                self._env0_crash_reason = "Ceiling Bounds Collision"
            elif hit_wall[0].item():
                self._env0_crash_reason = "Arena Wall Boundary"
            elif hit_dynamic_pillar[0].item() or (hit_physical_obstacle[0].item() and len(self._pillars) > 0 and torch.min(self._compute_obstacle_distances()[0]) < current_radius + 0.1):
                self._env0_crash_reason = "Dynamic Pillar Collision"
            elif hit_box_obstacle[0].item() or (hit_physical_obstacle[0].item() and torch.min(self._compute_map_obstacle_distances()[0]) < current_radius + 0.1):
                self._env0_crash_reason = "Static Wall Collision"
            elif hit_obstacle[0].item():
                # If contact sensor triggered, check height to distinguish floor/ceiling from obstacles
                z_val = pos_local[0, 2].item()
                if z_val < 0.25:
                    self._env0_crash_reason = "Floor Collision (Contact)"
                elif z_val > (2.5 * map_scale - 0.15):
                    self._env0_crash_reason = "Ceiling Collision (Contact)"
                else:
                    if len(self._pillars) > 0:
                        min_pillar_dist = torch.min(self._compute_obstacle_distances()[0]).item()
                    else:
                        min_pillar_dist = float('inf')
                    min_static_dist = torch.min(self._compute_map_obstacle_distances()[0]).item()
                    
                    if min_pillar_dist < min_static_dist:
                        self._env0_crash_reason = "Dynamic Pillar Collision (Contact)"
                    else:
                        self._env0_crash_reason = "Static Wall Collision (Contact)"
            elif hit_physical_obstacle[0].item():
                if len(self._pillars) > 0:
                    min_pillar_dist = torch.min(self._compute_obstacle_distances()[0]).item()
                else:
                    min_pillar_dist = float('inf')
                min_static_dist = torch.min(self._compute_map_obstacle_distances()[0]).item()
                
                if min_pillar_dist < min_static_dist:
                    self._env0_crash_reason = "Dynamic Pillar Collision (LiDAR)"
                else:
                    self._env0_crash_reason = "Static Wall Collision (LiDAR)"
            else:
                self._env0_crash_reason = None

        # Store exact crash reasons for all envs to support headless evaluation scripts
        self.crash_reasons = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.crash_reasons = torch.where(hit_floor, torch.tensor(1, device=self.device), self.crash_reasons)
        self.crash_reasons = torch.where(hit_ceiling, torch.tensor(2, device=self.device), self.crash_reasons)
        self.crash_reasons = torch.where(hit_wall, torch.tensor(3, device=self.device), self.crash_reasons)
        self.crash_reasons = torch.where(hit_obstacle | hit_physical_obstacle | hit_box_obstacle | hit_dynamic_pillar, torch.tensor(4, device=self.device), self.crash_reasons)

        died = (
            hit_floor_or_ceiling
            | hit_wall
            | hit_obstacle
            | hit_dynamic_pillar
        )
        if getattr(self, "is_brain_play", False):
            # In play mode, only reset on crash (died), do not reset when reaching intermediate waypoints
            terminated = died
        else:
            terminated = died | reached_goal
        return terminated, time_out

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset environments with randomized states."""
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        assert env_ids is not None
        env_count = env_ids.shape[0]

        # Capture env0 metrics before reset and clearing self._episode_sums
        if 0 in env_ids:
            self._env0_last_episode_length = float(self.episode_length_buf[0].item())
            self._env0_last_final_dist = torch.linalg.norm(
                self._desired_pos_w[0] - self._robot.data.root_pos_w[0]
            ).item()

        # Log metrics
        dist_per_env = torch.linalg.norm(
            self._desired_pos_w[env_ids] - self._robot.data.root_pos_w[env_ids], dim=1
        )
        final_dist = dist_per_env.mean()
        # Collect episode rewards for play script logging before resetting them
        import sys
        is_play_script = any("play.py" in arg or "play_saliency.py" in arg for arg in sys.argv)
        play_episode_sums = {}
        if is_play_script:
            for key in self._episode_sums.keys():
                play_episode_sums[key] = self._episode_sums[key][env_ids].clone()

        extras = dict()
        for key in self._episode_sums.keys():
            avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        
        # Collision & goal rates (across all envs in this reset batch)
        reached_goal_mask = dist_per_env < self.cfg.goal_radius
        crash_mask = self.reset_terminated[env_ids] & ~reached_goal_mask
        total_resets = max(len(env_ids), 1)
        batch_goal_rate = torch.count_nonzero(reached_goal_mask).item() / total_resets

        if not is_play_script:
            # Only count environments that were spawned at the current curriculum level
            valid_mask = self.env_curriculum_level[env_ids] == self.curriculum_level
            if torch.any(valid_mask):
                valid_env_ids = env_ids[valid_mask]
                valid_dist_per_env = dist_per_env[valid_mask]
                valid_reached_goal_mask = valid_dist_per_env < self.cfg.goal_radius
                valid_resets = len(valid_env_ids)
                valid_goal_rate = torch.count_nonzero(valid_reached_goal_mask).item() / valid_resets
                
                alpha = min(valid_resets / 800.0, 0.06)
                self.running_goal_rate = (1.0 - alpha) * self.running_goal_rate + alpha * valid_goal_rate
                
                # Advance curriculum level
                if self.running_goal_rate > 0.75 and self.curriculum_level < 5:
                    self.curriculum_level += 1
                    self.running_goal_rate = 0.50  # Start lower to prove performance; 0.25 margin above regression
                    print(f"\n[CURRICULUM] Advanced to Level {self.curriculum_level}! Running goal rate reset to 0.50.\n")
                    self._update_episode_length(self._get_episode_length_for_level(self.curriculum_level))
                # Regress curriculum level — only if truly failing (needs ~17 bad batches from 0.50)
                elif self.running_goal_rate < 0.25 and self.curriculum_level > getattr(self.cfg, "initial_curriculum_level", 1):
                    self.curriculum_level -= 1
                    self.running_goal_rate = 0.55  # Higher reset for quick re-advancement after brief dip
                    print(f"\n[CURRICULUM] Regressed to Level {self.curriculum_level}! Running goal rate reset to 0.55.\n")
                    self._update_episode_length(self._get_episode_length_for_level(self.curriculum_level))

            # Mark all reset environments as starting fresh at the current curriculum level
            self.env_curriculum_level[env_ids] = self.curriculum_level


        self.extras["log"]["Metrics/collision_rate"] = torch.count_nonzero(crash_mask).item() / total_resets
        self.extras["log"]["Metrics/goal_rate"] = batch_goal_rate
        self.extras["log"]["Metrics/curriculum_level"] = float(self.curriculum_level)
        self.extras["log"]["Metrics/running_goal_rate"] = float(self.running_goal_rate)
        
        # Log to CMD for PLAY scripts
        if is_play_script:
            for i, env_id in enumerate(env_ids):
                env_id_val = env_id.item()
                step_count = int(self.episode_length_buf[env_id].item())
                if reached_goal_mask[i]:
                    status = "SUCCESS (Reached Goal!)"
                elif crash_mask[i]:
                    status = "FAILED (Crashed!)"
                else:
                    status = "TIMEOUT"
                
                # Calculate total and detailed rewards for this episode
                ep_rewards = {key: play_episode_sums[key][i].item() for key in play_episode_sums}
                total_reward = sum(ep_rewards.values())
                reward_details = ", ".join([f"{k}: {val:.2f}" for k, val in ep_rewards.items()])
                
                print(
                    f"-------------\n"
                    f">> [AE] Attempt for Env {env_id_val}: {status} | STEPS = {step_count}\n"
                    f"   Total Reward: {total_reward:.2f}\n"
                    f"   Breakdown: {reward_details}\n"
                    f"-------------"
                )
        
        self.extras["log"]["Episode_Termination/died"] = torch.count_nonzero(crash_mask).item()
        self.extras["log"]["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"]["Metrics/final_distance_to_goal"] = final_dist.item()
        self.extras["log"]["Metrics/episode_length"] = torch.mean(self.episode_length_buf[env_ids].float()).item()
        self.extras["log"]["Metrics/total_steps"] = float(self.common_step_counter)

        # Always inject Env0 keys to ensure they are present in ep_extras[0]
        if hasattr(self, "_env0_log_values"):
            self.extras["log"].update(self._env0_log_values)

        # Reset robot
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        self._actions[env_ids] = 0.0

        # Spawn drone & target positions
        default_root_state = self._robot.data.default_root_state[env_ids].clone()

        if getattr(self.cfg, "corner_fine_tune", False):
            env_span = float(getattr(self.scene.cfg, "env_spacing", 6.0))
            corner_margin = float(getattr(self.cfg, "corner_margin", 0.2))
            max_offset = env_span / 2.0 - corner_margin

            diagonal_pairs = torch.tensor(
                [
                    [[-1.0, -1.0], [1.0, 1.0]],
                    [[1.0, -1.0], [-1.0, 1.0]],
                ],
                device=self.device,
            )
            pick = torch.randint(0, 2, (env_count,), device=self.device)
            chosen = diagonal_pairs[pick]

            drone_offsets = chosen[:, 0] * max_offset
            goal_offsets = chosen[:, 1] * max_offset

            default_root_state[:, 0] = self._terrain.env_origins[env_ids, 0] + drone_offsets[:, 0]
            default_root_state[:, 1] = self._terrain.env_origins[env_ids, 1] + drone_offsets[:, 1]
            default_root_state[:, 2] = 1.0

            self._desired_pos_w[env_ids, 0] = self._terrain.env_origins[env_ids, 0] + goal_offsets[:, 0]
            self._desired_pos_w[env_ids, 1] = self._terrain.env_origins[env_ids, 1] + goal_offsets[:, 1]
            self._desired_pos_w[env_ids, 2] = torch.ones(env_count, device=self.device) * getattr(self.cfg, "corner_goal_z", 1.0)
        else:
            map_scale = getattr(self.cfg, "map_scale", 1.0)
            spawn_limit = 20.0 * map_scale
            spawn_x = torch.zeros(env_count, device=self.device).uniform_(-spawn_limit, spawn_limit)
            spawn_y = torch.zeros(env_count, device=self.device).uniform_(-spawn_limit, spawn_limit)

            # Resample drone spawns that are inside map obstacles
            for _ in range(10):
                in_obstacle = self._is_inside_map_obstacle(spawn_x, spawn_y)
                if not torch.any(in_obstacle):
                    break
                n = torch.sum(in_obstacle).item()
                spawn_x[in_obstacle] = torch.zeros(n, device=self.device).uniform_(-spawn_limit, spawn_limit)
                spawn_y[in_obstacle] = torch.zeros(n, device=self.device).uniform_(-spawn_limit, spawn_limit)

            default_root_state[:, 0] = spawn_x + self._terrain.env_origins[env_ids, 0]
            default_root_state[:, 1] = spawn_y + self._terrain.env_origins[env_ids, 1]
            default_root_state[:, 2] = 1.0

            # Initialize goals at a distance determined by the curriculum level
            min_d, max_d = self.curriculum_distances[self.curriculum_level]
            
            # Generate a random angle and distance for each environment
            angle = torch.zeros(env_count, device=self.device).uniform_(0.0, 2 * 3.1415926535)
            dist = torch.zeros(env_count, device=self.device).uniform_(min_d, max_d)
            
            goal_x_local = spawn_x + dist * torch.cos(angle)
            goal_y_local = spawn_y + dist * torch.sin(angle)

            # Resample goals that are inside map obstacles or out of bounds [-23.7, 23.7] scaled
            # Wall at 23.90, collision at 23.85. Keep goal at least 23.70 to avoid forcing drone into walls.
            goal_limit = 23.7 * map_scale
            for attempt in range(25):
                in_obstacle = self._is_inside_map_obstacle(goal_x_local, goal_y_local)
                out_of_bounds = (goal_x_local.abs() > goal_limit) | (goal_y_local.abs() > goal_limit)
                bad = in_obstacle | out_of_bounds
                if not torch.any(bad):
                    break
                n = torch.sum(bad).item()
                
                # For bad environments, resample. Start scaling down distance if we struggle.
                scale = 1.0
                if attempt > 10:
                    scale = max(0.2, 1.0 - 0.05 * (attempt - 10))
                
                angle_resample = torch.zeros(n, device=self.device).uniform_(0.0, 2 * 3.1415926535)
                dist_resample = torch.zeros(n, device=self.device).uniform_(min_d * scale, max_d * scale)
                
                goal_x_local[bad] = spawn_x[bad] + dist_resample * torch.cos(angle_resample)
                goal_y_local[bad] = spawn_y[bad] + dist_resample * torch.sin(angle_resample)
            
            self._desired_pos_w[env_ids, 0] = goal_x_local + self._terrain.env_origins[env_ids, 0]
            self._desired_pos_w[env_ids, 1] = goal_y_local + self._terrain.env_origins[env_ids, 1]
            self._desired_pos_w[env_ids, 2] = 1.0

        # Orient drone to face target
        dx = self._desired_pos_w[env_ids, 0] - default_root_state[:, 0]
        dy = self._desired_pos_w[env_ids, 1] - default_root_state[:, 1]
        goal_yaw = torch.atan2(dy, dx)

        zeros = torch.zeros_like(goal_yaw)
        spawn_quat = quat_from_euler_xyz(zeros, zeros, goal_yaw)
        default_root_state[:, 3:7] = spawn_quat
        self._target_yaw[env_ids] = goal_yaw

        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)

        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        self._prev_dist_to_goal[env_ids] = torch.linalg.norm(
            self._desired_pos_w[env_ids] - default_root_state[:, :3], dim=1
        )

        # Randomize Pillar Positions and Orientations
        num_resets = env_count
        env_origins = self._terrain.env_origins[env_ids]

        # Shuffle the zone assignments for each env to randomize the spawn order
        zones_tensor = torch.tensor(self.cfg.pillar_x_zones, device=self.device)  # shape (6, 2)
        perms = torch.stack([torch.randperm(6, device=self.device) for _ in range(num_resets)], dim=0)  # shape (num_resets, 6)

        for i, pillar in enumerate(self._pillars):
            # Select randomized zone boundaries for this obstacle across all reset envs
            zone_idx = perms[:, i]
            chosen_zones = zones_tensor[zone_idx]
            x_lo = chosen_zones[:, 0]
            x_hi = chosen_zones[:, 1]

            y_lo, y_hi = self.cfg.pillar_y_range

            pillar_x = x_lo + torch.rand(num_resets, device=self.device) * (x_hi - x_lo)
            pillar_y = torch.zeros(num_resets, device=self.device).uniform_(y_lo, y_hi)

            state = pillar.data.default_root_state[env_ids].clone()
            state[:, 0] = pillar_x + env_origins[:, 0]
            state[:, 1] = pillar_y + env_origins[:, 1]
            state[:, 2] = self.cfg.pillar_z + env_origins[:, 2]

            # Randomize yaw (rotation around Z axis)
            pillar_yaw = torch.zeros(num_resets, device=self.device).uniform_(0.0, 2 * 3.141592653589793)
            zeros = torch.zeros_like(pillar_yaw)
            pillar_quat = quat_from_euler_xyz(zeros, zeros, pillar_yaw)
            state[:, 3:7] = pillar_quat

            state[:, 7:] = 0.0
            pillar.write_root_pose_to_sim(state[:, :7], env_ids)
            pillar.write_root_velocity_to_sim(state[:, 7:], env_ids)

    # ------------------------------------------------------------------
    # Debug vis
    # ------------------------------------------------------------------
    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                goal_marker_cfg = VisualizationMarkersCfg(
                    prim_path="/Visuals/Command/goal_position",
                    markers={
                        "sphere": sim_utils.SphereCfg(
                            radius=0.25,  # 50cm diameter
                            visual_material=sim_utils.PreviewSurfaceCfg(
                                diffuse_color=(0.1, 0.8, 1.0),  # Bright cyan
                                emissive_color=(0.0, 0.4, 0.5),  # Cyan glow
                                opacity=0.8,
                            ),
                        ),
                    },
                )
                self.goal_pos_visualizer = VisualizationMarkers(goal_marker_cfg)
            self.goal_pos_visualizer.set_visibility(True)

            if not hasattr(self, "drone_tracker_visualizer"):
                drone_marker_cfg = VisualizationMarkersCfg(
                    prim_path="/Visuals/DroneTracker",
                    markers={
                        "sphere": sim_utils.SphereCfg(
                            radius=0.25,  # 50cm diameter
                            visual_material=sim_utils.PreviewSurfaceCfg(
                                diffuse_color=(1.0, 0.1, 0.1),  # Bright red
                                emissive_color=(0.5, 0.0, 0.0),  # Red glow
                                opacity=0.7,
                            ),
                        ),
                    },
                )
                self.drone_tracker_visualizer = VisualizationMarkers(drone_marker_cfg)
            self.drone_tracker_visualizer.set_visibility(True)

            if not hasattr(self, "ceiling_visualizer"):
                map_scale = getattr(self.cfg, "map_scale", 1.0)
                ceiling_marker_cfg = VisualizationMarkersCfg(
                    prim_path="/Visuals/Ceiling",
                    markers={
                        "cuboid": sim_utils.CuboidCfg(
                            size=(48.0 * map_scale, 48.0 * map_scale, 0.02),  # Scaled thin glass sheet
                            visual_material=sim_utils.PreviewSurfaceCfg(
                                diffuse_color=(0.0, 0.8, 1.0),  # Cyan color
                                opacity=0.15,
                            ),
                        ),
                    },
                )
                self.ceiling_visualizer = VisualizationMarkers(ceiling_marker_cfg)
            self.ceiling_visualizer.set_visibility(True)

            if not hasattr(self, "pillar_zone_visualizer_list"):
                self.pillar_zone_visualizer_list = []
                offset = self._drone_collision_offset
                for i, shape in enumerate(self._obstacle_shapes):
                    if shape["type"] == "sphere":
                        marker_cfg = VisualizationMarkersCfg(
                            prim_path=f"/Visuals/ObstacleZone_{i}",
                            markers={
                                "sphere": sim_utils.SphereCfg(
                                    radius=shape["radius"] + offset,
                                    visual_material=sim_utils.PreviewSurfaceCfg(
                                        diffuse_color=(0.0, 1.0, 0.0),
                                        opacity=0.1,
                                    ),
                                ),
                            },
                        )
                    elif shape["type"] == "cylinder":
                        marker_cfg = VisualizationMarkersCfg(
                            prim_path=f"/Visuals/ObstacleZone_{i}",
                            markers={
                                "cylinder": sim_utils.CylinderCfg(
                                    radius=shape["radius"] + offset,
                                    height=shape["half_z"] * 2,
                                    visual_material=sim_utils.PreviewSurfaceCfg(
                                        diffuse_color=(0.0, 1.0, 0.0),
                                        opacity=0.1,
                                    ),
                                ),
                            },
                        )
                    elif shape["type"] == "box":
                        marker_cfg = VisualizationMarkersCfg(
                            prim_path=f"/Visuals/ObstacleZone_{i}",
                            markers={
                                "cuboid": sim_utils.CuboidCfg(
                                    size=(shape["size_x"] + 2 * offset, shape["size_y"] + 2 * offset, shape["half_z"] * 2),
                                    visual_material=sim_utils.PreviewSurfaceCfg(
                                        diffuse_color=(0.0, 1.0, 0.0),
                                        opacity=0.1,
                                    ),
                                ),
                            },
                        )
                    else:
                        raise ValueError(f"Unknown shape type {shape['type']}")
                    self.pillar_zone_visualizer_list.append(VisualizationMarkers(marker_cfg))
            for viz in self.pillar_zone_visualizer_list:
                viz.set_visibility(True)
            try:
                from omni.isaac.debug_draw import _debug_draw
                self._draw = _debug_draw.acquire_debug_draw_interface()
                if self._draw is None:
                    print("\n[DEBUG] omni.isaac.debug_draw.acquire_debug_draw_interface() returned None!\n")
                else:
                    print("\n[DEBUG] Debug draw interface acquired successfully!\n")
            except ImportError as e:
                print(f"\n[DEBUG] Failed to import omni.isaac.debug_draw: {e}\n")
                self._draw = None
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)
            if hasattr(self, "drone_tracker_visualizer"):
                self.drone_tracker_visualizer.set_visibility(False)
            if hasattr(self, "pillar_zone_visualizer_list"):
                for viz in self.pillar_zone_visualizer_list:
                    viz.set_visibility(False)
            if hasattr(self, "_draw") and self._draw is not None:
                self._draw.clear_lines()

    def _debug_vis_callback(self, event):
        # Prevent the visual goal sphere from spawning on top of the drone and blocking the camera
        goal_pos = self._desired_pos_w.clone()
        drone_pos = self._robot.data.root_pos_w[:, :3]
        dist_to_drone = torch.norm(goal_pos - drone_pos, dim=-1)
        too_close = dist_to_drone < 0.65
        goal_pos[too_close, 2] -= 10.0  # Temporarily hide underground
        self.goal_pos_visualizer.visualize(goal_pos)
        if hasattr(self, "drone_tracker_visualizer"):
            self.drone_tracker_visualizer.visualize(self._robot.data.root_pos_w[:, :3])
        if hasattr(self, "ceiling_visualizer"):
            map_scale = getattr(self.cfg, "map_scale", 1.0)
            z_ceil = 2.5 * map_scale + self._terrain.env_origins[0, 2].item()
            x_origin = self._terrain.env_origins[0, 0].item()
            y_origin = self._terrain.env_origins[0, 1].item()
            ceil_pos = torch.tensor([[x_origin, y_origin, z_ceil]], device=self.device)
            self.ceiling_visualizer.visualize(ceil_pos)
        if hasattr(self, "pillar_zone_visualizer_list") and len(self._pillars) > 0:
            for i, (pillar, viz) in enumerate(zip(self._pillars, self.pillar_zone_visualizer_list)):
                pos = pillar.data.root_pos_w[0, :3].unsqueeze(0)  # (1, 3)
                quat = pillar.data.root_quat_w[0, :4].unsqueeze(0)  # (1, 4)
                viz.visualize(pos, quat)

        # Draw 2D LiDAR ray lines in world frame for Env 0
        if hasattr(self, "_draw") and self._draw is not None and getattr(self, "_last_lidar_scan", None) is not None:
            self._draw.clear_lines()
            # Get drone position and orientation in world coordinates
            drone_pos = self._robot.data.root_pos_w[0, :3]  # (3,)
            drone_quat = self._robot.data.root_quat_w[0, :4]  # (4,)
            p_start = (drone_pos[0].item(), drone_pos[1].item(), drone_pos[2].item())

            num_rays = 24
            ray_angles_b = torch.linspace(-torch.pi, torch.pi, num_rays + 1, device=self.device)[:-1]

            # 1. Define ray directions in the drone's body frame
            cos_b = torch.cos(ray_angles_b)
            sin_b = torch.sin(ray_angles_b)
            zeros_b = torch.zeros_like(ray_angles_b)
            ray_dirs_b = torch.stack([cos_b, sin_b, zeros_b], dim=-1)  # (24, 3)

            # 2. Rotate ray directions to the world frame using the drone's quaternion
            quat_w = drone_quat.unsqueeze(0).repeat(num_rays, 1)  # (24, 4)
            ray_dirs_w = quat_rotate(quat_w, ray_dirs_b)  # (24, 3)

            start_points = []
            end_points = []
            colors = []
            thicknesses = []

            for i in range(num_rays):
                # Only draw rays that hit an actual obstacle within 9.0m (to exclude non-hits)
                raw_dist = self._last_lidar_scan[0, i].item()
                if raw_dist >= 9.0:
                    continue
                
                dir_w = ray_dirs_w[i]
                # Compute ray end point in world coordinates using rotated directions and real distance
                p_end = (
                    p_start[0] + raw_dist * dir_w[0].item(),
                    p_start[1] + raw_dist * dir_w[1].item(),
                    p_start[2] + raw_dist * dir_w[2].item(),
                )
                start_points.append(p_start)
                end_points.append(p_end)
                thicknesses.append(2.0)
                colors.append((0.0, 1.0, 0.0, 0.8))  # Translucent green for all active rays

            # --- Draw ceiling grid at Z = 2.5 * map_scale ---
            map_scale = getattr(self.cfg, "map_scale", 1.0)
            z_ceil = 2.5 * map_scale + self._terrain.env_origins[0, 2].item()
            x_origin = self._terrain.env_origins[0, 0].item()
            y_origin = self._terrain.env_origins[0, 1].item()
            x_min, x_max = x_origin - 24.0 * map_scale, x_origin + 24.0 * map_scale
            y_min, y_max = y_origin - 24.0 * map_scale, y_origin + 24.0 * map_scale

            ceil_color = (0.0, 0.8, 1.0, 0.4)  # cyan boundary
            ceil_thick = 3.0

            boundary_pts = [
                ((x_min, y_min, z_ceil), (x_max, y_min, z_ceil)),
                ((x_max, y_min, z_ceil), (x_max, y_max, z_ceil)),
                ((x_max, y_max, z_ceil), (x_min, y_max, z_ceil)),
                ((x_min, y_max, z_ceil), (x_min, y_min, z_ceil)),
            ]
            for p1, p2 in boundary_pts:
                start_points.append(p1)
                end_points.append(p2)
                colors.append(ceil_color)
                thicknesses.append(ceil_thick)

            # Draw grid lines inside ceiling
            grid_color = (0.0, 0.5, 0.8, 0.2)
            grid_thick = 1.0
            for x_val in range(-20, 25, 10):
                x_w = x_origin + x_val * map_scale
                start_points.append((x_w, y_min, z_ceil))
                end_points.append((x_w, y_max, z_ceil))
                colors.append(grid_color)
                thicknesses.append(grid_thick)
            for y_val in range(-20, 25, 10):
                y_w = y_origin + y_val * map_scale
                start_points.append((x_min, y_w, z_ceil))
                end_points.append((x_max, y_w, z_ceil))
                colors.append(grid_color)
                thicknesses.append(grid_thick)

            self._draw.draw_lines(start_points, end_points, colors, thicknesses)


