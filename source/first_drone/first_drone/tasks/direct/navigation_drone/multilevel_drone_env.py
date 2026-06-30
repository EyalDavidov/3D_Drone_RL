"""Multi-Level PPO+AE Drone Navigation Environment.

4 levels from final.usd. Each episode the agent is assigned a random level.
The target for each level is the spawn point of the next level (or the finish
point for level 4). The episode terminates when the drone reaches the target
or times out (10s for levels 1-2, 25s for levels 3-4).

Depth (128×72) → AE Encoder → z_img (32-dim)
z_img + target_rel_body + target_dist + lin_vel + ang_vel + gravity → PPO (45-dim)
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
from isaaclab.sensors import TiledCamera, ContactSensor
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import (
    subtract_frame_transforms,
    wrap_to_pi,
    quat_from_euler_xyz,
    euler_xyz_from_quat,
    quat_rotate_inverse,
)

from isaaclab.markers import CUBOID_MARKER_CFG, VisualizationMarkersCfg  # isort: skip

from .multilevel_drone_env_cfg import MultiLevelDroneEnvCfg
from first_drone.models.ae import AE


class MultiLevelDroneEnv(DirectRLEnv):
    """Multi-level PPO+AE drone navigation environment."""

    cfg: MultiLevelDroneEnvCfg

    def __init__(self, cfg: MultiLevelDroneEnvCfg, render_mode: str | None = None, **kwargs):
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

        # ----- Multi-level buffers -----
        self._current_level = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Precompute level data as tensors for fast indexing
        self._level_spawns_t = torch.tensor(self.cfg.level_spawns, dtype=torch.float, device=self.device)   # (4, 3)
        self._level_targets_t = torch.tensor(self.cfg.level_targets, dtype=torch.float, device=self.device)  # (4, 3)

        # Precompute max steps per level: duration / (dt * decimation)
        step_dt = self.cfg.sim.dt * self.cfg.decimation
        self._level_max_steps_lookup = torch.tensor(
            [int(d / step_dt) for d in self.cfg.level_durations], dtype=torch.long, device=self.device
        )  # (4,)

        # Initialize per-env max steps to the largest possible value (safe default)
        self._level_max_steps = torch.full(
            (self.num_envs,), self._level_max_steps_lookup.max().item(),
            dtype=torch.long, device=self.device
        )

        # ----- Episode reward logging -----
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "progress", "goal", "collision",
            ]
        }

        # AE visualization state
        self._ae_vis_step = 0

        # Debug visualization
        self.set_debug_vis(self.cfg.debug_vis)



    # ------------------------------------------------------------------
    # Scene setup
    # ------------------------------------------------------------------
    def _setup_scene(self):
        """Create drone, multi-level room, terrain, camera, and lighting."""
        self._robot = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self._robot

        room_cfg = sim_utils.UsdFileCfg(usd_path=self.cfg.room_usd_path)
        room_cfg.func("/World/envs/env_0/Room", room_cfg)

        # --- Poles (spawned as kinematic rigid objects) ---
        self._poles = []
        for i in range(self.cfg.num_poles):
            pole_cfg = RigidObjectCfg(
                prim_path=f"/World/envs/env_.*/Pole_{i}",
                spawn=self.cfg.pole_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.0, 0.0, -100.0)
                ),
            )
            pole = RigidObject(pole_cfg)
            self.scene.rigid_objects[f"pole_{i}"] = pole
            self._poles.append(pole)

        # --- Room 3 Obstacles (spawned as kinematic rigid objects) ---
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
                    init_state=RigidObjectCfg.InitialStateCfg(
                        pos=(0.0, 0.0, -100.0)
                    ),
                )
                obj = RigidObject(obj_cfg)
                self.scene.rigid_objects[f"room3_{name}_{i}"] = obj
                self._room3_obstacles.append(obj)

        # --- Room 4 Obstacles (spawned as kinematic rigid objects) ---
        self._corr1_obstacles = []
        for i in range(self.cfg.num_room4_corr1):
            cfg = RigidObjectCfg(
                prim_path=f"/World/envs/env_.*/Room4_corr1_{i}",
                spawn=self.cfg.corr1_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -100.0)),
            )
            obj = RigidObject(cfg)
            self.scene.rigid_objects[f"room4_corr1_{i}"] = obj
            self._corr1_obstacles.append(obj)

        self._corr2_obstacles = []
        for i in range(self.cfg.num_room4_corr2):
            cfg = RigidObjectCfg(
                prim_path=f"/World/envs/env_.*/Room4_corr2_{i}",
                spawn=self.cfg.corr2_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -100.0)),
            )
            obj = RigidObject(cfg)
            self.scene.rigid_objects[f"room4_corr2_{i}"] = obj
            self._corr2_obstacles.append(obj)

        # --- Color Room 3 Obstacles (Before cloning to other envs) ---
        try:
            import omni.usd
            from pxr import Usd, UsdGeom, Gf
            stage = omni.usd.get_context().get_stage()
            if stage:
                colors_dict = {
                    "wall": (0.15, 0.35, 0.95),          # Beautiful Blue
                    "cone": (0.95, 0.80, 0.05),          # Bright Yellow
                    "big_gate": (0.60, 0.10, 0.90),      # Royal Purple
                    "small_gate": (0.05, 0.80, 0.85),    # Vibrant Cyan
                    "poles_triangle": (0.10, 0.80, 0.25),# Emerald Green
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
                print("[INFO] Successfully colored Room 3 obstacles in source environment.")
        except Exception as e:
            print(f"[WARNING] Could not color Room 3 obstacles: {e}")

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self._tiled_camera = TiledCamera(self.cfg.tiled_camera)
        
        # Instantiate view camera
        from isaaclab.sensors import Camera
        self._view_camera = Camera(self.cfg.view_camera)

        # Contact sensor for collision detection with room geometry
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        self.scene.sensors["tiled_camera"] = self._tiled_camera

        # --- Viewport & Camera Configuration for env_0 ---
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
                    cam.GetFocalLengthAttr().Set(10.5)
                    cam.GetFocusDistanceAttr().Set(400.0)
                    
                    # Set as active viewport
                    viewport_window = omni.kit.viewport.utility.get_active_viewport_window()
                    if viewport_window is not None:
                        viewport_window.set_active_camera(cam_path)
                        print(f"[INFO] Successfully set active viewport camera to {cam_path}")
        except Exception as e:
            print(f"[WARNING] Could not set Viewport active camera (might be running headless): {e}")

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ------------------------------------------------------------------
    # Depth preprocessing
    # ------------------------------------------------------------------
    def _preprocess_depth(self) -> torch.Tensor:
        """Get, clamp, and normalize depth to [0, 1]."""
        raw = self._tiled_camera.data.output["depth"].clone()
        raw[raw == float("inf")] = self.cfg.depth_max
        raw[raw != raw] = self.cfg.depth_max  # handle NaN
        raw = (raw.clamp(0.0, self.cfg.depth_max) / self.cfg.depth_max) ** 1.7
        depth = raw.permute(0, 3, 1, 2)  # (B, 1, H, W)
        self._last_depth_processed = depth
        return depth

    # ------------------------------------------------------------------
    # Physics step
    # ------------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor):
        """Convert high-level navigation actions to low-level motor commands."""
        self._previous_actions = self._actions.clone()
        self._actions = actions.clone().clamp(-1.0, 1.0)

        # 6-DOF velocity commands
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

        if "log" not in self.extras:
            self.extras["log"] = {}
        self.extras["log"]["Metrics/total_steps"] = float(self.common_step_counter)

        return obs, rewards, terminated, truncated, info

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def _get_observations(self) -> dict:
        """Build 77-dim flat observation vector.

        Pipeline:
          1. Preprocess depth → (B, 1, 72, 128) normalized
          2. AE encode (detached) → z_img (B, 64)
          3. Compute state features → (B, 13)
          4. Concatenate → (B, 77) flat policy observation
        """
        # Step 1: preprocess depth
        depth = self._preprocess_depth()

        # Step 2: AE encode (no gradients for RL)
        z_img = self.ae.encode_detached(depth)  # (B, 64)

        if getattr(self.cfg, "show_ae_images", False):
            self._show_ae_images(depth)

        # Step 3: state features
        desired_pos_b, _ = subtract_frame_transforms(
            self._robot.data.root_pos_w, self._robot.data.root_quat_w, self._desired_pos_w
        )
        target_dist = torch.linalg.norm(
            self._desired_pos_w - self._robot.data.root_pos_w, dim=1, keepdim=True
        )  # (B, 1)

        # Step 4: concatenate all
        obs = torch.cat(
            [
                z_img,                                # (B, 64) — AE latent
                desired_pos_b,                        # (B, 3)  — target in body frame
                target_dist,                          # (B, 1)  — scalar distance
                self._robot.data.root_lin_vel_b,      # (B, 3)  — linear velocity
                self._robot.data.root_ang_vel_b,      # (B, 3)  — angular velocity
                self._robot.data.projected_gravity_b, # (B, 3)  — orientation summary
            ],
            dim=-1,
        )  # Total: 64 + 3 + 1 + 3 + 3 + 3 = 77

        return {"policy": obs}

    def _show_ae_images(self, depth: torch.Tensor) -> None:
        """Show AE input depth and reconstruction side by side."""
        if cv2 is None or np is None:
            return

        interval = getattr(self.cfg, "ae_image_display_interval", 20)
        if self._ae_vis_step % interval != 0:
            self._ae_vis_step += 1
            return

        self._ae_vis_step += 1
        with torch.no_grad():
            z = self.ae.encode(depth)
            recon = self.ae.decode(z)

        depth_img = depth[0, 0].detach().cpu().numpy()
        recon_img = recon[0, 0].detach().cpu().numpy()

        depth_vis = np.uint8(np.clip(depth_img * 255.0, 0, 255))
        recon_vis = np.uint8(np.clip(recon_img * 255.0, 0, 255))
        combined = np.hstack([depth_vis, recon_vis])
        scale = 4
        combined = cv2.resize(combined, (combined.shape[1] * scale, combined.shape[0] * scale),
                              interpolation=cv2.INTER_NEAREST)
        combined = cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR)

        label = "AE depth input (left) | reconstruction (right)"
        cv2.putText(combined, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("AE Input/Output", combined)
        cv2.waitKey(1)

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------
    def _get_rewards(self) -> torch.Tensor:
        """Compute reward for multi-level navigation."""
        curr_dist = torch.linalg.norm(
            self._desired_pos_w - self._robot.data.root_pos_w, dim=1
        )

        # 1. Progress reward
        progress = self._prev_dist_to_goal - curr_dist
        self._prev_dist_to_goal = curr_dist.clone()

        # Distance reward (1 - tanh)
        distance_reward = 1.0 - torch.tanh(curr_dist)

        # 2. Goal reached
        reached_goal = (curr_dist < self.cfg.goal_radius).float()

        # 3. Time penalty
        time_penalty = torch.ones(self.num_envs, device=self.device)

        # 4. Heading alignment
        dx = self._desired_pos_w[:, 0] - self._robot.data.root_pos_w[:, 0]
        dy = self._desired_pos_w[:, 1] - self._robot.data.root_pos_w[:, 1]
        target_yaw = torch.atan2(dy, dx)
        current_roll, current_pitch, current_yaw = euler_xyz_from_quat(self._robot.data.root_quat_w)
        heading_error = wrap_to_pi(target_yaw - current_yaw)
        heading_alignment = torch.cos(heading_error)

        # 5. Angular velocity penalty (roll + pitch stability)
        ang_vel_sq = torch.sum(self._robot.data.root_ang_vel_b ** 2, dim=1)

        # 6. Yaw rate penalty
        yaw_action_sq = self._actions[:, 3] ** 2

        # 7. Action magnitude penalty
        action_sq = torch.sum(self._actions ** 2, dim=1)

        # 8. Sideslip penalty
        lateral_vel = self._robot.data.root_lin_vel_b[:, 1]
        sideslip_sq = lateral_vel ** 2

        # 9. Forward velocity reward (encourages moving forward along local X)
        forward_vel = torch.clamp(self._robot.data.root_lin_vel_b[:, 0], min=0.0)

        # 10. Tilt angle penalty (dead-zone beyond ~18 degrees, i.e. cos(18) = 0.95)
        projected_gravity_b = self._robot.data.projected_gravity_b
        tilt_deviation = (0.95 - projected_gravity_b[:, 2].abs()).clamp(min=0.0)
        tilt_penalty = tilt_deviation ** 2

        # Collision — termination without reaching goal
        died_from_crash = (self.reset_terminated.float() - reached_goal).clamp(min=0.0)

        # Action rate penalty
        action_rate_sq = torch.sum((self._actions - self._previous_actions) ** 2, dim=1)

        rewards = {
            "progress": self.cfg.w_progress * progress,
            "distance": self.cfg.w_distance * distance_reward,
            "goal": self.cfg.w_goal * reached_goal,
            "time": self.cfg.w_time * time_penalty,
            "heading": self.cfg.w_heading * heading_alignment,
            "ang_vel": self.cfg.w_ang_vel * ang_vel_sq,
            "yaw_rate": self.cfg.w_yaw_rate * yaw_action_sq,
            "action": self.cfg.w_action * action_sq,
            "action_rate": self.cfg.w_action_rate * action_rate_sq,
            "sideslip": self.cfg.w_sideslip * sideslip_sq,
            "forward": self.cfg.w_forward * forward_vel,
            "tilt": self.cfg.w_tilt * tilt_penalty,
            "collision": self.cfg.collision_penalty * died_from_crash,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)



        # Accumulate for logging
        for key, value in rewards.items():
            if key not in self._episode_sums:
                self._episode_sums[key] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            self._episode_sums[key] += value

        return reward

    # ------------------------------------------------------------------
    # Termination
    # ------------------------------------------------------------------
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Terminate on collision, goal reached, or per-level timeout."""
        # Goal reached
        distance_to_goal = torch.linalg.norm(self._desired_pos_w - self._robot.data.root_pos_w, dim=1)
        reached_goal = distance_to_goal < self.cfg.goal_radius

        # Collision detection via contact sensor
        contact_forces = self._contact_sensor.data.net_forces_w
        hit_room = contact_forces.norm(dim=-1).max(dim=-1).values > self.cfg.contact_force_threshold

        # Height bounds collision: < 0.1m or > 1.9m relative to the environment's ground level
        relative_z = self._robot.data.root_pos_w[:, 2] - self._terrain.env_origins[:, 2]
        out_of_bounds = (relative_z < 0.1) | (relative_z > 1.9)
        hit_room = hit_room | out_of_bounds

        # If continuous_mode is enabled, handle in-flight level transitions
        if getattr(self.cfg, "continuous_mode", False):
            # We transition envs that reached their goal and are not at the final level (level 3)
            transition_mask = reached_goal & (self._current_level < 3)
            
            if torch.any(transition_mask):
                env_origins = self._terrain.env_origins
                # Transition those envs to the next level
                self._current_level[transition_mask] += 1
                
                # Update targets for transitioned envs
                new_targets = self._level_targets_t[self._current_level[transition_mask]]
                self._desired_pos_w[transition_mask] = new_targets + env_origins[transition_mask]
                
                # Reset previous distance for progress tracking
                self._prev_dist_to_goal[transition_mask] = torch.linalg.norm(
                    self._desired_pos_w[transition_mask] - self._robot.data.root_pos_w[transition_mask], dim=1
                )
                
                # Reset step counter for this level to avoid timeout
                self.episode_length_buf[transition_mask] = 0
                
                # Update max steps for the new level
                self._level_max_steps[transition_mask] = self._level_max_steps_lookup[self._current_level[transition_mask]]
                
                print(f"[INFO] Drone(s) transitioned in-flight to Level {(self._current_level[transition_mask] + 1).cpu().tolist()}")
                
                # Randomize and place obstacles for the new level in-flight
                transition_ids = torch.nonzero(transition_mask, as_tuple=False).flatten()
                self._randomize_obstacles(transition_ids)
                
            # In continuous mode, we only terminate on reaching the goal if it's the final level (level 3)
            final_goal_reached = reached_goal & (self._current_level == 3)
            terminated = final_goal_reached | hit_room
        else:
            terminated = reached_goal | hit_room

        # Per-level timeout: compare episode_length_buf against per-env max steps
        time_out = self.episode_length_buf >= self._level_max_steps - 1

        return terminated, time_out

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset environments with randomized level assignment."""
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        assert env_ids is not None
        env_count = env_ids.shape[0]



        # Log metrics
        dist_per_env = torch.linalg.norm(
            self._desired_pos_w[env_ids] - self._robot.data.root_pos_w[env_ids], dim=1
        )
        final_dist = dist_per_env.mean()
        extras = dict()
        for key in self._episode_sums.keys():
            avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)

        reached_goal_mask = dist_per_env < self.cfg.goal_radius
        crash_mask = self.reset_terminated[env_ids] & ~reached_goal_mask
        
        # Store outcomes for evaluation/tracking scripts
        if not hasattr(self, "last_completed_outcomes"):
            self.last_completed_outcomes = {}
        for idx, env_id in enumerate(env_ids.tolist()):
            if reached_goal_mask[idx]:
                self.last_completed_outcomes[env_id] = "success"
            elif self.reset_time_outs[env_id]:
                self.last_completed_outcomes[env_id] = "timeout"
            else:
                self.last_completed_outcomes[env_id] = "collision"

        total_resets = max(len(env_ids), 1)
        self.extras["log"]["Metrics/collision_rate"] = torch.count_nonzero(crash_mask).item() / total_resets
        self.extras["log"]["Metrics/goal_rate"] = torch.count_nonzero(reached_goal_mask).item() / total_resets

        self.extras["log"]["Episode_Termination/died"] = torch.count_nonzero(crash_mask).item()
        self.extras["log"]["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"]["Metrics/final_distance_to_goal"] = final_dist.item()
        self.extras["log"]["Metrics/episode_length"] = torch.mean(self.episode_length_buf[env_ids].float()).item()
        self.extras["log"]["Metrics/total_steps"] = float(self.common_step_counter)



        # Reset robot
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        self._actions[env_ids] = 0.0

        # --- Assign a level to each resetting env ---
        if getattr(self.cfg, "continuous_mode", False):
            levels = torch.zeros((env_count,), dtype=torch.long, device=self.device)
        elif hasattr(self.cfg, "force_level") and self.cfg.force_level is not None:
            levels = torch.full((env_count,), self.cfg.force_level, dtype=torch.long, device=self.device)
        else:
            levels = torch.randint(0, self.cfg.num_levels, (env_count,), device=self.device)
            
        self._current_level[env_ids] = levels

        # Set per-level max steps
        self._level_max_steps[env_ids] = self._level_max_steps_lookup[levels]



        # Get spawn and target positions for each env's level
        spawn_pos = self._level_spawns_t[levels]   # (env_count, 3)
        target_pos = self._level_targets_t[levels]  # (env_count, 3)

        # Set target (world frame = env_origin + local target)
        env_origins = self._terrain.env_origins[env_ids]
        self._desired_pos_w[env_ids] = target_pos + env_origins

        # Set drone spawn
        default_root_state = self._robot.data.default_root_state[env_ids].clone()
        
        # Add randomization for levels 1 and 2 (indices 0 and 1)
        # 2 meters range in Y => [-1.0, 1.0]
        # 1 meter range in Z => [-0.5, 0.5]
        rand_x = torch.zeros(env_count, device=self.device).uniform_(-1.0, 1.0)
        rand_z = torch.zeros(env_count, device=self.device).uniform_(-0.5, 0.5)
        
        # Disable randomization for levels 3 and 4
        mask_other = (levels > 1)
        rand_x[mask_other] = 0.0
        rand_z[mask_other] = 0.0

        default_root_state[:, 0] = spawn_pos[:, 0] + env_origins[:, 0] + rand_x
        default_root_state[:, 1] = spawn_pos[:, 1] + env_origins[:, 1] 
        default_root_state[:, 2] = spawn_pos[:, 2] + env_origins[:, 2] + rand_z

        # Fixed orientation from cf2x config: rot=(0.7071, 0.0, 0.0, -0.7071)
        default_root_state[:, 3] = 0.7071
        default_root_state[:, 4] = 0.0
        default_root_state[:, 5] = 0.0
        default_root_state[:, 6] = -0.7071

        # Reset yaw tracking to match the fixed orientation
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

        # Randomize all obstacles for the resetting environments
        self._randomize_obstacles(env_ids)

    def _randomize_obstacles(self, env_ids: torch.Tensor):
        """Randomize positions and states of all obstacles (Poles, Room 3, Room 4) for given env_ids."""
        num_resets = env_ids.shape[0]
        levels = self._current_level[env_ids]
        env_origins = self._terrain.env_origins[env_ids]

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
        
        is_level3 = (levels == 2)
        num_level3_resets = torch.count_nonzero(is_level3).item()
        
        if num_level3_resets > 0:
            # Generate a random permutation of the 12 grid cells for each Level 3 env
            perms = torch.stack([torch.randperm(12, device=self.device) for _ in range(num_level3_resets)])  # (num_level3_resets, 12)
            # Random yaw rotations for each obstacle
            import math
            rand_yaws = torch.zeros(num_level3_resets, 12, device=self.device).uniform_(0, 2 * math.pi)
            
        for j, obstacle in enumerate(self._room3_obstacles):
            state = obstacle.data.default_root_state[env_ids].clone()
            
            # Default: hide it under the ground
            obs_x = torch.zeros(num_resets, device=self.device)
            obs_y = torch.zeros(num_resets, device=self.device)
            obs_z = torch.full((num_resets,), -100.0, device=self.device)
            obs_qw = torch.ones(num_resets, device=self.device)
            obs_qz = torch.zeros(num_resets, device=self.device)
            
            if num_level3_resets > 0:
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
        is_level4 = (levels == 3)
        num_level4_resets = torch.count_nonzero(is_level4).item()

        if num_level4_resets > 0:
            # Generate random permutations of the 5 positions for each env
            perms_c1 = torch.stack([torch.randperm(5, device=self.device) for _ in range(num_level4_resets)])
            perms_c2 = torch.stack([torch.randperm(5, device=self.device) for _ in range(num_level4_resets)])
            
            # Shuffled heights for each env to ensure even distribution of heights (preventing all high/all low)
            perms_h1 = torch.stack([torch.randperm(5, device=self.device) for _ in range(num_level4_resets)])
            perms_h2 = torch.stack([torch.randperm(5, device=self.device) for _ in range(num_level4_resets)])
            
            # 5 distinct base positions along the corridor length to prevent overlap
            y_positions_c1 = torch.linspace(-17.2, -19.45, 5, device=self.device)
            x_positions_c2 = torch.linspace(-3.8, -0.65, 5, device=self.device)
            
            # 5 distinct height bins spanning [0.4, 1.6]
            z_positions = torch.linspace(0.4, 1.6, 5, device=self.device)

        # 1. Room 4.1 Corridor Obstacles (corr1)
        for j, obstacle in enumerate(self._corr1_obstacles):
            state = obstacle.data.default_root_state[env_ids].clone()
            
            obs_x = torch.zeros(num_resets, device=self.device)
            obs_y = torch.zeros(num_resets, device=self.device)
            obs_z = torch.full((num_resets,), -100.0, device=self.device)
            
            if num_level4_resets > 0:
                # All 5 obstacles are active
                # Assign position
                assigned_y = y_positions_c1[perms_c1[:, j]]
                # Add small noise (±0.05m) to Y
                noise_y = torch.zeros(num_level4_resets, device=self.device).uniform_(-0.05, 0.05)
                
                # Assign height from its shuffled bin
                assigned_z = z_positions[perms_h1[:, j]]
                # Add small noise (±0.05m) to Z to prevent perfect alignment
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
                # Assign position
                assigned_x = x_positions_c2[perms_c2[:, j]]
                # Add small noise (±0.05m) to X
                noise_x = torch.zeros(num_level4_resets, device=self.device).uniform_(-0.05, 0.05)
                
                # Assign height from its shuffled bin
                assigned_z = z_positions[perms_h2[:, j]]
                # Add small noise (±0.05m) to Z
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

    # ------------------------------------------------------------------
    # Debug vis
    # ------------------------------------------------------------------
    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                size = self.cfg.goal_radius * 1.5
                marker_cfg.markers["cuboid"].size = (size, size, size)
                marker_cfg.prim_path = "/Visuals/Command/goal_position"
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
            self.goal_pos_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        self.goal_pos_visualizer.visualize(self._desired_pos_w)
