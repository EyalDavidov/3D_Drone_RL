"""PPO+AE Drone Environment.

Depth (128×72) → AE Encoder → z_img (32-dim)
z_img + target_rel_body + target_dist + lin_vel + ang_vel + gravity → PPO (45-dim)

The environment returns a flat 45-dim observation vector (NOT raw images).
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
    quat_rotate_inverse,
)

from isaaclab.markers import CUBOID_MARKER_CFG, VisualizationMarkersCfg  # isort: skip

from .ae_ppo_drone_env_cfg import AEPPODroneEnvCfg
from first_drone.models.ae import AE


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
        """Create drone, room (empty), dynamic pillars, terrain, camera, and lighting."""
        self._robot = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self._robot

        room_cfg = sim_utils.UsdFileCfg(usd_path=self.cfg.room_usd_path)
        room_cfg.func("/World/envs/env_0/Room", room_cfg)

        # --- Dynamic pillars (spawned as kinematic rigid objects) ---
        zone_centers = [(lo + hi) / 2.0 for lo, hi in self.cfg.pillar_x_zones]
        self._pillars = []
        for i in range(self.cfg.num_pillars):
            pillar_cfg = RigidObjectCfg(
                prim_path=f"/World/envs/env_.*/Pillar_{i}",
                spawn=self.cfg.pillar_spawn,
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

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------
    def _get_observations(self) -> dict:
        """Build 45-dim flat observation vector.

        Pipeline:
          1. Preprocess depth → (B, 1, 72, 128) normalized
          2. AE encode (detached) → z_img (B, 32)
          3. Compute state features → (B, 13)
          4. Concatenate → (B, 45) flat policy observation
        """
        # Step 1: preprocess depth
        depth = self._preprocess_depth()

        # Step 2: AE encode (no gradients for RL)
        z_img = self.ae.encode_detached(depth)  # (B, 32)
        
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
                z_img,                                # (B, 32) — AE latent
                desired_pos_b,                        # (B, 3)  — target in body frame
                target_dist,                          # (B, 1)  — scalar distance
                self._robot.data.root_lin_vel_b,      # (B, 3)  — linear velocity
                self._robot.data.root_ang_vel_b,      # (B, 3)  — angular velocity
                self._robot.data.projected_gravity_b,  # (B, 3)  — orientation summary
            ],
            dim=-1,
        )  # Total: 32 + 3 + 1 + 3 + 3 + 3 = 45

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
        forward_vel = self._robot.data.root_lin_vel_b[:, 0].clamp(min=0.0)
        forward_speed_bonus = forward_vel * heading_alignment.clamp(min=0.0)

        # Velocity alignment
        vel_w = self._robot.data.root_lin_vel_w
        to_goal_w = self._desired_pos_w - self._robot.data.root_pos_w
        speed = torch.linalg.norm(vel_w, dim=1)
        dot = torch.sum(vel_w * to_goal_w, dim=1)
        vel_align_denom = speed * curr_dist + 1e-6
        cos_sim = dot / vel_align_denom
        vel_align_max_speed = getattr(self.cfg, "vel_align_max_speed", self.cfg.vel_limit[0])
        speed_factor = (speed / vel_align_max_speed).clamp(0.0, 1.0)
        velocity_alignment = cos_sim * speed_factor

        # Proximity penalty
        proximity_penalty = torch.zeros(self.num_envs, device=self.device)
        proximity_radius = getattr(self.cfg, "pillar_proximity_radius", 0.5)
        for pillar in self._pillars:
            pillar_pos = pillar.data.root_pos_w[:, :3]
            dist = torch.linalg.norm((self._robot.data.root_pos_w[:, :2] - pillar_pos[:, :2]), dim=1)
            scaled = ((proximity_radius - dist) / (proximity_radius + 1e-6)).clamp(min=0.0)
            proximity_penalty = torch.maximum(proximity_penalty, scaled)
        proximity_penalty = -proximity_penalty

        # Collision
        died_from_crash = (self.reset_terminated.float() - reached_goal).clamp(min=0.0)

        # Stuck penalty
        stuck_mask = (
            (progress.abs() < 1e-4) & (speed < 0.05) & (heading_alignment > 0.9)
        )
        stuck_penalty = stuck_mask.float() * -0.2

        # Action rate penalty — smooths out commands, reduces shaking
        action_rate_sq = torch.sum((self._actions - self._previous_actions) ** 2, dim=1)

        rewards = {
            "progress": self.cfg.w_progress * progress,
            "goal": self.cfg.w_goal * reached_goal,
            "time": self.cfg.w_time * time_penalty,
            "heading": self.cfg.w_heading * heading_alignment,
            "vel_align": getattr(self.cfg, "w_vel_align", 0.5) * velocity_alignment,
            "ang_vel": self.cfg.w_ang_vel * ang_vel_sq,
            "yaw_rate": getattr(self.cfg, "w_yaw_rate", -0.1) * yaw_action_sq,
            "forward_speed": getattr(self.cfg, "w_forward_speed", 0.3) * forward_speed_bonus,
            "action": self.cfg.w_action * action_sq,
            "action_rate": getattr(self.cfg, "w_action_rate", -0.02) * action_rate_sq,
            "sideslip": self.cfg.w_sideslip * sideslip_sq,
            "proximity": getattr(self.cfg, "w_proximity", 1.0) * proximity_penalty,
            "stuck": stuck_penalty,
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
        """Terminate on floor/ceiling/wall collision or timeout."""
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        pos_local = self._robot.data.root_pos_w[:, :3] - self._terrain.env_origins

        hit_floor_or_ceiling = (pos_local[:, 2] < 0.1) | (pos_local[:, 2] > 1.9)
        hit_wall = (
            (pos_local[:, 0] > 1.87) | (pos_local[:, 0] < -1.87)
            | (pos_local[:, 1] > 1.87) | (pos_local[:, 1] < -1.87)
        )

        # Check dynamic pillar collisions
        pillar_radius = self.cfg.pillar_collision_radius
        hit_pillar = torch.zeros_like(hit_wall)
        for pillar in self._pillars:
            pillar_pos = pillar.data.root_pos_w[:, :3]
            dist_sq = torch.sum((self._robot.data.root_pos_w[:, :2] - pillar_pos[:, :2]) ** 2, dim=1)
            hit_pillar = hit_pillar | (dist_sq < (pillar_radius ** 2))

        distance_to_goal = torch.linalg.norm(self._desired_pos_w - self._robot.data.root_pos_w, dim=1)
        reached_goal = distance_to_goal < self.cfg.goal_radius

        died = hit_floor_or_ceiling | hit_wall | hit_pillar
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

        # Log metrics
        final_dist = torch.linalg.norm(
            self._desired_pos_w[env_ids] - self._robot.data.root_pos_w[env_ids], dim=1
        ).mean()
        extras = dict()
        for key in self._episode_sums.keys():
            avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        self.extras["log"]["Episode_Termination/died"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        self.extras["log"]["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"]["Metrics/final_distance_to_goal"] = final_dist.item()
        self.extras["log"]["Metrics/episode_length"] = torch.mean(self.episode_length_buf[env_ids].float()).item()

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
            default_root_state[:, 2] = torch.zeros(env_count, device=self.device).uniform_(0.5, 1.5)

            self._desired_pos_w[env_ids, 0] = self._terrain.env_origins[env_ids, 0] + goal_offsets[:, 0]
            self._desired_pos_w[env_ids, 1] = self._terrain.env_origins[env_ids, 1] + goal_offsets[:, 1]
            self._desired_pos_w[env_ids, 2] = torch.ones(env_count, device=self.device) * getattr(self.cfg, "corner_goal_z", 1.0)
        else:
            self._desired_pos_w[env_ids, 0] = torch.zeros_like(self._desired_pos_w[env_ids, 0]).uniform_(-1.0, 1.0) + self._terrain.env_origins[env_ids, 0]
            self._desired_pos_w[env_ids, 1] = -1.0 + self._terrain.env_origins[env_ids, 1]
            self._desired_pos_w[env_ids, 2] = torch.zeros_like(self._desired_pos_w[env_ids, 2]).uniform_(0.5, 1.5)
            spawn_x = torch.zeros(env_count, device=self.device).uniform_(-1.0, 1.0)
            default_root_state[:, 0] = spawn_x + self._terrain.env_origins[env_ids, 0]
            default_root_state[:, 1] = self.cfg.spawn_y_offset + self._terrain.env_origins[env_ids, 1]
            default_root_state[:, 2] = torch.zeros(env_count, device=self.device).uniform_(0.5, 1.5)

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

        # Randomize Pillar Positions
        num_resets = env_count
        env_origins = self._terrain.env_origins[env_ids]
        for i, pillar in enumerate(self._pillars):
            x_lo, x_hi = self.cfg.pillar_x_zones[i]
            y_lo, y_hi = self.cfg.pillar_y_range

            state = pillar.data.default_root_state[env_ids].clone()
            pillar_x = torch.zeros(num_resets, device=self.device).uniform_(x_lo, x_hi)
            pillar_y = torch.zeros(num_resets, device=self.device).uniform_(y_lo, y_hi)

            state[:, 0] = pillar_x + env_origins[:, 0]
            state[:, 1] = pillar_y + env_origins[:, 1]
            state[:, 2] = self.cfg.pillar_z + env_origins[:, 2]
            state[:, 3] = 1.0
            state[:, 4:7] = 0.0
            state[:, 7:] = 0.0
            pillar.write_root_pose_to_sim(state[:, :7], env_ids)
            pillar.write_root_velocity_to_sim(state[:, 7:], env_ids)

    # ------------------------------------------------------------------
    # Debug vis
    # ------------------------------------------------------------------
    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.05, 0.05, 0.05)
                marker_cfg.prim_path = "/Visuals/Command/goal_position"
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
            self.goal_pos_visualizer.set_visibility(True)

            if not hasattr(self, "pillar_zone_visualizers"):
                r = self.cfg.pillar_collision_radius
                pillar_marker_cfg = VisualizationMarkersCfg(
                    prim_path="/Visuals/PillarZones",
                    markers={
                        "cylinder": sim_utils.CylinderCfg(
                            radius=r,
                            height=2.5,
                            visual_material=sim_utils.PreviewSurfaceCfg(
                                diffuse_color=(0.0, 1.0, 0.0),
                                opacity=0.1,
                            ),
                        ),
                    },
                )
                self.pillar_zone_visualizers = VisualizationMarkers(pillar_marker_cfg)
            self.pillar_zone_visualizers.set_visibility(True)
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)
            if hasattr(self, "pillar_zone_visualizers"):
                self.pillar_zone_visualizers.set_visibility(False)

    def _debug_vis_callback(self, event):
        self.goal_pos_visualizer.visualize(self._desired_pos_w)
        if hasattr(self, "pillar_zone_visualizers") and len(self._pillars) > 0:
            pillar_positions = torch.stack(
                [p.data.root_pos_w[0, :3] for p in self._pillars], dim=0
            )
            self.pillar_zone_visualizers.visualize(pillar_positions)
