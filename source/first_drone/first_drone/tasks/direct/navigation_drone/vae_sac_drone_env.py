"""SAC+VAE Drone Environment.

This environment implements the pipeline from "Vision Based Drone Obstacle
Avoidance by Deep RL", adapted for goal-directed navigation with hovering:

  Depth (128×72) → VAE Encoder → z_img (32-dim)
  z_img + target_rel_body + target_dist + lin_vel + ang_vel + gravity → SAC (45-dim)

The environment returns a flat 45-dim observation vector (NOT raw images).
The VAE is owned by the environment, trained online via the training script,
and its detached latent is fed to SAC.

Key differences from the PPO camera env:
  - Observations are flat vectors (VAE-encoded), not raw images
   - Reward function has 7 terms (progress, goal, hover, clearance, ang_vel, tilt, action)
  - Tracks previous distance for progress reward
  - Exposes raw depth for external VAE training
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

from .vae_sac_drone_env_cfg import SACDroneEnvCfg
from first_drone.models.vae import VAE


class SACDroneEnv(DirectRLEnv):
    """SAC+VAE drone navigation environment.

    Actions (4 continuous, clamped to [-1, 1]):
      - action[0]: desired body x velocity
      - action[1]: desired body y velocity
      - action[2]: desired body z velocity
      - action[3]: desired yaw rate

    The SAC agent is a high-level navigator. A frozen low-level flight
    controller converts the SAC actions into thrust/moment commands.
    """

    cfg: SACDroneEnvCfg

    def __init__(self, cfg: SACDroneEnvCfg, render_mode: str | None = None, **kwargs):
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

        # ----- VAE (owned by env, trained externally) -----
        self.vae = VAE(latent_dim=self.cfg.vae_latent_dim, beta=self.cfg.vae_beta).to(self.device)
        if hasattr(self.cfg, "vae_checkpoint_path") and self.cfg.vae_checkpoint_path is not None:
            import os
            if os.path.exists(self.cfg.vae_checkpoint_path):
                self.vae.load_state_dict(torch.load(self.cfg.vae_checkpoint_path, map_location=self.device))
                print(f"\n[INFO] VAE model loaded successfully from {self.cfg.vae_checkpoint_path}\n")
            else:
                print(f"\n[WARNING] VAE checkpoint not found at {self.cfg.vae_checkpoint_path}\n")
        self.vae.eval()

        # ----- Low-level flight controller -----
        self.llc = torch.jit.load(self.cfg.llc_checkpoint_path, map_location=self.device)
        self.llc.eval()
        for param in self.llc.parameters():
            param.requires_grad = False

        # ----- High-level navigator buffers -----
        self._desired_vel_b = torch.zeros(self.num_envs, 3, device=self.device)
        self._target_yaw = torch.zeros(self.num_envs, device=self.device)

        # ----- Depth image buffer (exposed for external VAE training) -----
        self._last_depth_processed = None

        # ----- Episode reward logging -----
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "progress", "goal", "collision",
            ]
        }

        # VAE visualization state
        self._vae_vis_step = 0

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
        """Get, clamp, and normalize depth to [0, 1].

        Returns:
            Preprocessed depth, shape (B, 1, 72, 128), values in [0, 1].
            Also stores it in self._last_depth_processed for VAE training.
        """
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
        """Convert high-level SAC navigation actions to low-level motor commands."""
        self._actions = actions.clone().clamp(-1.0, 1.0)

        # =========================================================================
        # CURRICULUM LEARNING (Toggle between Phase 1 and Phase 2)
        # =========================================================================

        # --- PHASE 1: Forced Forward Flight (Overcoming Hesitation) ---
        # Goal: Teach the drone that moving forward is good, ignoring lateral dodges.
        # UNCOMMENT THIS BLOCK FOR PHASE 1:
        # self._actions[:, 0] = self._actions[:, 0].clamp(0.0, 1.0)  # No reverse
        # self._actions[:, 1] = 0.0                                  # No strafing (lateral)
        # self._desired_vel_b[:, 0] = self._actions[:, 0] * self.cfg.vel_limit[0]
        # self._desired_vel_b[:, 1] = 0.0
        # self._desired_vel_b[:, 2] = self._actions[:, 2] * self.cfg.vel_limit[2]
        # self._target_yaw = wrap_to_pi(self._target_yaw + self._actions[:, 3] * self.cfg.yaw_rate_limit)

        # --- PHASE 2: 6-DOF Release (Agile Navigation & Dodging) ---
        # Goal: Full freedom to dodge randomized pillars.
        # UNCOMMENT THIS BLOCK FOR PHASE 2 (Current Default):
        self._desired_vel_b[:, 0] = self._actions[:, 0] * self.cfg.vel_limit[0]
        self._desired_vel_b[:, 1] = self._actions[:, 1] * self.cfg.vel_limit[1]
        self._desired_vel_b[:, 2] = self._actions[:, 2] * self.cfg.vel_limit[2]
        self._target_yaw = wrap_to_pi(self._target_yaw + self._actions[:, 3] * self.cfg.yaw_rate_limit)
        # =========================================================================

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
          2. VAE encode (detached) → z_img (B, 32)
          3. Compute state features → (B, 13)
          4. Concatenate → (B, 45) flat policy observation

        Returns dict with:
          - "policy": (B, 45) flat vector for SAC actor/critic
        """
        # Step 1: preprocess depth
        depth = self._preprocess_depth()

        # Step 2: VAE encode (no gradients for RL)
        z_img = self.vae.encode_detached(depth)  # (B, 32)
        #z_img=torch.zeros_like(z_img)

        if getattr(self.cfg, "show_vae_images", False):
            self._show_vae_images(depth)

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
                z_img,                                # (B, 32) — VAE latent
                desired_pos_b,                        # (B, 3)  — target in body frame
                target_dist,                          # (B, 1)  — scalar distance
                self._robot.data.root_lin_vel_b,      # (B, 3)  — linear velocity
                self._robot.data.root_ang_vel_b,      # (B, 3)  — angular velocity
                self._robot.data.projected_gravity_b,  # (B, 3)  — orientation summary
            ],
            dim=-1,
        )  # Total: 32 + 3 + 1 + 3 + 3 + 3 = 45

        return {"policy": obs}

    def _show_vae_images(self, depth: torch.Tensor) -> None:
        """Show VAE input depth and reconstruction side by side."""
        if cv2 is None or np is None:
            return

        interval = getattr(self.cfg, "vae_image_display_interval", 20)
        if self._vae_vis_step % interval != 0:
            self._vae_vis_step += 1
            return

        self._vae_vis_step += 1
        with torch.no_grad():
            # Explicitly encode and then use the DECODER to get the reconstruction
            mu, _ = self.vae.encode(depth)
            recon = self.vae.decode(mu)

        depth_img = depth[0, 0].detach().cpu().numpy()
        recon_img = recon[0, 0].detach().cpu().numpy()

        depth_vis = np.uint8(np.clip(depth_img * 255.0, 0, 255))
        recon_vis = np.uint8(np.clip(recon_img * 255.0, 0, 255))
        combined = np.hstack([depth_vis, recon_vis])
        # Scale up 4x so the window is clearly visible (128+128=256 wide, 72 tall → 1024x288)
        scale = 4
        combined = cv2.resize(combined, (combined.shape[1] * scale, combined.shape[0] * scale),
                              interpolation=cv2.INTER_NEAREST)
        combined = cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR)

        label = "VAE depth input (left) | reconstruction (right)"
        cv2.putText(combined, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("VAE Input/Output", combined)

        # Allow window to refresh without blocking simulation
        cv2.waitKey(1)

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------
    def _get_rewards(self) -> torch.Tensor:
        """Compute reward — Phase 2: 6-DOF with heading lock.

        Design principles:
          - Full movement freedom (forward, backward, lateral)
          - Strong heading reward locks gaze on goal at all times
          - Soft sideslip penalty discourages constant strafing
          - SUCCESS > CRASH > HOVER ordering preserved

        Terms:
          1. progress  — dense: getting closer to goal
          2. goal      — terminal: one-time bonus for reaching goal
          3. time      — per-step cost (anti-hesitation)
          4. heading   — per-step: cos(angle to goal) — gaze lock
          5. ang_vel   — tiny penalty for spinning
          6. action    — tiny penalty for jerky actions
          7. sideslip  — soft penalty for lateral velocity
          + collision  — one-time penalty on crash
        """
        # Current distance to goal (2D on X/Y to avoid height mismatch)
        curr_dist = torch.linalg.norm(
            (self._desired_pos_w - self._robot.data.root_pos_w)[:, :2], dim=1
        )

        # 1. Progress reward (dense breadcrumbs)
        progress = self._prev_dist_to_goal - curr_dist
        self._prev_dist_to_goal = curr_dist.clone()

        # 2. Goal reached (one-time terminal)
        reached_goal = (curr_dist < self.cfg.goal_radius).float()

        # 3. Time penalty (constant per-step cost)
        time_penalty = torch.ones(self.num_envs, device=self.device)

        # 4. Heading alignment — cos(angle between drone heading and goal direction)
        dx = self._desired_pos_w[:, 0] - self._robot.data.root_pos_w[:, 0]
        dy = self._desired_pos_w[:, 1] - self._robot.data.root_pos_w[:, 1]
        target_yaw = torch.atan2(dy, dx)
        _, _, current_yaw = euler_xyz_from_quat(self._robot.data.root_quat_w)
        heading_error = wrap_to_pi(target_yaw - current_yaw)
        heading_alignment = torch.cos(heading_error)  # [-1, +1]

        # 5. Angular velocity penalty (smooth flight)
        ang_vel_sq = torch.sum(self._robot.data.root_ang_vel_b ** 2, dim=1)

        # 6. Action magnitude penalty (smooth commands)
        action_sq = torch.sum(self._actions ** 2, dim=1)

        # 7. Sideslip penalty (lateral velocity damping)
        lateral_vel = self._robot.data.root_lin_vel_b[:, 1]
        sideslip_sq = lateral_vel ** 2

        # --- Velocity alignment (encourages moving toward goal) ---
        vel_w = self._robot.data.root_lin_vel_w  # world-frame linear vel (B,3)
        to_goal_w = self._desired_pos_w - self._robot.data.root_pos_w  # (B,3)
        speed = torch.linalg.norm(vel_w, dim=1)  # (B,)
        dot = torch.sum(vel_w * to_goal_w, dim=1)  # (B,)
        vel_align_denom = speed * curr_dist + 1e-6
        cos_sim = dot / vel_align_denom
        vel_align_max_speed = getattr(self.cfg, "vel_align_max_speed", self.cfg.vel_limit[0])
        speed_factor = (speed / vel_align_max_speed).clamp(0.0, 1.0)
        velocity_alignment = cos_sim * speed_factor

        # --- Proximity penalty to pillars (graduated) ---
        proximity_penalty = torch.zeros(self.num_envs, device=self.device)
        proximity_radius = getattr(self.cfg, "pillar_proximity_radius", getattr(self.cfg, "pillar_proximity_radius", 0.5))
        for pillar in self._pillars:
            pillar_pos = pillar.data.root_pos_w[:, :3]
            dist = torch.linalg.norm((self._robot.data.root_pos_w[:, :2] - pillar_pos[:, :2]), dim=1)
            inside = (dist < proximity_radius).float()
            # linear scale 0 at edge -> 1 at center
            scaled = ((proximity_radius - dist) / (proximity_radius + 1e-6)).clamp(min=0.0)
            # keep the max penalty across pillars to avoid double counting
            proximity_penalty = torch.maximum(proximity_penalty, scaled)

        # scale to [-1, 0] (edge ~ -0.0..-0.5, center -> -1.0)
        proximity_penalty = -proximity_penalty  # negative penalty

        # Collision (only on crash, not goal reach)
        died_from_crash = (self.reset_terminated.float() - reached_goal).clamp(min=0.0)

        # --- Small stuck penalty: facing goal but not moving or making progress ---
        stuck_mask = (
            (progress.abs() < 1e-4) & (speed < 0.05) & (heading_alignment > 0.9)
        )
        stuck_penalty = stuck_mask.float() * -0.2

        rewards = {
            "progress": self.cfg.w_progress * progress,
            "goal": self.cfg.w_goal * reached_goal,

            "time": self.cfg.w_time * time_penalty,
            "heading": self.cfg.w_heading * heading_alignment,
            "vel_align": getattr(self.cfg, "w_vel_align", 0.5) * velocity_alignment,
            "ang_vel": self.cfg.w_ang_vel * ang_vel_sq,
            "action": self.cfg.w_action * action_sq,
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
            if self._episode_sums[key].is_inference():
                self._episode_sums[key] = self._episode_sums[key].clone()
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
        # Box bounds for Empty_Room.usd: Y is from -2.0 to ~2.0, X is from -2.5 to ~2.5
        hit_wall = (
            (pos_local[:, 0] > 1.87) | (pos_local[:, 0] < -1.87)
            | (pos_local[:, 1] > 1.87) | (pos_local[:, 1] < -1.87)
        )

        # Check dynamic pillar collisions
        pillar_radius = self.cfg.pillar_collision_radius
        hit_pillar = torch.zeros_like(hit_wall)
        for pillar in self._pillars:
            pillar_pos = pillar.data.root_pos_w[:, :3]
            # 2D distance (XY) between drone and pillar
            dist_sq = torch.sum((self._robot.data.root_pos_w[:, :2] - pillar_pos[:, :2]) ** 2, dim=1)
            hit_pillar = hit_pillar | (dist_sq < (pillar_radius ** 2))

        # Use 2D (X/Y) distance to match reward's reached_goal calculation
        distance_to_goal = torch.linalg.norm((self._desired_pos_w - self._robot.data.root_pos_w)[:, :2], dim=1)
        reached_goal = distance_to_goal < self.cfg.goal_radius

        died = hit_floor_or_ceiling | hit_wall | hit_pillar
        # Terminate on goal reach (SUCCESS — not a crash)
        terminated = died | reached_goal
        return terminated, time_out

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset environments with 4 randomized states to prevent directional bias."""
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        # --- Logging ---
        # Final distance logged in 2D to match success/reward termination criteria
        final_dist = torch.linalg.norm(
            (self._desired_pos_w[env_ids] - self._robot.data.root_pos_w[env_ids])[:, :2], dim=1
        ).mean()
        extras = dict()
        for key in self._episode_sums.keys():
            avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = avg / self.max_episode_length_s
            if self._episode_sums[key].is_inference():
                self._episode_sums[key] = self._episode_sums[key].clone()
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        self.extras["log"]["Episode_Termination/died"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        self.extras["log"]["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"]["Metrics/final_distance_to_goal"] = final_dist.item()
        self.extras["log"]["Metrics/episode_length"] = torch.mean(self.episode_length_buf[env_ids].float()).item()

        # --- Reset robot ---
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        self._actions[env_ids] = 0.0

        # =========================================================================
        # SPAWN RANDOMIZATION (PHASE 1 vs PHASE 2 NEW)
        # =========================================================================

        # --- PHASE 1: Single Direction Spawn ---
        # UNCOMMENT THIS BLOCK FOR PHASE 1:
        default_root_state = self._robot.data.default_root_state[env_ids].clone()
        self._desired_pos_w[env_ids, 0] = torch.zeros_like(self._desired_pos_w[env_ids, 0]).uniform_(-1.0, 1.0) + self._terrain.env_origins[env_ids, 0]
        self._desired_pos_w[env_ids, 1] = -1.0 + self._terrain.env_origins[env_ids, 1]
        self._desired_pos_w[env_ids, 2] = torch.zeros_like(self._desired_pos_w[env_ids, 2]).uniform_(0.5, 1.5)
        spawn_x = torch.zeros(len(env_ids), device=self.device).uniform_(-1.0, 1.0)
        default_root_state[:, 0] = spawn_x + self._terrain.env_origins[env_ids, 0]
        # Drive spawn Y offset from configuration so it can spawn closer to target Y = -1.0
        default_root_state[:, 1] = self.cfg.spawn_y_offset + self._terrain.env_origins[env_ids, 1]
        default_root_state[:, 2] = torch.zeros(len(env_ids), device=self.device).uniform_(0.5, 1.5)

        # --- Orient drone to face the goal ---
        # Compute yaw angle from drone spawn → goal
        dx = self._desired_pos_w[env_ids, 0] - default_root_state[:, 0]
        dy = self._desired_pos_w[env_ids, 1] - default_root_state[:, 1]
        goal_yaw = torch.atan2(dy, dx)  # world-frame yaw toward goal

        # Set physical orientation using quat_from_euler_xyz (roll=0, pitch=0, yaw=goal_yaw)
        zeros = torch.zeros_like(goal_yaw)
        spawn_quat = quat_from_euler_xyz(zeros, zeros, goal_yaw)
        default_root_state[:, 3:7] = spawn_quat

        # Initialize target_yaw for the low-level controller
        self._target_yaw[env_ids] = goal_yaw

        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)

        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # --- Initialize previous distance (2D X/Y) ---
        self._prev_dist_to_goal[env_ids] = torch.linalg.norm(
            (self._desired_pos_w[env_ids] - default_root_state[:, :3])[:, :2], dim=1
        )

        # --- Randomize Pillar Positions (Domain Randomization) ---
        num_resets = env_ids.shape[0]
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
            # Identity quaternion (upright)
            state[:, 3] = 1.0
            state[:, 4:7] = 0.0
            # Zero velocity
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

            # Pillar kill-zone visualizers — translucent green cylinders
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
            # Gather real-time pillar positions for env 0 visualization
            pillar_positions = torch.stack(
                [p.data.root_pos_w[0, :3] for p in self._pillars], dim=0
            )
            self.pillar_zone_visualizers.visualize(pillar_positions)
