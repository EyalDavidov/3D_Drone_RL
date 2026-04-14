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

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sensors import TiledCamera
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import subtract_frame_transforms

from isaaclab.markers import CUBOID_MARKER_CFG  # isort: skip

from .sac_drone_env_cfg import SACDroneEnvCfg
from first_drone.models.vae import VAE


class SACDroneEnv(DirectRLEnv):
    """SAC+VAE drone navigation environment.

    Actions (4 continuous, clamped to [-1, 1]):
      - action[0]: z_thrust (mapped to hover ± scale)
      - action[1]: roll moment
      - action[2]: pitch moment
      - action[3]: yaw moment
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

        # ----- Depth image buffer (exposed for external VAE training) -----
        self._last_depth_processed = None

        # ----- Episode reward logging -----
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "progress", "goal", "hover", "clearance",
                "ang_vel", "tilt", "action", "collision",
            ]
        }

        # Debug visualization
        self.set_debug_vis(self.cfg.debug_vis)

    # ------------------------------------------------------------------
    # Scene setup
    # ------------------------------------------------------------------
    def _setup_scene(self):
        """Create drone, room, terrain, camera, and lighting."""
        self._robot = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self._robot

        room_cfg = sim_utils.UsdFileCfg(usd_path=self.cfg.room_usd_path)
        room_cfg.func("/World/envs/env_0/Room", room_cfg)

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
        """Convert RL actions → thrust and moment commands."""
        self._actions = actions.clone().clamp(-1.0, 1.0)
        self._thrust[:, 0, 2] = (
            self.cfg.thrust_to_weight * self._robot_weight * (self._actions[:, 0] + 1.0) / 2.0
        )
        self._moment[:, 0, :] = self.cfg.moment_scale * self._actions[:, 1:]

    def _apply_action(self):
        """Apply wrench to the drone body."""
        self._robot.permanent_wrench_composer.set_forces_and_torques(
            body_ids=self._body_id, forces=self._thrust, torques=self._moment
        )

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

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------
    def _get_rewards(self) -> torch.Tensor:
        """Compute 7-term reward.

        1. progress   — getting closer to goal (dense)
        2. goal       — bonus for being inside goal radius
        3. hover      — bonus for low velocity near goal
        4. clearance  — depth-center vs depth-mean (obstacle awareness)
        5. ang_vel    — penalty for spinning
        6. tilt       — penalty for excessive roll/pitch
        7. action     — penalty for large actions
        + collision   — one-time penalty on death
        """
        # Current distance to goal
        curr_dist = torch.linalg.norm(
            self._desired_pos_w - self._robot.data.root_pos_w, dim=1
        )

        # 1. Progress reward
        progress = self._prev_dist_to_goal - curr_dist
        self._prev_dist_to_goal = curr_dist.clone()

        # 2. Goal reached
        reached_goal = (curr_dist < self.cfg.goal_radius).float()

        # 3. Hover bonus (only near goal) - Disabled for now
        # vel_sq = torch.sum(self._robot.data.root_lin_vel_b ** 2, dim=1)
        # hover_bonus = at_goal * torch.exp(-2.0 * vel_sq)
        hover_bonus = torch.zeros_like(curr_dist)

        # 4. Depth clearance (center brighter = more open ahead)
        depth = self._last_depth_processed  # (B, 1, H, W)
        if depth is not None:
            h, w = depth.shape[2], depth.shape[3]
            # Center crop: middle 16×16 region
            ch, cw = h // 2, w // 2
            center = depth[:, :, ch - 8: ch + 8, cw - 8: cw + 8]
            clearance = center.mean(dim=(1, 2, 3)) - depth.mean(dim=(1, 2, 3))
        else:
            clearance = torch.zeros(self.num_envs, device=self.device)

        # 5. Angular velocity penalty
        ang_vel_sq = torch.sum(self._robot.data.root_ang_vel_b ** 2, dim=1)

        # 6. Tilt penalty (projected gravity deviation from straight down)
        # projected_gravity_b for a level drone is (0, 0, -1)
        # tilt = 1 - |gravity_z| (0 when level, ~1 when flipped)
        gravity_b = self._robot.data.projected_gravity_b
        tilt = 1.0 - gravity_b[:, 2].abs()

        # 7. Action magnitude penalty
        action_sq = torch.sum(self._actions ** 2, dim=1)

        # Check if died from collision (reset_terminated and NOT reached_goal)
        died_from_crash = (self.reset_terminated.float() - reached_goal).clamp(min=0.0)

        rewards = {
            "progress": self.cfg.w_progress * progress,
            "goal": self.cfg.w_goal * reached_goal,
            "hover": self.cfg.w_hover * hover_bonus,
            "clearance": self.cfg.w_clearance * clearance,
            "ang_vel": -self.cfg.w_ang_vel * ang_vel_sq,
            "tilt": -self.cfg.w_tilt * tilt,
            "action": -self.cfg.w_action * action_sq,
            "collision": self.cfg.collision_penalty * died_from_crash,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        # Accumulate for logging
        for key, value in rewards.items():
            self._episode_sums[key] += value

        return reward

    # ------------------------------------------------------------------
    # Termination
    # ------------------------------------------------------------------
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Terminate on floor/ceiling/wall collision or timeout."""
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        pos_local = self._robot.data.root_pos_w[:, :3] - self._terrain.env_origins

        hit_floor_or_ceiling = (pos_local[:, 2] < 0.1) | (pos_local[:, 2] > 2.0)
        hit_wall = (
            (pos_local[:, 0] > 1.9) | (pos_local[:, 0] < -1.9)
            | (pos_local[:, 1] > 1.9) | (pos_local[:, 1] < -1.9)
        )

        distance_to_goal = torch.linalg.norm(self._desired_pos_w - self._robot.data.root_pos_w, dim=1)
        reached_goal = distance_to_goal < self.cfg.goal_radius

        died = hit_floor_or_ceiling | hit_wall | reached_goal
        return died, time_out

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset environments. Drone at y=+1, goal at y=-1."""
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        # --- Logging ---
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

        # --- Reset robot ---
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        if len(env_ids) == self.num_envs:
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions[env_ids] = 0.0

        # --- Goal position ---
        self._desired_pos_w[env_ids, 0] = torch.zeros_like(self._desired_pos_w[env_ids, 0]).uniform_(-1.0, 1.0)
        self._desired_pos_w[env_ids, 0] += self._terrain.env_origins[env_ids, 0]
        self._desired_pos_w[env_ids, 1] = -1.0 + self._terrain.env_origins[env_ids, 1]
        self._desired_pos_w[env_ids, 2] = torch.zeros_like(self._desired_pos_w[env_ids, 2]).uniform_(0.5, 1.5)

        # --- Robot spawn position ---
        default_root_state = self._robot.data.default_root_state[env_ids].clone()
        default_root_state[:, 0] = torch.zeros(len(env_ids), device=self.device).uniform_(-1.0, 1.0)
        default_root_state[:, 0] += self._terrain.env_origins[env_ids, 0]
        default_root_state[:, 1] = 1.0 + self._terrain.env_origins[env_ids, 1]
        default_root_state[:, 2] = torch.zeros(len(env_ids), device=self.device).uniform_(0.5, 1.5)

        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)

        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # --- Initialize previous distance ---
        self._prev_dist_to_goal[env_ids] = torch.linalg.norm(
            self._desired_pos_w[env_ids] - default_root_state[:, :3], dim=1
        )

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
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        self.goal_pos_visualizer.visualize(self._desired_pos_w)
