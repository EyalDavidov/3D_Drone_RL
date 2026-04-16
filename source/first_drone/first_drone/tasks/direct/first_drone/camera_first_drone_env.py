# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sensors import TiledCamera
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import subtract_frame_transforms

from isaaclab.markers import CUBOID_MARKER_CFG  # isort: skip

from .camera_first_drone_env_cfg import CameraFirstDroneEnvCfg


class CameraFirstDroneEnv(DirectRLEnv):
    """First Drone environment with wrench-based quadcopter physics and camera sensor.

    Uses asymmetric actor-critic:
      - Actor (policy) receives the depth image from a body-mounted camera.
      - Critic receives the 12-dim body-frame state vector during training.

    The drone starts at y=+1.0 and must fly to a goal at y=-1.0,
    navigating through a room with poles in between.

    Actions (4 continuous, clamped to [-1, 1]):
      - action[0]: normalized thrust  (-1..1, mapped to 0..max_thrust)
      - action[1]: roll  moment       (-1..1, scaled by moment_scale)
      - action[2]: pitch moment       (-1..1, scaled by moment_scale)
      - action[3]: yaw   moment       (-1..1, scaled by moment_scale)
    """

    cfg: CameraFirstDroneEnvCfg

    def __init__(self, cfg: CameraFirstDroneEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # ----- Action / wrench buffers -----
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)   # force applied to body (only Z used)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)   # torque applied to body (x, y, z)

        # ----- Goal position (world frame) -----
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)

        # ----- Episode reward logging -----
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "distance_to_goal",
                "died",
                "reached_goal",
            ]
        }

        # ----- Physical constants (computed once) -----
        self._body_id = self._robot.find_bodies("body")[0]
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        # ----- Debug visualization (goal markers) -----
        self.set_debug_vis(self.cfg.debug_vis)

        if len(self.cfg.tiled_camera.data_types) != 1:
            raise ValueError(
                "The camera environment only supports one image type at a time but the following were"
                f" provided: {self.cfg.tiled_camera.data_types}"
            )

    # ------------------------------------------------------------------
    # Scene setup
    # ------------------------------------------------------------------
    def _setup_scene(self):
        """Create the drone articulation, room, terrain, camera, and lighting."""
        self._robot = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self._robot

        # Room with poles — spawned as a static USD prim into env_0 (gets cloned to all envs)
        room_cfg = sim_utils.UsdFileCfg(usd_path=self.cfg.room_usd_path)
        room_cfg.func("/World/envs/env_0/Room", room_cfg)

        # Terrain (ground plane)
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # Camera
        self._tiled_camera = TiledCamera(self.cfg.tiled_camera)

        # Clone environments
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        self.scene.sensors["tiled_camera"] = self._tiled_camera

        # Lighting
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ------------------------------------------------------------------
    # Physics step: convert actions → forces/torques
    # ------------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor):
        """Convert RL actions into thrust and moment commands.

        Actions are clamped to [-1, 1].
        - Thrust (Z-axis force):
            thrust = thrust_to_weight * weight * (action[0] + 1) / 2
            This maps action[0] = -1 → 0 thrust, action[0] = +1 → max thrust.
        - Moments (torques on x, y, z):
            moment = moment_scale * action[1:4]
            These directly control roll, pitch, and yaw.
        """

        # # לא משתמשים ב־actions
        # self._actions = torch.zeros_like(actions)

        # # איפוס
        # self._thrust[:, 0, :] = 0.0
        # self._moment[:, 0, :] = 0.0

        # # thrust קבוע = משקל → hover
        # self._thrust[:, 0, 2] = 1.00 * self._robot_weight  # 1.05 ליציבות

        self._actions = actions.clone().clamp(-1.0, 1.0)
        self._thrust[:, 0, 2] = (
            self.cfg.thrust_to_weight * self._robot_weight * (self._actions[:, 0] + 1.0) / 2.0
        )
        self._moment[:, 0, :] = self.cfg.moment_scale * self._actions[:, 1:]

    def _apply_action(self):
        """Apply the computed thrust and moment as an external wrench on the drone body."""
        self._robot.permanent_wrench_composer.set_forces_and_torques(
            body_ids=self._body_id, forces=self._thrust, torques=self._moment
        )

    # ------------------------------------------------------------------
    # Observations (asymmetric actor-critic)
    # ------------------------------------------------------------------
    def _get_observations(self) -> dict:
        """Build observations for asymmetric actor-critic.

        Returns a dict with three groups:
          - "policy": depth image (B, 1, H, W) — body-mounted camera, channels-first.
          - "imu":    6-dim IMU-like data [ang_vel_b(3), projected_gravity_b(3)].
                      Paired with "policy" for the actor. Realistic for deployment
                      (a real drone always has an IMU).
          - "critic": 12-dim privileged state vector (used only during training):
                      [lin_vel_b(3), ang_vel_b(3), projected_gravity_b(3), goal_pos_b(3)]
        """
        # --- Actor observation 1: depth image ---
        depth_image = self._tiled_camera.data.output["depth"].clone()
        # Replace inf (no hit / sky) with 0 so the network gets clean inputs
        depth_image[depth_image == float("inf")] = 0.0
        # Permute from (B, H, W, 1) → (B, 1, H, W) for RSL-RL CNN
        depth_image = depth_image.permute(0, 3, 1, 2)

        # --- Actor observation 2: IMU data (available on real hardware) ---
        imu_obs = torch.cat(
            [
                self._robot.data.root_ang_vel_b,
                self._robot.data.projected_gravity_b,
            ],
            dim=-1,
        )

        # --- Critic observation: full privileged state ---
        desired_pos_b, _ = subtract_frame_transforms(
            self._robot.data.root_pos_w, self._robot.data.root_quat_w, self._desired_pos_w
        )
        critic_obs = torch.cat(
            [
                self._robot.data.root_lin_vel_b,
                self._robot.data.root_ang_vel_b,
                self._robot.data.projected_gravity_b,
                desired_pos_b,
            ],
            dim=-1,
        )

        return {"policy": depth_image, "imu": imu_obs, "critic": critic_obs}

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------
    def _get_rewards(self) -> torch.Tensor:
        """Compute the per-step reward.

        Three terms are summed:
          1. distance_to_goal — penalty for being far from goal
          2. died     — one-time penalty when the drone crashes (floor/ceiling/walls)
          3. reached_goal - one-time huge bonus for successfully reaching the target
        """
        distance_to_goal = torch.linalg.norm(self._desired_pos_w - self._robot.data.root_pos_w, dim=1)
        # distance_to_goal_mapped = 1 - torch.tanh(distance_to_goal / 1.6)
        distance_to_goal_mapped = 0.1 * distance_to_goal

        # Check if reached goal this step
        reached_goal = (distance_to_goal < self.cfg.goal_radius).float()
        
        # Check if died from collision (which is reset_terminated AND NOT reached_goal)
        # because we will make reached_goal trigger reset_terminated in _get_dones
        died_from_crash = (self.reset_terminated.float() - reached_goal).clamp(min=0.0)

        rewards = {
            "distance_to_goal": distance_to_goal_mapped * self.cfg.distance_to_goal_reward_scale * self.step_dt,
            "died": died_from_crash * self.cfg.died_reward_scale,
            "reached_goal": reached_goal * self.cfg.reached_goal_reward_scale,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        # Accumulate for episode-level logging
        for key, value in rewards.items():
            self._episode_sums[key] += value

        return reward

    # ------------------------------------------------------------------
    # Termination conditions
    # ------------------------------------------------------------------
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Determine which environments should reset.

        Returns:
            died: True if drone hits obstacles OR reaches the goal
            time_out: True if episode exceeded max length
        """
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # Compute position relative to env origin (so wall checks work across the grid)
        pos_local = self._robot.data.root_pos_w[:, :3] - self._terrain.env_origins

        # Floor / ceiling
        hit_floor_or_ceiling = (pos_local[:, 2] < 0.1) | (pos_local[:, 2] > 2.0)
        # Walls (room is 4×4 centered at origin → walls at ±2, trigger slightly inside)
        hit_wall = (
            (pos_local[:, 0] > 1.85) | (pos_local[:, 0] < -1.85)
            | (pos_local[:, 1] > 1.85) | (pos_local[:, 1] < -1.85)
        )
        
        # Check if reached goal
        distance_to_goal = torch.linalg.norm(self._desired_pos_w - self._robot.data.root_pos_w, dim=1)
        reached_goal = distance_to_goal < self.cfg.goal_radius

        # Terminate if crashed OR reached goal
        died = hit_floor_or_ceiling | hit_wall | reached_goal
        return died, time_out

    # ------------------------------------------------------------------
    # Reset logic
    # ------------------------------------------------------------------
    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset environments that terminated or timed out.

        Drone spawns at y=+1.0 (behind the poles), goal is at y=-1.0 (other side).
        X is randomized in [-1, 1], Z in [0.5, 1.5].
        """
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        # --- Logging ---
        final_distance_to_goal = torch.linalg.norm(
            self._desired_pos_w[env_ids] - self._robot.data.root_pos_w[env_ids], dim=1
        ).mean()
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/died"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        extras["Metrics/final_distance_to_goal"] = final_distance_to_goal.item()
        self.extras["log"].update(extras)

        # --- Reset robot ---
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        # Spread out resets to avoid training spikes when all envs reset together
        if len(env_ids) == self.num_envs:
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions[env_ids] = 0.0

        # --- Sample new goal position ---
        # Goal: y=-1.0 (far side of poles), x in [-1, 1], z in [0.5, 1.5]
        self._desired_pos_w[env_ids, 0] = torch.zeros_like(self._desired_pos_w[env_ids, 0]).uniform_(-1.0, 1.0)
        self._desired_pos_w[env_ids, 0] += self._terrain.env_origins[env_ids, 0]
        self._desired_pos_w[env_ids, 1] = -1.0 + self._terrain.env_origins[env_ids, 1]
        self._desired_pos_w[env_ids, 2] = torch.zeros_like(self._desired_pos_w[env_ids, 2]).uniform_(0.5, 1.5)

        # --- Reset robot state ---
        # Drone starts at y=+1.0 (near side), x in [-1, 1], z in [0.5, 1.5]
        default_root_state = self._robot.data.default_root_state[env_ids].clone()
        # Randomize starting X
        default_root_state[:, 0] = torch.zeros(len(env_ids), device=self.device).uniform_(-1.0, 1.0)
        default_root_state[:, 0] += self._terrain.env_origins[env_ids, 0]
        # Y = +1.0 (near side of poles)
        default_root_state[:, 1] = 1.0 + self._terrain.env_origins[env_ids, 1]
        # Randomize starting Z
        default_root_state[:, 2] = torch.zeros(len(env_ids), device=self.device).uniform_(0.5, 1.5)

        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)

        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    # ------------------------------------------------------------------
    # Debug visualization (goal markers)
    # ------------------------------------------------------------------
    def _set_debug_vis_impl(self, debug_vis: bool):
        """Create or toggle visibility of goal position markers."""
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
        """Update goal marker positions each frame."""
        self.goal_pos_visualizer.visualize(self._desired_pos_w)
