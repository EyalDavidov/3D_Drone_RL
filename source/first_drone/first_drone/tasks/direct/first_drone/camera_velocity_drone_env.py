from __future__ import annotations

import torch

from .camera_first_drone_env import CameraFirstDroneEnv


class CameraVelocityDroneEnv(CameraFirstDroneEnv):
    """Environment variant where the agent commands body-frame velocities
    and a yaw rate: action = [vx, vy, vz, yaw_rate]. The env uses a simple
    PD-style controller to convert velocity errors into forces and a P
    controller for yaw-rate to torque.
    """

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Controller gains (tunable)
        self._kp_lin = 5.0
        self._kd_lin = 1.0
        self._kp_yaw = 2.0
        self._kd_ang_xy = 0.1

        # Limits for commanded velocities (safety clipping)
        self._vel_limit = torch.tensor([3.0, 3.0, 2.0], device=self.device)
        self._yaw_rate_limit = 3.0

    def _pre_physics_step(self, actions: torch.Tensor):
        """Interpret actions as motor-level commands and apply as forces/torques.

        Expected action layout (same as original camera env):
          - action[0]: normalized thrust (-1..1 mapped to 0..max_thrust)
          - action[1]: roll moment (-1..1 scaled by moment_scale)
          - action[2]: pitch moment
          - action[3]: yaw moment
        """
        self._actions = actions.clone().clamp(-1.0, 1.0)
        # thrust (Z) mapping -> 0..max_thrust
        self._thrust[:, 0, 2] = (
            self.cfg.thrust_to_weight * self._robot_weight * (self._actions[:, 0] + 1.0) / 2.0
        )
        # moments scaled by cfg.moment_scale
        self._moment[:, 0, :] = self.cfg.moment_scale * self._actions[:, 1:]

    def _apply_action(self):
        """Apply the computed thrust and moment as an external wrench on the drone body."""
        self._robot.permanent_wrench_composer.set_forces_and_torques(
            body_ids=self._body_id, forces=self._thrust, torques=self._moment
        )

    # ------------------------------------------------------------------
    # Rewards / dones / reset (copied/adapted from CameraFirstDroneEnv)
    # ------------------------------------------------------------------
    def _get_rewards(self) -> torch.Tensor:
        """Reward based on velocity-matching to the desired target.

        Primary objective: minimize squared error between current body velocities and desired target.
        Also include penalties for ang/lin velocity magnitude and collisions.
        """
        cur_vb = self._robot.data.root_lin_vel_b
        cur_wb = self._robot.data.root_ang_vel_b

        desired_vb = getattr(self, "_desired_vel_b", torch.zeros_like(cur_vb))
        desired_yaw = getattr(self, "_desired_yaw_rate", torch.zeros(self.num_envs, device=self.device))

        vel_err_sq = torch.sum(torch.square(cur_vb - desired_vb), dim=1)
        yaw_err_sq = torch.square(cur_wb[:, 2] - desired_yaw)

        # velocity-match reward (negative squared error scaled)
        vel_match_reward = -5.0 * vel_err_sq * self.step_dt
        yaw_match_reward = -2.0 * yaw_err_sq * self.step_dt

        # stability penalties (as before)
        ang_vel = torch.sum(torch.square(cur_wb), dim=1)
        lin_vel = torch.sum(torch.square(cur_vb), dim=1)

        pos_local = self._robot.data.root_pos_w[:, :3] - self._terrain.env_origins
        hit_pole = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for px, py in self.cfg.pole_positions:
            dist_sq = torch.square(pos_local[:, 0] - px) + torch.square(pos_local[:, 1] - py)
            hit_pole |= (dist_sq < ((self.cfg.pole_radius + self.cfg.drone_radius) ** 2))

        died_from_crash = self.reset_terminated.float()

        rewards = {
            "vel_match": vel_match_reward,
            "yaw_match": yaw_match_reward,
            "died": died_from_crash * self.cfg.died_reward_scale,
            "hit_pole": hit_pole.float() * self.cfg.hit_pole_reward_scale * self.step_dt,
            "ang_vel": ang_vel * self.cfg.ang_vel_reward_scale * self.step_dt,
            "lin_vel": lin_vel * self.cfg.lin_vel_reward_scale * self.step_dt,
        }

        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        for key, value in rewards.items():
            # ensure episode sums exist for new keys
            if key not in self._episode_sums:
                self._episode_sums[key] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            self._episode_sums[key] += value

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        pos_local = self._robot.data.root_pos_w[:, :3] - self._terrain.env_origins

        hit_floor_or_ceiling = (pos_local[:, 2] < 0.1) | (pos_local[:, 2] > 2.0)
        wall_bound = 1.9 - self.cfg.drone_radius
        hit_wall = (
            (pos_local[:, 0] > wall_bound) | (pos_local[:, 0] < -wall_bound)
            | (pos_local[:, 1] > wall_bound) | (pos_local[:, 1] < -wall_bound)
        )
        died = hit_floor_or_ceiling | hit_wall
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        # final velocity error (mean L2 error) for logging
        cur_vb = self._robot.data.root_lin_vel_b[env_ids]
        desired_vb = getattr(self, "_desired_vel_b", torch.zeros_like(cur_vb))[env_ids]
        vel_err = torch.linalg.norm(cur_vb - desired_vb, dim=1)
        final_velocity_error = vel_err.mean()
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
        extras["Metrics/final_velocity_error"] = final_velocity_error.item()
        self.extras["log"].update(extras)

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        if len(env_ids) == self.num_envs:
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions[env_ids] = 0.0

        # --- Sample new desired velocity target (body frame) ---
        # vx, vy in [-1, 1] m/s, vz in [-0.5, 0.5] m/s, yaw_rate in [-1.0, 1.0] rad/s
        self._desired_vel_b = torch.zeros(self.num_envs, 3, device=self.device)
        self._desired_vel_b[env_ids, 0] = torch.zeros_like(self._desired_vel_b[env_ids, 0]).uniform_(-1.0, 1.0)
        self._desired_vel_b[env_ids, 1] = torch.zeros_like(self._desired_vel_b[env_ids, 1]).uniform_(-1.0, 1.0)
        self._desired_vel_b[env_ids, 2] = torch.zeros_like(self._desired_vel_b[env_ids, 2]).uniform_(-0.5, 0.5)
        self._desired_yaw_rate = torch.zeros(self.num_envs, device=self.device)
        self._desired_yaw_rate[env_ids] = torch.zeros_like(self._desired_yaw_rate[env_ids]).uniform_(-1.0, 1.0)

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

        # no previous-distance tracking for velocity task

    # Override scene setup to avoid creating a camera
    def _setup_scene(self):
        """Create the drone articulation, room (floor), terrain, and lighting — no camera."""
        self._robot = self._robot = __import__(
            "isaaclab.assets", fromlist=["Articulation"]
        ).Articulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self._robot

        # Room (floor) — spawn the provided USD file into env_0 (cloned to all envs)
        room_cfg = __import__("isaaclab.sim", fromlist=["UsdFileCfg"]).UsdFileCfg(usd_path=self.cfg.room_usd_path)
        room_cfg.func("/World/envs/env_0/Room", room_cfg)

        # Terrain (ground plane)
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # Clone environments
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # Lighting
        light_cfg = __import__("isaaclab.sim", fromlist=["DomeLightCfg"]).DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _get_observations(self) -> dict:
        """Return observations for the velocity-control policy.

        Observations returned:
          - "policy": body-frame linear and angular velocities (B, 6)
          - "imu": same as policy (kept for compatibility)
          - "critic": privileged state (lin_vel_b(3), ang_vel_b(3), projected_gravity_b(3), desired_vel_b(3))
        """
        # Body velocities
        lin_vel_b = self._robot.data.root_lin_vel_b
        ang_vel_b = self._robot.data.root_ang_vel_b

        # desired velocity target (body frame + yaw_rate) previously sampled in reset
        desired_vb = getattr(self, "_desired_vel_b", torch.zeros_like(lin_vel_b))
        desired_yaw = getattr(self, "_desired_yaw_rate", torch.zeros(self.num_envs, device=self.device))

        # Policy observation: the agent receives the desired velocities as its input
        # shape: (B, 4) -> [vx, vy, vz, yaw_rate]
        policy_obs = torch.cat([desired_vb, desired_yaw.unsqueeze(-1)], dim=-1)

        # IMU observation (kept for compatibility): current body lin/ang vel
        imu_obs = torch.cat([lin_vel_b, ang_vel_b], dim=-1)

        # Critic: privileged state includes current velocities, projected gravity, and desired target
        critic_obs = torch.cat([lin_vel_b, ang_vel_b, self._robot.data.projected_gravity_b, desired_vb], dim=-1)

        return {"policy": policy_obs, "imu": imu_obs, "critic": critic_obs}
