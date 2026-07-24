from __future__ import annotations

import gymnasium as gym
import torch

from isaaclab.envs import DirectRLEnv
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.utils.math import wrap_to_pi, quat_from_euler_xyz, euler_xyz_from_quat
from isaaclab.markers import RED_ARROW_X_MARKER_CFG, BLUE_ARROW_X_MARKER_CFG
from isaaclab.markers import VisualizationMarkers

class FlightControllerDroneEnv(DirectRLEnv):
    """Environment variant where the agent commands body-frame velocities
    and a yaw rate: action = [vx, vy, vz, yaw_rate]. The env uses a simple
    PD-style controller to convert velocity errors into forces and a P
    controller for yaw-rate to torque.
    """

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Limits for commanded velocities (safety clipping)
        self._vel_limit = torch.tensor([1.0, 1.0, 0.5], device=self.device)
        self._yaw_rate_limit = 3.0

        # ----- Action / wrench buffers -----
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._previous_actions = torch.zeros_like(self._actions)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)   # force applied to body (only Z used)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)   # torque applied to body (x, y, z)

        # ----- Target velocity and yaw -----
        self._desired_vel_b = torch.zeros(self.num_envs, 3, device=self.device)
        self._target_yaw = torch.zeros(self.num_envs, device=self.device)

        # ----- Episode reward logging -----
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "progress",
                "died",
                "ang_vel",
            ]
        }
        
        # ----- Physical constants (computed once) -----
        self._body_id = self._robot.find_bodies("body")[0]
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        # ----- Debug visualization (goal markers) -----
        self.set_debug_vis(self.cfg.debug_vis)

        # ----- Color front propellers -----
        self._color_front_propellers()

    def _color_front_propellers(self):
        """Paint the front propellers (m1 and m4) bright green so the front is identifiable."""
        import isaaclab.sim as sim_utils
        from isaaclab.sim.utils import bind_visual_material
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        
        # Create a bright green material
        green_material_path = "/World/Materials/BrightGreen"
        green_material_cfg = sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.0, 1.0, 0.0),  # Bright green
            metallic=0.1,
            roughness=0.4
        )
        
        if not stage.GetPrimAtPath(green_material_path).IsValid():
            green_material_cfg.func(green_material_path, green_material_cfg)

        # Iterate over all environments and apply the material
        for env_id in range(self.num_envs):
            # Front propellers for X-configuration CF2X are typically m1_prop and m4_prop
            for prop in ["m1_prop", "m4_prop"]:
                prop_path = f"/World/envs/env_{env_id}/Drone/{prop}"
                if stage.GetPrimAtPath(prop_path).IsValid():
                    # Set stronger_than_descendants to True to override existing colors
                    bind_visual_material(prop_path, green_material_path, stage=stage, stronger_than_descendants=True)

    # Override scene setup to avoid creating a camera
    def _setup_scene(self):
        """Create the drone articulation, room (floor), terrain, and lighting — no camera."""
        self._robot = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self._robot

        # Terrain (ground plane)
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # Clone environments
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # Lighting
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
    
    def _pre_physics_step(self, actions: torch.Tensor):
        """Interpret actions as motor-level commands and apply as forces/torques.

        Expected action layout (same as original camera env):
          - action[0]: normalized thrust (-1..1 mapped to 0..max_thrust)
          - action[1]: roll moment (-1..1 scaled by moment_scale)
          - action[2]: pitch moment
          - action[3]: yaw moment
        """
        self._previous_actions = self._actions.clone()
        self._actions = actions.clone().clamp(-1.0, 1.0)
        # thrust (Z) mapping -> 0..max_thrust
        self._thrust[:, 0, 2] = (
            self.cfg.thrust_to_weight * self._robot_weight * (self._actions[:, 0] + 1.0) / 2.0
        )
        # moments scaled by cfg.moment_scale
        self._moment[:, 0, :] = self.cfg.moment_scale * self._actions[:, 1:]

        # Determine which environments should get a new navigation decision (every 5 steps)
        update_mask = (self.episode_length_buf % 10 == 0)

        # Random walk for dynamic target yaw (updated every 5 steps)
        yaw_drift = torch.zeros_like(self._target_yaw).uniform_(-0.5, 0.5)
        self._target_yaw = torch.where(update_mask, wrap_to_pi(self._target_yaw + yaw_drift), self._target_yaw)

        # Random walk for dynamic desired velocity (updated every 5 steps)
        vel_drift = torch.zeros_like(self._desired_vel_b).uniform_(-0.05, 0.05)
        new_vel = torch.clamp(self._desired_vel_b + vel_drift, min=-self._vel_limit, max=self._vel_limit)
        self._desired_vel_b[update_mask] = new_vel[update_mask]

    def _get_drone_yaw(self) -> torch.Tensor:
        """Returns the drone's current yaw angle in world frame."""
        _, _, yaw = euler_xyz_from_quat(self._robot.data.root_quat_w)
        return yaw

    def _apply_action(self):
        """Apply the computed thrust and moment as an external wrench on the drone body."""
        self._robot.permanent_wrench_composer.set_forces_and_torques(
            body_ids=self._body_id, forces=self._thrust, torques=self._moment
        )

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
        projected_gravity_b = self._robot.data.projected_gravity_b

        # desired velocity target (body frame) and absolute yaw error
        desired_vb = self._desired_vel_b

        # Compute the error between the target absolute yaw and the current yaw
        current_yaw = self._get_drone_yaw()
        target_yaw_tensor = self._target_yaw
        yaw_err = wrap_to_pi(target_yaw_tensor - current_yaw)

        # Policy observation: the agent receives the desired velocities and the yaw error to correct.
        # shape: (B, 4) -> [vx, vy, vz, yaw_err]
        policy_obs = torch.cat([desired_vb, yaw_err.unsqueeze(-1)], dim=-1)

        # IMU observation (kept for compatibility): current body lin/ang vel
        imu_obs = torch.cat([lin_vel_b, ang_vel_b, projected_gravity_b], dim=-1)

        # Critic: privileged state includes current velocities, projected gravity, desired target and current yaw error
        critic_obs = torch.cat([lin_vel_b, ang_vel_b, projected_gravity_b, desired_vb, yaw_err.unsqueeze(-1)], dim=-1)

        return {"policy": policy_obs, "imu": imu_obs, "critic": critic_obs}

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

        current_yaw = self._get_drone_yaw()
        target_yaw_tensor = self._target_yaw
        yaw_err = wrap_to_pi(target_yaw_tensor - current_yaw)

        vel_err_sq = torch.sum(torch.square(cur_vb - desired_vb), dim=1)
        # yaw_match penalizes absolute yaw error (squared) rather than just yaw rate
        yaw_err_sq = torch.square(yaw_err)
        
        # Gaussian rewards for tracking
        vel_match_reward = torch.exp(-vel_err_sq / 0.5)
        yaw_match_reward = torch.exp(-yaw_err_sq / 0.5)

        # Action penalties
        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)

        # stability penalties
        ang_vel = torch.sum(torch.square(cur_wb), dim=1)

        # Distinguish between "loss of control" crash vs "navigation" crash
        # If projected gravity Z is > -0.5, it means drone is tilted more than 60 degrees.
        unstable = self._robot.data.projected_gravity_b[:, 2] > -0.5
        
        # Only apply the -50 died penalty if it crashed because it lost control.
        # If it hits a wall/floor while upright, we just reset (from _get_dones) but don't penalize!
        died_from_instability = (self.reset_terminated & unstable).float()

        rewards = {
            "vel_match": self.cfg.vel_match_reward_scale * vel_match_reward * self.step_dt,
            "yaw_match": self.cfg.yaw_match_reward_scale * yaw_match_reward * self.step_dt,
            "died": died_from_instability * self.cfg.died_reward_scale,
            "ang_vel": ang_vel * self.cfg.ang_vel_reward_scale * self.step_dt,
            "action_rate": action_rate * getattr(self.cfg, "action_rate_reward_scale", 0.0) * self.step_dt,
        }

        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        for key, value in rewards.items():
            # ensure episode sums exist for new keys
            if key not in self._episode_sums:
                self._episode_sums[key] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            if self._episode_sums[key].is_inference():
                self._episode_sums[key] = self._episode_sums[key].clone()
            self._episode_sums[key] += value

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        pos_local = self._robot.data.root_pos_w[:, :3] - self._terrain.env_origins

        # The drone only "dies" (terminates) if it hits the floor. No walls or ceiling.
        hit_floor = (pos_local[:, 2] < 0.1)
        
        return hit_floor, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        # final velocity error (mean L2 error) for logging
        cur_vb = self._robot.data.root_lin_vel_b[env_ids]
        desired_vb = getattr(self, "_desired_vel_b", torch.zeros_like(cur_vb))[env_ids]
        vel_err = torch.linalg.norm(cur_vb - desired_vb, dim=1)
        final_velocity_error = vel_err.mean()

        #To log
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            if self._episode_sums[key].is_inference():
                self._episode_sums[key] = self._episode_sums[key].clone()
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
        self._desired_vel_b[env_ids, 0] = torch.zeros(len(env_ids), device=self.device).uniform_(-1.0, 1.0)
        self._desired_vel_b[env_ids, 1] = torch.zeros(len(env_ids), device=self.device).uniform_(-1.0, 1.0)
        self._desired_vel_b[env_ids, 2] = torch.zeros(len(env_ids), device=self.device).uniform_(-0.5, 0.5)
        self._target_yaw[env_ids] = torch.zeros(len(env_ids), device=self.device).uniform_(-torch.pi, torch.pi)
        # No more _desired_yaw_rate, we use cfg.target_yaw

        default_root_state = self._robot.data.default_root_state[env_ids].clone()

        # Randomize spawn location
        default_root_state[:, 0] = torch.zeros(len(env_ids), device=self.device).uniform_(-1.0, 1.0)
        default_root_state[:, 0] += self._terrain.env_origins[env_ids, 0]
        default_root_state[:, 1] = 1.0 + self._terrain.env_origins[env_ids, 1]
        default_root_state[:, 2] = torch.zeros(len(env_ids), device=self.device).uniform_(0.5, 1.5)

        # Randomize initial yaw
        rand_yaw = torch.zeros(len(env_ids), device=self.device).uniform_(-torch.pi, torch.pi)
        rand_roll_pitch = torch.zeros(len(env_ids), 2, device=self.device)  # roll=0, pitch=0
        default_root_state[:, 3:7] = quat_from_euler_xyz(rand_roll_pitch[:, 0], rand_roll_pitch[:, 1], rand_yaw)

        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)

        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # no previous-distance tracking for velocity task

    # ------------------------------------------------------------------
    # Debug visualization (goal markers)
    # ------------------------------------------------------------------
    def _set_debug_vis_impl(self, debug_vis: bool):
        """Create or toggle visibility of goal position markers."""
        if debug_vis:
            if not hasattr(self, "target_vel_visualizer"):
                marker_cfg = RED_ARROW_X_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Command/target_vel"
                marker_cfg.markers["arrow"].scale = (0.2, 0.2, 0.2)
                self.target_vel_visualizer = VisualizationMarkers(marker_cfg)
            self.target_vel_visualizer.set_visibility(True)

            if not hasattr(self, "current_vel_visualizer"):
                marker_cfg_current = BLUE_ARROW_X_MARKER_CFG.copy()
                marker_cfg_current.prim_path = "/Visuals/Command/current_vel"
                marker_cfg_current.markers["arrow"].scale = (0.2, 0.2, 0.2)
                self.current_vel_visualizer = VisualizationMarkers(marker_cfg_current)
            self.current_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "target_vel_visualizer"):
                self.target_vel_visualizer.set_visibility(False)
            if hasattr(self, "current_vel_visualizer"):
                self.current_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        """Update goal marker positions each frame."""
        if not hasattr(self, "scene"):
            return
            
        current_yaw = self._get_drone_yaw()
        zeros = torch.zeros_like(current_yaw)

        if hasattr(self, "target_vel_visualizer") and self.target_vel_visualizer is not None:
            # target velocity in body frame
            des_vb = self._desired_vel_b
            
            # Rotate body frame desired velocity by drone's yaw to get world frame vector
            cos_yaw = torch.cos(current_yaw)
            sin_yaw = torch.sin(current_yaw)
            
            vx_w = des_vb[:, 0] * cos_yaw - des_vb[:, 1] * sin_yaw
            vy_w = des_vb[:, 0] * sin_yaw + des_vb[:, 1] * cos_yaw
            vz_w = des_vb[:, 2]
            
            target_vel_mag = torch.sqrt(vx_w**2 + vy_w**2 + vz_w**2) + 1e-6
            target_yaw_dir = torch.atan2(vy_w, vx_w)
            target_pitch = torch.atan2(-vz_w, torch.sqrt(vx_w**2 + vy_w**2))
            
            target_quat = quat_from_euler_xyz(zeros, target_pitch, target_yaw_dir)
            
            marker_pos_target = self._robot.data.root_pos_w.clone()
            marker_pos_target[:, 2] += 0.1  # Offset 0.1m above the drone
            
            scale_target = torch.ones(self.num_envs, 3, device=self.device)
            scale_target[:, 0] = target_vel_mag * 1.5  # Stretch arrow length proportional to velocity
            scale_target[:, 1] = 0.15 # Thickness
            scale_target[:, 2] = 0.15
            
            self.target_vel_visualizer.visualize(marker_pos_target, target_quat, scales=scale_target)

        if hasattr(self, "current_vel_visualizer") and self.current_vel_visualizer is not None:
            # Current velocity in world frame
            cur_vw = self._robot.data.root_lin_vel_w
            
            vx_c = cur_vw[:, 0]
            vy_c = cur_vw[:, 1]
            vz_c = cur_vw[:, 2]
            
            current_vel_mag = torch.sqrt(vx_c**2 + vy_c**2 + vz_c**2) + 1e-6
            current_yaw_dir = torch.atan2(vy_c, vx_c)
            current_pitch_dir = torch.atan2(-vz_c, torch.sqrt(vx_c**2 + vy_c**2))
            
            current_quat = quat_from_euler_xyz(zeros, current_pitch_dir, current_yaw_dir)
            
            marker_pos_current = self._robot.data.root_pos_w.clone()
            marker_pos_current[:, 2] += 0.11  # Offset slightly higher to avoid Z-fighting with red arrow
            
            scale_current = torch.ones(self.num_envs, 3, device=self.device)
            scale_current[:, 0] = current_vel_mag * 1.5
            scale_current[:, 1] = 0.15
            scale_current[:, 2] = 0.15
            
            self.current_vel_visualizer.visualize(marker_pos_current, current_quat, scales=scale_current)
