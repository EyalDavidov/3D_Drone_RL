from __future__ import annotations

import gymnasium as gym
import torch

from isaaclab.envs import DirectRLEnv
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.utils.math import wrap_to_pi, quat_from_euler_xyz, euler_xyz_from_quat
from isaaclab.markers import RED_ARROW_X_MARKER_CFG
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
        self._vel_limit = torch.tensor([3.0, 3.0, 2.0], device=self.device)
        self._yaw_rate_limit = 3.0

        # ----- Action / wrench buffers -----
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)   # force applied to body (only Z used)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)   # torque applied to body (x, y, z)

        # ----- Goal position (world frame) (for debug visualization) -----
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._desired_vel_b = torch.zeros(self.num_envs, 3, device=self.device)

        # ----- Episode reward logging -----
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "progress",
                "died",
                "ang_vel",
                "lin_vel",
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

        # Room (floor) — spawn the provided USD file into env_0 (cloned to all envs)
        # room_cfg = sim_utils.UsdFileCfg(usd_path=self.cfg.room_usd_path)
        # room_cfg.func("/World/envs/env_0/Room", room_cfg)

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
        self._actions = actions.clone().clamp(-1.0, 1.0)
        # thrust (Z) mapping -> 0..max_thrust
        self._thrust[:, 0, 2] = (
            self.cfg.thrust_to_weight * self._robot_weight * (self._actions[:, 0] + 1.0) / 2.0
        )
        # moments scaled by cfg.moment_scale
        self._moment[:, 0, :] = self.cfg.moment_scale * self._actions[:, 1:]

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
        target_yaw_tensor = torch.full_like(current_yaw, self.cfg.target_yaw)
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
        target_yaw_tensor = torch.full_like(current_yaw, self.cfg.target_yaw)
        yaw_err = wrap_to_pi(target_yaw_tensor - current_yaw)

        vel_err_sq = torch.sum(torch.square(cur_vb - desired_vb), dim=1)
        # yaw_match penalizes absolute yaw error (squared) rather than just yaw rate
        yaw_err_sq = torch.square(yaw_err)


        # stability penalties (as before)
        ang_vel = torch.sum(torch.square(cur_wb), dim=1)
        lin_vel = torch.sum(torch.square(cur_vb), dim=1)

        died_from_crash = self.reset_terminated.float()

        rewards = {
            "vel_match": self.cfg.vel_match_reward_scale * vel_err_sq * self.step_dt,
            "yaw_match": self.cfg.yaw_match_reward_scale * yaw_err_sq * self.step_dt,
            "died": died_from_crash * self.cfg.died_reward_scale,
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
        self._desired_vel_b[env_ids, 0] = 0.0
        self._desired_vel_b[env_ids, 1] = 0.0
        self._desired_vel_b[env_ids, 2] = 0.0
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
            if not hasattr(self, "yaw_arrow_visualizer"):
                marker_cfg = RED_ARROW_X_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Command/target_yaw"
                marker_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
                # You might need to adjust the arrow's orientation if it points up by default
                self.yaw_arrow_visualizer = VisualizationMarkers(marker_cfg)
            self.yaw_arrow_visualizer.set_visibility(True)
        else:
            if hasattr(self, "yaw_arrow_visualizer"):
                self.yaw_arrow_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        """Update goal marker positions each frame."""
        if hasattr(self, "yaw_arrow_visualizer") and self.yaw_arrow_visualizer is not None:
            if not hasattr(self, "scene"):
                return
            # Place the arrow above the drone, pointing in the target yaw direction
            marker_pos = self._robot.data.root_pos_w.clone()
            marker_pos[:, 2] += 0.3  # Offset 0.3m above the drone
            
            # The arrow points along the local X-axis. Create a quaternion for target yaw.
            target_yaw_tensor = torch.full((marker_pos.shape[0],), self.cfg.target_yaw, device=self.device)
            zeros = torch.zeros_like(target_yaw_tensor)
            
            # Construct a quaternion that rotates the local X-axis (the arrow) to match the target yaw
            marker_quat = quat_from_euler_xyz(zeros, zeros, target_yaw_tensor)
            
            self.yaw_arrow_visualizer.visualize(marker_pos, marker_quat)
