import gymnasium as gym
import torch
from isaaclab.envs import DirectRLEnv
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.utils.math import wrap_to_pi, quat_from_euler_xyz, euler_xyz_from_quat, quat_rotate_inverse
from isaaclab.markers import CUBOID_MARKER_CFG
from isaaclab.markers import VisualizationMarkers
from .low_level_controller import LowLevelController

class NavigationDroneEnv(DirectRLEnv):
    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # 1. Load the frozen Flight Controller!
        self.llc = LowLevelController(cfg.llc_checkpoint_path, device=self.device)
        
        # 2. Limits for High-Level agent's commands
        self._vel_limit = torch.tensor([1.0, 1.0, 0.5], device=self.device)
        self._yaw_rate_limit = 0.05 # 0.05 rad per step at 50Hz = 2.5 rad/s

        # 3. Buffers for passing info from High-Level to Low-Level
        self._desired_vel_b = torch.zeros(self.num_envs, 3, device=self.device)
        self._target_yaw = torch.zeros(self.num_envs, device=self.device)

        # 4. Navigation specific buffers
        self._goal_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._distance_to_goal = torch.zeros(self.num_envs, device=self.device)
        self._previous_distance = torch.zeros(self.num_envs, device=self.device)

        # 5. Physics / Motor buffers
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._body_id = self._robot.find_bodies("body")[0]
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        self._episode_sums = {
            "progress": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "reached_goal": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "died": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
#"died_unstable": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "died_hit_floor": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "distance_to_goal_mapped": torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        }

        self.set_debug_vis(self.cfg.debug_vis)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self._robot

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        
    def _get_drone_yaw(self) -> torch.Tensor:
        _, _, yaw = euler_xyz_from_quat(self._robot.data.root_quat_w)
        return yaw

    def _pre_physics_step(self, actions: torch.Tensor):
        # The High-Level Agent runs at 50Hz! (Because decimation = 2)
        actions = actions.clamp(-1.0, 1.0)
        
        # Translate NN outputs to physical goals for the Low-Level Controller
        self._desired_vel_b[:, 0] = actions[:, 0] * self._vel_limit[0]
        self._desired_vel_b[:, 1] = actions[:, 1] * self._vel_limit[1]
        self._desired_vel_b[:, 2] = actions[:, 2] * self._vel_limit[2]
        self._target_yaw = wrap_to_pi(self._target_yaw + actions[:, 3] * self._yaw_rate_limit)

        # --- LOW LEVEL CONTROLLER EXECUTION ---
        # We query the physics state exactly once every 50Hz step (just like during LLC training)
        lin_vel_b = self._robot.data.root_lin_vel_b
        ang_vel_b = self._robot.data.root_ang_vel_b
        projected_gravity_b = self._robot.data.projected_gravity_b
        current_yaw = self._get_drone_yaw()
        
        yaw_err = wrap_to_pi(self._target_yaw - current_yaw)
        
        # 1. Prepare 13-dim observation for Low-Level Controller (MUST MATCH EXACT TRAINING ORDER!)
        ll_obs = torch.cat([lin_vel_b, ang_vel_b, projected_gravity_b, self._desired_vel_b, yaw_err.unsqueeze(-1)], dim=-1)
        
        # 2. Ask the frozen Flight Controller for motor actions
        ll_actions = self.llc(ll_obs)
        
        # 3. Calculate forces and torques
        self._thrust[:, 0, 2] = self.cfg.thrust_to_weight * self._robot_weight * (ll_actions[:, 0] + 1.0) / 2.0
        self._moment[:, 0, :] = self.cfg.moment_scale * ll_actions[:, 1:]

    def _apply_action(self):
        # THIS RUNS AT 100Hz! It just applies the forces calculated by the 50Hz step above
        self._robot.permanent_wrench_composer.set_forces_and_torques(
            body_ids=self._body_id, forces=self._thrust, torques=self._moment
        )

    def _get_observations(self) -> dict:
        # High-Level Agent Observation (7-dim)
        pos_w = self._robot.data.root_pos_w[:, :3]
        
        # Calculate vector to goal in WORLD frame
        pos_err_w = self._goal_pos_w - pos_w
        self._distance_to_goal = torch.norm(pos_err_w, dim=1)
        
        # Transform vector to goal into BODY frame
        quat_w = self._robot.data.root_quat_w
        pos_err_b = quat_rotate_inverse(quat_w, pos_err_w)
        
        # Use body frame velocity
        lin_vel_b = self._robot.data.root_lin_vel_b
        
        # Now the agent sees everything relative to itself! (Egocentric)
        obs = torch.cat([pos_err_b, lin_vel_b, self._distance_to_goal.unsqueeze(-1)], dim=-1)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        pos_w = self._robot.data.root_pos_w[:, :3]
        self._distance_to_goal = torch.norm(self._goal_pos_w - pos_w, dim=1)
        distance_to_goal_mapped = 1.0 - torch.tanh(self._distance_to_goal / 5.0)
        
        # 1. Progress reward (Reward it for getting closer to the goal)
        progress = self._previous_distance - self._distance_to_goal
        progress_reward = progress * self.cfg.progress_reward_scale
        
        # 2. Reached goal reward
        reached_goal = self._distance_to_goal < self.cfg.goal_radius
        goal_reward = reached_goal.float() * self.cfg.reached_goal_reward
        
        # 3. Crash penalty
      #  unstable = self._robot.data.projected_gravity_b[:, 2] > -0.5
        hit_floor = (pos_w[:, 2] - self._terrain.env_origins[:, 2]) < 0.1
        #died = (unstable | hit_floor)
        died = (hit_floor)
        died_reward = died.float() * self.cfg.died_reward_scale
        
        self._previous_distance = self._distance_to_goal.clone()
        
        rewards = {
            "progress": progress_reward,
            "reached_goal": goal_reward,
            "died": died_reward,
            "distance_to_goal_mapped": distance_to_goal_mapped * self.cfg.distance_to_goal_mapped_reward_scale * self.step_dt,  # Just to have a nice number to log for this metric
        }
        
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        
        for key, value in rewards.items():
            self._episode_sums[key] += value
            
        #self._episode_sums["died_unstable"] += unstable.float()
        self._episode_sums["died_hit_floor"] += hit_floor.float()
            
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        pos_w = self._robot.data.root_pos_w[:, :3]
        #unstable = self._robot.data.projected_gravity_b[:, 2] > -0.5
        hit_floor = (pos_w[:, 2] - self._terrain.env_origins[:, 2]) < 0.1
        #died = (unstable | hit_floor)
        died = (hit_floor)

        
        success = self._distance_to_goal < self.cfg.goal_radius
        
        return (died | success), time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
            
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
            
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        if len(env_ids) == self.num_envs:
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._desired_vel_b[env_ids] = 0.0
        
        default_root_state = self._robot.data.default_root_state[env_ids].clone()
        
        # Spawn random offset
        default_root_state[:, 0] += torch.zeros(len(env_ids), device=self.device).uniform_(-2.0, 2.0)
        default_root_state[:, 1] += torch.zeros(len(env_ids), device=self.device).uniform_(-2.0, 2.0)
        default_root_state[:, 2] = torch.zeros(len(env_ids), device=self.device).uniform_(0.5, 1.5)
        
        # Random initial yaw
        rand_yaw = torch.zeros(len(env_ids), device=self.device).uniform_(-torch.pi, torch.pi)
        rand_roll_pitch = torch.zeros(len(env_ids), 2, device=self.device)
        default_root_state[:, 3:7] = quat_from_euler_xyz(rand_roll_pitch[:, 0], rand_roll_pitch[:, 1], rand_yaw)
        self._target_yaw[env_ids] = rand_yaw

        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)

        # Randomize target goal
        self._goal_pos_w[env_ids, 0] = self._terrain.env_origins[env_ids, 0] + torch.zeros(len(env_ids), device=self.device).uniform_(-3.0, 3.0)
        self._goal_pos_w[env_ids, 1] = self._terrain.env_origins[env_ids, 1] + torch.zeros(len(env_ids), device=self.device).uniform_(-3.0, 3.0)
        self._goal_pos_w[env_ids, 2] = torch.zeros(len(env_ids), device=self.device).uniform_(0.5, 2.0)
        
        pos_w = default_root_state[:, :3]
        self._distance_to_goal[env_ids] = torch.norm(self._goal_pos_w[env_ids] - pos_w, dim=1)
        self._previous_distance[env_ids] = self._distance_to_goal[env_ids].clone()

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Create or toggle visibility of goal position markers."""
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.1, 0.1, 0.1) # A bit larger so it's easier to see
                marker_cfg.prim_path = "/Visuals/Command/goal_position"
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
            self.goal_pos_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        """Update goal marker positions each frame."""
        if hasattr(self, "goal_pos_visualizer") and self.goal_pos_visualizer is not None:
            # We use self._goal_pos_w instead of self._desired_pos_w
            self.goal_pos_visualizer.visualize(self._goal_pos_w)
