"""Brain-driven Navigation Drone Environment.

Self-contained environment that integrates:
  1. Pretrained PPO navigator policy (frozen)
  2. Brain state machine (lawnmower coverage + YOLO search)
  3. Perception module (YOLO person detection)
  4. AE depth encoder + Low-level flight controller (inherited from AEPPODroneEnv)

The Brain module generates high-level waypoints, the frozen PPO policy navigates
to them, and the LLC converts velocity commands to motor forces. The external
caller simply calls env.step() — all intelligence is internal.

Architecture:
    Camera RGB/Depth → Perception (YOLO) → Brain State Machine → Waypoint
    → Frozen PPO(obs) → High-level actions → LLC → Motor forces → Physics
"""
from __future__ import annotations

import os
import sys
import torch
import gymnasium as gym

from .ae_ppo_drone_env import AEPPODroneEnv
from .brain_nav_drone_env_cfg import BrainNavDroneEnvCfg
from first_drone.models.perception import PerceptionModule
from first_drone.models.brain import BrainModule


class BrainNavDroneEnv(AEPPODroneEnv):
    """Self-contained Brain + PPO navigation environment.

    Subclasses AEPPODroneEnv to inherit the full physical simulation
    (drone, room, camera, AE, LLC, LiDAR, contact sensors). Overrides
    step() to run the Brain → PPO pipeline internally instead of
    accepting external RL actions.
    """

    cfg: BrainNavDroneEnvCfg

    def __init__(self, cfg: BrainNavDroneEnvCfg, render_mode: str | None = None, **kwargs):
        # Initialize full AEPPODroneEnv scene (drone, camera, AE, LLC, room, terrain, etc.)
        super().__init__(cfg, render_mode, **kwargs)

        # Mark as brain-play mode so AEPPODroneEnv doesn't reset on goal-reached
        self.is_brain_play = True

        # ---------- Load Frozen PPO Navigator Policy ----------
        self._load_navigator_policy()

        # ---------- Initialize Perception Module ----------
        self._perception = PerceptionModule(use_mock=self.cfg.use_mock_perception)
        print(f"\n[BrainNavEnv] Perception initialized (use_mock={self.cfg.use_mock_perception})\n")

        # ---------- Initialize Brain Module ----------
        # The Brain needs a wrapper that looks like the RslRlVecEnvWrapper
        # but we can pass self directly since it has .unwrapped access
        self._brain = BrainModule(
            _BrainEnvAdapter(self),
            step_size=self.cfg.brain_step_size,
            safety_margin=self.cfg.brain_safety_margin,
        )
        print(f"\n[BrainNavEnv] Brain initialized (step_size={self.cfg.brain_step_size}m, "
              f"margin={self.cfg.brain_safety_margin}m, waypoints={len(self._brain.waypoints)})\n")

        # ---------- Internal State Buffers ----------
        self._timestep = 0
        self._last_person_found = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._last_person_world_xyz = torch.zeros((self.num_envs, 3), device=self.device)

    def _load_navigator_policy(self):
        """Load the pretrained PPO navigator policy as a frozen inference module."""
        checkpoint_path = self.cfg.navigator_checkpoint_path

        if not checkpoint_path:
            raise ValueError(
                "\n[BrainNavEnv] ERROR: navigator_checkpoint_path is empty!\n"
                "You must provide the path to a trained PPO navigator checkpoint.\n"
                "Example: set navigator_checkpoint_path to the directory containing 'exported/policy.pt'\n"
                "  or directly to a model_*.pt file.\n"
            )

        # Try loading as JIT-exported policy first (fastest inference)
        try:
            policy_dir = os.path.dirname(checkpoint_path)
            jit_policy_path = os.path.join(policy_dir, "exported", "policy.pt")

            if os.path.exists(jit_policy_path):
                print(f"\n[BrainNavEnv] Loading JIT navigator policy: {jit_policy_path}\n")
                self._navigator_policy = torch.jit.load(jit_policy_path, map_location=self.device)
                self._navigator_policy.eval()
                self._navigator_policy_expects_dict = False
                return

            # If checkpoint_path itself is the JIT policy
            if checkpoint_path.endswith("policy.pt") and os.path.exists(checkpoint_path):
                print(f"\n[BrainNavEnv] Loading JIT navigator policy: {checkpoint_path}\n")
                self._navigator_policy = torch.jit.load(checkpoint_path, map_location=self.device)
                self._navigator_policy.eval()
                self._navigator_policy_expects_dict = False
                return

            raise FileNotFoundError(f"No exported JIT policy found at: {jit_policy_path}")

        except Exception as jit_err:
            print(f"\n[BrainNavEnv] JIT loading failed ({jit_err}). Trying RSL-RL Runner...\n")

        # Fallback: Load via RSL-RL OnPolicyRunner
        try:
            from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
            from rsl_rl.runners import OnPolicyRunner
            from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

            agent_cfg = load_cfg_from_registry("AE-PPO-Drone-Direct-v0", "rsl_rl_cfg_entry_point")
            agent_cfg.device = self.sim.cfg.device

            agent_dict = agent_cfg.to_dict()
            for model_key in ["actor", "critic"]:
                if model_key in agent_dict:
                    agent_dict[model_key].pop("stochastic", None)
                    agent_dict[model_key].pop("init_noise_std", None)
                    agent_dict[model_key].pop("noise_std_type", None)
                    agent_dict[model_key].pop("state_dependent_std", None)

            # Create a temporary wrapper for the runner
            temp_gym_env = _RslRlCompatWrapper(self)
            runner = OnPolicyRunner(temp_gym_env, agent_dict, log_dir=None, device=agent_cfg.device)
            runner.load(checkpoint_path)
            self._navigator_policy = runner.get_inference_policy(device=self.device)
            self._navigator_policy_expects_dict = True
            print(f"\n[BrainNavEnv] Loaded navigator via RSL-RL Runner: {checkpoint_path}\n")

        except Exception as runner_err:
            raise RuntimeError(
                f"\n[BrainNavEnv] Failed to load navigator policy from: {checkpoint_path}\n"
                f"  JIT error: {jit_err}\n"
                f"  Runner error: {runner_err}\n"
            )

    def step(self, action):
        """Override step to run the internal Brain → PPO pipeline.

        The external `action` argument is ignored — the Brain module
        determines all high-level commands, and the frozen PPO policy
        generates the low-level navigation actions.
        """
        with torch.inference_mode():
            # 1. Grab camera outputs and drone state
            rgb_image = self._tiled_camera.data.output["rgb"].clone()
            depth_image = self._tiled_camera.data.output["depth"].clone()
            drone_pos = self._robot.data.root_pos_w.clone()
            drone_quat = self._robot.data.root_quat_w.clone()
            # Replace infinity values in depth
            depth_image[depth_image == float("inf")] = 10.0

            # 2. Run Perception (YOLO + de-projection)
            run_yolo = self._brain.state != "SCAN" or (self._timestep % self.cfg.brain_yolo_interval == 0)
            if run_yolo:
                person_found, person_world_xyz = self._perception.process_camera_data(
                    rgb_image, depth_image, drone_pos, drone_quat
                )
                self._last_person_found = person_found
                self._last_person_world_xyz = person_world_xyz
            else:
                person_found = self._last_person_found
                person_world_xyz = self._last_person_world_xyz

            # 3. Update Brain State Machine
            desired_pos_w, target_yaw = self._brain.update(
                person_found, person_world_xyz, drone_pos, drone_quat
            )

            # 4. Set the high-level commands directly in the environment
            self._desired_pos_w[:, :] = desired_pos_w
            self._target_yaw[:] = target_yaw

            # 5. Re-evaluate observations for the navigator policy with the new targets
            obs_dict = self._get_observations()

            # 6. Action determination (bypassing PPO during high-level Brain states)
            if self._brain.state == "SCAN":
                # Spin in place: zero translation velocities, positive yaw rate command
                ppo_actions = torch.zeros((self.num_envs, 4), device=self.device)
                ppo_actions[:, 3] = 0.25  # Smooth yaw rate rotation command
            elif self._brain.state == "COMPLETE":
                # Hover in place at final target coordinates
                ppo_actions = torch.zeros((self.num_envs, 4), device=self.device)
            else:
                # Normal navigation: query frozen PPO policy
                policy_obs = obs_dict if self._navigator_policy_expects_dict else obs_dict["policy"]
                ppo_actions = self._navigator_policy(policy_obs)

        # 7. Step the parent environment with the PPO-generated actions
        obs, rewards, terminated, truncated, info = super().step(ppo_actions)

        # 8. Reset handling: if drone reset, recreate the Brain for the new scenario
        dones = terminated | truncated
        if dones.any():
            for env_id in range(self.num_envs):
                if dones[env_id].item():
                    print(f"[BrainNavEnv] Environment {env_id} reset detected. Re-initializing Brain.")

            # Recreate Brain module with fresh waypoints
            self._brain = BrainModule(
                _BrainEnvAdapter(self),
                step_size=self.cfg.brain_step_size,
                safety_margin=self.cfg.brain_safety_margin,
            )

        # 9. Periodic status logging
        if self._timestep % 100 == 0:
            d_pos = drone_pos[0]
            g_pos = desired_pos_w[0]
            dist = torch.norm(d_pos - g_pos).item()
            print(
                f"[BrainNavEnv Step {self._timestep}] State: {self._brain.state} | "
                f"Drone: ({d_pos[0].item():.2f}, {d_pos[1].item():.2f}, {d_pos[2].item():.2f}) | "
                f"Target: ({g_pos[0].item():.2f}, {g_pos[1].item():.2f}, {g_pos[2].item():.2f}) | "
                f"Dist: {dist:.2f}m"
            )

        self._timestep += 1
        return obs, rewards, terminated, truncated, info


class _BrainEnvAdapter:
    """Lightweight adapter so BrainModule can access env internals.

    BrainModule expects an object with `.unwrapped` returning the
    DirectRLEnv instance. When Brain is used inside the env itself,
    we provide this thin wrapper.
    """

    def __init__(self, env: BrainNavDroneEnv):
        self._env = env

    @property
    def unwrapped(self):
        return self._env

    @property
    def device(self):
        return self._env.device


class _RslRlCompatWrapper:
    """Minimal compatibility wrapper for RSL-RL OnPolicyRunner initialization.

    OnPolicyRunner needs an env-like object with specific attributes
    during construction. This wrapper provides just enough interface
    to load a checkpoint without a full RslRlVecEnvWrapper.
    """

    def __init__(self, env: BrainNavDroneEnv):
        self._env = env
        self.num_envs = env.num_envs
        self.device = env.device
        self.num_obs = env.cfg.observation_space
        self.num_actions = env.cfg.action_space
        self.max_episode_length = env.max_episode_length
        self.num_privileged_obs = None
        self.obs_dict = {"policy": torch.zeros(env.num_envs, env.cfg.observation_space, device=env.device)}

    @property
    def unwrapped(self):
        return self._env

    def get_observations(self):
        return self.obs_dict, {}

    def reset(self):
        return self.obs_dict, {}
