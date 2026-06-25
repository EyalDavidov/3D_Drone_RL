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

import numpy as np
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
        self._closing = False
        self._mission_complete = False

        if getattr(self, "_room_bounds_local", None) is not None:
            print(f"[BrainNavEnv] Spawn bounds from USD room: {self._room_bounds_local}\n")

        self._sync_map_geometry_from_usd()

        # ---------- Load Frozen PPO Navigator Policy ----------
        self._load_navigator_policy()

        # ---------- Initialize Perception Module ----------
        self._perception = PerceptionModule(
            use_mock=self.cfg.use_mock_perception,
            person_conf_threshold=self.cfg.yolo_person_conf_threshold,
            min_bbox_area_frac=self.cfg.yolo_min_bbox_area_frac,
        )
        print(
            f"\n[BrainNavEnv] Perception initialized (use_mock={self.cfg.use_mock_perception}, "
            f"person_conf>={self.cfg.yolo_person_conf_threshold:.0%})\n"
        )

        # ---------- Initialize Brain Module ----------
        # The Brain needs a wrapper that looks like the RslRlVecEnvWrapper
        # but we can pass self directly since it has .unwrapped access
        self._brain = BrainModule(
            _BrainEnvAdapter(self),
            step_size=self.cfg.brain_step_size,
            safety_margin=self.cfg.brain_safety_margin,
        )
        print(f"\n[BrainNavEnv] Brain initialized (step_size={self.cfg.brain_step_size}m, "
              f"margin={self.cfg.brain_safety_margin}m, "
              f"sequential={getattr(self.cfg, 'brain_use_sequential_spawns', False)})\n")

        # ---------- Internal State Buffers ----------
        self._timestep = 0
        self._last_person_found = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._last_person_world_xyz = torch.zeros((self.num_envs, 3), device=self.device)

        # Re-apply debug vis with brain-safe implementation (clears parent post-update subscription)
        self.set_debug_vis(self.cfg.debug_vis)

        # Brain play: use long episodes; parent init overwrites cfg.episode_length_s via curriculum.
        if getattr(self.cfg, "brain_disable_episode_timeout", True):
            play_len = 3600.0
            self._update_episode_length(play_len)
            print(f"[BrainNavEnv] Episode timeout disabled for play (length={play_len}s).\n")

        # Initial spawn + Brain SLAM target (parent reset uses random PPO goals — override now).
        self._apply_brain_spawn_and_goal(self._robot._ALL_INDICES, mission_snapshot=None)

    def _build_sequential_spawn_sequence(self) -> tuple[tuple, list[str]]:
        """Build scan/nav waypoints: rooms in order, corridor insert, finish at Worker."""
        zones = getattr(self, "_map_zones", None) or {}
        room_keys = sorted(
            (k for k in zones if k.startswith("room_")),
            key=lambda k: int(k.split("_", 1)[1]),
        )
        if len(room_keys) < 2:
            centers = getattr(self, "_room_segment_centers", None) or []
            if len(centers) < 2:
                return tuple(), []
            seq = [tuple(c) for c in centers]
            labels = [f"room_{i + 1}" for i in range(len(centers))]
            finish = list(seq[-1])
            segments = getattr(self, "_room_segments", {})
            if segments:
                last_idx = max(segments.keys())
                _, _, ly0, ly1 = segments[last_idx]["bounds"]
                finish[1] = ly0 + 0.35
                finish[2] = 1.0
            seq.append(tuple(finish))
            labels.append("finish")
            return tuple(seq), labels

        seq: list[tuple] = []
        labels: list[str] = []
        for key in room_keys:
            seq.append(zones[key]["center"])
            labels.append(key)

        if "corridor" in zones:
            seq.insert(3, zones["corridor"]["center"])
            labels.insert(3, "corridor")
        elif "side_coridors" in zones:
            seq.insert(3, zones["side_coridors"]["center"])
            labels.insert(3, "side_coridors")

        finish = getattr(self, "_finish_point_local", None)
        if finish is None and room_keys:
            lx0, lx1, ly0, ly1 = zones[room_keys[-1]]["bounds"]
            finish = (0.5 * (lx0 + lx1), ly0 + 0.35, 1.0)
        if finish is not None:
            seq.append(tuple(finish))
            labels.append("finish (Worker)")

        return tuple(seq), labels

    def _sync_map_geometry_from_usd(self) -> None:
        """Apply measured final_flat.usd dimensions to config and Brain spawn sequence."""
        raw = getattr(self, "_map_bounds_raw_local", None)
        if raw is not None:
            self.cfg.map_bounds = tuple(raw)
            print(
                f"[BrainNavEnv] map_bounds synced from USD: "
                f"X=[{raw[0]:.2f}, {raw[1]:.2f}] Y=[{raw[2]:.2f}, {raw[3]:.2f}]\n"
            )

        if not getattr(self.cfg, "brain_auto_room_spawns", True):
            return

        seq, labels = self._build_sequential_spawn_sequence()
        if len(seq) < 2:
            print("[BrainNavEnv] WARNING: Could not parse map zones from USD — using cfg spawn sequence.\n")
            return

        self.cfg.brain_spawn_sequence = seq
        print("[BrainNavEnv] Sequential spawn sequence from USD map zones:")
        for label, pt in zip(labels, seq):
            print(f"  • {label}: ({pt[0]:.2f}, {pt[1]:.2f}, {pt[2]:.2f})")
        print()

    def _capture_brain_mission(self):
        if not hasattr(self, "_brain") or self._brain.found_person:
            return None
        if getattr(self.cfg, "brain_use_sequential_spawns", False):
            return None  # crash always restarts at spawn1
        if not getattr(self.cfg, "brain_preserve_mission_on_crash", True):
            return None
        return self._brain.capture_mission_snapshot()

    def _get_spawn1_local(self) -> tuple[float, float, float]:
        """Always return spawn1 from the configured sequence (env-local)."""
        seq = getattr(self.cfg, "brain_spawn_sequence", None)
        if seq and len(seq) > 0:
            return float(seq[0][0]), float(seq[0][1]), float(seq[0][2])
        return 0.0, 0.0, 1.0

    def _sample_brain_spawn_xyz(self, env_count: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Spawn at spawn1 for sequential mission, else random navigable floor."""
        device = self.device
        if getattr(self.cfg, "brain_use_sequential_spawns", False):
            sx, sy, sz = self._get_spawn1_local()
            spawn_x = torch.full((env_count,), sx, device=device)
            spawn_y = torch.full((env_count,), sy, device=device)
            spawn_z = torch.full((env_count,), sz, device=device)
            return spawn_x, spawn_y, spawn_z
        sx, sy = self._sample_navigable_spawn_xy(env_count)
        spawn_z = torch.full((env_count,), 1.0, device=device)
        return sx, sy, spawn_z

    def _apply_brain_spawn_and_goal(self, env_ids, mission_snapshot=None):
        """Place drone on navigable floor and set PPO target from the Brain SLAM state."""
        from isaaclab.utils.math import quat_from_euler_xyz

        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        env_count = env_ids.shape[0]

        spawn_x, spawn_y, spawn_z = self._sample_brain_spawn_xyz(env_count)

        default_root_state = self._robot.data.default_root_state[env_ids].clone()
        default_root_state[:, 0] = spawn_x + self._terrain.env_origins[env_ids, 0]
        default_root_state[:, 1] = spawn_y + self._terrain.env_origins[env_ids, 1]
        default_root_state[:, 2] = spawn_z
        default_root_state[:, 7:] = 0.0

        if hasattr(self, "_brain"):
            if mission_snapshot is not None:
                self._brain.restore_mission_snapshot(mission_snapshot)
            elif getattr(self.cfg, "brain_use_sequential_spawns", False):
                self._brain.reset_mission_from_start()
            elif not hasattr(self, "_brain_spawn_initialized"):
                self._brain_spawn_initialized = True

        goal_local = np.zeros((env_count, 3), dtype=np.float32)
        for i in range(env_count):
            if hasattr(self, "_brain"):
                gl = self._brain.get_brain_goal_local(
                    drone_local_xy=(float(spawn_x[i].item()), float(spawn_y[i].item()))
                )
                goal_local[i] = gl
            else:
                goal_local[i] = (float(spawn_x[i].item()), float(spawn_y[i].item()), 1.0)

        self._desired_pos_w[env_ids, 0] = (
            torch.tensor(goal_local[:, 0], device=self.device) + self._terrain.env_origins[env_ids, 0]
        )
        self._desired_pos_w[env_ids, 1] = (
            torch.tensor(goal_local[:, 1], device=self.device) + self._terrain.env_origins[env_ids, 1]
        )
        self._desired_pos_w[env_ids, 2] = torch.tensor(goal_local[:, 2], device=self.device)

        dx = self._desired_pos_w[env_ids, 0] - default_root_state[:, 0]
        dy = self._desired_pos_w[env_ids, 1] - default_root_state[:, 1]
        dist_xy = torch.sqrt(dx * dx + dy * dy)
        # SCAN locks goal on spawn — atan2(0, 0) is NaN and corrupts physics.
        goal_yaw = torch.where(dist_xy > 1e-3, torch.atan2(dy, dx), torch.zeros_like(dist_xy))
        zeros = torch.zeros_like(goal_yaw)
        default_root_state[:, 3:7] = quat_from_euler_xyz(zeros, zeros, goal_yaw)
        self._target_yaw[env_ids] = goal_yaw

        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)

        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        self._prev_dist_to_goal[env_ids] = torch.linalg.norm(
            self._desired_pos_w[env_ids] - default_root_state[:, :3], dim=1
        )

        sx, sy = float(spawn_x[0].item()), float(spawn_y[0].item())
        sz = float(spawn_z[0].item())
        gx, gy = float(goal_local[0, 0]), float(goal_local[0, 1])
        state = getattr(self._brain, "state", "?") if hasattr(self, "_brain") else "?"
        seg = getattr(self._brain, "segment_idx", "?") if hasattr(self, "_brain") else "?"
        print(
            f"[BrainNavEnv] Spawned at spawn1 local ({sx:.2f}, {sy:.2f}, {sz:.2f}) | "
            f"SLAM target ({gx:.2f}, {gy:.2f}) | state={state} segment={seg}\n"
        )

    def _reset_idx(self, env_ids: torch.Tensor | None = None):
        """Crash/episode reset: respawn at spawn1 and restart sequential mission."""
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        mission = self._capture_brain_mission()
        if getattr(self.cfg, "brain_use_sequential_spawns", False) and hasattr(self, "_brain"):
            self._brain.reset_mission_from_start()
            mission = None

        super()._reset_idx(env_ids)
        self._apply_brain_spawn_and_goal(env_ids, mission_snapshot=mission)
        self._stuck_step_count = 0
        if env_ids.shape[0] > 0:
            self._prev_drone_pos_xy = self._robot.data.root_pos_w[0, :2].clone()
        else:
            self._prev_drone_pos_xy = None

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

            # 2. Run Perception (YOLO + de-projection) — always run during SCAN spin so we don't miss a person
            run_yolo = (self._timestep % max(1, self.cfg.brain_yolo_interval)) == 0
            if run_yolo:
                person_found, person_world_xyz = self._perception.process_camera_data(
                    rgb_image, depth_image, drone_pos, drone_quat
                )
                self._last_person_found = person_found
                self._last_person_world_xyz = person_world_xyz
                if person_found.any():
                    p = person_world_xyz[0].cpu().numpy()
                    print(
                        f"[BrainNavEnv] YOLO person detected at world "
                        f"({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f}) | brain state={self._brain.state}"
                    )
            else:
                person_found = self._last_person_found
                person_world_xyz = self._last_person_world_xyz

            # 3. Update Brain State Machine
            desired_pos_w, target_yaw = self._brain.update(
                person_found, person_world_xyz, drone_pos, drone_quat
            )

            if (
                self._brain.state == "COMPLETE"
                and self._brain.found_person
                and not self._mission_complete
            ):
                self._mission_complete = True
                target = self._brain.target_person_pos
                print(
                    "\n[BrainNavEnv] MISSION COMPLETE — high-confidence person reached.\n"
                    f"  Rescue coordinates (local): X:{target[0]:.2f} Y:{target[1]:.2f} Z:{target[2]:.2f}\n"
                )
            elif (
                self._brain.state == "COMPLETE"
                and getattr(self._brain, "mission_finished", False)
                and not self._brain.found_person
                and not self._mission_complete
            ):
                self._mission_complete = True
                finish = self.cfg.brain_spawn_sequence[-1]
                print(
                    "\n[BrainNavEnv] MISSION COMPLETE — reached finish point (no person found).\n"
                    f"  Finish coordinates (local): X:{finish[0]:.2f} Y:{finish[1]:.2f} Z:{finish[2]:.2f}\n"
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
            elif self._brain.state == "APPROACH_TARGET":
                # Rescue: use PPO navigator to fly toward detected person (not spin)
                policy_obs = obs_dict if self._navigator_policy_expects_dict else obs_dict["policy"]
                ppo_actions = self._navigator_policy(policy_obs)
            else:
                # Normal navigation: query frozen PPO policy
                policy_obs = obs_dict if self._navigator_policy_expects_dict else obs_dict["policy"]
                ppo_actions = self._navigator_policy(policy_obs)

        # 7. Step the parent environment with the PPO-generated actions
        obs, rewards, terminated, truncated, info = super().step(ppo_actions)

        # Detect stuck drone (wedged against wall, no XY movement) and force reset
        pos_now = self._robot.data.root_pos_w[0]
        if self._prev_drone_pos_xy is None:
            self._prev_drone_pos_xy = pos_now[:2].clone()
            self._stuck_step_count = 0
        else:
            moved = torch.norm(pos_now[:2] - self._prev_drone_pos_xy).item()
            if moved < 0.03 and self._brain.state in ("GOTO_WAYPOINT", "APPROACH_TARGET"):
                self._stuck_step_count += 1
            else:
                self._stuck_step_count = 0
            self._prev_drone_pos_xy = pos_now[:2].clone()

        if self._stuck_step_count > 120:
            print("[BrainNavEnv] Drone stuck — resetting to spawn1 and restarting mission.")
            if getattr(self.cfg, "brain_use_sequential_spawns", False):
                self._brain.reset_mission_from_start()
            self._reset_idx(torch.tensor([0], device=self.device))

        # Log crash without reset when brain_reset_on_crash=False
        if not getattr(self.cfg, "brain_reset_on_crash", True):
            last_died = getattr(self, "_last_died", None)
            if last_died is not None and last_died.any() and not getattr(self, "_logged_crash_no_reset", False):
                self._logged_crash_no_reset = True
                print("[BrainNavEnv] Drone crashed — sim kept running (brain_reset_on_crash=False). Press Ctrl+C to exit.")

        # 8. Reset handling (_reset_idx preserves SLAM mission and respawns on map floor)
        dones = terminated | truncated
        if dones.any():
            last_died = getattr(self, "_last_died", None)
            for env_id in range(self.num_envs):
                if not dones[env_id].item():
                    continue
                if last_died is not None and last_died[env_id].item():
                    reason = getattr(self, "_env0_crash_reason", "unknown")
                    if getattr(self.cfg, "brain_use_sequential_spawns", False):
                        print(
                            f"[BrainNavEnv] Crash ({reason}) — respawned at spawn1, "
                            f"mission restarted from beginning."
                        )
                    else:
                        wp = getattr(self._brain, "current_wp_idx", 0)
                        print(
                            f"[BrainNavEnv] Crash — respawned on map floor, "
                            f"continuing SLAM (state={self._brain.state}, wp={wp})."
                        )
                else:
                    print(f"[BrainNavEnv] Environment {env_id} reset (timeout).")

        # 9. Periodic status logging
        if self._timestep % 100 == 0:
            d_pos = drone_pos[0]
            g_pos = desired_pos_w[0]
            dist = torch.norm(d_pos - g_pos).item()
            yolo_conf = getattr(self._perception, "last_best_person_conf", 0.0)
            yolo_note = (
                f"YOLO best person={yolo_conf:.0%} (need {self.cfg.yolo_person_conf_threshold:.0%})"
            )
            seg = getattr(self._brain, "segment_idx", 0)
            visited, total = self._brain.coverage_stats() if hasattr(self._brain, "coverage_stats") else (0, 0)
            cov_note = f"coverage={visited}/{total}" if total > 0 else ""
            print(
                f"[BrainNavEnv Step {self._timestep}] State: {self._brain.state} | seg={seg} | "
                f"Drone: ({d_pos[0].item():.2f}, {d_pos[1].item():.2f}, {d_pos[2].item():.2f}) | "
                f"Target: ({g_pos[0].item():.2f}, {g_pos[1].item():.2f}, {g_pos[2].item():.2f}) | "
                f"Dist: {dist:.2f}m | {yolo_note} {cov_note}"
            )

        self._timestep += 1
        return obs, rewards, terminated, truncated, info

    def set_debug_vis(self, debug_vis: bool) -> None:
        """Brain play: use step()-time debug draw only (no post-update subscription).

        The post-update weakref callback can fire during Omniverse shutdown and crash
        omni.syntheticdata / omni.graph.core plugins.
        """
        if debug_vis:
            if hasattr(self, "_debug_vis_handle") and self._debug_vis_handle is not None:
                self._debug_vis_handle.unsubscribe()
                self._debug_vis_handle = None
            self._set_debug_vis_impl(True)
        else:
            if hasattr(self, "_debug_vis_handle") and self._debug_vis_handle is not None:
                self._debug_vis_handle.unsubscribe()
                self._debug_vis_handle = None
            self._set_debug_vis_impl(False)

    def close(self):
        """Disable debug draw and OpenCV windows before Isaac Sim tears down."""
        self._closing = True
        try:
            self.set_debug_vis(False)
        except Exception:
            pass
        try:
            import cv2

            cv2.destroyAllWindows()
        except Exception:
            pass
        super().close()


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
