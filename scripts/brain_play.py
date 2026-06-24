# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run the Brain module (Deterministic Room Coverage + YOLO Search & Rescue) on top of the PPO navigator policy."""

import argparse
import sys
import os
import time
import cv2
import random
import torch
import gymnasium as gym

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Run search and rescue brain module on AE-PPO-Drone environment.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate (default: 1 for demo).")
parser.add_argument("--task", type=str, default="AE-PPO-Drone-Direct-v0", help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to PPO model checkpoint (.pt).")
parser.add_argument("--use_mock", action="store_true", default=False, help="Use mock perception instead of real YOLO.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--real-time", action="store_true", default=True, help="Run in real-time, if possible.")
parser.add_argument("--step_size", type=float, default=10.0, help="Lawnmower path corridor step size (meters, matches Level 5 target distance).")
parser.add_argument("--safety_margin", type=float, default=0.7, help="Safety margin from obstacles/walls (meters).")
parser.add_argument("--curriculum_level", type=int, default=5, help="Curriculum level to run (1-5, default: 5).")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# Force camera rendering on since YOLO needs it
args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Enable the debug_draw extension programmatically so search path lines and sensors are rendered
try:
    import omni.kit.app
    ext_manager = omni.kit.app.get_app().get_extension_manager()
    if not ext_manager.is_extension_enabled("omni.isaac.debug_draw"):
        print("\n[INFO] Programmatically enabling 'omni.isaac.debug_draw' extension...\n")
        ext_manager.set_extension_enabled_immediate("omni.isaac.debug_draw", True)
except Exception as e:
    print(f"\n[WARNING] Could not enable 'omni.isaac.debug_draw' programmatically: {e}\n")

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from packaging import version

import first_drone.tasks  # noqa: F401
from first_drone.models.perception import PerceptionModule
from first_drone.models.brain import BrainModule

def main():
    if args_cli.num_envs != 1:
        raise ValueError("brain_play.py only supports --num_envs 1 (Brain controls a single drone mission).")

    # 1. Parse config
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=True
    )
    
    # Force single-agent / play seed settings
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed
    else:
        env_cfg.seed = random.randint(0, 100000)
    
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.initial_curriculum_level = args_cli.curriculum_level
    
    # Auto-enable debug visualization in play mode so LiDAR and targets are displayed
    env_cfg.debug_vis = True
    env_cfg.show_ae_images = False
    env_cfg.spawn_person = True

    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    agent_cfg.device = env_cfg.sim.device
    agent_cfg.logger = "tensorboard"
    
    # 2. Instantiate gymnasium environment
    print(f"[INFO] Creating environment: {args_cli.task}")
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.unwrapped.is_brain_play = True
    
    # 3. Wrap environment for rsl-rl compatibility
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    env.unwrapped.is_brain_play = True
    
    # 4. Resolve PPO model checkpoint path
    if args_cli.checkpoint:
        checkpoint_path = args_cli.checkpoint
    else:
        # Search for latest checkpoint in logs
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        log_root_path = os.path.join(project_root, "logs", "rsl_rl", "Navigation_Drone_Direct")
        if not os.path.exists(log_root_path):
            log_root_path = os.path.join(project_root, "logs", "ppo", "navigation_drone_direct")
        
        try:
            checkpoint_path = get_checkpoint_path(log_root_path, run_dir=".*", checkpoint=".*")
        except Exception as e:
            # Fallback to some specific known checkpoints if exist
            print(f"[WARN] Automatic checkpoint resolution failed ({e}). Attempting hardcoded fallback.")
            checkpoint_path = os.path.join(project_root, "logs", "ppo", "navigation_drone_direct", "24-06_01-12", "model_800.pt")
            
    print(f"[INFO] Loading PPO policy checkpoint from: {checkpoint_path}")
    
    # 5. Load low-level navigation policy (JIT or Runner)
    # RSL-RL policies are exported or loaded using torch.jit or runners
    try:
        # Load JIT policy if possible (exported policies are standard torchscript)
        policy_dir = os.path.dirname(checkpoint_path)
        jit_policy_path = os.path.join(policy_dir, "exported", "policy.pt")
        if not os.path.exists(jit_policy_path):
            raise FileNotFoundError(f"No exported JIT policy found at: {jit_policy_path}")
            
        print(f"[INFO] Loading JIT Policy: {jit_policy_path}")
        policy = torch.jit.load(jit_policy_path, map_location=env.unwrapped.device)
        policy.eval()
        policy_expects_obs_dict = False
    except Exception as ex:
        print(f"[WARN] Failed to load policy as JIT ({ex}). Trying to load using RSL-RL Runner.")
        from rsl_rl.runners import OnPolicyRunner

        agent_dict = agent_cfg.to_dict()
        for model_key in ["actor", "critic"]:
            if model_key in agent_dict:
                agent_dict[model_key].pop("stochastic", None)
                agent_dict[model_key].pop("init_noise_std", None)
                agent_dict[model_key].pop("noise_std_type", None)
                agent_dict[model_key].pop("state_dependent_std", None)

        runner = OnPolicyRunner(env, agent_dict, log_dir=None, device=agent_cfg.device)
        runner.load(checkpoint_path)
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        policy_expects_obs_dict = True

    # 6. Initialize perception and brain modules
    print(f"[INFO] Initializing Perception Module (use_mock={args_cli.use_mock})")
    perception = PerceptionModule(use_mock=args_cli.use_mock)
    
    print(f"[INFO] Initializing Brain Module (step_size={args_cli.step_size}m, margin={args_cli.safety_margin}m)")
    brain = BrainModule(env, step_size=args_cli.step_size, safety_margin=args_cli.safety_margin)
    
    # --- Switch the viewport camera to the drone's perspective ---
    try:
        import omni.kit.viewport.utility
        viewport_api = omni.kit.viewport.utility.get_active_viewport()
        if viewport_api is not None:
            viewport_api.camera_path = "/World/envs/env_0/Drone/body/Camera"
            print(f"[INFO] Viewport camera switched to drone camera: {viewport_api.camera_path}")
    except Exception as e:
        print(f"[WARN] Failed to set viewport camera to drone view: {e}")
    # -------------------------------------------------------------

    # Initialize environment
    print(f"[INFO] Running Search & Rescue demonstration. Press Ctrl+C to exit.")
    env.reset()
    
    # Store initial search/waypoint lines in debug draw interface for clear visualization
    draw_search_path(env, brain.waypoints)
    
    timestep = 0
    dt = env.unwrapped.step_dt
    yolo_interval = 5
    last_person_found = torch.zeros(args_cli.num_envs, dtype=torch.bool, device=env.unwrapped.device)
    last_person_world_xyz = torch.zeros((args_cli.num_envs, 3), device=env.unwrapped.device)
    
    # Main simulation loop
    while simulation_app.is_running():
        start_time = time.time()
        
        with torch.inference_mode():
            # 1. Grab camera outputs and drone state
            rgb_image = env.unwrapped._tiled_camera.data.output["rgb"].clone()
            depth_image = env.unwrapped._tiled_camera.data.output["depth"].clone()
            drone_pos = env.unwrapped._robot.data.root_pos_w.clone()
            drone_quat = env.unwrapped._robot.data.root_quat_w.clone()
            # Replace infinity values in depth
            depth_image[depth_image == float("inf")] = 10.0
            

            
            # 2. Run Perception (YOLO + de-projection)
            run_yolo = brain.state != "SCAN" or (timestep % yolo_interval == 0)
            if run_yolo:
                person_found, person_world_xyz = perception.process_camera_data(
                    rgb_image, depth_image, drone_pos, drone_quat
                )
                last_person_found = person_found
                last_person_world_xyz = person_world_xyz
            else:
                person_found = last_person_found
                person_world_xyz = last_person_world_xyz
            
            # 3. Update Brain State Machine
            desired_pos_w, target_yaw = brain.update(
                person_found, person_world_xyz, drone_pos, drone_quat
            )
            
            # 4. Set the high-level commands directly in the environment
            env.unwrapped._desired_pos_w[:, :] = desired_pos_w
            env.unwrapped._target_yaw[:] = target_yaw
            
            # 5. Re-evaluate observations for the actor policy with the new targets
            obs_dict = env.unwrapped._get_observations()
            
            # 6. Action determination (bypassing high-level policy during high-level states)
            if brain.state == "SCAN":
                # Spin in place: zero translation velocities, positive yaw rate command
                actions = torch.zeros((args_cli.num_envs, 4), device=env.unwrapped.device)
                actions[:, 3] = 0.25  # Smooth yaw rate rotation command
            elif brain.state == "COMPLETE":
                # Hover in place at final target coordinates
                actions = torch.zeros((args_cli.num_envs, 4), device=env.unwrapped.device)
            else:
                # Normal high-level navigation toward active waypoints/targets
                policy_obs = obs_dict if policy_expects_obs_dict else obs_dict["policy"]
                actions = policy(policy_obs)
            
            # 7. Step the simulator
            obs, _, dones, _ = env.step(actions)
            
            # 8. Reset handling: if drone reset, recreate the brain for the new scenario
            if dones[0].item():
                print(f"[Brain] Environment reset detected. Re-initializing search parameters.")
                brain = BrainModule(env, step_size=args_cli.step_size, safety_margin=args_cli.safety_margin)
                draw_search_path(env, brain.waypoints)
            elif timestep % 100 == 0:
                d_pos = drone_pos[0]
                g_pos = desired_pos_w[0]
                dist = torch.norm(d_pos - g_pos).item()
                print(f"[Brain Step {timestep}] State: {brain.state} | Drone: ({d_pos[0].item():.2f}, {d_pos[1].item():.2f}, {d_pos[2].item():.2f}) | Target: ({g_pos[0].item():.2f}, {g_pos[1].item():.2f}, {g_pos[2].item():.2f}) | Dist: {dist:.2f}m")
                
        # Handle OpenCV visualization windows
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
        timestep += 1
        
        # Maintain real-time speed if requested
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)
            
    # Cleanup
    cv2.destroyAllWindows()
    env.close()
    simulation_app.close()

def draw_search_path(env, waypoints):
    """
    Draw cyan lines in the simulator view showing the generated lawnmower waypoints.
    """
    try:
        from omni.isaac.debug_draw import _debug_draw
        draw = _debug_draw.acquire_debug_draw_interface()
        if draw is None or len(waypoints) < 2:
            return
            
        env_origin = env.unwrapped._terrain.env_origins[0].cpu().numpy()
        
        start_pts = []
        end_pts = []
        colors = []
        thicknesses = []
        
        # Draw search path segments
        for i in range(len(waypoints) - 1):
            wp1 = waypoints[i]
            wp2 = waypoints[i+1]
            
            p1 = (wp1[0] + env_origin[0], wp1[1] + env_origin[1], wp1[2] + 0.1)
            p2 = (wp2[0] + env_origin[0], wp2[1] + env_origin[1], wp2[2] + 0.1)
            
            start_pts.append(p1)
            end_pts.append(p2)
            colors.append((0.0, 1.0, 1.0, 0.75))  # cyan color path
            thicknesses.append(3.0)
            
        # Draw waypoint nodes (spheres/crosses)
        for wp in waypoints:
            p = (wp[0] + env_origin[0], wp[1] + env_origin[1], wp[2])
            # Draw tiny local cross or marker at waypoint by batching into lines
            start_pts.append((p[0] - 0.15, p[1], p[2]))
            end_pts.append((p[0] + 0.15, p[1], p[2]))
            colors.append((0.0, 1.0, 1.0, 1.0))
            thicknesses.append(3.0)
            
            start_pts.append((p[0], p[1] - 0.15, p[2]))
            end_pts.append((p[0], p[1] + 0.15, p[2]))
            colors.append((0.0, 1.0, 1.0, 1.0))
            thicknesses.append(3.0)
            
        # Draw all lines to stage in a single batch
        draw.draw_lines(start_pts, end_pts, colors, thicknesses)
        
    except Exception as e:
        print(f"[Brain] Could not draw debug search paths on stage: {e}")

if __name__ == "__main__":
    main()
