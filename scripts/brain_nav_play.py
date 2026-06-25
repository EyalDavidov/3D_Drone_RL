# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run the self-contained Brain Navigation environment.

This is a simplified launch script for the BrainNavDroneEnv, which embeds
the Brain module, YOLO perception, and frozen PPO navigator internally.
The script simply creates the env and loops env.step() — all intelligence
is inside the environment.

Usage:
    python scripts/brain_nav_play.py --navigator_checkpoint <path_to_checkpoint_dir_or_model.pt> [--use_mock]
"""

import argparse
import sys
import os
import time
import traceback
import torch
import gymnasium as gym

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Run the self-contained Brain Navigation environment.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments (default: 1).")
parser.add_argument("--task", type=str, default="Brain-Nav-Drone-Direct-v0", help="Name of the task.")
parser.add_argument(
    "--navigator_checkpoint", type=str, required=True,
    help="Path to the trained PPO navigator checkpoint (.pt file or directory containing exported/policy.pt)."
)
parser.add_argument("--use_mock", action="store_true", default=False, help="Use mock perception instead of real YOLO.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--real-time", action="store_true", default=True, help="Run in real-time, if possible.")
parser.add_argument("--step_size", type=float, default=10.0, help="Lawnmower path corridor step size (meters).")
parser.add_argument("--safety_margin", type=float, default=0.7, help="Safety margin from obstacles/walls (meters).")
parser.add_argument(
    "--yolo_conf", type=float, default=0.95,
    help="Minimum YOLO confidence (0-1) to accept a person detection (default: 0.95).",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# Force camera rendering on since the env needs depth/RGB
args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Enable the debug_draw extension programmatically
try:
    import omni.kit.app
    ext_manager = omni.kit.app.get_app().get_extension_manager()
    if not ext_manager.is_extension_enabled("omni.isaac.debug_draw"):
        print("\n[INFO] Programmatically enabling 'omni.isaac.debug_draw' extension...\n")
        ext_manager.set_extension_enabled_immediate("omni.isaac.debug_draw", True)
except Exception as e:
    print(f"\n[WARNING] Could not enable 'omni.isaac.debug_draw' programmatically: {e}\n")

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import first_drone.tasks  # noqa: F401

import random
try:
    import cv2
except ImportError:
    cv2 = None


def main():
    # 1. Parse config
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=True
    )

    # Apply CLI overrides to the config
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed
    else:
        env_cfg.seed = random.randint(0, 100000)

    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.navigator_checkpoint_path = args_cli.navigator_checkpoint
    env_cfg.use_mock_perception = args_cli.use_mock
    env_cfg.brain_step_size = args_cli.step_size
    env_cfg.brain_safety_margin = args_cli.safety_margin
    env_cfg.yolo_person_conf_threshold = args_cli.yolo_conf
    env_cfg.debug_vis = True
    env_cfg.show_ae_images = False

    # 2. Create the self-contained environment
    print(f"[INFO] Creating environment: {args_cli.task}")
    env = gym.make(args_cli.task, cfg=env_cfg)

    # --- Switch the viewport camera to the drone's perspective ---
    try:
        import omni.kit.viewport.utility
        viewport_api = omni.kit.viewport.utility.get_active_viewport()
        if viewport_api is not None:
            viewport_api.camera_path = "/World/envs/env_0/Drone/body/Camera"
            print(f"[INFO] Viewport camera switched to drone camera: {viewport_api.camera_path}")
    except Exception as e:
        print(f"[WARN] Failed to set viewport camera to drone view: {e}")

    # 3. Reset and run
    print(f"[INFO] Running Brain Navigation. Press Ctrl+C or 'q' in YOLO window to exit.")
    obs, info = env.reset()

    dt = env.unwrapped.step_dt
    dummy_action = torch.zeros((args_cli.num_envs, 4), device=env.unwrapped.device)
    exit_reason = "unknown"

    try:
        while simulation_app.is_running():
            start_time = time.time()

            # The environment is self-driving — dummy action is ignored
            obs, rewards, terminated, truncated, info = env.step(dummy_action)

            # Handle OpenCV visualization windows
            if cv2 is not None and cv2.waitKey(1) & 0xFF == ord('q'):
                exit_reason = "user_pressed_q"
                break

            if getattr(env.unwrapped, "_mission_complete", False):
                exit_reason = "mission_complete"
                if getattr(env.unwrapped._brain, "found_person", False):
                    print("[INFO] Person rescued — stopping play loop.")
                else:
                    print("[INFO] Finish point reached — stopping play loop.")
                break

            # Maintain real-time speed if requested
            sleep_time = dt - (time.time() - start_time)
            if args_cli.real_time and sleep_time > 0:
                time.sleep(sleep_time)
        else:
            exit_reason = "simulation_app_stopped"
    except KeyboardInterrupt:
        exit_reason = "keyboard_interrupt"
        print("\n[INFO] Interrupted by user.")
    except Exception as exc:
        exit_reason = "error"
        print(f"\n[ERROR] Play loop crashed: {exc}")
        traceback.print_exc()
    finally:
        print(f"[INFO] Play loop ended ({exit_reason}).")
        # Tear down debug draw / YOLO windows before Omniverse shutdown (prevents native crash on exit)
        try:
            env.unwrapped.set_debug_vis(False)
        except Exception:
            pass
        if cv2 is not None:
            cv2.destroyAllWindows()
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
