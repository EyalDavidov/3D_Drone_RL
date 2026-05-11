# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to fly the drone smoothly and spin to scan the room with YOLO."""

import argparse
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Hover and scan agent for YOLO testing.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import first_drone.tasks  # noqa: F401

def main():
    # create environment configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    print(f"[INFO]: Starting YOLO Visual Test Flight...")
    env.reset()
    
    # --- Switch the UI viewport explicitly to the drone's camera ---
    try:
        import omni.kit.viewport.utility
        viewport_api = omni.kit.viewport.utility.get_active_viewport()
        if viewport_api is not None:
            viewport_api.camera_path = "/World/envs/env_0/Drone/body/Camera"
    except Exception as e:
        print(f"[WARN] Failed to set viewport camera: {e}")
    # -------------------------------------------------------------

    # simulate environment
    step_count = 0
    import math

    while simulation_app.is_running():
        with torch.inference_mode():
            # Create an action tensor of shape [num_envs, 4]
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            
            # --- Kinematic Orbit ---
            # Instead of physical flight, force the drone along a perfect circle.
            actions[:, :] = 0.0
            
            center_x, center_y = 3.5, 0.0  # Person location
            r = 3.5  # Radius of orbit
            
            # Change angle slowly over time
            theta = math.pi - (step_count * 0.005)
            
            x = center_x + r * math.cos(theta)
            y = center_y + r * math.sin(theta)
            yaw = theta - math.pi  # Drone is always looking at the center
            
            qw = math.cos(yaw / 2.0)
            qz = math.sin(yaw / 2.0)
            
            # Move all drones precisely in the orbit path
            root_state = env.unwrapped._robot.data.default_root_state.clone()
            for i in range(env.unwrapped.num_envs):
                origin_x = env.unwrapped._terrain.env_origins[i, 0]
                origin_y = env.unwrapped._terrain.env_origins[i, 1]
                root_state[i, 0] = origin_x + x
                root_state[i, 1] = origin_y + y
                root_state[i, 2] = 1.0  # Constant height (chest / head level)
                root_state[i, 3] = qw
                root_state[i, 4] = 0.0
                root_state[i, 5] = 0.0
                root_state[i, 6] = qz
            
            env_ids = env.unwrapped._robot._ALL_INDICES
            env.unwrapped._robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
            env.unwrapped._robot.write_root_velocity_to_sim(torch.zeros_like(root_state[:, 7:]), env_ids)
            # -----------------------
            
            env.step(actions)
            step_count += 1

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()