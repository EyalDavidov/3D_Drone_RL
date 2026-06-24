# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import sys
import os

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate policy termination reasons.")
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments to simulate.")
parser.add_argument("--num_runs", type=int, default=100, help="Number of episodes to run.")
parser.add_argument("--task", type=str, default="AE-PPO-Drone-Direct-v0", help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
# append RSL-RL cli arguments
import cli_args
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# force headless mode
args_cli.headless = True
args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab.utils.assets import retrieve_file_path

import first_drone.tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.initial_curriculum_level = 5  # Force Curriculum Level 5 for evaluation
    env_cfg.debug_vis = False
    env_cfg.show_ae_images = False
    
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    
    agent_dict = agent_cfg.to_dict()
    for model_key in ["actor", "critic"]:
        if model_key in agent_dict:
            agent_dict[model_key].pop("stochastic", None)
            agent_dict[model_key].pop("init_noise_std", None)
            agent_dict[model_key].pop("noise_std_type", None)
            agent_dict[model_key].pop("state_dependent_std", None)
            
    runner = OnPolicyRunner(env, agent_dict, log_dir=None, device=agent_cfg.device)
    
    resume_path = retrieve_file_path(args_cli.checkpoint)
    print(f"[INFO] Loading checkpoint: {resume_path}")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    
    obs = env.get_observations()
    
    results = []
    num_runs = getattr(args_cli, "num_runs", 100)
    print(f"[INFO] Running evaluation for exactly {num_runs} episodes in headless mode...")
    
    # Warmup step to populate sensors/buffers
    with torch.inference_mode():
        actions = policy(obs)
        obs, _, dones, _ = env.step(actions)
        policy.reset(dones)
    
    step_count = 0
    while len(results) < num_runs:
        with torch.inference_mode():
            actions = policy(obs)
            
            # Step env
            obs, _, dones, _ = env.step(actions)
            policy.reset(dones)
            
            # Check who terminated in this step
            reset_envs = torch.where(dones)[0]
            if len(reset_envs) > 0:
                truncated_tensor = env.unwrapped.reset_time_outs
                terminated_tensor = env.unwrapped.reset_terminated
                crash_reasons = env.unwrapped.crash_reasons
                
                for env_idx in reset_envs:
                    idx = env_idx.item()
                    if len(results) >= num_runs:
                        break
                    
                    if truncated_tensor[idx]:
                        results.append("Timeout")
                    elif terminated_tensor[idx]:
                        crash_code = crash_reasons[idx].item()
                        if crash_code == 0:
                            results.append("Success")
                        elif crash_code == 1:
                            results.append("Floor Collision")
                        elif crash_code == 2:
                            results.append("Ceiling Collision")
                        elif crash_code == 3:
                            results.append("Wall Boundary Collision")
                        elif crash_code == 4:
                            results.append("Obstacle / Mesh Impact")
                        else:
                            results.append("Other Impact")
                    else:
                        results.append("Other Impact")
            
            step_count += 1
            if step_count % 100 == 0:
                print(f"[PROGRESS] Completed {len(results)}/{num_runs} episodes...")
                
    # Print results
    from collections import Counter
    counts = Counter(results)
    
    print("\n" + "="*50)
    print(f"EVALUATION RESULTS FOR {num_runs} EPISODES (HEADLESS) AT LEVEL 5")
    print("="*50)
    print(f"Success (Reached Goal):      {counts['Success']} ({counts['Success']/num_runs*100:.2f}%)")
    print(f"Timeouts:                    {counts['Timeout']} ({counts['Timeout']/num_runs*100:.2f}%)")
    print(f"Floor Collisions:            {counts['Floor Collision']} ({counts['Floor Collision']/num_runs*100:.2f}%)")
    print(f"Ceiling Collisions:          {counts['Ceiling Collision']} ({counts['Ceiling Collision']/num_runs*100:.2f}%)")
    print(f"Wall Boundary Collisions:    {counts['Wall Boundary Collision']} ({counts['Wall Boundary Collision']/num_runs*100:.2f}%)")
    print(f"Obstacle / Mesh Impacts:     {counts['Obstacle / Mesh Impact']} ({counts['Obstacle / Mesh Impact']/num_runs*100:.2f}%)")
    print(f"Other Impacts:               {counts['Other Impact']} ({counts['Other Impact']/num_runs*100:.2f}%)")
    print(f"Total Crashes:               {num_runs - counts['Success'] - counts['Timeout']} ({(num_runs - counts['Success'] - counts['Timeout'])/num_runs*100:.2f}%)")
    print("="*50)

if __name__ == "__main__":
    try:
        main()
    finally:
        import sys
        import os
        sys.stdout.flush()
        simulation_app.close()
        os._exit(0)
