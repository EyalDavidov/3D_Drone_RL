"""Script to train RL agent with PPO (RSL-RL) + VAE.

Usage:
    c:\Isaac\IsaacLab\isaaclab.bat -p scripts/sac/train_ppo_vae.py --task VAE-SAC-Drone-Direct-v0 --num_envs 512 --enable_cameras --freeze_vae 
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import os
from datetime import datetime

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with PPO on VAE env.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="VAE-SAC-Drone-Direct-v0", help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="first_drone.tasks.direct.navigation_drone.agents.rsl_rl_ppo_cfg:NavigationPPOCfg", help="Path to PPO config."
)
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment.")
parser.add_argument("--max_iterations", type=int, default=None, help="Total environment steps for training.")
parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging.")
parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from.")
parser.add_argument("--vae_checkpoint", type=str, default=None, help="Path to pretrained VAE weights (.pt) to load before training.")
parser.add_argument("--freeze_vae", action="store_true", help="If set, do not update the VAE during training.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

# clear out sys.argv
sys.argv = [sys.argv[0]]

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import first_drone.tasks  # noqa: F401 — registers environments

from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry


def main():
    """Train with PPO agent."""
    # ---- Resolve configurations from the gym registry ----
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    
    # Import the PPO config directly since it's not registered under VAE-SAC-Drone-Direct-v0
    from first_drone.tasks.direct.navigation_drone.agents.rsl_rl_ppo_cfg import NavigationPPOCfg
    agent_cfg = NavigationPPOCfg()

    # ---- Override with CLI arguments ----
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations

    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = "cuda:0"
    agent_cfg.device = "cuda:0"

    # ---- Logging directory ----
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    log_root_path = os.path.join(project_root, "logs", "ppo", agent_cfg.experiment_name)
    log_dir = os.path.join(log_root_path, datetime.now().strftime("%d-%m_%H-%M"))
    os.makedirs(log_dir, exist_ok=True)
    print(f"[INFO] Logging experiment in directory: {log_dir}")

    # ---- Create environment ----
    env = gym.make(args_cli.task, cfg=env_cfg)

    env.unwrapped.cfg.show_vae_images = False

    # ---- Load VAE weights ----
    if args_cli.vae_checkpoint is not None:
        pretrained = torch.load(args_cli.vae_checkpoint, map_location=env_cfg.sim.device)
        if "vae" in pretrained:
            env.unwrapped.vae.load_state_dict(pretrained["vae"])
        else:
            env.unwrapped.vae.load_state_dict(pretrained)
        env.unwrapped.vae.eval()
        print(f"[INFO] Loaded pretrained VAE weights from: {args_cli.vae_checkpoint}")

    if args_cli.freeze_vae:
        print("[INFO] VAE updates disabled for training (freeze_vae=True)")
        # In PPO, we don't train the VAE anyway (rsl_rl doesn't know about it), 
        # so freeze_vae just means we ensure it's in eval mode and doesn't require grads
        env.unwrapped.vae.eval()
        for param in env.unwrapped.vae.parameters():
            param.requires_grad = False

    # ---- Wrap env for rsl-rl ----
    env = RslRlVecEnvWrapper(env)

    # ---- Create runner ----
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)

    # ---- Load checkpoint ----
    if args_cli.resume:
        print(f"[INFO]: Loading model checkpoint from: {args_cli.resume}")
        runner.load(args_cli.resume)

    # ---- Run training ----
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # ---- Cleanup ----
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
