"""Script to train RL agent with SAC+VAE.

Usage:
    c:\\Isaac\\IsaacLab\\isaaclab.bat -p scripts/sac/train.py --task VAE-SAC-Drone-Direct-v0 --num_envs 64 --enable_cameras --freeze_vae 
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with SAC.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="sac_cfg_entry_point", help="Name of the SAC agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment.")
parser.add_argument("--max_iterations", type=int, default=None, help="Total environment steps for training.")
parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging.")
parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from.")
parser.add_argument("--vae_checkpoint", type=str, default=None, help="Path to pretrained VAE weights (.pt) to load before training.")
parser.add_argument("--freeze_vae", action="store_true", help="If set, do not update the VAE during SAC training.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

# clear out sys.argv
sys.argv = [sys.argv[0]]

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
from datetime import datetime

import gymnasium as gym
import torch

import first_drone.tasks  # noqa: F401 — registers environments

from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

from first_drone.models.sac_runner import SACRunner


def main():
    """Train with SAC agent."""
    # ---- Resolve configurations from the gym registry ----
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(args_cli.task, args_cli.agent)

    # ---- Override with CLI arguments ----
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    max_iterations = args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.sac_max_iterations

    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = "cuda:0"

    # ---- Logging directory ----
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    log_root_path = os.path.join(project_root, "logs", "sac", agent_cfg.experiment_name)
    log_dir = os.path.join(log_root_path, datetime.now().strftime("%d-%m_%H-%M"))
    os.makedirs(log_dir, exist_ok=True)
    print(f"[INFO] Logging experiment in directory: {log_dir}")

    # ---- Create environment ----
    env = gym.make(args_cli.task, cfg=env_cfg)

    env.unwrapped.cfg.show_vae_images = False

    if args_cli.vae_checkpoint is not None:
        pretrained = torch.load(args_cli.vae_checkpoint, map_location=env_cfg.sim.device)
        if "vae" in pretrained:
            env.unwrapped.vae.load_state_dict(pretrained["vae"])
        else:
            env.unwrapped.vae.load_state_dict(pretrained)
        env.unwrapped.vae.eval()
        print(f"[INFO] Loaded pretrained VAE weights from: {args_cli.vae_checkpoint}")

    if args_cli.freeze_vae:
        agent_cfg.train_vae = False
        print("[INFO] VAE updates disabled for training (freeze_vae=True)")

    # ---- Create runner ----
    runner = SACRunner(env, agent_cfg, log_dir=log_dir, device=env_cfg.sim.device)

    # ---- Load checkpoint ----
    if args_cli.resume:
        print(f"[INFO]: Loading model checkpoint from: {args_cli.resume}")
        runner.load(args_cli.resume)

    # ---- Run training ----
    runner.learn(num_learning_iterations=max_iterations, no_wandb=args_cli.no_wandb,
                 resume_path=args_cli.resume)

    # ---- Cleanup ----
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
