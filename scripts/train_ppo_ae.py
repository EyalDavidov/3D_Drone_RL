"""Script to train RL agent with PPO (RSL-RL) + AE.

Usage:
    c:\\Isaac\\IsaacLab\\isaaclab.bat -p scripts/train_ppo_ae.py --task AE-PPO-Drone-Direct-v0 --num_envs 512 --enable_cameras --freeze_ae
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import os
from datetime import datetime

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with PPO on AE env.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="AE-PPO-Drone-Direct-v0", help="Name of the task.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment.")
parser.add_argument("--max_iterations", type=int, default=None, help="Total environment steps for training.")
parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging (uses tensorboard instead).")
parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from.")
parser.add_argument("--ae_checkpoint", type=str, default=None, help="Path to pretrained AE weights (.pt) to load before training.")
parser.add_argument("--freeze_ae", action="store_true", default=True, help="If set, do not update the AE during training. Default is True.")
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
    
    # Import the PPO config directly
    from first_drone.tasks.direct.navigation_drone.agents.rsl_rl_ppo_cfg import NavigationPPOCfg
    agent_cfg = NavigationPPOCfg()

    # ---- Override with CLI arguments ----
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations

    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = "cuda:0"
    agent_cfg.device = "cuda:0"

    # Disable wandb if flag is set
    if args_cli.no_wandb:
        agent_cfg.logger = "tensorboard"
        agent_cfg.wandb_project = None

    # ---- Logging directory ----
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    log_root_path = os.path.join(project_root, "logs", "ppo", agent_cfg.experiment_name)
    log_dir = os.path.join(log_root_path, datetime.now().strftime("%d-%m_%H-%M"))
    os.makedirs(log_dir, exist_ok=True)
    print(f"[INFO] Logging experiment in directory: {log_dir}")

    # ---- Create environment ----
    env = gym.make(args_cli.task, cfg=env_cfg)

    env.unwrapped.cfg.show_ae_images = False

    # ---- Load AE weights ----
    # Try custom path first, then fall back to config path
    ae_path = args_cli.ae_checkpoint or env.unwrapped.cfg.ae_checkpoint_path
    if ae_path is not None and os.path.exists(ae_path):
        pretrained = torch.load(ae_path, map_location=env_cfg.sim.device)
        if "ae" in pretrained:
            env.unwrapped.ae.load_state_dict(pretrained["ae"])
        else:
            env.unwrapped.ae.load_state_dict(pretrained)
        env.unwrapped.ae.eval()
        print(f"[INFO] Loaded pretrained AE weights from: {ae_path}")
    elif ae_path is not None:
        print(f"[WARNING] AE checkpoint path was specified but file not found: {ae_path}")

    if args_cli.freeze_ae:
        print("[INFO] AE updates disabled for training (freeze_ae=True)")
        env.unwrapped.ae.eval()
        for param in env.unwrapped.ae.parameters():
            param.requires_grad = False

    # ---- Wrap env for rsl-rl ----
    env = RslRlVecEnvWrapper(env)

    # ---- Create runner ----
    agent_dict = agent_cfg.to_dict()
    for model_key in ["actor", "critic"]:
        if model_key in agent_dict:
            agent_dict[model_key].pop("stochastic", None)
            agent_dict[model_key].pop("init_noise_std", None)
            agent_dict[model_key].pop("noise_std_type", None)
            agent_dict[model_key].pop("state_dependent_std", None)

    runner = OnPolicyRunner(env, agent_dict, log_dir=log_dir, device=agent_cfg.device)

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
