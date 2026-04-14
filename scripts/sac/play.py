"""SAC+VAE Evaluation Script.

Loads a trained checkpoint and runs the policy in the environment.

Usage:
    c:\\Isaac\\IsaacLab\\isaaclab.bat -p scripts/sac/play.py --checkpoint logs/sac/.../checkpoint_final.pt --num_envs 4 --enable_cameras
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate SAC+VAE drone agent.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt file.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments.")
parser.add_argument("--task", type=str, default="SAC-First-Drone-Direct-v0", help="Task name.")
parser.add_argument("--num_episodes", type=int, default=20, help="Number of episodes to run.")
parser.add_argument("--viewer", action="store_true", help="Enable OpenCV debug window to view drone camera.")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

sys.argv = [sys.argv[0]]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Everything after AppLauncher."""

import os
import cv2
import gymnasium as gym
import torch

import first_drone.tasks  # noqa: F401

from first_drone.models.sac import SACActorCritic


def main():
    device = "cuda:0"

    # Create environment
    env_cfg_class = gym.spec(args_cli.task).kwargs["env_cfg_entry_point"]
    module_path, class_name = env_cfg_class.rsplit(":", 1)
    import importlib
    module = importlib.import_module(module_path)
    env_cfg = getattr(module, class_name)()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = device

    env = gym.make(args_cli.task, cfg=env_cfg)
    unwrapped = env.unwrapped

    obs_dim = env_cfg.observation_space
    action_dim = env_cfg.action_space
    num_envs = unwrapped.num_envs

    # Load checkpoint
    print(f"[INFO] Loading checkpoint: {args_cli.checkpoint}")
    ckpt = torch.load(args_cli.checkpoint, map_location=device)

    # Restore VAE
    unwrapped.vae.load_state_dict(ckpt["vae"])
    unwrapped.vae.eval()

    # Restore SAC
    sac = SACActorCritic(obs_dim=obs_dim, action_dim=action_dim).to(device)
    sac.load_state_dict(ckpt["sac"])
    sac.eval()

    print(f"[INFO] Loaded checkpoint from step {ckpt.get('step', '?')}")
    print(f"[INFO] Running {args_cli.num_episodes} episodes with {num_envs} envs...")

    # Run evaluation
    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]

    episode_rewards = torch.zeros(num_envs, device=device)
    completed = 0
    total_reward = 0.0
    total_success = 0

    while completed < args_cli.num_episodes:
        with torch.no_grad():
            actions = sac.act(obs, deterministic=True)

        next_obs_dict, rewards, terminated, truncated, infos = env.step(actions)
        obs = next_obs_dict["policy"]
        episode_rewards += rewards
        
        # Update debug viewer if enabled
        if args_cli.viewer:
            if unwrapped._last_depth_processed is not None:
                # Get the latest depth from env_0
                # Shape is (B, 1, 72, 128), we want (72, 128)
                raw_depth = unwrapped._last_depth_processed[0, 0].clone()
                # Normalize to 0-255 uint8 format (invert so closer is brighter)
                normalized_img = ((1.0 - raw_depth) * 255.0).byte().cpu().numpy()
                
                # Upscale strictly for the debug viewer (128x72 -> 640x360)
                display_img = cv2.resize(normalized_img, (640, 360), interpolation=cv2.INTER_NEAREST)
                
                cv2.imshow("Drone Env 0 Camera", display_img)
                cv2.waitKey(1)

        dones = terminated | truncated
        done_ids = dones.nonzero(as_tuple=False).squeeze(-1)

        for idx in done_ids:
            ep_reward = episode_rewards[idx].item()
            total_reward += ep_reward
            completed += 1

            # Check success
            dist = torch.linalg.norm(
                unwrapped._desired_pos_w[idx] - unwrapped._robot.data.root_pos_w[idx]
            ).item()
            was_success = dist < env_cfg.goal_radius and not terminated[idx]

            status = "SUCCESS" if was_success else ("CRASH" if terminated[idx] else "TIMEOUT")
            if was_success:
                total_success += 1

            print(f"  Episode {completed:>3d}: reward={ep_reward:>7.2f}, dist={dist:.2f}m, {status}")

            if completed >= args_cli.num_episodes:
                break

        episode_rewards[done_ids] = 0.0

    avg = total_reward / max(completed, 1)
    success_rate = total_success / max(completed, 1) * 100
    print(f"\n[RESULTS] {completed} episodes | Avg reward: {avg:.2f} | Success rate: {success_rate:.1f}%")

    if args_cli.viewer:
        cv2.destroyAllWindows()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
