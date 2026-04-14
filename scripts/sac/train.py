"""SAC+VAE Training Script for the drone navigation task.

Usage:
    c:\\Isaac\\IsaacLab\\isaaclab.bat -p scripts/sac/train.py --task SAC-First-Drone-Direct-v0 --num_envs 64 --enable_cameras

This script bypasses RSL-RL entirely and runs a custom SAC training loop with:
  - Online VAE training on depth images
  - Split replay buffer (regular + success)
  - Wandb logging
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train SAC+VAE drone agent.")
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments.")
parser.add_argument("--task", type=str, default="SAC-First-Drone-Direct-v0", help="Task name.")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument("--max_steps", type=int, default=None, help="Override max env steps.")
parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging.")
parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from.")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

# Clear sys.argv so gymnasium doesn't choke
sys.argv = [sys.argv[0]]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Everything after AppLauncher."""

import os
import time
from datetime import datetime

import gymnasium as gym
import torch

import first_drone.tasks  # noqa: F401 — registers environments

from first_drone.models.vae import VAE
from first_drone.models.sac import SACActorCritic
from first_drone.models.replay_buffer import SplitReplayBuffer


def main():
    # ---- Config ----
    torch.manual_seed(args_cli.seed)
    device = "cuda:0"

    # Create environment
    env_cfg_class = gym.spec(args_cli.task).kwargs["env_cfg_entry_point"]
    # Resolve the config class from the entry point string
    module_path, class_name = env_cfg_class.rsplit(":", 1)
    import importlib
    module = importlib.import_module(module_path)
    env_cfg = getattr(module, class_name)()

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs

    env_cfg.sim.device = device
    env_cfg.seed = args_cli.seed

    env = gym.make(args_cli.task, cfg=env_cfg)
    unwrapped = env.unwrapped

    # Read dimensions
    obs_dim = env_cfg.observation_space   # 45
    action_dim = env_cfg.action_space     # 4
    num_envs = unwrapped.num_envs
    max_steps = args_cli.max_steps or env_cfg.sac_max_iterations

    print(f"[INFO] obs_dim={obs_dim}, action_dim={action_dim}, num_envs={num_envs}")

    # ---- Models ----
    # VAE is owned by the environment (env.unwrapped.vae)
    vae = unwrapped.vae
    vae_optimizer = torch.optim.Adam(vae.parameters(), lr=env_cfg.vae_lr)

    # SAC
    sac = SACActorCritic(
        obs_dim=obs_dim,
        action_dim=action_dim,
        actor_lr=env_cfg.sac_actor_lr,
        critic_lr=env_cfg.sac_critic_lr,
        alpha_lr=env_cfg.sac_alpha_lr,
        gamma=env_cfg.sac_gamma,
        tau=env_cfg.sac_tau,
    ).to(device)

    # Replay buffer
    replay = SplitReplayBuffer(
        obs_dim=obs_dim,
        action_dim=action_dim,
        max_size=env_cfg.sac_replay_size,
        success_ratio=env_cfg.sac_success_ratio,
        device=device,
    )

    # ---- Logging ----
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(project_root, "logs", "sac", env_cfg.experiment_name, timestamp)
    os.makedirs(log_dir, exist_ok=True)
    print(f"[INFO] Logging to: {log_dir}")

    wandb_run = None
    if not args_cli.no_wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=env_cfg.wandb_project,
                name=f"{env_cfg.experiment_name}_{timestamp}",
                config={
                    "obs_dim": obs_dim,
                    "action_dim": action_dim,
                    "num_envs": num_envs,
                    **{k: v for k, v in env_cfg.__dict__.items() if isinstance(v, (int, float, str, bool))},
                },
            )
            print(f"[INFO] Wandb run: {wandb_run.url}")
        except Exception as e:
            print(f"[WARN] Wandb init failed: {e}. Continuing without wandb.")

    # ---- Resume checkpoint ----
    start_step = 0
    if args_cli.resume:
        print(f"[INFO] Resuming from: {args_cli.resume}")
        ckpt = torch.load(args_cli.resume, map_location=device)
        sac.load_state_dict(ckpt["sac"])
        vae.load_state_dict(ckpt["vae"])
        vae_optimizer.load_state_dict(ckpt["vae_optimizer"])
        start_step = ckpt.get("step", 0)
        print(f"[INFO] Resumed at step {start_step}")

    # ---- Training loop ----
    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]  # (num_envs, 45)

    # Depth buffer for VAE training (collect depths from recent steps)
    depth_buffer = []
    max_depth_buffer = 2000  # keep last N depth frames for VAE batches

    episode_rewards = torch.zeros(num_envs, device=device)
    episode_lengths = torch.zeros(num_envs, device=device)
    completed_episodes = 0
    total_reward_sum = 0.0

    start_time = time.time()

    for step in range(start_step, max_steps):
        # ---- Act ----
        if step < env_cfg.sac_warmup_steps:
            # Random actions during warmup (explore & collect VAE data)
            actions = torch.rand(num_envs, action_dim, device=device) * 2 - 1
        else:
            actions = sac.act(obs)

        # ---- Step environment ----
        next_obs_dict, rewards, terminated, truncated, infos = env.step(actions)
        next_obs = next_obs_dict["policy"]
        dones = terminated | truncated

        # Track episode stats
        episode_rewards += rewards
        episode_lengths += 1

        # Determine success (reached goal)
        dist_to_goal = torch.linalg.norm(
            unwrapped._desired_pos_w - unwrapped._robot.data.root_pos_w, dim=1
        )
        success = dist_to_goal < env_cfg.goal_radius

        # ---- Store transitions ----
        replay.add(obs, actions, rewards, next_obs, dones.float(), success)

        # ---- Collect depth for VAE training ----
        if unwrapped._last_depth_processed is not None:
            depth_buffer.append(unwrapped._last_depth_processed.clone())
            if len(depth_buffer) > max_depth_buffer:
                depth_buffer.pop(0)

        # ---- Log completed episodes ----
        done_mask = dones.nonzero(as_tuple=False).squeeze(-1)
        if len(done_mask) > 0:
            for idx in done_mask:
                total_reward_sum += episode_rewards[idx].item()
                completed_episodes += 1
            episode_rewards[done_mask] = 0.0
            episode_lengths[done_mask] = 0.0

        # ---- Update networks ----
        if step >= env_cfg.sac_warmup_steps and step % env_cfg.sac_update_every == 0:
            if replay.can_sample(env_cfg.sac_batch_size):
                sac_logs = {}
                vae_logs = {}

                for _ in range(env_cfg.sac_gradient_steps):
                    batch = replay.sample(env_cfg.sac_batch_size)

                    # Update SAC critic
                    critic_info = sac.update_critic(
                        batch["obs"], batch["action"], batch["reward"],
                        batch["next_obs"], batch["done"],
                    )

                    # Update SAC actor + alpha
                    actor_info = sac.update_actor_and_alpha(batch["obs"])

                    # Soft update target
                    sac.soft_update_target()

                    # Accumulate logs
                    for k, v in {**critic_info, **actor_info}.items():
                        sac_logs[k] = sac_logs.get(k, 0.0) + v

                # Average over gradient steps
                for k in sac_logs:
                    sac_logs[k] /= env_cfg.sac_gradient_steps

                # ---- Train VAE ----
                if len(depth_buffer) >= 4:
                    # Sample a random batch of depth frames
                    n_vae = min(env_cfg.sac_batch_size, len(depth_buffer) * num_envs)
                    depth_all = torch.cat(depth_buffer[-8:], dim=0)  # recent frames
                    idx = torch.randint(0, depth_all.shape[0], (n_vae,))
                    depth_batch = depth_all[idx]

                    recon, mu, logvar = vae(depth_batch)
                    vae_loss, recon_loss, kl_loss = vae.loss(recon, depth_batch, mu, logvar)

                    vae_optimizer.zero_grad()
                    vae_loss.backward()
                    vae_optimizer.step()

                    vae_logs = {
                        "vae_loss": vae_loss.item(),
                        "vae_recon_loss": recon_loss.item(),
                        "vae_kl_loss": kl_loss.item(),
                    }

                # ---- Log ----
                if step % 500 == 0:
                    elapsed = time.time() - start_time
                    fps = step / max(elapsed, 1)
                    avg_reward = total_reward_sum / max(completed_episodes, 1)

                    print(
                        f"Step {step:>7d}/{max_steps} | "
                        f"Ep: {completed_episodes:>5d} | "
                        f"Avg R: {avg_reward:>7.2f} | "
                        f"Alpha: {sac.alpha.item():.3f} | "
                        f"FPS: {fps:.0f}"
                    )

                    if wandb_run:
                        log_data = {
                            "step": step,
                            "episode/avg_reward": avg_reward,
                            "episode/completed": completed_episodes,
                            "fps": fps,
                            **{f"sac/{k}": v for k, v in sac_logs.items()},
                            **{f"vae/{k}": v for k, v in vae_logs.items()},
                        }
                        # Include env extras if available
                        if "log" in unwrapped.extras:
                            for k, v in unwrapped.extras["log"].items():
                                log_data[k] = v if isinstance(v, (int, float)) else v
                        wandb.log(log_data, step=step)

        # ---- Save checkpoint ----
        if step > 0 and step % env_cfg.save_interval == 0:
            ckpt_path = os.path.join(log_dir, f"checkpoint_{step}.pt")
            torch.save({
                "step": step,
                "sac": sac.state_dict(),
                "vae": vae.state_dict(),
                "vae_optimizer": vae_optimizer.state_dict(),
            }, ckpt_path)
            print(f"[INFO] Saved checkpoint: {ckpt_path}")

        # Advance observation
        obs = next_obs

    # ---- Final save ----
    final_path = os.path.join(log_dir, "checkpoint_final.pt")
    torch.save({
        "step": max_steps,
        "sac": sac.state_dict(),
        "vae": vae.state_dict(),
        "vae_optimizer": vae_optimizer.state_dict(),
    }, final_path)
    print(f"[INFO] Final checkpoint: {final_path}")

    elapsed = time.time() - start_time
    print(f"Training complete. Total time: {elapsed:.1f}s, Episodes: {completed_episodes}")

    if wandb_run:
        wandb_run.finish()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
