"""SAC+VAE Training Runner.

Generic runner class for SAC training, analogous to RSL-RL's OnPolicyRunner.
The runner encapsulates the entire training loop and is instantiated by the
generic train.py script.

Usage (from train.py):
    runner = SACRunner(env, agent_cfg, log_dir, device)
    runner.learn(max_iterations=agent_cfg.sac_max_iterations)
"""

from __future__ import annotations

import os
import time
from datetime import datetime

import torch

from first_drone.models.sac import SACActorCritic
from first_drone.models.replay_buffer import SplitReplayBuffer


class SACRunner:
    """Self-contained SAC+VAE training runner.

    Mirrors the interface of rsl_rl.runners.OnPolicyRunner so that
    the training script can treat all runners identically:
        runner = SACRunner(env, cfg, ...)
        runner.learn(num_learning_iterations=N)
    """

    def __init__(self, env, agent_cfg, log_dir: str, device: str = "cuda:0"):
        """Initialize runner, models, buffers, and optimizer.

        Args:
            env: Gymnasium environment (unwrapped must expose .vae, ._last_depth_processed, etc.)
            agent_cfg: SACAgentCfg instance with all hyper-parameters.
            log_dir: Directory for checkpoints.
            device: Torch device string.
        """
        self.env = env
        self.cfg = agent_cfg
        self.log_dir = log_dir
        self.device = device
        self.unwrapped = env.unwrapped

        # Read dimensions from the environment
        self.obs_dim = self.unwrapped.cfg.observation_space
        self.action_dim = self.unwrapped.cfg.action_space
        self.num_envs = self.unwrapped.num_envs

        # VAE (owned by the environment)
        self.vae = self.unwrapped.vae
        self.vae_optimizer = torch.optim.Adam(self.vae.parameters(), lr=self.cfg.vae_lr)

        # SAC actor-critic
        self.sac = SACActorCritic(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            actor_lr=self.cfg.sac_actor_lr,
            critic_lr=self.cfg.sac_critic_lr,
            alpha_lr=self.cfg.sac_alpha_lr,
            gamma=self.cfg.sac_gamma,
            tau=self.cfg.sac_tau,
        ).to(self.device)

        # Replay buffer
        self.replay = SplitReplayBuffer(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            max_size=self.cfg.sac_replay_size,
            success_ratio=self.cfg.sac_success_ratio,
            device=self.device,
        )

        # VAE warmup length
        self.vae_training_steps = getattr(self.cfg, "vae_training_steps", self.cfg.sac_warmup_steps)

        # Wandb handle (initialized lazily in learn())
        self._wandb_run = None

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------
    def save(self, path: str, step: int):
        """Save a training checkpoint."""
        torch.save({
            "step": step,
            "sac": self.sac.state_dict(),
            "vae": self.vae.state_dict(),
            "vae_optimizer": self.vae_optimizer.state_dict(),
        }, path)

    def load(self, path: str) -> int:
        """Load a training checkpoint. Returns the step number."""
        ckpt = torch.load(path, map_location=self.device)
        self.sac.load_state_dict(ckpt["sac"])
        
        # Only load VAE from SAC checkpoint if we are actively training it.
        # If it's frozen, we keep the properly loaded pre-trained weights from the environment setup.
        if getattr(self.cfg, "train_vae", True):
            if "vae" in ckpt:
                self.vae.load_state_dict(ckpt["vae"])
            if "vae_optimizer" in ckpt:
                self.vae_optimizer.load_state_dict(ckpt["vae_optimizer"])
        else:
            print("[INFO] train_vae is False (frozen). Skipping VAE load from SAC checkpoint.")
            
        return ckpt.get("step", 0)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    def _init_wandb(self):
        """Try to initialize wandb. Returns the run or None."""
        try:
            import wandb
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            run = wandb.init(
                project=self.cfg.wandb_project,
                name=f"{self.cfg.experiment_name}_{timestamp}",
                config={
                    "obs_dim": self.obs_dim,
                    "action_dim": self.action_dim,
                    "num_envs": self.num_envs,
                    **{k: v for k, v in self.cfg.__dict__.items() if isinstance(v, (int, float, str, bool))},
                },
            )
            print(f"[INFO] Wandb run: {run.url}")
            return run
        except Exception as e:
            print(f"[WARN] Wandb init failed: {e}. Continuing without wandb.")
            return None

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------
    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False,
              resume_path: str | None = None, no_wandb: bool = False):
        """Run the full SAC+VAE training loop.

        Args:
            num_learning_iterations: Total environment steps.
            init_at_random_ep_len: Unused, kept for API compatibility with OnPolicyRunner.
            resume_path: Optional path to a checkpoint to resume from.
            no_wandb: If True, skip wandb initialization.
        """
        max_steps = num_learning_iterations

        # Resume
        start_step = 0
        if resume_path is not None:
            print(f"[INFO] Resuming from: {resume_path}")
            start_step = self.load(resume_path)
            print(f"[INFO] Resumed at step {start_step}")

        # Wandb
        if not no_wandb:
            self._wandb_run = self._init_wandb()

        # Reset environment
        obs_dict, _ = self.env.reset()
        obs = obs_dict["policy"]

        # Depth buffer for VAE training (stored on CPU to avoid GPU OOM)
        depth_buffer = []
        max_depth_buffer = 50  # ~115 MB on CPU for 1024 envs (was 100)

        # Episode tracking
        episode_rewards = torch.zeros(self.num_envs, device=self.device)
        episode_lengths = torch.zeros(self.num_envs, device=self.device)
        completed_episodes = 0
        total_reward_sum = 0.0

        start_time = time.time()
        print(f"[INFO] Starting SAC training for {max_steps} steps "
              f"(VAE warmup: {self.vae_training_steps} steps)")

        for step in range(start_step, max_steps):
            # ---- Act ----
            if step < self.vae_training_steps:
                actions = torch.rand(self.num_envs, self.action_dim, device=self.device) * 2 - 1
            else:
                actions = self.sac.act(obs)

            # ---- Step environment ----
            next_obs_dict, rewards, terminated, truncated, infos = self.env.step(actions)
            next_obs = next_obs_dict["policy"]
            dones = terminated | truncated

            # Track episode stats
            episode_rewards += rewards
            episode_lengths += 1

            # Determine success
            # Use 2D (X/Y) distance for success determination to match env logic
            dist_to_goal = torch.linalg.norm(
                (self.unwrapped._desired_pos_w - self.unwrapped._robot.data.root_pos_w)[:, :2], dim=1
            )
            success = dist_to_goal < self.unwrapped.cfg.goal_radius

            # Store transition
            self.replay.add(obs, actions, rewards, next_obs, dones.float(), success)

            # Collect depth for VAE training (store on CPU to save GPU memory)
            # We ONLY collect this if we are actively training the VAE, otherwise this GPU->CPU transfer blocks and slows down training heavily!
            if getattr(self.cfg, "train_vae", True) and self.unwrapped._last_depth_processed is not None:
                depth_buffer.append(self.unwrapped._last_depth_processed.detach().cpu())
                if len(depth_buffer) > max_depth_buffer:
                    depth_buffer.pop(0)

            # Log completed episodes
            done_mask = dones.nonzero(as_tuple=False).squeeze(-1)
            if len(done_mask) > 0:
                for idx in done_mask:
                    total_reward_sum += episode_rewards[idx].item()
                    completed_episodes += 1
                episode_rewards[done_mask] = 0.0
                episode_lengths[done_mask] = 0.0

            # ---- Update networks ----
            sac_logs = {}
            vae_logs = {}

            if step < self.vae_training_steps:
                # VAE-only warmup: update based on vae_update_every (default 1)
                vae_update_every = getattr(self.cfg, "vae_update_every", 1)
                if step % vae_update_every == 0:
                    vae_logs = self._update_vae(depth_buffer)
            else:
                # SAC training: update SAC and VAE based on sac_update_every
                if step % self.cfg.sac_update_every == 0:
                    sac_logs = self._update_sac(step, max_steps)
                    vae_logs = self._update_vae(depth_buffer)

            # ---- Console & Wandb logging ----
            if step % 500 == 0:
                self._log_step(step, max_steps, start_time, completed_episodes,
                               total_reward_sum, sac_logs, vae_logs)

            # ---- Save checkpoint ----
            if step > 0 and step % self.cfg.save_interval == 0:
                ckpt_path = os.path.join(self.log_dir, f"checkpoint_{step}.pt")
                self.save(ckpt_path, step)
                print(f"[INFO] Saved checkpoint: {ckpt_path}")

            # Advance observation
            obs = next_obs

        # ---- Final save ----
        final_path = os.path.join(self.log_dir, "checkpoint_final.pt")
        self.save(final_path, max_steps)
        print(f"[INFO] Final checkpoint: {final_path}")

        elapsed = time.time() - start_time
        print(f"Training complete. Total time: {elapsed:.1f}s, Episodes: {completed_episodes}")

        if self._wandb_run:
            self._wandb_run.finish()

    # ------------------------------------------------------------------
    # Private update helpers
    # ------------------------------------------------------------------
    def _update_vae(self, depth_buffer: list) -> dict:
        """Run one VAE training step on collected depth frames."""
        if not getattr(self.cfg, "train_vae", True):
            return {}
        if len(depth_buffer) < 4:
            return {}

        n_vae = min(self.cfg.sac_batch_size, len(depth_buffer) * self.num_envs)
        depth_all = torch.cat(depth_buffer[-8:], dim=0)  # on CPU
        idx = torch.randint(0, depth_all.shape[0], (n_vae,))
        depth_batch = depth_all[idx].to(self.device)  # move selected batch to GPU

        recon, mu, logvar = self.vae(depth_batch)
        vae_loss, recon_loss, kl_loss = self.vae.loss(recon, depth_batch, mu, logvar)

        self.vae_optimizer.zero_grad()
        vae_loss.backward()
        self.vae_optimizer.step()

        return {
            "vae_loss": vae_loss.item(),
            "vae_recon_loss": recon_loss.item(),
            "vae_kl_loss": kl_loss.item(),
        }

    def _update_sac(self, step: int, max_steps: int) -> dict:
        """Run SAC gradient updates on sampled replay batches."""
        if not self.replay.can_sample(self.cfg.sac_batch_size):
            return {}

        start_steps = getattr(self.cfg, "sac_gradient_steps_start", getattr(self.cfg, "sac_gradient_steps", 20))
        end_steps = getattr(self.cfg, "sac_gradient_steps_end", getattr(self.cfg, "sac_gradient_steps", 1))
        
        # Calculate dynamic gradient steps
        progress = min(1.0, max(0.0, step / max(1, max_steps)))
        current_gradient_steps = int(start_steps - (start_steps - end_steps) * progress)
        current_gradient_steps = max(1, current_gradient_steps)

        sac_logs: dict[str, float] = {}
        for _ in range(current_gradient_steps):
            batch = self.replay.sample(self.cfg.sac_batch_size)

            critic_info = self.sac.update_critic(
                batch["obs"], batch["action"], batch["reward"],
                batch["next_obs"], batch["done"],
            )
            actor_info = self.sac.update_actor_and_alpha(batch["obs"])
            self.sac.soft_update_target()

            for k, v in {**critic_info, **actor_info}.items():
                sac_logs[k] = sac_logs.get(k, 0.0) + v

        for k in sac_logs:
            sac_logs[k] /= current_gradient_steps

        return sac_logs

    def _log_step(self, step: int, max_steps: int, start_time: float,
                  completed_episodes: int, total_reward_sum: float,
                  sac_logs: dict, vae_logs: dict):
        """Print console log and optionally push to wandb."""
        elapsed = time.time() - start_time
        fps = step / max(elapsed, 1)
        avg_reward = total_reward_sum / max(completed_episodes, 1)

        if step < self.vae_training_steps:
            phase = "VAE+SAC Warmup" if getattr(self.cfg, "train_vae", True) else "SAC Random Warmup"
        else:
            phase = "SAC Train"
            
        print(
            f"Step {step:>7d}/{max_steps} | "
            f"Phase: {phase} | "
            f"Ep: {completed_episodes:>5d} | "
            f"Avg R: {avg_reward:>7.2f} | "
            f"Alpha: {self.sac.alpha.item():.3f} | "
            f"FPS: {fps:.0f}"
        )

        if self._wandb_run:
            import wandb
            log_data = {
                "step": step,
                "episode/avg_reward": avg_reward,
                "episode/completed": completed_episodes,
                "fps": fps,
                **{f"sac/{k}": v for k, v in sac_logs.items()},
                **{f"vae/{k}": v for k, v in vae_logs.items()},
            }
            if "log" in self.unwrapped.extras:
                for k, v in self.unwrapped.extras["log"].items():
                    log_data[k] = v if isinstance(v, (int, float)) else v
            wandb.log(log_data, step=step)
