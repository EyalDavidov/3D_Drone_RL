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

        # Per-env episode trajectory accumulators (for full-trajectory success buffer)
        self._ep_obs = [[] for _ in range(self.num_envs)]
        self._ep_act = [[] for _ in range(self.num_envs)]
        self._ep_rew = [[] for _ in range(self.num_envs)]
        self._ep_nobs = [[] for _ in range(self.num_envs)]
        self._ep_done = [[] for _ in range(self.num_envs)]

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

        # Episode tracking (for interval statistics)
        episode_rewards = torch.zeros(self.num_envs, device=self.device)
        episode_lengths = torch.zeros(self.num_envs, device=self.device)
        interval_completed_episodes = 0
        interval_reward_sum = 0.0
        interval_length_sum = 0.0

        # Timing
        start_time = time.time()
        interval_start_time = time.time()
        collection_time = 0.0
        learning_time = 0.0

        print(f"[INFO] Starting SAC training for {max_steps} steps "
              f"(VAE warmup: {self.vae_training_steps} steps)")

        for step in range(start_step, max_steps):
            t_col_start = time.time()

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

            # Determine success (per-env, at this instant)
            dist_to_goal = torch.linalg.norm(
                self.unwrapped._desired_pos_w - self.unwrapped._robot.data.root_pos_w, dim=1
            )
            success = dist_to_goal < self.unwrapped.cfg.goal_radius

            # Store transition in regular buffer
            self.replay.add(obs, actions, rewards, next_obs, dones.float())

            # Collect depth for VAE training (store on CPU to save GPU memory)
            if getattr(self.cfg, "train_vae", True) and self.unwrapped._last_depth_processed is not None:
                depth_buffer.append(self.unwrapped._last_depth_processed.detach().cpu())
                if len(depth_buffer) > max_depth_buffer:
                    depth_buffer.pop(0)

            # Accumulate per-env trajectory and flush on episode end
            for i in range(self.num_envs):
                self._ep_obs[i].append(obs[i].unsqueeze(0))
                self._ep_act[i].append(actions[i].unsqueeze(0))
                self._ep_rew[i].append(rewards[i].unsqueeze(0))
                self._ep_nobs[i].append(next_obs[i].unsqueeze(0))
                self._ep_done[i].append(dones[i].float().unsqueeze(0))

            # Flush completed episodes
            done_mask = dones.nonzero(as_tuple=False).squeeze(-1)
            if len(done_mask) > 0:
                for idx in done_mask:
                    i = idx.item()
                    # If this episode was successful, flush full trajectory to success buffer
                    if success[i]:
                        traj_obs = torch.cat(self._ep_obs[i], dim=0)
                        traj_act = torch.cat(self._ep_act[i], dim=0)
                        traj_rew = torch.cat(self._ep_rew[i], dim=0)
                        traj_nobs = torch.cat(self._ep_nobs[i], dim=0)
                        traj_done = torch.cat(self._ep_done[i], dim=0)
                        self.replay.add_success_trajectory(
                            traj_obs, traj_act, traj_rew, traj_nobs, traj_done
                        )
                    # Clear episode accumulator
                    self._ep_obs[i].clear()
                    self._ep_act[i].clear()
                    self._ep_rew[i].clear()
                    self._ep_nobs[i].clear()
                    self._ep_done[i].clear()

                    interval_reward_sum += episode_rewards[i].item()
                    interval_length_sum += episode_lengths[i].item()
                    interval_completed_episodes += 1
                episode_rewards[done_mask] = 0.0
                episode_lengths[done_mask] = 0.0

            t_col_end = time.time()
            collection_time += (t_col_end - t_col_start)

            # ---- Update networks ----
            t_learn_start = time.time()
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
                    sac_logs = self._update_sac()
                    vae_logs = self._update_vae(depth_buffer)

            t_learn_end = time.time()
            learning_time += (t_learn_end - t_learn_start)

            # ---- Console & Wandb logging ----
            log_interval = getattr(self.cfg, "log_interval", 100)
            if step > start_step and step % log_interval == 0:
                self._log_step(step, max_steps, start_time, interval_start_time,
                               collection_time, learning_time, interval_completed_episodes,
                               interval_reward_sum, interval_length_sum, sac_logs, vae_logs, log_interval)
                
                # Reset interval stats
                interval_reward_sum = 0.0
                interval_length_sum = 0.0
                interval_completed_episodes = 0
                collection_time = 0.0
                learning_time = 0.0
                interval_start_time = time.time()

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

    def _update_sac(self) -> dict:
        """Run SAC gradient updates on sampled replay batches."""
        if not self.replay.can_sample(self.cfg.sac_batch_size):
            return {}

        sac_logs: dict[str, float] = {}
        for _ in range(self.cfg.sac_gradient_steps):
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
            sac_logs[k] /= self.cfg.sac_gradient_steps

        return sac_logs

    def _log_step(self, step: int, max_steps: int, start_time: float,
                  interval_start_time: float, collection_time: float, learning_time: float,
                  completed_episodes: int, total_reward_sum: float, total_length_sum: float,
                  sac_logs: dict, vae_logs: dict, log_interval: int = 100):
        """Print console log and optionally push to wandb."""
        elapsed_interval = time.time() - interval_start_time
        # Multiply by num_envs because step is *environment step* (one step = num_envs transitions)
        fps = int((log_interval * self.num_envs) / max(elapsed_interval, 1e-6))
        
        avg_reward = total_reward_sum / max(completed_episodes, 1)
        avg_length = total_length_sum / max(completed_episodes, 1)

        if step < self.vae_training_steps:
            phase = "VAE+SAC Warmup" if getattr(self.cfg, "train_vae", True) else "SAC Random Warmup"
        else:
            phase = "SAC Train"
            
        elapsed_total = time.time() - start_time
        eta = (elapsed_total / max(step, 1)) * (max_steps - step)
        
        def format_time(t):
            return f"{int(t // 3600):02d}:{int((t % 3600) // 60):02d}:{int(t % 60):02d}"

        width = 45
        print("-" * width)
        print(f"{'Phase:':<25} {phase}")
        print(f"{'Iteration:':<25} {step} / {max_steps}")
        print(f"{'Total Timesteps:':<25} {step * self.num_envs}")
        print(f"{'ETA:':<25} {format_time(eta)}")
        print("-" * width)
        print(f"{'FPS:':<25} {fps}")
        print(f"{'Collection Time:':<25} {collection_time:.2f}s")
        print(f"{'Learning Time:':<25} {learning_time:.2f}s")
        if completed_episodes > 0:
            print(f"{'Mean Episode Reward:':<25} {avg_reward:.2f}")
            print(f"{'Mean Episode Length:':<25} {avg_length:.1f}")
        print("-" * width)
        
        if sac_logs:
            for k, v in sac_logs.items():
                print(f"{f'sac/{k}:':<25} {v:.4f}")
        if vae_logs:
            for k, v in vae_logs.items():
                print(f"{f'vae/{k}:':<25} {v:.4f}")
        print("-" * width)
        print()

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
            # Replay buffer debug info
            log_data.update(self.replay.get_debug_info())
            wandb.log(log_data, step=step)
