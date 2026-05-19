"""Split Replay Buffer for SAC training.

Maintains two separate buffers:
  - Regular buffer: stores all transitions
  - Success buffer: stores FULL trajectories from successful episodes

Samples are drawn from both in a configurable ratio (e.g. 75% regular, 25% success)
to ensure the agent keeps learning from rare successful experiences.

The success buffer receives entire episode trajectories (not just the terminal
transition) via `add_success_trajectory()`, called by the runner when an episode
ends successfully.
"""

from __future__ import annotations

import torch


class SplitReplayBuffer:
    """GPU-resident replay buffer with regular/success split sampling."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        max_size: int = 200_000,
        success_ratio: float = 0.25,
        device: str = "cuda:0",
    ):
        """Initialize the replay buffer.

        Args:
            obs_dim: Dimension of the observation vector (after VAE encoding).
            action_dim: Dimension of the action space.
            max_size: Maximum number of transitions per buffer.
            success_ratio: Fraction of each batch drawn from the success buffer.
            device: Torch device.
        """
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.max_size = max_size
        self.success_ratio = success_ratio
        self.device = device

        assert 0.0 <= success_ratio <= 1.0, (
            f"sac_success_ratio must be in [0.0, 1.0], got {success_ratio}"
        )

        # Regular buffer — stores ALL transitions
        self._reg = _Buffer(obs_dim, action_dim, max_size, device)
        # Success buffer — stores full trajectories from successful episodes
        self._suc = _Buffer(obs_dim, action_dim, max_size, device)

        # Debug counters
        self._total_added = 0
        self._total_success_transitions = 0

    @property
    def total_size(self) -> int:
        return self._reg.size + self._suc.size

    @property
    def reg_size(self) -> int:
        return self._reg.size

    @property
    def suc_size(self) -> int:
        return self._suc.size

    def add(self, obs: torch.Tensor, action: torch.Tensor, reward: torch.Tensor,
            next_obs: torch.Tensor, done: torch.Tensor):
        """Add a batch of transitions to the regular buffer.

        All tensors: shape (B, ...) where B = num_envs.

        Args:
            obs: Current observation, (B, obs_dim).
            action: Action taken, (B, action_dim).
            reward: Reward received, (B, 1) or (B,).
            next_obs: Next observation, (B, obs_dim).
            done: Episode termination flag, (B, 1) or (B,).
        """
        reward = reward.view(-1, 1) if reward.dim() == 1 else reward
        done = done.view(-1, 1).float() if done.dim() == 1 else done.float()

        self._reg.add(obs, action, reward, next_obs, done)
        self._total_added += obs.shape[0]

    def add_success_trajectory(self, obs: torch.Tensor, action: torch.Tensor,
                               reward: torch.Tensor, next_obs: torch.Tensor,
                               done: torch.Tensor):
        """Add a full successful episode trajectory to the success buffer.

        Called by the runner when an episode ends with success. All transitions
        from that episode are flushed here so the agent can re-learn the entire
        approach path, not just the final lucky step.

        Args:
            obs: (T, obs_dim) — full episode observations.
            action: (T, action_dim) — full episode actions.
            reward: (T, 1) or (T,) — full episode rewards.
            next_obs: (T, obs_dim) — full episode next-observations.
            done: (T, 1) or (T,) — full episode done flags.
        """
        if obs.shape[0] == 0:
            return
        reward = reward.view(-1, 1) if reward.dim() == 1 else reward
        done = done.view(-1, 1).float() if done.dim() == 1 else done.float()

        self._suc.add(obs, action, reward, next_obs, done)
        self._total_success_transitions += obs.shape[0]

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        """Sample a mixed batch from both buffers, shuffled.

        Returns:
            Dictionary with keys: obs, action, reward, next_obs, done.
            Each tensor has shape (batch_size, ...).
        """
        if self._suc.size == 0 or self.success_ratio == 0:
            # No successes yet, sample entirely from regular
            return self._reg.sample(batch_size)

        n_suc = int(batch_size * self.success_ratio)
        n_reg = batch_size - n_suc

        reg_batch = self._reg.sample(n_reg)
        suc_batch = self._suc.sample(n_suc)

        # Concatenate
        combined = {
            key: torch.cat([reg_batch[key], suc_batch[key]], dim=0)
            for key in reg_batch.keys()
        }

        # Shuffle the combined batch to prevent ordering bias
        perm = torch.randperm(batch_size, device=self.device)
        return {key: val[perm] for key, val in combined.items()}

    def can_sample(self, batch_size: int) -> bool:
        """Check if there are enough transitions to sample a full batch."""
        return self._reg.size >= batch_size

    def get_debug_info(self) -> dict[str, int | float]:
        """Return buffer statistics for logging/debugging."""
        return {
            "replay/reg_size": self._reg.size,
            "replay/suc_size": self._suc.size,
            "replay/total_added": self._total_added,
            "replay/total_success_transitions": self._total_success_transitions,
            "replay/suc_fill_pct": self._suc.size / self.max_size * 100,
        }


class _Buffer:
    """Simple ring buffer for transitions, stored on GPU."""

    def __init__(self, obs_dim: int, action_dim: int, max_size: int, device: str):
        self.max_size = max_size
        self.size = 0
        self.ptr = 0  # next write position
        self.device = device

        self.obs = torch.zeros(max_size, obs_dim, device=device)
        self.action = torch.zeros(max_size, action_dim, device=device)
        self.reward = torch.zeros(max_size, 1, device=device)
        self.next_obs = torch.zeros(max_size, obs_dim, device=device)
        self.done = torch.zeros(max_size, 1, device=device)

    def add(self, obs: torch.Tensor, action: torch.Tensor, reward: torch.Tensor,
            next_obs: torch.Tensor, done: torch.Tensor):
        """Add a batch of transitions to the ring buffer."""
        b = obs.shape[0]
        if b == 0:
            return

        # Ensure tensors are on the correct device
        obs = obs.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        next_obs = next_obs.to(self.device)
        done = done.to(self.device)

        # Handle wrap-around
        if self.ptr + b <= self.max_size:
            self.obs[self.ptr: self.ptr + b] = obs
            self.action[self.ptr: self.ptr + b] = action
            self.reward[self.ptr: self.ptr + b] = reward
            self.next_obs[self.ptr: self.ptr + b] = next_obs
            self.done[self.ptr: self.ptr + b] = done
        else:
            # Split across the boundary
            first = self.max_size - self.ptr
            self.obs[self.ptr:] = obs[:first]
            self.action[self.ptr:] = action[:first]
            self.reward[self.ptr:] = reward[:first]
            self.next_obs[self.ptr:] = next_obs[:first]
            self.done[self.ptr:] = done[:first]

            rem = b - first
            self.obs[:rem] = obs[first:]
            self.action[:rem] = action[first:]
            self.reward[:rem] = reward[first:]
            self.next_obs[:rem] = next_obs[first:]
            self.done[:rem] = done[first:]

        self.ptr = (self.ptr + b) % self.max_size
        self.size = min(self.size + b, self.max_size)

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        """Sample uniformly from stored transitions."""
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)
        return {
            "obs": self.obs[idx],
            "action": self.action[idx],
            "reward": self.reward[idx],
            "next_obs": self.next_obs[idx],
            "done": self.done[idx],
        }
