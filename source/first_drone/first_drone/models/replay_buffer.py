"""Split Replay Buffer for SAC training.

Maintains two separate buffers:
  - Regular buffer: stores all transitions
  - Success buffer: stores transitions from successful episodes

Samples are drawn from both in a configurable ratio (e.g. 75% regular, 25% success)
to ensure the agent keeps learning from rare successful experiences.
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

        # Regular buffer
        self._reg = _Buffer(obs_dim, action_dim, max_size, device)
        # Success buffer
        self._suc = _Buffer(obs_dim, action_dim, max_size, device)

    @property
    def total_size(self) -> int:
        return self._reg.size + self._suc.size

    def add(self, obs: torch.Tensor, action: torch.Tensor, reward: torch.Tensor,
            next_obs: torch.Tensor, done: torch.Tensor, success: torch.Tensor | None = None):
        """Add a batch of transitions.

        All tensors: shape (B, ...) where B = num_envs.

        Args:
            obs: Current observation, (B, obs_dim).
            action: Action taken, (B, action_dim).
            reward: Reward received, (B, 1) or (B,).
            next_obs: Next observation, (B, obs_dim).
            done: Episode termination flag, (B, 1) or (B,).
            success: Optional per-env success flag, (B,). If provided, transitions
                     where success=True are also added to the success buffer.
        """
        reward = reward.view(-1, 1) if reward.dim() == 1 else reward
        done = done.view(-1, 1).float() if done.dim() == 1 else done.float()

        # Add all to regular buffer
        self._reg.add(obs, action, reward, next_obs, done)

        # Add successes to success buffer
        if success is not None and success.any():
            mask = success.bool()
            self._suc.add(
                obs[mask], action[mask], reward[mask], next_obs[mask], done[mask]
            )

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        """Sample a mixed batch from both buffers.

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

        return {
            key: torch.cat([reg_batch[key], suc_batch[key]], dim=0)
            for key in reg_batch.keys()
        }

    def can_sample(self, batch_size: int) -> bool:
        """Check if there are enough transitions to sample a full batch."""
        return self._reg.size >= batch_size


class _Buffer:
    """Simple ring buffer for transitions, stored on GPU."""

    def __init__(self, obs_dim: int, action_dim: int, max_size: int, device: str):
        self.max_size = max_size
        self.size = 0
        self.ptr = 0  # next write position

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
        idx = torch.randint(0, self.size, (batch_size,), device=self.obs.device)
        return {
            "obs": self.obs[idx],
            "action": self.action[idx],
            "reward": self.reward[idx],
            "next_obs": self.next_obs[idx],
            "done": self.done[idx],
        }
