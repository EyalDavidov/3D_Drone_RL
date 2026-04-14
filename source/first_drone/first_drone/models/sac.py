"""Soft Actor-Critic (SAC) with auto-tuned temperature.

Implements the SAC algorithm components:
  - GaussianActor: tanh-squashed Gaussian policy (4 actions)
  - TwinCritic: two independent Q-networks for stable Q estimation
  - Alpha (temperature): auto-tuned entropy coefficient

Based on "Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL" (Haarnoja et al.)
and adapted for drone navigation with 45-dim observation (VAE latent + state).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


class GaussianActor(nn.Module):
    """Tanh-squashed Gaussian policy network.

    4 hidden layers of 256 units (as specified in the paper).
    Outputs mean and log_std for each action dimension.

    Actions are squashed through tanh to lie in [-1, 1], then scaled
    to physical drone commands by the environment.
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: list[int] | None = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256, 256, 256]

        # Build MLP trunk
        layers = []
        in_dim = obs_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
        self.trunk = nn.Sequential(*layers)

        # Output heads
        self.fc_mean = nn.Linear(in_dim, action_dim)
        self.fc_log_std = nn.Linear(in_dim, action_dim)

    def forward(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute action and log-probability.

        Args:
            obs: Observation tensor, shape (B, obs_dim).
            deterministic: If True, use mean action (no sampling). For evaluation.

        Returns:
            action: Tanh-squashed action in [-1, 1], shape (B, action_dim).
            log_prob: Log-probability of the action, shape (B, 1).
        """
        h = self.trunk(obs)
        mean = self.fc_mean(h)
        log_std = self.fc_log_std(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()

        # Sample from Gaussian
        dist = Normal(mean, std)
        if deterministic:
            z = mean
        else:
            z = dist.rsample()  # reparameterized sample

        # Squash through tanh
        action = torch.tanh(z)

        # Compute log_prob with tanh correction
        # log π(a|s) = log π(u|s) - Σ log(1 - tanh²(u))
        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob


class QNetwork(nn.Module):
    """Single Q-network: Q(obs, action) → scalar value.

    3 hidden layers of 256 units as specified in the paper.
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: list[int] | None = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256, 256]

        layers = []
        in_dim = obs_dim + action_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Compute Q-value for (obs, action) pair.

        Args:
            obs: Observation, shape (B, obs_dim).
            action: Action, shape (B, action_dim).

        Returns:
            Q-value, shape (B, 1).
        """
        return self.net(torch.cat([obs, action], dim=-1))


class TwinCritic(nn.Module):
    """Twin Q-networks for SAC (reduces overestimation bias).

    Uses min(Q1, Q2) for the target to provide a conservative Q estimate.
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: list[int] | None = None):
        super().__init__()
        self.q1 = QNetwork(obs_dim, action_dim, hidden_dims)
        self.q2 = QNetwork(obs_dim, action_dim, hidden_dims)

    def forward(
        self, obs: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute both Q-values.

        Returns:
            q1_value: Q1(obs, action), shape (B, 1).
            q2_value: Q2(obs, action), shape (B, 1).
        """
        return self.q1(obs, action), self.q2(obs, action)


class SACActorCritic(nn.Module):
    """Complete SAC agent with auto-tuned temperature α.

    Contains:
      - GaussianActor (policy)
      - TwinCritic (Q-functions)
      - TwinCritic target (soft-updated copy)
      - Learnable log_alpha (entropy temperature)

    The target entropy is set to -action_dim (standard heuristic).
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        actor_hidden: list[int] | None = None,
        critic_hidden: list[int] | None = None,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        init_alpha: float = 0.2,
    ):
        """Initialize SAC components and optimizers.

        Args:
            obs_dim: Dimension of the observation vector.
            action_dim: Dimension of the action space.
            actor_hidden: Hidden layer sizes for actor. Default: [256]*4.
            critic_hidden: Hidden layer sizes for each critic. Default: [256]*3.
            actor_lr: Learning rate for the actor.
            critic_lr: Learning rate for the critics.
            alpha_lr: Learning rate for the temperature parameter.
            gamma: Discount factor.
            tau: Soft update coefficient for target networks.
            init_alpha: Initial entropy temperature.
        """
        super().__init__()
        self.gamma = gamma
        self.tau = tau
        self.action_dim = action_dim

        # Networks
        self.actor = GaussianActor(obs_dim, action_dim, actor_hidden)
        self.critic = TwinCritic(obs_dim, action_dim, critic_hidden)
        self.critic_target = TwinCritic(obs_dim, action_dim, critic_hidden)

        # Copy critic params to target (hard copy)
        self.critic_target.load_state_dict(self.critic.state_dict())
        # Freeze target (only updated via polyak averaging)
        for p in self.critic_target.parameters():
            p.requires_grad = False

        # Auto-tuned temperature
        self.log_alpha = nn.Parameter(torch.tensor(init_alpha, dtype=torch.float32).log())
        self.target_entropy = -float(action_dim)

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

    @property
    def alpha(self) -> torch.Tensor:
        """Current entropy temperature."""
        return self.log_alpha.exp()

    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Select action for environment interaction (no gradients).

        Args:
            obs: Observation, shape (B, obs_dim).
            deterministic: If True, use mean action.

        Returns:
            Action in [-1, 1], shape (B, action_dim).
        """
        action, _ = self.actor(obs, deterministic=deterministic)
        return action

    def update_critic(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
    ) -> dict[str, float]:
        """Update twin critics using Bellman backup.

        Target: y = r + γ(1-d) * (min(Q1_tgt, Q2_tgt) - α * log_prob)

        Returns dict of logging metrics.
        """
        with torch.no_grad():
            next_action, next_log_prob = self.actor(next_obs)
            q1_tgt, q2_tgt = self.critic_target(next_obs, next_action)
            q_tgt = torch.min(q1_tgt, q2_tgt) - self.alpha * next_log_prob
            target = reward + self.gamma * (1.0 - done) * q_tgt

        q1, q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return {
            "critic_loss": critic_loss.item(),
            "q1_mean": q1.mean().item(),
            "q2_mean": q2.mean().item(),
        }

    def update_actor_and_alpha(
        self, obs: torch.Tensor
    ) -> dict[str, float]:
        """Update actor to maximize Q - α*log_prob, and tune α.

        Returns dict of logging metrics.
        """
        action, log_prob = self.actor(obs)
        q1, q2 = self.critic(obs, action)
        q_min = torch.min(q1, q2)

        # Actor loss: minimize -Q + α*log_prob (= maximize Q - α*log_prob)
        actor_loss = (self.alpha.detach() * log_prob - q_min).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Alpha loss: tune temperature to match target entropy
        alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        return {
            "actor_loss": actor_loss.item(),
            "alpha": self.alpha.item(),
            "alpha_loss": alpha_loss.item(),
            "entropy": -log_prob.mean().item(),
        }

    @torch.no_grad()
    def soft_update_target(self):
        """Polyak-average the critic target: θ_tgt ← τ*θ + (1-τ)*θ_tgt."""
        for p, p_tgt in zip(self.critic.parameters(), self.critic_target.parameters()):
            p_tgt.data.mul_(1.0 - self.tau).add_(self.tau * p.data)
