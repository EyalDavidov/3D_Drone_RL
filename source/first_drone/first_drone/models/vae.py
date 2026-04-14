"""Variational Autoencoder for depth image compression.

Encodes 128×72 grayscale depth images into a compact 32-dimensional latent code.
Based on the architecture from "Vision Based Drone Obstacle Avoidance by Deep RL".

Usage:
    vae = VAE(latent_dim=32).to(device)

    # Training (with gradients):
    recon, mu, logvar = vae(depth_batch)       # depth_batch: (B, 1, 72, 128)
    loss = vae.loss(recon, depth_batch, mu, logvar)

    # Inference (for RL policy):
    z = vae.encode_detached(depth_batch)       # (B, 32), no gradients
"""

from __future__ import annotations

import torch
import torch.nn as nn


class VAE(nn.Module):
    """Convolutional VAE for depth image compression.

    Architecture follows the paper: 4 conv layers in encoder, 4 transposed conv
    layers in decoder. Input is (B, 1, 72, 128), latent is 32-dimensional.
    """

    def __init__(self, latent_dim: int = 32, beta: float = 1e-3):
        """Initialize the VAE.

        Args:
            latent_dim: Dimension of the latent space. Default: 32.
            beta: Weight of KL divergence loss relative to reconstruction loss.
                  Start small (1e-4 to 1e-3) to avoid posterior collapse.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.beta = beta

        # ----- Encoder -----
        # Input: (B, 1, 72, 128) → after 4 conv layers → (B, 256, 4, 8) = 8192 features
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),    # → (B, 32, 36, 64)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),   # → (B, 64, 18, 32)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # → (B, 128, 9, 16)
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), # → (B, 256, 4, 8)
            nn.ReLU(),
            nn.Flatten(),                                             # → (B, 8192)
        )

        self._encoder_out_dim = 256 * 4 * 8  # 8192

        self.fc_mu = nn.Linear(self._encoder_out_dim, latent_dim)
        self.fc_logvar = nn.Linear(self._encoder_out_dim, latent_dim)

        # ----- Decoder -----
        # Latent (B, 32) → FC → reshape → 4 transposed conv layers → (B, 1, 72, 128)
        self.fc_decode = nn.Linear(latent_dim, self._encoder_out_dim)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # → (B, 128, 8, 16)
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # → (B, 64, 16, 32)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=(5, 4), stride=2, padding=1),  # → (B, 32, 35, 64)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=(6, 4), stride=2, padding=1),   # → (B, 1, 72, 128)
            nn.Sigmoid(),  # Output normalized to [0, 1] to match normalized depth input
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent distribution parameters.

        Args:
            x: Depth images, shape (B, 1, 72, 128), values in [0, 1].

        Returns:
            mu: Mean of latent distribution, shape (B, latent_dim).
            logvar: Log-variance of latent distribution, shape (B, latent_dim).
        """
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample from the latent distribution using the reparameterization trick.

        z = mu + eps * exp(0.5 * logvar), where eps ~ N(0, I).
        This allows gradients to flow through the sampling operation.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent code back to depth image.

        Args:
            z: Latent code, shape (B, latent_dim).

        Returns:
            Reconstructed depth image, shape (B, 1, 72, 128), values in [0, 1].
        """
        h = self.fc_decode(z)
        h = h.view(-1, 256, 4, 8)
        return self.decoder(h)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass: encode → sample → decode.

        Args:
            x: Depth images, shape (B, 1, 72, 128).

        Returns:
            recon: Reconstructed depth image, shape (B, 1, 72, 128).
            mu: Latent mean, shape (B, latent_dim).
            logvar: Latent log-variance, shape (B, latent_dim).
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    def encode_detached(self, x: torch.Tensor) -> torch.Tensor:
        """Encode to latent mean, detached from the VAE computation graph.

        Use this during RL to prevent RL gradients from corrupting the VAE.

        Args:
            x: Depth images, shape (B, 1, 72, 128).

        Returns:
            Latent code (mu), shape (B, latent_dim). No gradients attached.
        """
        with torch.no_grad():
            mu, _ = self.encode(x)
        return mu

    def loss(
        self,
        recon: torch.Tensor,
        target: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute VAE loss = reconstruction + beta * KL divergence.

        Args:
            recon: Reconstructed images from decoder, shape (B, 1, 72, 128).
            target: Original depth images, shape (B, 1, 72, 128).
            mu: Latent mean, shape (B, latent_dim).
            logvar: Latent log-variance, shape (B, latent_dim).

        Returns:
            total_loss: Combined loss scalar.
            recon_loss: MSE reconstruction loss scalar.
            kl_loss: KL divergence loss scalar.
        """
        recon_loss = nn.functional.mse_loss(recon, target, reduction="mean")
        # KL(q(z|x) || p(z)) where p(z) = N(0, I)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        total_loss = recon_loss + self.beta * kl_loss
        return total_loss, recon_loss, kl_loss
