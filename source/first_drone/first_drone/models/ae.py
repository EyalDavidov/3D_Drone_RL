"""Tiny Autoencoder for simple geometric depth image compression.

Optimized for 128×72 grayscale depth images containing basic geometric structures.
Extremely lightweight (2 layers) for fast training and RL efficiency.
Unlike VAE, this is a standard, deterministic Autoencoder (no KL loss, no sampling).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class AE(nn.Module):
    """Convolutional Autoencoder for depth image compression.

    Input is (B, 1, 72, 128), latent is 32-dimensional.
    """

    def __init__(self, latent_dim: int = 32):
        """Initialize the AE.

        Args:
            latent_dim: Dimension of the latent space. Default: 32.
        """
        super().__init__()
        self.latent_dim = latent_dim

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

        # Deterministic bottleneck mapping to latent code z
        self.fc_z = nn.Linear(self._encoder_out_dim, latent_dim)
        self.ln_z = nn.LayerNorm(latent_dim)

        # ----- Decoder -----
        # Latent (B, 32) → FC → reshape → 4 transposed conv layers → (B, 1, 72, 128)
        self.fc_decode = nn.Linear(latent_dim, self._encoder_out_dim)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, output_padding=(1, 0)),  # → (B, 128, 9, 16)
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # → (B, 64, 18, 32)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),    # → (B, 32, 36, 64)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),     # → (B, 1, 72, 128)
            nn.Sigmoid(),  # Output normalized to [0, 1] to match normalized depth input
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to deterministic latent code z.

        Args:
            x: Depth images, shape (B, 1, 72, 128), values in [0, 1].

        Returns:
            z: Latent code, shape (B, latent_dim).
        """
        h = self.encoder(x)
        return self.ln_z(self.fc_z(h))

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

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Full forward pass: encode → decode.

        Args:
            x: Depth images, shape (B, 1, 72, 128).

        Returns:
            recon: Reconstructed depth image, shape (B, 1, 72, 128).
            z: Latent code, shape (B, latent_dim).
        """
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z

    def encode_detached(self, x: torch.Tensor) -> torch.Tensor:
        """Encode to latent code, detached from the computation graph.

        Use this during RL to prevent RL gradients from corrupting the encoder.

        Args:
            x: Depth images, shape (B, 1, 72, 128).

        Returns:
            Latent code (z), shape (B, latent_dim). No gradients attached.
        """
        with torch.no_grad():
            z = self.encode(x)
        return z

    def loss(
        self,
        recon: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute standard AE reconstruction loss (MSE).

        Args:
            recon: Reconstructed images from decoder, shape (B, 1, 72, 128).
            target: Original depth images, shape (B, 1, 72, 128).

        Returns:
            total_loss: Reconstruction loss scalar.
            recon_loss: Reconstruction loss scalar (kept for API compatibility).
        """
        recon_loss = nn.functional.mse_loss(recon, target, reduction="mean")
        return recon_loss, recon_loss
