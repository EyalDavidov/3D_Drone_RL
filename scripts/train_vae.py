"""Standalone VAE training script — no Isaac Sim required.

This script:
1. Loads depth images (.npy) from a directory.
2. Trains the VAE architecture (imported directly, bypassing Isaac Lab).
3. Logs training progress to WandB.
4. Saves model weights to a 'checkpoints' directory.
5. Periodically shows reconstruction vs original images via OpenCV.

Usage:
    python scripts/train_vae.py --data_dir data/depth_collection --epochs 50 --batch_size 64
"""

import argparse
import os

# Repository root (one level up from scripts/)
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
import sys
import glob

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
import cv2

# ── Import VAE directly (bypass first_drone package __init__) ───────
# This avoids pulling in Isaac Lab / pxr dependencies.
_VAE_MODULE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__),
                 "..", "source", "first_drone", "first_drone", "models")
)
sys.path.insert(0, _VAE_MODULE_DIR)
from vae import VAE  # noqa: E402
sys.path.pop(0)

# WandB is optional
try:
    import wandb
except ImportError:
    wandb = None


# ═══════════════════════════════════════════════════════════════════
#  Dataset
# ═══════════════════════════════════════════════════════════════════

class DepthDataset(Dataset):
    """Dataset for loading pre-collected depth images (.npy)."""

    def __init__(self, data_dir: str):
        self.file_paths = sorted(glob.glob(os.path.join(data_dir, "*.npy")))
        if not self.file_paths:
            raise FileNotFoundError(f"No .npy files found in {data_dir}")
        print(f"[INFO] Found {len(self.file_paths)} depth images in {data_dir}")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        depth = np.load(self.file_paths[idx])       # (72, 128), float32, [0,1]
        return torch.from_numpy(depth).unsqueeze(0)  # (1, 72, 128)


# ═══════════════════════════════════════════════════════════════════
#  Visualization helper
# ═══════════════════════════════════════════════════════════════════

def visualize_reconstruction(epoch: int, original: torch.Tensor,
                             reconstruction: torch.Tensor,
                             window_name: str = "VAE Reconstruction"):
    """Show original vs reconstructed depth side by side (OpenCV window)."""
    orig = original[0, 0].cpu().numpy()
    recon = reconstruction[0, 0].detach().cpu().numpy()

    orig_vis = (orig * 255).astype(np.uint8)
    recon_vis = (recon * 255).astype(np.uint8)
    combined = np.hstack((orig_vis, recon_vis))

    scale = 4
    combined = cv2.resize(
        combined,
        (combined.shape[1] * scale, combined.shape[0] * scale),
        interpolation=cv2.INTER_NEAREST,
    )
    cv2.putText(
        combined,
        f"Epoch {epoch}: Original (L) | Reconstructed (R)",
        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
    )
    cv2.imshow(window_name, combined)
    cv2.waitKey(1)


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Train VAE on collected depth data (standalone).")
    parser.add_argument("--data_dir", type=str, default=os.path.join(_REPO_ROOT, "data", "depth_collection"),
                        help="Path to collected .npy depth images.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--latent_dim", type=int, default=32, help="VAE latent dimension.")
    parser.add_argument("--beta", type=float, default=1e-3, help="Beta weight for KL loss.")
    parser.add_argument("--vis_interval", type=int, default=50,
                        help="Show reconstruction every N batches.")
    parser.add_argument("--save_dir", type=str, default=os.path.join(_REPO_ROOT, "logs", "vae"),
                        help="Directory to save model weights.")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to a .pt checkpoint to resume training from.")
    parser.add_argument("--no_wandb", action="store_true", help="Disable WandB logging.")
    args = parser.parse_args()

    # ── Device ──────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    os.makedirs(args.save_dir, exist_ok=True)

    # ── Dataset ─────────────────────────────────────────────────────
    try:
        dataset = DepthDataset(args.data_dir)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    # ── Model ───────────────────────────────────────────────────────
    model = VAE(latent_dim=args.latent_dim, beta=args.beta).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    start_epoch = 1

    # ── Resume from checkpoint ──────────────────────────────────────
    if args.resume:
        if not os.path.isfile(args.resume):
            print(f"[ERROR] Checkpoint not found: {args.resume}")
            return
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint)
        # Try to infer epoch from filename (e.g. vae_epoch_50.pt → 50)
        basename = os.path.basename(args.resume)
        for part in basename.replace(".", "_").split("_"):
            if part.isdigit():
                start_epoch = int(part) + 1
                break
        print(f"[INFO] Resumed from {args.resume} → starting at epoch {start_epoch}")

    # ── WandB ───────────────────────────────────────────────────────
    use_wandb = (not args.no_wandb) and (wandb is not None)
    if use_wandb:
        wandb.init(project="drone-vae-training", config=vars(args))
        wandb.watch(model)
    elif not args.no_wandb and wandb is None:
        print("[WARN] wandb not installed — logging disabled. pip install wandb")

    # ── Training loop ───────────────────────────────────────────────
    end_epoch = start_epoch + args.epochs - 1
    print(f"[INFO] Training epochs {start_epoch}→{end_epoch}  |  "
          f"train={train_size}  val={val_size}  batch={args.batch_size}")

    for epoch in range(start_epoch, end_epoch + 1):
        # --- Train ---
        model.train()
        t_loss, t_recon, t_kl = 0.0, 0.0, 0.0

        for batch_idx, data in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()

            recon, mu, logvar = model(data)
            loss, recon_loss, kl_loss = model.loss(recon, data, mu, logvar)

            loss.backward()
            optimizer.step()

            t_loss += loss.item()
            t_recon += recon_loss.item()
            t_kl += kl_loss.item()

            if batch_idx % args.vis_interval == 0:
                visualize_reconstruction(epoch, data, recon)
                print(f"  Epoch {epoch} [{batch_idx}/{len(train_loader)}] "
                      f"loss={loss.item():.6f}")

        # --- Validate ---
        model.eval()
        v_loss, v_recon, v_kl = 0.0, 0.0, 0.0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                recon, mu, logvar = model(data)
                loss, rl, kl = model.loss(recon, data, mu, logvar)
                v_loss += loss.item()
                v_recon += rl.item()
                v_kl += kl.item()

        n_train = len(train_loader)
        n_val = len(val_loader)
        avg_t = t_loss / n_train
        avg_v = v_loss / n_val

        # --- Log ---
        if use_wandb:
            wandb.log({
                "train/total_loss": avg_t,
                "train/recon_loss": t_recon / n_train,
                "train/kl_loss": t_kl / n_train,
                "val/total_loss": avg_v,
                "val/recon_loss": v_recon / n_val,
                "val/kl_loss": v_kl / n_val,
                "epoch": epoch,
            })

        print(f"===> Epoch {epoch}/{end_epoch}  "
              f"train_loss={avg_t:.6f}  val_loss={avg_v:.6f}")

        # --- Checkpoint ---
        if epoch % 10 == 0 or epoch == end_epoch:
            path = os.path.join(args.save_dir, f"vae_epoch_{epoch}.pt")
            torch.save(model.state_dict(), path)
            print(f"[INFO] Saved checkpoint → {path}")

    # Save final model separately for easy loading
    final_path = os.path.join(args.save_dir, "vae_final.pt")
    torch.save(model.state_dict(), final_path)
    print(f"[INFO] Final model saved → {final_path}")

    cv2.destroyAllWindows()
    print("[DONE] Training finished!")


if __name__ == "__main__":
    main()
