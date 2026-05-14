import torch
import sys
import os

sys.path.append(r"d:\isaac\3D_Drone_RL\source\first_drone")
from first_drone.models.vae import VAE

path = r"D:\isaac\3D_Drone_RL\logs\vae\vae_final.pt"
print(f"Testing loading of {path}")

vae = VAE(latent_dim=32, beta=1e-3)
try:
    ckpt = torch.load(path, map_location='cpu')
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        print("Found 'model_state_dict' in checkpoint.")
        vae.load_state_dict(ckpt['model_state_dict'])
    elif isinstance(ckpt, dict) and 'encoder.0.weight' in ckpt:
        print("Found direct state dict.")
        vae.load_state_dict(ckpt)
    else:
        print("Unknown checkpoint format. Keys:", getattr(ckpt, 'keys', lambda: None)())
        vae.load_state_dict(ckpt)
    
    print("Successfully loaded VAE!")
except Exception as e:
    print(f"Error loading: {e}")

vae.eval()
with torch.no_grad():
    x = torch.rand(2, 1, 72, 128)
    recon, mu, logvar = vae(x)
    print(f"Recon shape: {recon.shape}, min: {recon.min().item():.4f}, max: {recon.max().item():.4f}, mean: {recon.mean().item():.4f}, std: {recon.std().item():.4f}")
