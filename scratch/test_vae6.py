import torch
import sys
import os

import importlib.util
spec = importlib.util.spec_from_file_location("vae", r"d:\isaac\3D_Drone_RL\source\first_drone\first_drone\models\vae.py")
vae_module = importlib.util.module_from_spec(spec)
sys.modules["vae"] = vae_module
spec.loader.exec_module(vae_module)

VAE = vae_module.VAE

path = r"D:\isaac\3D_Drone_RL\logs\vae\vae_final.pt"
vae = VAE(latent_dim=32, beta=1e-3)
vae.load_state_dict(torch.load(path, map_location='cpu'))
vae.eval()

with torch.no_grad():
    x = torch.rand(2, 1, 72, 128)
    recon, mu, logvar = vae(x)
    print(f"Recon min: {recon.min().item():.4f}, max: {recon.max().item():.4f}, mean: {recon.mean().item():.4f}, std: {recon.std().item():.4f}")
    
    # Check if the output is perfectly uniform
    diff = recon.max() - recon.min()
    print("Diff max-min:", diff.item())
