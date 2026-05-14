import torch
import sys
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
    x = torch.zeros(2, 1, 72, 128)
    recon, mu, logvar = vae(x)
    print(f"Zeros input - Recon min: {recon.min().item():.4f}, max: {recon.max().item():.4f}")
    
    x = torch.ones(2, 1, 72, 128)
    recon, mu, logvar = vae(x)
    print(f"Ones input - Recon min: {recon.min().item():.4f}, max: {recon.max().item():.4f}")
