import os
import glob
import numpy as np
import torch
import sys

repo_root = r"d:\isaac\3D_Drone_RL"
sys.path.insert(0, os.path.join(repo_root, "source", "first_drone", "first_drone", "models"))
from ae import AE
sys.path.pop(0)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = os.path.join(repo_root, "logs", "ae_arena", "ae_final.pt")
    
    model = AE(latent_dim=32).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    data_dir = os.path.join(repo_root, "data", "depth_arena_15m")
    file_paths = sorted(glob.glob(os.path.join(data_dir, "*.npy")))[:200]
    
    losses = []
    with torch.no_grad():
        for path in file_paths:
            img_np = np.load(path)
            img_tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).to(device)
            recon, z = model(img_tensor)
            loss = torch.nn.functional.mse_loss(recon, img_tensor).item()
            losses.append(loss)
            
    print(f"AE reconstruction MSE on 200 files: mean={np.mean(losses):.6f}, std={np.std(losses):.6f}")

if __name__ == '__main__':
    main()
