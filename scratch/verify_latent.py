import os
import glob
import numpy as np
import torch
import sys

# Import AE directly to avoid omni/isaaclab dependencies
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ae_dir = os.path.abspath(os.path.join(repo_root, "source", "first_drone", "first_drone", "models"))
sys.path.insert(0, ae_dir)
from ae import AE
sys.path.pop(0)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model_path = os.path.join(repo_root, "logs", "ae_arena", "ae_final.pt")
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
        
    model = AE(latent_dim=32).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Model loaded successfully.")
    
    # Load dataset files
    data_dir = os.path.join(repo_root, "data", "depth_arena_15m")
    file_paths = sorted(glob.glob(os.path.join(data_dir, "*.npy")))[:200]  # first 200 images
    if not file_paths:
        print(f"Error: No .npy files found in {data_dir}")
        return
    print(f"Loaded {len(file_paths)} files.")
    
    # Encode images
    latents = []
    with torch.no_grad():
        for path in file_paths:
            img_np = np.load(path)  # (72, 128)
            img_tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).to(device) # (1, 1, 72, 128)
            z = model.encode(img_tensor) # (1, 32)
            latents.append(z[0].cpu())
            
    latents = torch.stack(latents)  # (200, 32)
    
    # Compute statistics
    norms = latents.norm(dim=-1)
    stds = latents.std(dim=-1)
    means = latents.mean(dim=-1)
    
    # Compute delta between consecutive frames
    deltas = (latents[1:] - latents[:-1]).norm(dim=-1)
    
    print("\n" + "=" * 50)
    print("LATENT SPACE STATISTICS (200 frames)")
    print("=" * 50)
    print(f"Norm:  mean={norms.mean().item():.4f}, std={norms.std().item():.4f}, min={norms.min().item():.4f}, max={norms.max().item():.4f}")
    print(f"Std:   mean={stds.mean().item():.4f}, std={stds.std().item():.4f}, min={stds.min().item():.4f}, max={stds.max().item():.4f}")
    print(f"Mean:  mean={means.mean().item():.4f}, std={means.std().item():.4f}, min={means.min().item():.4f}, max={means.max().item():.4f}")
    print(f"Delta between consecutive frames: mean={deltas.mean().item():.4f}, std={deltas.std().item():.4f}, min={deltas.min().item():.4f}, max={deltas.max().item():.4f}")
    print("=" * 50)

if __name__ == "__main__":
    main()
