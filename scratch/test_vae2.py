import torch

path = r"D:\isaac\3D_Drone_RL\logs\vae\vae_final.pt"
print(f"Testing loading of {path}")

try:
    ckpt = torch.load(path, map_location='cpu')
    if isinstance(ckpt, dict):
        print("Found dict. Keys:", ckpt.keys())
        if 'model_state_dict' in ckpt:
            print("Has model_state_dict!")
            state_dict = ckpt['model_state_dict']
            print("First few keys:", list(state_dict.keys())[:5])
        else:
            print("First few keys:", list(ckpt.keys())[:5])
    else:
        print("Unknown format:", type(ckpt))
except Exception as e:
    print(f"Error loading: {e}")
