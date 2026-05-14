import torch

path = r"D:\isaac\3D_Drone_RL\logs\vae\vae_final.pt"
ckpt = torch.load(path, map_location='cpu')
print("Mean weight of fc_decode:", ckpt['fc_decode.weight'].mean().item())
print("Std weight of fc_decode:", ckpt['fc_decode.weight'].std().item())
