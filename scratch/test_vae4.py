import torch

path = r"D:\isaac\3D_Drone_RL\logs\vae\vae_final.pt"
ckpt = torch.load(path, map_location='cpu')

import sys
sys.path.append(r"d:\isaac\3D_Drone_RL\source\first_drone")

# We just want to check if the bias of the output layer is uniform, or if `mu` is always 0.
print("Decoder last layer bias mean:", ckpt['decoder.6.bias'].mean().item())
print("Decoder last layer bias std:", ckpt['decoder.6.bias'].std().item())

print("Encoder mu weight std:", ckpt['fc_mu.weight'].std().item())
print("Encoder mu bias max:", ckpt['fc_mu.bias'].max().item())
