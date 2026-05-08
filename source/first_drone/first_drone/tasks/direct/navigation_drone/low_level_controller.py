import torch
import torch.nn as nn
import os

class LowLevelController(nn.Module):
    def __init__(self, checkpoint_path, device="cuda"):
        super().__init__()
        self.device = device
        
        # Exact architecture of our trained Low-Level Actor [32, 16]
        self.mlp = nn.Sequential(
            nn.Linear(13, 32),
            nn.ELU(),
            nn.Linear(32, 16),
            nn.ELU(),
            nn.Linear(16, 4)
        ).to(self.device)
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"[Error] Low Level Checkpoint not found at: {checkpoint_path}")
            
        print(f"[NavigationDroneEnv] Successfully loaded Flight Controller weights from: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        actor_state = ckpt['actor_state_dict']
        
        # Extract only the weights belonging to the MLP and strip the "mlp." prefix
        filtered_state = {k.replace("mlp.", ""): v for k, v in actor_state.items() if k.startswith("mlp.")}
        self.mlp.load_state_dict(filtered_state)
        
        # Freeze weights! The low level controller should not train anymore.
        for param in self.mlp.parameters():
            param.requires_grad = False
        self.eval()

    def forward(self, obs):
        """
        obs: shape (B, 13) -> [lin_vel_b(3), ang_vel_b(3), projected_gravity_b(3), desired_vel_b(3), yaw_err(1)]
        returns: actions (B, 4) -> mapped to [-1, 1] for motors
        """
        with torch.no_grad():
            actions = self.mlp(obs)
            return torch.clamp(actions, -1.0, 1.0)
