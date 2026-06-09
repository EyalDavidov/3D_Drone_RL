import torch

def main():
    path = r"d:\isaac\3D_Drone_RL\logs\rsl_rl\navigation_drone_direct\09-06_00-39\model_1100.pt"
    checkpoint = torch.load(path, map_location="cpu")
    print("Keys in checkpoint:", checkpoint.keys())
    
    actor_sd = checkpoint.get("actor_state_dict", {})
    critic_sd = checkpoint.get("critic_state_dict", {})
    
    print("\n--- Actor State Dict Shapes ---")
    for k, v in actor_sd.items():
        if hasattr(v, "shape"):
            print(f"{k:45s}: shape {v.shape}")
            
    print("\n--- Critic State Dict Shapes ---")
    for k, v in critic_sd.items():
        if hasattr(v, "shape"):
            print(f"{k:45s}: shape {v.shape}")

if __name__ == '__main__':
    main()
