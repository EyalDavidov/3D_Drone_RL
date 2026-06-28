import torch
import argparse
import os

def extract_hyperparams(checkpoint_path):
    print(f"Loading checkpoint: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        print("File not found!")
        return

    try:
        # For standard checkpoints, weights_only=False avoids issues in PyTorch 2.6
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"Failed to load with standard torch.load: {e}")
        return

    # Check if the loaded object is a JIT ScriptModule (exported policy)
    if isinstance(checkpoint, torch.jit.ScriptModule) or hasattr(checkpoint, "graph"):
        print("\n" + "="*50)
        print("🚀 EXPORTED TORCHSCRIPT POLICY DETECTED")
        print("="*50)
        print("This is an exported policy (.pt) used for inference/deployment.")
        print("Exported policies ONLY contain the computational graph and weights.")
        print("They DO NOT contain training hyperparameters (Learning Rate, Optimizer, Iterations).")
        print("To see training hyperparameters, please point this script to the original model_XXXX.pt checkpoint instead!")
        print("="*50 + "\n")
        return
    
    print("\n" + "="*50)
    print("🎯 CHECKPOINT METADATA")
    print("="*50)
    
    if "iter" in checkpoint:
        print(f"Iteration / Epochs saved: {checkpoint['iter']}")
    else:
        print("Iteration data not found in checkpoint.")

    print("\n" + "="*50)
    print("🧠 MODEL ARCHITECTURE (ACTOR)")
    print("="*50)
    
    if "actor_state_dict" in checkpoint:
        state_dict = checkpoint["actor_state_dict"]
        # Extract weight matrices to infer hidden layer sizes
        layers = []
        for key, tensor in state_dict.items():
            if "weight" in key and "mlp" in key:
                layers.append(tensor.shape)
            elif "weight" in key:
                print(f"  - {key}: {tensor.shape}")
                
        if layers:
            print(f"  - MLP Input Dimension: {layers[0][1]}")
            for i, shape in enumerate(layers):
                print(f"  - Hidden Layer {i+1} Output: {shape[0]}")
    elif "model_state_dict" in checkpoint:
        print("Found model_state_dict instead of actor_state_dict. Keys:")
        for k, v in checkpoint["model_state_dict"].items():
            print(f"  - {k}: {v.shape}")
    else:
        print("No standard actor_state_dict found. Available keys:", checkpoint.keys())

    print("\n" + "="*50)
    print("⚙️ OPTIMIZER (LEARNING RATE)")
    print("="*50)
    
    if "optimizer_state_dict" in checkpoint:
        opt_dict = checkpoint["optimizer_state_dict"]
        if "param_groups" in opt_dict:
            for i, group in enumerate(opt_dict["param_groups"]):
                lr = group.get("lr", "Unknown")
                print(f"  - Parameter Group {i} Learning Rate: {lr}")
    else:
        print("Optimizer state not saved in this checkpoint.")
        
    print("\n" + "="*50)
    print("📝 NOTE ON OTHER HYPERPARAMETERS")
    print("="*50)
    print("Things like 'gamma', 'entropy_coef', 'num_mini_batches' are NOT saved inside the .pt file by RSL-RL.")
    print("To find those, look in the same folder as the checkpoint for a 'config.yaml', 'env.yaml', or 'params.json' file!")
    print("="*50 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract hyperparameters from a PT checkpoint")
    parser.add_argument("checkpoint", type=str, help="Path to the .pt file")
    args = parser.parse_args()
    
    extract_hyperparams(args.checkpoint)
