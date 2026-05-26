"""Play script with Saliency Map (Jacobian) visualization.

Computes the gradient of the Actor's output actions with respect to the raw
depth image pixels: nabla_x pi(a|s). Visualizes these gradients as a heatmap
overlaying the depth image to show which pixels drive the agent's decisions.

Usage:
    D:\\Isaac\\IsaacLab\\isaaclab.bat -p scripts/play_saliency.py ^
        --task AE-PPO-Drone-Direct-v0 --num_envs 4 --enable_cameras ^
        --checkpoint logs/ppo/navigation_drone_direct/24-05_22-30/model_1499.pt
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play with saliency map visualization.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="AE-PPO-Drone-Direct-v0", help="Name of the task.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt).")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time.")
parser.add_argument("--saliency_target", type=str, default="action", choices=["action", "value"],
                    help="Compute saliency w.r.t. actor output (action) or critic output (value).")
parser.add_argument("--saliency_env", type=int, default=0, help="Which environment index to visualize.")
parser.add_argument("--update_interval", type=int, default=2, help="Update saliency every N steps.")
parser.add_argument("--saliency_method", type=str, default="smoothgrad", choices=["vanilla", "smoothgrad", "integrated"],
                    help="Method to compute saliency map.")
parser.add_argument("--saliency_samples", type=int, default=15, help="Number of samples for SmoothGrad/Integrated Gradients.")
parser.add_argument("--saliency_noise", type=float, default=0.1, help="Noise level (std) for SmoothGrad.")
parser.add_argument("--no_saliency_focus", action="store_true", default=False,
                    help="Disable focusing the heatmap on close obstacles (shows raw gradients).")

AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

sys.argv = [sys.argv[0]]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import time
import cv2
import numpy as np
import gymnasium as gym
import torch

from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import first_drone.tasks  # noqa: F401

from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry


def compute_saliency(
    depth_image: torch.Tensor,
    ae_encoder: torch.nn.Module,
    ae_fc_z: torch.nn.Module,
    state_features: torch.Tensor,
    actor_mlp: torch.nn.Module,
    actor_obs_normalizer: torch.nn.Module,
    actor_deterministic_output: torch.nn.Module | None = None,
    method: str = "smoothgrad",
    num_samples: int = 15,
    noise_level: float = 0.1,
    focus_obstacles: bool = True,
) -> torch.Tensor:
    """Compute saliency map of policy output w.r.t. depth pixels.

    Supports:
      - 'vanilla': standard Jacobian gradient (often noisy in deep nets)
      - 'smoothgrad': averages gradients over inputs corrupted by Gaussian noise
      - 'integrated': path integral of gradients from a black baseline
    """
    if method == "vanilla":
        depth_input = depth_image.clone().detach().requires_grad_(True)
        h = ae_encoder(depth_input)
        z_img = ae_fc_z(h)
        obs = torch.cat([z_img, state_features], dim=-1)
        obs = actor_obs_normalizer(obs)
        output = actor_mlp(obs)
        if actor_deterministic_output is not None:
            output = actor_deterministic_output(output)
        loss = output.abs().sum()
        loss.backward()
        saliency = depth_input.grad.abs().squeeze(0).squeeze(0)

    elif method == "smoothgrad":
        total_grad = torch.zeros_like(depth_image[0, 0])
        for _ in range(num_samples):
            noise = torch.randn_like(depth_image) * noise_level
            noisy_depth = (depth_image + noise).clamp(0.0, 1.0)
            depth_input = noisy_depth.clone().detach().requires_grad_(True)

            h = ae_encoder(depth_input)
            z_img = ae_fc_z(h)
            obs = torch.cat([z_img, state_features], dim=-1)
            obs = actor_obs_normalizer(obs)
            output = actor_mlp(obs)
            if actor_deterministic_output is not None:
                output = actor_deterministic_output(output)
            loss = output.abs().sum()
            loss.backward()

            if depth_input.grad is not None:
                total_grad += depth_input.grad.abs().squeeze(0).squeeze(0)
        saliency = total_grad / num_samples

    elif method == "integrated":
        total_grad = torch.zeros_like(depth_image[0, 0])
        baseline = torch.zeros_like(depth_image)  # completely black baseline
        for step in range(1, num_samples + 1):
            alpha = step / num_samples
            interpolated = baseline + alpha * (depth_image - baseline)
            depth_input = interpolated.clone().detach().requires_grad_(True)

            h = ae_encoder(depth_input)
            z_img = ae_fc_z(h)
            obs = torch.cat([z_img, state_features], dim=-1)
            obs = actor_obs_normalizer(obs)
            output = actor_mlp(obs)
            if actor_deterministic_output is not None:
                output = actor_deterministic_output(output)
            loss = output.abs().sum()
            loss.backward()

            if depth_input.grad is not None:
                total_grad += depth_input.grad.abs().squeeze(0).squeeze(0)
        # Average and multiply by differences from baseline
        saliency = (total_grad / num_samples) * (depth_image - baseline).abs().squeeze(0).squeeze(0)

    else:
        raise ValueError(f"Unknown saliency method: {method}")

    # Focus saliency on close obstacles (pillars) by weighting with proximity (1.0 - depth)
    if focus_obstacles:
        proximity = 1.0 - depth_image[0, 0]
        saliency = saliency * proximity

    # Normalize to [0, 1]
    sal_min, sal_max = saliency.min(), saliency.max()
    if sal_max - sal_min > 1e-8:
        saliency = (saliency - sal_min) / (sal_max - sal_min)
    else:
        saliency = torch.zeros_like(saliency)

    return saliency


def main():
    """Play with saliency map visualization."""
    # ---- Resolve configurations ----
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")

    from first_drone.tasks.direct.navigation_drone.agents.rsl_rl_ppo_cfg import NavigationPPOCfg
    agent_cfg = NavigationPPOCfg()

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = 42
    env_cfg.sim.device = "cuda:0"
    agent_cfg.device = "cuda:0"
    agent_cfg.logger = "tensorboard"  # no wandb for play

    # ---- Create environment ----
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.unwrapped.cfg.show_ae_images = False  # we handle visualization ourselves

    # ---- Wrap env ----
    env = RslRlVecEnvWrapper(env)

    # ---- Create runner and load checkpoint ----
    agent_dict = agent_cfg.to_dict()
    for model_key in ["actor", "critic"]:
        if model_key in agent_dict:
            agent_dict[model_key].pop("stochastic", None)
            agent_dict[model_key].pop("init_noise_std", None)
            agent_dict[model_key].pop("noise_std_type", None)
            agent_dict[model_key].pop("state_dependent_std", None)

    runner = OnPolicyRunner(env, agent_dict, log_dir=None, device=agent_cfg.device)
    print(f"[INFO] Loading checkpoint: {args_cli.checkpoint}")
    runner.load(args_cli.checkpoint)

    # ---- Extract actor components for saliency computation ----
    # rsl_rl structure: PPO.actor is MLPModel with .obs_normalizer, .mlp, .distribution
    actor_model = runner.alg.actor
    actor_model.eval()
    actor_mlp = actor_model.mlp
    actor_obs_normalizer = actor_model.obs_normalizer

    # Get the deterministic output function from the distribution
    actor_deterministic_output = None
    if actor_model.distribution is not None:
        actor_deterministic_output = actor_model.distribution.as_deterministic_output_module()

    print(f"[INFO] Actor MLP: {actor_mlp}")
    print(f"[INFO] Actor has distribution: {actor_model.distribution is not None}")

    # Get the inference policy for normal stepping
    inference_policy = runner.get_inference_policy(device=env.unwrapped.device)

    # ---- AE components (for gradient-enabled forward pass) ----
    ae = env.unwrapped.ae
    ae_encoder = ae.encoder  # CNN part (Sequential ending with Flatten)
    ae_fc_z = ae.fc_z        # Linear bottleneck

    dt = env.unwrapped.step_dt
    env_idx = args_cli.saliency_env

    # ---- Main loop ----
    obs = env.get_observations()
    timestep = 0

    print(f"\n[INFO] Saliency visualization running (env {env_idx})")
    print(f"[INFO] Target: {args_cli.saliency_target}")
    print("[INFO] Press 'q' in OpenCV window to quit\n")

    while simulation_app.is_running():
        start_time = time.time()

        # Step with inference policy (no gradients)
        with torch.inference_mode():
            actions = inference_policy(obs)
            obs, _, dones, _ = env.step(actions)

        # Compute and display saliency
        if timestep % args_cli.update_interval == 0:
            # Get depth image for the target environment
            depth_proc = env.unwrapped._last_depth_processed
            if depth_proc is not None and depth_proc.shape[0] > env_idx:
                # Extract single env depth image
                depth_single = depth_proc[env_idx:env_idx+1].clone()  # (1, 1, 72, 128)

                # Extract state features for the target environment
                # obs is typically a TensorDict, so we extract the "policy" tensor first
                if isinstance(obs, dict) or hasattr(obs, "keys"):
                    policy_obs = obs["policy"]
                else:
                    policy_obs = obs
                full_obs = policy_obs[env_idx:env_idx+1]  # (1, 45)
                state_features = full_obs[:, 32:].clone().detach()  # (1, 13)

                # Compute saliency (needs gradients — outside inference_mode context)
                try:
                    saliency = compute_saliency(
                        depth_single, ae_encoder, ae_fc_z,
                        state_features, actor_mlp,
                        actor_obs_normalizer, actor_deterministic_output,
                        method=args_cli.saliency_method,
                        num_samples=args_cli.saliency_samples,
                        noise_level=args_cli.saliency_noise,
                        focus_obstacles=not args_cli.no_saliency_focus,
                    )

                    # === Visualization ===
                    depth_np = depth_single[0, 0].detach().cpu().numpy()  # (72, 128)
                    saliency_np = saliency.detach().cpu().numpy()          # (72, 128)

                    # Convert depth to BGR for overlay
                    depth_vis = np.uint8(np.clip(depth_np * 255, 0, 255))
                    depth_bgr = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR)

                    # Convert saliency to heatmap (jet colormap)
                    saliency_uint8 = np.uint8(np.clip(saliency_np * 255, 0, 255))
                    heatmap = cv2.applyColorMap(saliency_uint8, cv2.COLORMAP_JET)

                    # Overlay: blend depth + heatmap
                    overlay = cv2.addWeighted(depth_bgr, 0.4, heatmap, 0.6, 0)

                    # Scale up for visibility
                    scale = 4
                    depth_large = cv2.resize(depth_bgr, (128 * scale, 72 * scale),
                                             interpolation=cv2.INTER_NEAREST)
                    saliency_large = cv2.resize(heatmap, (128 * scale, 72 * scale),
                                                interpolation=cv2.INTER_NEAREST)
                    overlay_large = cv2.resize(overlay, (128 * scale, 72 * scale),
                                               interpolation=cv2.INTER_NEAREST)

                    # Create combined display: [depth | saliency | overlay]
                    combined = np.hstack([depth_large, saliency_large, overlay_large])

                    # Add labels
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(combined, "Depth Input", (20, 30),
                                font, 0.8, (255, 255, 255), 2)
                    cv2.putText(combined, "Saliency (Jacobian)", (128*scale + 20, 30),
                                font, 0.8, (255, 255, 255), 2)
                    cv2.putText(combined, "Overlay", (2*128*scale + 20, 30),
                                font, 0.8, (255, 255, 255), 2)

                    cv2.imshow("Saliency Map - Depth Camera Influence on Policy", combined)

                    # Also show the AE reconstruction for reference
                    with torch.no_grad():
                        z = ae.encode(depth_single)
                        recon = ae.decode(z)
                    recon_np = recon[0, 0].cpu().numpy()
                    recon_vis = np.uint8(np.clip(recon_np * 255, 0, 255))
                    recon_large = cv2.resize(recon_vis, (128 * scale, 72 * scale),
                                             interpolation=cv2.INTER_NEAREST)
                    recon_bgr = cv2.cvtColor(recon_large, cv2.COLOR_GRAY2BGR)
                    cv2.putText(recon_bgr, "AE Reconstruction", (20, 30),
                                font, 0.8, (0, 255, 0), 2)
                    cv2.imshow("AE Reconstruction", recon_bgr)

                except Exception as e:
                    if timestep % 100 == 0:
                        print(f"[WARNING] Saliency computation failed: {e}")
                        import traceback
                        traceback.print_exc()

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        timestep += 1

        # Real-time pacing
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    cv2.destroyAllWindows()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
