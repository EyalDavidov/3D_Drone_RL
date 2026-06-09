"""Diagnostic script: check contact sensor forces during play to detect false positives.

Run:
  D:\isaac\env_isaaclab\Scripts\python.exe scratch/diagnose_contact.py --task AE-PPO-Drone-Direct-v0 --num_envs 1 --enable_cameras --headless
"""
import argparse, sys, os
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, default="AE-PPO-Drone-Direct-v0")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
sys.argv = [sys.argv[0]]
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import first_drone.tasks  # noqa

OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "diagnose_output.txt")

def log(msg, f):
    print(msg)
    f.write(msg + "\n")
    f.flush()

def main():
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = "cuda:0"

    env = gym.make(args_cli.task, cfg=env_cfg)
    unwrapped = env.unwrapped

    obs, _ = env.reset()

    with open(OUTPUT_FILE, "w") as f:
        log("=" * 80, f)
        log("DIAGNOSTIC: Monitoring contact forces with ZERO actions (hovering)", f)
        log("=" * 80, f)

        contact_force_history = []
        for step in range(200):
            action = torch.zeros(args_cli.num_envs, 4, device="cuda:0")
            obs, reward, terminated, truncated, info = env.step(action)

            # Get contact force
            contact_force = torch.linalg.norm(
                unwrapped._contact_sensor.data.net_forces_w[:, 0, :], dim=-1
            )
            force_val = contact_force[0].item()
            contact_force_history.append(force_val)

            # Get drone position
            pos_local = (unwrapped._robot.data.root_pos_w[0] - unwrapped._terrain.env_origins[0]).cpu().tolist()

            # Get depth stats
            depth = unwrapped._last_depth_processed
            if depth is not None:
                depth_mean = depth[0].mean().item()
                depth_min = depth[0].min().item()
                depth_max_val = depth[0].max().item()
            else:
                depth_mean = depth_min = depth_max_val = -1

            if step % 10 == 0 or force_val > 0.5:
                flag = " *** CONTACT! ***" if force_val > 1.0 else ""
                log(
                    "Step %04d: pos_local=(%.2f, %.2f, %.2f), contact_force=%.3fN, "
                    "depth(min=%.3f, mean=%.3f, max=%.3f)%s" % (
                        step, pos_local[0], pos_local[1], pos_local[2],
                        force_val, depth_min, depth_mean, depth_max_val, flag
                    ), f
                )

            if terminated[0].item() or truncated[0].item():
                hit_floor_ceiling = pos_local[2] < 0.1 or pos_local[2] > 2.5
                hit_wall = abs(pos_local[0]) > 24.5 or abs(pos_local[1]) > 24.5
                log("\n*** TERMINATED at step %d! ***" % step, f)
                log("  terminated=%s, truncated=%s" % (terminated[0].item(), truncated[0].item()), f)
                log("  hit_floor_ceiling=%s (z=%.3f)" % (hit_floor_ceiling, pos_local[2]), f)
                log("  hit_wall=%s (x=%.3f, y=%.3f)" % (hit_wall, pos_local[0], pos_local[1]), f)
                log("  contact_force=%.3fN (threshold=1.0)" % force_val, f)
                break

        forces = torch.tensor(contact_force_history)
        log("\n--- Contact Force Summary ---", f)
        log("  Mean: %.4f N" % forces.mean().item(), f)
        log("  Max:  %.4f N" % forces.max().item(), f)
        log("  Min:  %.4f N" % forces.min().item(), f)
        log("  Std:  %.4f N" % forces.std().item(), f)
        log("  Steps > 0.5N: %d" % (forces > 0.5).sum().item(), f)
        log("  Steps > 1.0N: %d" % (forces > 1.0).sum().item(), f)
        log("\nOutput saved to: %s" % OUTPUT_FILE, f)

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
