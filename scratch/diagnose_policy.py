"""Diagnostic 2: Check AE latent space stability and policy actions during flight.

Loads a trained model and monitors:
1. AE latent vector stability (does it change erratically?)
2. Policy action outputs (what does the policy actually command?)
3. Position/velocity to understand drone behavior
"""
import argparse, sys, os
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, default="AE-PPO-Drone-Direct-v0")
parser.add_argument("--checkpoint", type=str, required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
sys.argv = [sys.argv[0], "play.py"]
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
import first_drone.tasks  # noqa

OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "diagnose_policy_output.txt")

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
    unwrapped.curriculum_level = 4
    print(f"\n[DIAGNOSTIC] Forced curriculum level to {unwrapped.curriculum_level}\n")
    
    # Load AE
    ae_path = unwrapped.cfg.ae_checkpoint_path
    if os.path.exists(ae_path):
        pretrained = torch.load(ae_path, map_location="cuda:0")
        if "ae" in pretrained:
            unwrapped.ae.load_state_dict(pretrained["ae"])
        else:
            unwrapped.ae.load_state_dict(pretrained)
        unwrapped.ae.eval()
    
    # Wrap and load policy
    env_wrapped = RslRlVecEnvWrapper(env)
    
    from first_drone.tasks.direct.navigation_drone.agents.rsl_rl_ppo_cfg import NavigationPPOCfg
    agent_cfg = NavigationPPOCfg()
    agent_dict = agent_cfg.to_dict()
    for model_key in ["actor", "critic"]:
        if model_key in agent_dict:
            agent_dict[model_key].pop("stochastic", None)
            agent_dict[model_key].pop("init_noise_std", None)
            agent_dict[model_key].pop("noise_std_type", None)
            agent_dict[model_key].pop("state_dependent_std", None)
    
    runner = OnPolicyRunner(env_wrapped, agent_dict, log_dir=None, device="cuda:0")
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device="cuda:0")
    
    obs = env_wrapped.get_observations()

    with open(OUTPUT_FILE, "w") as f:
        log("=" * 100, f)
        log("DIAGNOSTIC 2: Policy actions, AE latent, and drone behavior", f)
        log("Checkpoint: %s" % args_cli.checkpoint, f)
        log("=" * 100, f)
        
        prev_latent = None
        
        for step in range(1500):
            with torch.inference_mode():
                actions = policy(obs)
                obs, _, dones, _ = env_wrapped.step(actions)
            
            # Get raw actions
            act = unwrapped._actions[0].cpu().tolist()
            
            # Get position and velocity
            pos_local = (unwrapped._robot.data.root_pos_w[0] - unwrapped._terrain.env_origins[0]).cpu().tolist()
            vel_b = unwrapped._robot.data.root_lin_vel_b[0].cpu().tolist()
            ang_vel_b = unwrapped._robot.data.root_ang_vel_b[0].cpu().tolist()
            
            # Get AE latent
            depth = unwrapped._last_depth_processed
            if depth is not None:
                with torch.no_grad():
                    z = unwrapped.ae.encode(depth[:1])
                latent = z[0].cpu()
                latent_mean = latent.mean().item()
                latent_std = latent.std().item()
                latent_norm = latent.norm().item()
                
                if prev_latent is not None:
                    latent_diff = (latent - prev_latent).norm().item()
                else:
                    latent_diff = 0.0
                prev_latent = latent.clone()
            else:
                latent_mean = latent_std = latent_norm = latent_diff = -1
            
            # Distance to goal
            dist_to_goal = torch.linalg.norm(
                unwrapped._desired_pos_w[0] - unwrapped._robot.data.root_pos_w[0]
            ).item()
            
            # Goal direction
            goal_local = (unwrapped._desired_pos_w[0] - unwrapped._robot.data.root_pos_w[0]).cpu().tolist()
            
            if step % 10 == 0:
                log("Step %04d:" % step, f)
                log("  pos_local=(%.2f, %.2f, %.2f)" % (pos_local[0], pos_local[1], pos_local[2]), f)
                log("  vel_body=(%.3f, %.3f, %.3f) speed=%.3f" % (vel_b[0], vel_b[1], vel_b[2], sum(v**2 for v in vel_b)**0.5), f)
                log("  ang_vel_body=(%.3f, %.3f, %.3f)" % (ang_vel_b[0], ang_vel_b[1], ang_vel_b[2]), f)
                log("  actions=(fwd=%.3f, lat=%.3f, vert=%.3f, yaw=%.3f)" % (act[0], act[1], act[2], act[3]), f)
                log("  dist_to_goal=%.2f  goal_dir=(%.1f, %.1f, %.1f)" % (dist_to_goal, goal_local[0], goal_local[1], goal_local[2]), f)
                log("  AE latent: mean=%.4f, std=%.4f, norm=%.3f, delta=%.4f" % (latent_mean, latent_std, latent_norm, latent_diff), f)
                log("", f)
            
            if dones[0].item():
                log("*** EPISODE ENDED at step %d ***" % step, f)
                log("  pos_local=(%.2f, %.2f, %.2f)" % (pos_local[0], pos_local[1], pos_local[2]), f)
                log("  dist_to_goal=%.3f" % dist_to_goal, f)
                reached = dist_to_goal < unwrapped.cfg.goal_radius
                log("  reached_goal=%s" % reached, f)
                
                contact_force = torch.linalg.norm(
                    unwrapped._contact_sensor.data.net_forces_w[:, 0, :], dim=-1
                )[0].item()
                log("  contact_force=%.3fN" % contact_force, f)
                
                hit_fc = pos_local[2] < 0.1 or pos_local[2] > 2.5
                hit_wall = abs(pos_local[0]) > 24.5 or abs(pos_local[1]) > 24.5
                log("  hit_floor_ceiling=%s (z=%.3f)" % (hit_fc, pos_local[2]), f)
                log("  hit_wall=%s" % hit_wall, f)
                log("", f)
                # Continue to see multiple episodes
        
        log("Output saved to: %s" % OUTPUT_FILE, f)

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
