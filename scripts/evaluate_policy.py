"""Script to evaluate a trained RL policy and print detailed performance statistics.

This script runs the policy in the environment for a specified number of episodes,
tracks the outcome of each episode (Success, Collision, or Timeout) per curriculum level,
and prints a summary table at the end.

Usage:
    isaaclab.bat -p scripts/evaluate_policy.py --task MultiLevel-Drone-Direct-v0 --checkpoint logs/ppo/navigation_drone_direct/29-06_15-48/model_1500.pt --num_episodes 100
"""

# ── Launch Isaac Sim first ──────────────────────────────────────────
import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate a trained RL agent with statistics.")
parser.add_argument("--num_episodes", type=int, default=100, help="Number of episodes to evaluate.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to the PPO policy checkpoint (.pt file).")
parser.add_argument("--num_envs", type=int, default=16, help="Number of parallel environments for evaluation.")
parser.add_argument("--task", type=str, default="MultiLevel-Drone-Direct-v0", help="Name of the task.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration.")
parser.add_argument("--web_viewer", action="store_true", default=False, help="Enable web viewer if running headless.")
parser.add_argument("--output_report", type=str, default="logs/evaluation_report.md", help="Path to save the markdown report.")
parser.add_argument("--continuous", action="store_true", default=False, help="Enable continuous navigation from start to finish (all rooms sequentially).")

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# Clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Imports (after sim launch) ──────────────────────────────────────
import os
import torch
import numpy as np
import gymnasium as gym
from collections import defaultdict
from datetime import datetime

from isaaclab.envs import DirectRLEnv
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

import first_drone.tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config


# ═══════════════════════════════════════════════════════════════════
#  Evaluation loop
# ═══════════════════════════════════════════════════════════════════

@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    # Override env config
    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.continuous:
        env_cfg.continuous_mode = True
    # Disable video recording/rendering overhead if not needed, but keep viewport active if viewer is enabled
    env_cfg.viewer.eye = [5.0, 5.0, 5.0]
    env_cfg.viewer.lookat = [0.0, 0.0, 1.0]

    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    unwrapped_env = env.unwrapped

    # Wrap for RSL-RL
    rsl_env = RslRlVecEnvWrapper(env)

    # Load agent config and policy
    agent_dict = agent_cfg.to_dict()
    # Remove unused keys for compatibility
    for model_key in ["actor", "critic"]:
        if model_key in agent_dict:
            agent_dict[model_key].pop("stochastic", None)
            agent_dict[model_key].pop("init_noise_std", None)
            agent_dict[model_key].pop("noise_std_type", None)
            agent_dict[model_key].pop("state_dependent_std", None)

    # Initialize RSL-RL runner to load the policy
    # We use a dummy log_dir since we are only evaluating
    runner = OnPolicyRunner(rsl_env, agent_dict, log_dir="/tmp/eval", device=agent_cfg.device)
    
    print(f"[INFO] Loading policy checkpoint from: {args_cli.checkpoint}")
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=agent_cfg.device)

    # ── Statistics trackers ──────────────────────────────────────────
    episodes_completed = 0
    stats = {
        "total": {"success": 0, "collision": 0, "timeout": 0, "steps": []},
        "per_level": defaultdict(lambda: {"success": 0, "collision": 0, "timeout": 0, "steps": []})
    }

    # Track active episode data per environment
    # Each env tracks: (start_level, step_count)
    env_episode_levels = {}
    env_step_counts = np.zeros(args_cli.num_envs, dtype=int)

    # Reset environment
    obs, _ = env.reset()
    
    # Initialize levels for the first episode
    for i in range(args_cli.num_envs):
        env_episode_levels[i] = unwrapped_env._current_level[i].item()

    print(f"\n[INFO] Starting evaluation for {args_cli.num_episodes} episodes...")
    print("=" * 70)

    while episodes_completed < args_cli.num_episodes and simulation_app.is_running():
        # Get action from policy
        with torch.no_grad():
            actions = policy(obs)

        # Step environment
        obs, rewards, terminated, truncated, infos = env.step(actions)
        env_step_counts += 1

        # Check completed episodes
        for i in range(args_cli.num_envs):
            is_terminated = terminated[i].item()
            is_truncated = truncated[i].item()

            if is_terminated or is_truncated:
                # Episode finished! Determine outcome
                level = env_episode_levels[i]
                steps = env_step_counts[i]
                
                # Retrieve the exact outcome recorded by the environment before it was reset
                outcome = "collision"
                if hasattr(unwrapped_env, "last_completed_outcomes") and i in unwrapped_env.last_completed_outcomes:
                    outcome = unwrapped_env.last_completed_outcomes[i]

                # Record statistics
                stats["total"][outcome] += 1
                stats["total"]["steps"].append(steps)
                
                stats["per_level"][level][outcome] += 1
                stats["per_level"][level]["steps"].append(steps)

                episodes_completed += 1
                
                if episodes_completed % 10 == 0 or episodes_completed == 1:
                    print(f"Episode {episodes_completed:3d}/{args_cli.num_episodes} | "
                          f"Level {level+1} | Outcome: {outcome.upper():9s} | Steps: {steps:3d}")

                # Reset tracker for this env
                env_step_counts[i] = 0
                # The environment auto-resets on term/trunc, so we grab the new level assigned to this env
                env_episode_levels[i] = unwrapped_env._current_level[i].item()

                if episodes_completed >= args_cli.num_episodes:
                    break

    # ── Print beautiful summary table ────────────────────────────────
    print("\n" + "=" * 70)
    print("                    EVALUATION SUMMARY STATISTICS                    ")
    print("=" * 70)
    
    total_runs = sum([stats["total"]["success"], stats["total"]["collision"], stats["total"]["timeout"]])
    if total_runs == 0:
        print("[WARNING] No episodes completed.")
        env.close()
        return

    def print_row(label, success, collision, timeout, steps_list):
        total = success + collision + timeout
        if total == 0:
            print(f"{label:12s} | No episodes recorded.")
            return
        sr = (success / total) * 100
        cr = (collision / total) * 100
        tr = (timeout / total) * 100
        avg_steps = np.mean(steps_list) if steps_list else 0.0
        print(f"{label:12s} | Episodes: {total:4d} | Success: {sr:5.1f}% | Collision: {cr:5.1f}% | Timeout: {tr:5.1f}% | Avg Steps: {avg_steps:5.1f}")

    # Print overall stats
    print_row("OVERALL", stats["total"]["success"], stats["total"]["collision"], stats["total"]["timeout"], stats["total"]["steps"])
    print("-" * 70)

    # Print per-level stats
    for level in sorted(stats["per_level"].keys()):
        level_data = stats["per_level"][level]
        print_row(f"Level {level+1}", level_data["success"], level_data["collision"], level_data["timeout"], level_data["steps"])

    print("=" * 70 + "\n")

    # ── Write Markdown Report ────────────────────────────────────────
    try:
        report_dir = os.path.dirname(os.path.abspath(args_cli.output_report))
        os.makedirs(report_dir, exist_ok=True)
        
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open(args_cli.output_report, "w", encoding="utf-8") as f:
            f.write(f"# 📊 דוח הערכת ביצועי סוכן (RL Policy Evaluation Report)\n\n")
            f.write(f"* **תאריך ריצה:** {now_str}\n")
            f.write(f"* **קובץ מודל (Checkpoint):** `{args_cli.checkpoint}`\n")
            f.write(f"* **סך הכל אפיזודות שנבחנו:** {total_runs}\n\n")
            
            # Overall table
            f.write(f"## 📈 סיכום ביצועים כללי\n\n")
            f.write(f"| מדד | אחוז מהריצות | כמות ריצות |\n")
            f.write(f"| :--- | :---: | :---: |\n")
            
            s_pct = (stats["total"]["success"] / total_runs) * 100
            c_pct = (stats["total"]["collision"] / total_runs) * 100
            t_pct = (stats["total"]["timeout"] / total_runs) * 100
            avg_steps = np.mean(stats["total"]["steps"]) if stats["total"]["steps"] else 0.0
            
            f.write(f"| **הגעה למטרה (Success Rate)** | **{s_pct:.1f}%** | {stats['total']['success']} |\n")
            f.write(f"| **התנגשויות (Collision Rate)** | {c_pct:.1f}% | {stats['total']['collision']} |\n")
            f.write(f"| **חריגת זמן (Timeout Rate)** | {t_pct:.1f}% | {stats['total']['timeout']} |\n\n")
            f.write(f"* **ממוצע צעדים לאפיזודה:** {avg_steps:.1f} צעדים\n\n")
            
            # Per-level table
            f.write(f"## 🗺️ פירוט לפי שלבי המסלול (Curriculum Levels)\n\n")
            f.write(f"| שלב | אפיזודות | הצלחה (Success) | התנגשות (Collision) | חריגת זמן (Timeout) | ממוצע צעדים |\n")
            f.write(f"| :---: | :---: | :---: | :---: | :---: | :---: |\n")
            
            for level in sorted(stats["per_level"].keys()):
                level_data = stats["per_level"][level]
                l_total = level_data["success"] + level_data["collision"] + level_data["timeout"]
                if l_total > 0:
                    ls_pct = (level_data["success"] / l_total) * 100
                    lc_pct = (level_data["collision"] / l_total) * 100
                    lt_pct = (level_data["timeout"] / l_total) * 100
                    l_avg_steps = np.mean(level_data["steps"]) if level_data["steps"] else 0.0
                    f.write(f"| **שלב {level+1}** | {l_total} | {ls_pct:.1f}% | {lc_pct:.1f}% | {lt_pct:.1f}% | {l_avg_steps:.1f} |\n")
                else:
                    f.write(f"| **שלב {level+1}** | 0 | - | - | - | - |\n")
            
            f.write(f"\n---\n*הדוח הופק אוטומטית על ידי סקריפט ההערכה של הפרויקט.*")
            
        print(f"[INFO] Saved evaluation report to: {args_cli.output_report}")
    except Exception as e:
        print(f"[WARNING] Could not save evaluation report: {e}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
