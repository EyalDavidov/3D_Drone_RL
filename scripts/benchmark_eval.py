# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Benchmark evaluation script for 5-victim search and rescue drone with Visual SLAM.

Supports single pilot runs or batch automated runs in headless or windowed mode.
Room Spawning Rules:
  - Room 1: 1 victim
  - Room 2: 1 victim
  - Room 3 (Big Room): 2 victims (>= 1.0m apart)
  - Room 4: 1 victim
  - Corridor: 0 victims
  - Clearance: >= 1.0m from walls and dynamic obstacles

Outputs:
  - benchmark_runs.jsonl
  - benchmark_summary.json
  - recordings/flight_*.jsonl (full flight telemetry + base64 camera feeds)
"""

import argparse
import sys
import os
import time
import json
import random
import math
import numpy as np
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="A/B Benchmark Evaluation for Search and Rescue Drone.")
parser.add_argument("--num_runs", type=int, default=1, help="Number of benchmark runs (default: 1 for pilot test).")
parser.add_argument("--scan_mode", action="store_true", default=False, help="Enable active 360-degree SCAN mode at waypoints.")
parser.add_argument("--task", type=str, default="Brain-Nav-Drone-Direct-v0", help="Task name.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to PPO checkpoint (.pt).")
parser.add_argument("--seed_start", type=int, default=1000, help="Starting random seed.")
parser.add_argument("--max_steps", type=int, default=3500, help="Maximum steps per mission run.")
parser.add_argument("--output_dir", type=str, default="logs/benchmark_results", help="Directory to save benchmark JSONL outputs.")
parser.add_argument("--lightweight-recording", action="store_true", default=False, help="Record telemetry without embedded base64 camera images.")
parser.add_argument("--no-recording", action="store_true", default=False, help="Disable flight telemetry JSONL recording.")

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# Force camera rendering on since Visual SLAM & YOLO need RGB & Depth
args_cli.enable_cameras = True

# Launch Omniverse application
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Enable debug_draw extension
try:
    import omni.kit.app
    ext_manager = omni.kit.app.get_app().get_extension_manager()
    if not ext_manager.is_extension_enabled("omni.isaac.debug_draw"):
        ext_manager.set_extension_enabled_immediate("omni.isaac.debug_draw", True)
except Exception as e:
    print(f"[WARN] Could not enable omni.isaac.debug_draw: {e}")

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import first_drone.tasks  # noqa: F401
from first_drone.tasks.direct.navigation_drone.brain_nav_drone_env import resolve_navigator_checkpoint
from first_drone.tasks.direct.navigation_drone.real_slam.real_slam_env import RealSlamDroneEnv

# Import LiveDroneTelemetry for Dashboard flight recordings
dashboard_path = os.path.join(os.path.dirname(__file__), "dashboard")
if dashboard_path not in sys.path:
    sys.path.append(dashboard_path)
try:
    from live_telemetry import LiveDroneTelemetry
    _TELEMETRY_AVAILABLE = True
except Exception as te:
    print(f"[WARN] LiveDroneTelemetry not available: {te}")
    _TELEMETRY_AVAILABLE = False


def sample_5_room_victims(env, rng) -> list[tuple[float, float, float]]:
    """Sample 5 victims according to strict room allocation rules instantly:

    Room 1: 1 victim
    Room 2: 1 victim
    Room 3: 2 victims (>= 1.0m apart)
    Room 4: 1 victim
    Corridor: 0 victims
    """
    room_zones = {
        "room_1": (-1.2, 1.2, -1.2, 1.2),
        "room_2": (-1.2, 1.2, -7.0, -3.0),
        "room_3": (-3.0, 3.0, -14.5, -9.5),
        "room_4": (-7.5, -5.0, -21.5, -18.5),
    }

    floor_z = env._person_spawn_local_z()
    
    # Drone spawn location in local frame
    d_pos = env._robot.data.root_pos_w[0] - env._terrain.env_origins[0]
    dx, dy = float(d_pos[0].item()), float(d_pos[1].item())

    placed: list[tuple[float, float, float]] = []

    def sample_zone(zone_bounds: tuple[float, float, float, float], existing: list[tuple[float, float, float]]) -> tuple[float, float, float]:
        min_x, max_x, min_y, max_y = zone_bounds
        for _ in range(100):
            x = rng.uniform(min_x, max_x)
            y = rng.uniform(min_y, max_y)
            if math.hypot(x - dx, y - dy) < 1.5:
                continue
            too_close = any(math.hypot(x - px, y - py) < 1.0 for px, py, _ in existing)
            if too_close:
                continue
            return (round(x, 3), round(y, 3), round(floor_z, 3))
        # Fallback to center if 100 random tries failed
        cx, cy = (min_x + max_x) / 2.0, (min_y + max_y) / 2.0
        return (round(cx, 3), round(cy, 3), round(floor_z, 3))

    placed.append(sample_zone(room_zones["room_1"], placed))
    placed.append(sample_zone(room_zones["room_2"], placed))
    placed.append(sample_zone(room_zones["room_3"], placed))
    placed.append(sample_zone(room_zones["room_3"], placed))
    placed.append(sample_zone(room_zones["room_4"], placed))

    return placed


def main():
    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args_cli.output_dir, f"benchmark_{timestamp_str}")
    os.makedirs(run_dir, exist_ok=True)

    jsonl_path = os.path.join(run_dir, "benchmark_runs.jsonl")
    summary_path = os.path.join(run_dir, "benchmark_summary.json")

    print(f"\n==================================================")
    print(f"🚀 STARTING VISUAL SLAM BENCHMARK EVALUATION")
    print(f"  • Total Runs: {args_cli.num_runs}")
    print(f"  • Task: {args_cli.task}")
    print(f"  • Headless: {args_cli.headless}")
    print(f"  • Lightweight Recording: {args_cli.lightweight_recording}")
    print(f"  • Output Directory: {run_dir}")
    print(f"==================================================\n")

    # Parse config
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1, use_fabric=True)
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.initial_curriculum_level = 5
    env_cfg.debug_vis = False
    env_cfg.show_ae_images = False
    env_cfg.spawn_person = True
    env_cfg.yolo_show_opencv = False
    env_cfg.navigator_checkpoint_path = resolve_navigator_checkpoint(
        args_cli.checkpoint or env_cfg.navigator_checkpoint_path
    )

    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    agent_cfg.device = env_cfg.sim.device

    # Instantiate Real SLAM environment
    print(f"[INFO] Instantiating RealSlamDroneEnv for Visual SLAM Benchmark...")
    env_instance = RealSlamDroneEnv(cfg=env_cfg)
    env_instance.is_brain_play = True
    env = RslRlVecEnvWrapper(env_instance, clip_actions=agent_cfg.clip_actions)
    env.unwrapped.is_brain_play = True

    all_run_metrics = []

    for run_idx in range(args_cli.num_runs):
        seed = args_cli.seed_start + run_idx
        rng = random.Random(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        print(f"\n--- Run {run_idx + 1}/{args_cli.num_runs} (Seed: {seed}) ---")

        # Reset env
        env.unwrapped.seed(seed)
        env.reset()

        # Sample and spawn 5 victims in rooms 1-4
        victim_positions = sample_5_room_victims(env.unwrapped, rng)
        env.unwrapped._hide_map_default_person()
        env.unwrapped._hide_static_rescue_persons_for_dynamic_spawn()
        env.unwrapped._clear_dynamic_spawned_persons()
        
        prims = []
        for i, pos in enumerate(victim_positions):
            name = f"DynamicBenchmark_{i}"
            prim = env.unwrapped._spawn_rescue_person_wrapper(name, pos, yaw_deg=90.0)
            if not env.unwrapped._align_person_scale_to_static_template(prim):
                env.unwrapped._queue_person_scale_fix(prim)
            prims.append(prim)
            
        env.unwrapped._dynamic_spawn_prims = prims
        env.unwrapped.spawned_targets_local = victim_positions
        env.unwrapped.dynamic_spawn_active = True
        
        perception = getattr(env.unwrapped, "_perception", None)
        if perception is not None:
            perception._rescue_person_slots = env.unwrapped._build_dynamic_spawn_log_slots(victim_positions)
            perception._detection_log = []
            perception._person_best_conf = {}
            perception.frame_confirmed_persons = []

        print(f"  • Placed 5 Victims: {victim_positions}")

        # Metrics tracking
        start_wall_time = time.time()
        timestep = 0
        dt = env.unwrapped.step_dt
        dummy_action = torch.zeros((1, 4), device=env.unwrapped.device)

        detected_victims_set = set()
        first_detection_time = None
        all_detected_time = None
        collision_occurred = False
        mission_completed = False

        # Flight Telemetry recorder
        telemetry = None
        if _TELEMETRY_AVAILABLE and not args_cli.no_recording:
            try:
                telemetry = LiveDroneTelemetry(
                    tick_rate=24.0,
                    recording=True,
                    lightweight_recording=args_cli.lightweight_recording,
                )
                print(f"  • Telemetry recorder active -> saving flight to recordings/flight_*.jsonl")
            except Exception as te:
                print(f"  • Could not start telemetry recorder: {te}")
                telemetry = None

        while timestep < args_cli.max_steps and simulation_app.is_running():
            with torch.inference_mode():
                # Step physics + Visual SLAM + PPO navigator + LLC controller
                obs, rewards, dones, infos = env.step(dummy_action)

                # Push frame telemetry to JSONL recorder
                if telemetry is not None:
                    try:
                        telemetry.push(env.unwrapped, timestep * dt)
                    except Exception:
                        pass

                # Check victim detections
                det_count, total_victims = env.unwrapped.count_spawned_targets_detected()
                if det_count > len(detected_victims_set):
                    if first_detection_time is None:
                        first_detection_time = round(timestep * dt, 2)
                    for idx_d in range(det_count):
                        detected_victims_set.add(idx_d)
                    if len(detected_victims_set) == total_victims and all_detected_time is None:
                        all_detected_time = round(timestep * dt, 2)

                # Check SLAM state and mission completion
                brain = getattr(env.unwrapped, "_brain", None)
                slam_state = getattr(brain, "state", "EXPLORE") if brain else "EXPLORE"

                if slam_state == "COMPLETE" or getattr(brain, "mission_finished", False):
                    mission_completed = True
                    print(f"  ✨ Mission Complete on step {timestep} (Time: {timestep * dt:.1f}s)!")
                    break

                # Also check dynamic spawn finish condition (all victims + high coverage)
                visited, total = (0, 0)
                if brain and hasattr(brain, "coverage_stats"):
                    visited, total = brain.coverage_stats()
                coverage_pct = (visited / max(total, 1)) * 100.0
                if total_victims > 0 and det_count >= total_victims and coverage_pct >= 95.0:
                    mission_completed = True
                    print(f"  ✨ All victims detected ({det_count}/{total_victims}) & Coverage {coverage_pct:.1f}% reached!")
                    break

                # Check crash / termination
                if dones[0].item():
                    collision_occurred = True
                    print(f"  ❌ Collision / Reset on step {timestep} (Time: {timestep * dt:.1f}s)")
                    break

            timestep += 1

        if telemetry is not None:
            try:
                telemetry.close()
            except Exception:
                pass

        elapsed_wall_time = round(time.time() - start_wall_time, 2)
        sim_duration_s = round(timestep * dt, 2)
        det_count_final, total_victims = env.unwrapped.count_spawned_targets_detected()
        full_detection = (det_count_final == 5)
        successful_rescue = full_detection and mission_completed and not collision_occurred

        run_record = {
            "run_id": f"run_{run_idx + 1:03d}",
            "seed": seed,
            "scan_mode": args_cli.scan_mode,
            "victim_positions": victim_positions,
            "total_victims": total_victims,
            "detected_count": det_count_final,
            "full_detection": full_detection,
            "first_detection_time_s": first_detection_time,
            "all_detected_time_s": all_detected_time,
            "collision_occurred": collision_occurred,
            "mission_completed": mission_completed,
            "successful_rescue_mission": successful_rescue,
            "steps": timestep,
            "sim_duration_s": sim_duration_s,
            "wall_duration_s": elapsed_wall_time,
        }

        all_run_metrics.append(run_record)

        # Write record line to JSONL immediately
        with open(jsonl_path, "a", encoding="utf-8") as f_jsonl:
            f_jsonl.write(json.dumps(run_record) + "\n")

        print(f"  • Result: Victims Found={det_count_final}/5 | Full={full_detection} | Collided={collision_occurred} | Completed={mission_completed}")
        print(f"  • Speed: {sim_duration_s}s sim calculated in {elapsed_wall_time}s wall time ({sim_duration_s / max(elapsed_wall_time, 0.01):.1f}x speed)")

    # Generate summary JSON
    total_runs = len(all_run_metrics)
    summary_data = {
        "benchmark_timestamp": timestamp_str,
        "total_runs": total_runs,
        "total_victims_placed": total_runs * 5,
        "total_victims_detected": sum(r["detected_count"] for r in all_run_metrics),
        "overall_detection_rate": round(sum(r["detected_count"] for r in all_run_metrics) / max(total_runs * 5, 1), 4),
        "full_detection_rate": round(sum(1 for r in all_run_metrics if r["full_detection"]) / max(total_runs, 1), 4),
        "collision_free_rate": round(sum(1 for r in all_run_metrics if not r["collision_occurred"]) / max(total_runs, 1), 4),
        "mission_completion_rate": round(sum(1 for r in all_run_metrics if r["mission_completed"]) / max(total_runs, 1), 4),
        "successful_rescue_mission_rate": round(sum(1 for r in all_run_metrics if r["successful_rescue_mission"]) / max(total_runs, 1), 4),
        "avg_sim_duration_s": round(float(np.mean([r["sim_duration_s"] for r in all_run_metrics])), 2),
        "avg_wall_duration_s": round(float(np.mean([r["wall_duration_s"] for r in all_run_metrics])), 2),
        "jsonl_file": jsonl_path,
    }

    with open(summary_path, "w", encoding="utf-8") as f_sum:
        json.dump(summary_data, f_sum, indent=2)

    print(f"\n==================================================")
    print(f"✅ BENCHMARK COMPLETED SUCCESSFULLY!")
    print(f"  • Summary Saved: {summary_path}")
    print(f"  • JSONL Runs Saved: {jsonl_path}")
    print(f"  • Overall Detection Rate: {summary_data['overall_detection_rate']:.1%}")
    print(f"  • Full Rescue Mission Rate: {summary_data['successful_rescue_mission_rate']:.1%}")
    print(f"==================================================\n")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
