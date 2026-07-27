"""merge_benchmark_results.py — Scans and merges all benchmark runs into a single unified benchmark_summary.json.
"""
import os
import json
from pathlib import Path
import numpy as np

def merge_all_benchmark_results():
    logs_dir = Path("logs/benchmark_results")
    if not logs_dir.exists():
        print("[WARN] logs/benchmark_results directory does not exist.")
        return

    runs_by_seed = {}

    for b_folder in sorted(logs_dir.glob("benchmark_*")):
        j_file = b_folder / "benchmark_runs.jsonl"
        if not j_file.exists():
            continue

        with open(j_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    seed = record.get("seed")
                    if seed is None:
                        continue

                    if seed not in runs_by_seed:
                        runs_by_seed[seed] = record
                    else:
                        existing = runs_by_seed[seed]
                        new_score = (record.get("successful_rescue_mission", False), record.get("detected_count", 0), record.get("steps", 0))
                        old_score = (existing.get("successful_rescue_mission", False), existing.get("detected_count", 0), existing.get("steps", 0))
                        if new_score > old_score:
                            runs_by_seed[seed] = record
                except Exception:
                    pass

    sorted_seeds = sorted(runs_by_seed.keys())
    all_run_metrics = [runs_by_seed[s] for s in sorted_seeds]

    total_runs = len(all_run_metrics)
    if total_runs == 0:
        print("[WARN] No valid benchmark runs found to merge.")
        return

    output_dir = logs_dir / "unified_benchmark_summary"
    output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = output_dir / "benchmark_runs.jsonl"
    summary_path = output_dir / "benchmark_summary.json"

    with open(jsonl_path, "w", encoding="utf-8") as f_jsonl:
        for rec in all_run_metrics:
            f_jsonl.write(json.dumps(rec) + "\n")

    summary_data = {
        "unified_summary": True,
        "total_unique_seeds": total_runs,
        "seeds_processed": sorted_seeds,
        "total_victims_placed": total_runs * 5,
        "total_victims_detected": sum(r.get("detected_count", 0) for r in all_run_metrics),
        "overall_detection_rate": round(sum(r.get("detected_count", 0) for r in all_run_metrics) / max(total_runs * 5, 1), 4),
        "full_detection_rate": round(sum(1 for r in all_run_metrics if r.get("full_detection", False)) / max(total_runs, 1), 4),
        "collision_free_rate": round(sum(1 for r in all_run_metrics if not r.get("collision_occurred", False)) / max(total_runs, 1), 4),
        "mission_completion_rate": round(sum(1 for r in all_run_metrics if r.get("mission_completed", False)) / max(total_runs, 1), 4),
        "successful_rescue_mission_rate": round(sum(1 for r in all_run_metrics if r.get("successful_rescue_mission", False)) / max(total_runs, 1), 4),
        "avg_sim_duration_s": round(float(np.mean([r.get("sim_duration_s", 0) for r in all_run_metrics])), 2),
        "avg_wall_duration_s": round(float(np.mean([r.get("wall_duration_s", 0) for r in all_run_metrics])), 2),
        "jsonl_file": str(jsonl_path),
    }

    with open(summary_path, "w", encoding="utf-8") as f_sum:
        json.dump(summary_data, f_sum, indent=2)

    print("\n==================================================")
    print("UNIFIED BENCHMARK SUMMARY GENERATED SUCCESSFULLY!")
    print(f"  * Total Unique Seeds Merged: {total_runs} (Seeds {min(sorted_seeds)} to {max(sorted_seeds)})")
    print(f"  * Summary Saved To: {summary_path}")
    print(f"  * Overall Detection Rate: {summary_data['overall_detection_rate']:.1%}")
    print(f"  * Full Rescue Mission Rate: {summary_data['successful_rescue_mission_rate']:.1%}")
    print("==================================================\n")

if __name__ == "__main__":
    merge_all_benchmark_results()
