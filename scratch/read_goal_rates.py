import os
from tensorboard.backend.event_processing import event_accumulator

def main():
    ppo_dir = r"d:\isaac\3D_Drone_RL\logs\rsl_rl\navigation_drone_direct"
    if not os.path.exists(ppo_dir):
        print(f"Directory not found: {ppo_dir}")
        return
    import glob
    dirs = [d for d in glob.glob(os.path.join(ppo_dir, "*")) if os.path.isdir(d)]
    if not dirs:
        print("No log directories found.")
        return
    log_dir = max(dirs, key=os.path.getmtime)
    print(f"Reading from: {log_dir}")

    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()

    tags = ['Metrics/goal_rate', 'Metrics/collision_rate', 'Metrics/running_goal_rate', 'Policy/mean_std']
    scalars = ea.Tags().get('scalars', [])
    
    print(f"--- Raw Goal and Collision Rates ---")
    if 'Metrics/goal_rate' in scalars:
        goal_events = ea.Scalars('Metrics/goal_rate')
        coll_events = ea.Scalars('Metrics/collision_rate')
        rgr_events = ea.Scalars('Metrics/running_goal_rate')
        std_events = ea.Scalars('Policy/mean_std')
        
        print(f"Total steps: {len(goal_events)}")
        print("Step | Raw Goal Rate | Collision Rate | Running Goal Rate | Mean STD")
        print("-" * 75)
        
        # Print the last 35 steps to see the most recent history
        start_idx = max(0, len(goal_events) - 35)
        for idx in range(start_idx, len(goal_events)):
            step = goal_events[idx].step
            gr = goal_events[idx].value
            cr = coll_events[idx].value if idx < len(coll_events) else 0.0
            rgr = rgr_events[idx].value if idx < len(rgr_events) else 0.0
            std = std_events[idx].value if idx < len(std_events) else 0.0
            print(f"{step:<4d} | {gr:.4f}         | {cr:.4f}         | {rgr:.4f}            | {std:.4f}")


if __name__ == '__main__':
    main()
