import os
from tensorboard.backend.event_processing import event_accumulator

def main():
    log_dir = r"d:\isaac\3D_Drone_RL\logs\rsl_rl\navigation_drone_direct\09-06_01-06"
    if not os.path.exists(log_dir):
        print(f"Log directory not found: {log_dir}")
        return

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
        
        # Print every 10 steps to see the trend clearly
        for idx in range(0, len(goal_events), 10):
            step = goal_events[idx].step
            gr = goal_events[idx].value
            cr = coll_events[idx].value if idx < len(coll_events) else 0.0
            rgr = rgr_events[idx].value if idx < len(rgr_events) else 0.0
            std = std_events[idx].value if idx < len(std_events) else 0.0
            print(f"{step:<4d} | {gr:.4f}         | {cr:.4f}         | {rgr:.4f}            | {std:.4f}")
            
        # Also print the last step explicitly
        last_idx = len(goal_events) - 1
        if last_idx >= 0 and last_idx % 10 != 0:
            step = goal_events[last_idx].step
            gr = goal_events[last_idx].value
            cr = coll_events[last_idx].value if last_idx < len(coll_events) else 0.0
            rgr = rgr_events[last_idx].value if last_idx < len(rgr_events) else 0.0
            std = std_events[last_idx].value if last_idx < len(std_events) else 0.0
            print(f"{step:<4d} | {gr:.4f}         | {cr:.4f}         | {rgr:.4f}            | {std:.4f}")

if __name__ == '__main__':
    main()
