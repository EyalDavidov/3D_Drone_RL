import os
from tensorboard.backend.event_processing import event_accumulator

def main():
    log_dir = r"d:\isaac\3D_Drone_RL\logs\rsl_rl\navigation_drone_direct\09-06_11-19"
    if not os.path.exists(log_dir):
        print(f"Log directory not found: {log_dir}")
        return

    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()

    tags = ['Metrics/curriculum_level', 'Metrics/running_goal_rate', 'Loss/entropy', 'Policy/mean_std', 'Metrics/goal_rate', 'Metrics/collision_rate']
    scalars = ea.Tags().get('scalars', [])
    
    print(f"--- History of Run 09-06_11-19 ---")
    if 'Policy/mean_std' in scalars:
        events = ea.Scalars('Policy/mean_std')
        print(f"Total steps: {len(events)}")
        print("Step | Curriculum | Raw Goal Rate | Collision Rate | Running Goal Rate | Mean STD")
        print("-" * 90)
        
        def get_val_at_idx(tag, idx):
            if tag in scalars:
                evs = ea.Scalars(tag)
                target_step = events[idx].step
                for e in evs:
                    if e.step == target_step:
                        return e.value
            return None

        # Print all steps to see the exact progression
        for idx in range(len(events)):
            step = events[idx].step
            curr = get_val_at_idx('Metrics/curriculum_level', idx)
            gr = get_val_at_idx('Metrics/goal_rate', idx)
            cr = get_val_at_idx('Metrics/collision_rate', idx)
            rgr = get_val_at_idx('Metrics/running_goal_rate', idx)
            std = events[idx].value
            
            curr_str = f"{curr:.2f}" if curr is not None else "N/A"
            gr_str = f"{gr:.4f}" if gr is not None else "N/A"
            cr_str = f"{cr:.4f}" if cr is not None else "N/A"
            rgr_str = f"{rgr:.4f}" if rgr is not None else "N/A"
            print(f"{step:<4d} | {curr_str:<10s} | {gr_str:<13s} | {cr_str:<14s} | {rgr_str:<17s} | {std:.4f}")

if __name__ == '__main__':
    main()
