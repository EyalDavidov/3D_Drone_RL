import os
from tensorboard.backend.event_processing import event_accumulator

def main():
    log_dir = r"d:\isaac\3D_Drone_RL\logs\rsl_rl\navigation_drone_direct\15-06_20-54"
    if not os.path.exists(log_dir):
        print(f"Log directory not found: {log_dir}")
        return

    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()

    tags = [
        'Metrics/curriculum_level', 
        'Metrics/goal_rate', 
        'Metrics/running_goal_rate', 
        'Metrics/collision_rate', 
        'Loss/entropy', 
        'Policy/mean_std',
        'Loss/value'
    ]
    scalars = ea.Tags().get('scalars', [])
    
    print(f"--- History of 09-06_21-34 ---")
    if 'Policy/mean_std' in scalars:
        events = ea.Scalars('Policy/mean_std')
        print(f"Total steps: {len(events)}")
        
        # Print a few sparse events to see the progression
        interval = max(1, len(events) // 20)
        print("Step | Level | Goal Rate | Run Goal Rate | Col Rate | Value Loss | Entropy | Mean STD")
        print("-" * 105)
        
        def get_val_at_idx(tag, idx):
            if tag in scalars:
                evs = ea.Scalars(tag)
                # find closest step
                target_step = events[idx].step
                for e in evs:
                    if e.step == target_step:
                        return e.value
            return None

        for idx in range(0, len(events), interval):
            step = events[idx].step
            curr = get_val_at_idx('Metrics/curriculum_level', idx)
            gr = get_val_at_idx('Metrics/goal_rate', idx)
            rgr = get_val_at_idx('Metrics/running_goal_rate', idx)
            col = get_val_at_idx('Metrics/collision_rate', idx)
            val_loss = get_val_at_idx('Loss/value', idx)
            ent = get_val_at_idx('Loss/entropy', idx)
            std = events[idx].value
            
            curr_str = f"{curr:.1f}" if curr is not None else "N/A"
            gr_str = f"{gr:.4f}" if gr is not None else "N/A"
            rgr_str = f"{rgr:.4f}" if rgr is not None else "N/A"
            col_str = f"{col:.4f}" if col is not None else "N/A"
            val_str = f"{val_loss:.1f}" if val_loss is not None else "N/A"
            ent_str = f"{ent:.4f}" if ent is not None else "N/A"
            std_str = f"{std:.4f}" if std is not None else "N/A"
            print(f"{step:<4d} | {curr_str:<5s} | {gr_str:<9s} | {rgr_str:<13s} | {col_str:<8s} | {val_str:<10s} | {ent_str:<7s} | {std_str:<8s}")
            
        # Also print the last event explicitly
        last_idx = len(events) - 1
        step = events[last_idx].step
        curr = get_val_at_idx('Metrics/curriculum_level', last_idx)
        gr = get_val_at_idx('Metrics/goal_rate', last_idx)
        rgr = get_val_at_idx('Metrics/running_goal_rate', last_idx)
        col = get_val_at_idx('Metrics/collision_rate', last_idx)
        val_loss = get_val_at_idx('Loss/value', last_idx)
        ent = get_val_at_idx('Loss/entropy', last_idx)
        std = events[last_idx].value
        curr_str = f"{curr:.1f}" if curr is not None else "N/A"
        gr_str = f"{gr:.4f}" if gr is not None else "N/A"
        rgr_str = f"{rgr:.4f}" if rgr is not None else "N/A"
        col_str = f"{col:.4f}" if col is not None else "N/A"
        val_str = f"{val_loss:.1f}" if val_loss is not None else "N/A"
        ent_str = f"{ent:.4f}" if ent is not None else "N/A"
        std_str = f"{std:.4f}" if std is not None else "N/A"
        print(f"{step:<4d} | {curr_str:<5s} | {gr_str:<9s} | {rgr_str:<13s} | {col_str:<8s} | {val_str:<10s} | {ent_str:<7s} | {std_str:<8s}")

if __name__ == '__main__':
    main()
