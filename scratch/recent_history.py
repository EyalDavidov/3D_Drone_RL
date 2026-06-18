import os
import glob
from tensorboard.backend.event_processing import event_accumulator

def main():
    ppo_dir = r"d:\isaac\3D_Drone_RL\logs\rsl_rl\navigation_drone_direct"
    dirs = [d for d in glob.glob(os.path.join(ppo_dir, "*")) if os.path.isdir(d)]
    if not dirs:
        print("No log directories found.")
        return
    log_dir = max(dirs, key=os.path.getmtime)
    print(f"Reading from: {log_dir}")

    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()

    tags = [
        'Metrics/curriculum_level', 
        'Metrics/goal_rate', 
        'Metrics/running_goal_rate', 
        'Metrics/collision_rate', 
        'Loss/entropy', 
        'Policy/mean_std',
        'Loss/value',
        'Episode_Termination/died',
        'Episode_Termination/time_out'
    ]
    
    data = {}
    for tag in tags:
        if tag in ea.Tags().get('scalars', []):
            data[tag] = ea.Scalars(tag)
        else:
            data[tag] = []

    if data['Policy/mean_std']:
        steps = [e.step for e in data['Policy/mean_std']]
        print("\nStep | Level | Goal Rate | Run Goal Rate | Col Rate | Died (Col) | Timeouts | Value Loss | Mean STD")
        print("-" * 115)
        
        # Helper to get value at step
        def get_val_at_step(tag_events, step):
            for e in tag_events:
                if e.step == step:
                    return e.value
            return None

        # Print all steps from step 1700 onwards
        steps_to_show = [s for s in steps if s >= 1700]
        if not steps_to_show:
            steps_to_show = steps[-30:]

        for s in steps_to_show:
            curr = get_val_at_step(data['Metrics/curriculum_level'], s)
            gr = get_val_at_step(data['Metrics/goal_rate'], s)
            rgr = get_val_at_step(data['Metrics/running_goal_rate'], s)
            col = get_val_at_step(data['Metrics/collision_rate'], s)
            died = get_val_at_step(data['Episode_Termination/died'], s)
            timeouts = get_val_at_step(data['Episode_Termination/time_out'], s)
            val_loss = get_val_at_step(data['Loss/value'], s)
            std = get_val_at_step(data['Policy/mean_std'], s)
            
            curr_str = f"{curr:.1f}" if curr is not None else "N/A"
            gr_str = f"{gr:.4f}" if gr is not None else "N/A"
            rgr_str = f"{rgr:.4f}" if rgr is not None else "N/A"
            col_str = f"{col:.4f}" if col is not None else "N/A"
            died_str = f"{died:.1f}" if died is not None else "N/A"
            timeouts_str = f"{timeouts:.1f}" if timeouts is not None else "N/A"
            val_str = f"{val_loss:.1f}" if val_loss is not None else "N/A"
            std_str = f"{std:.4f}" if std is not None else "N/A"
            
            print(f"{s:<4d} | {curr_str:<5s} | {gr_str:<9s} | {rgr_str:<13s} | {col_str:<8s} | {died_str:<10s} | {timeouts_str:<8s} | {val_str:<10s} | {std_str:<8s}")
            
if __name__ == "__main__":
    main()
