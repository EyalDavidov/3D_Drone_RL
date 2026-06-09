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

    tags = ['Metrics/curriculum_level', 'Metrics/running_goal_rate', 'Loss/entropy', 'Policy/mean_std']
    
    data = {}
    for tag in tags:
        if tag in ea.Tags().get('scalars', []):
            data[tag] = ea.Scalars(tag)
        else:
            data[tag] = []

    # Print last 30 steps
    if data['Policy/mean_std']:
        steps = [e.step for e in data['Policy/mean_std']]
        # Print the last 30 points
        print("\nStep | Curriculum Level | Running Goal Rate | Entropy Loss | Policy Mean STD")
        print("-" * 75)
        
        # We align by step
        steps_to_show = steps[-30:]
        
        # Helper to get value at step
        def get_val_at_step(tag_events, step):
            for e in tag_events:
                if e.step == step:
                    return e.value
            return None

        for s in steps_to_show:
            curr = get_val_at_step(data['Metrics/curriculum_level'], s)
            rgr = get_val_at_step(data['Metrics/running_goal_rate'], s)
            ent = get_val_at_step(data['Loss/entropy'], s)
            std = get_val_at_step(data['Policy/mean_std'], s)
            
            curr_str = f"{curr:.2f}" if curr is not None else "N/A"
            rgr_str = f"{rgr:.4f}" if rgr is not None else "N/A"
            ent_str = f"{ent:.4f}" if ent is not None else "N/A"
            std_str = f"{std:.4f}" if std is not None else "N/A"
            
            print(f"{s:<4d} | {curr_str:<16s} | {rgr_str:<17s} | {ent_str:<12s} | {std_str:<15s}")
            
if __name__ == "__main__":
    main()
