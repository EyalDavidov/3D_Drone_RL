import os
import glob
from tensorboard.backend.event_processing import event_accumulator

def main():
    ppo_dir = r"d:\isaac\3D_Drone_RL\logs\rsl_rl\navigation_drone_direct"
    if not os.path.exists(ppo_dir):
        print(f"Directory not found: {ppo_dir}")
        return
        
    dirs = [d for d in glob.glob(os.path.join(ppo_dir, "*")) if os.path.isdir(d)]
    if not dirs:
        print("No run directories found.")
        return
        
    log_dir = max(dirs, key=os.path.getmtime)
    print(f"Latest log directory: {log_dir}")

    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()

    target_tags = [
        'Metrics/curriculum_level',
        'Metrics/goal_rate',
        'Metrics/running_goal_rate',
        'Metrics/collision_rate',
        'Metrics/episode_length',
        'Episode_Reward/progress',
        'Episode_Reward/goal',
        'Episode_Reward/collision',
        'Episode_Reward/heading',
        'Episode_Reward/vel_align',
        'Episode_Reward/proximity',
        'Episode_Reward/speed_proximity',
        'Episode_Reward/z_deviation',
        'Episode_Reward/action_rate',
        'Episode_Reward/sideslip',
        'Episode_Reward/tilt',
        'Policy/mean_std',
        'Loss/entropy',
        'Loss/learning_rate',
        'Loss/value',
        'Loss/surrogate'
    ]
    
    scalars = ea.Tags().get('scalars', [])
    
    steps = [3700, 3750, 3800, 3850, 3900]
    print("--- Detailed Training Metrics at Steps ---")
    
    def get_val_closest_to_step(tag, target_step):
        if tag in scalars:
            events = ea.Scalars(tag)
            # Find the event with step closest to target_step
            closest_event = min(events, key=lambda e: abs(e.step - target_step))
            return closest_event.step, closest_event.value
        return None, None

    header = f"{'Tag':30s} | " + " | ".join([f"Step {s}" for s in steps])
    print(header)
    print("-" * len(header))
    
    for tag in target_tags:
        vals = []
        for s in steps:
            actual_step, val = get_val_closest_to_step(tag, s)
            if val is not None:
                vals.append(f"{val:.4f} ({actual_step})")
            else:
                vals.append("N/A")
        print(f"{tag:30s} | " + " | ".join(vals))

if __name__ == '__main__':
    main()


