import sys
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
        
    # Get the latest directory by modification time
    log_dir = max(dirs, key=os.path.getmtime)
    print(f"Latest log directory: {log_dir}")

    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()

    tags = ea.Tags()
    scalars = tags.get('scalars', [])
    print("All scalar tags:", scalars)
    
    # We will query and print target training metrics
    target_tags = [
        'Metrics/curriculum_level',
        'Metrics/goal_rate',
        'Metrics/running_goal_rate',
        'Metrics/collision_rate',
        'Episode_Reward/progress',
        'Episode_Reward/goal',
        'Episode_Reward/collision',
        'Episode_Reward/heading',
        'Episode_Reward/vel_align',
        'Episode_Reward/forward_speed',
        'Episode_Reward/proximity',
        'Episode_Reward/speed_proximity',
        'Episode_Reward/z_deviation',
        'Episode_Reward/action_rate',
        'Episode_Reward/sideslip',
        'Loss/entropy',
        'Policy/mean_std',
        'Loss/value',
        'Loss/surrogate',
        'Env0_Metrics/collision_rate',
        'Env0_Reward/z_deviation',
        'Loss/learning_rate',
        'Env0_Reward/inside_obstacle'
    ]
    
    # Print curriculum changes
    if 'Metrics/curriculum_level' in scalars:
        events = ea.Scalars('Metrics/curriculum_level')
        last_val = None
        print("\n--- Curriculum Level Changes ---")
        for e in events:
            if last_val is None or e.value != last_val:
                print(f"Step {e.step}: Curriculum Level {e.value:.1f}")
                last_val = e.value
    else:
        print("Metrics/curriculum_level not found")

    print("\n--- Latest Values for Reward Components ---")
    for tag in target_tags:
        if tag in scalars:
            events = ea.Scalars(tag)
            print(f"{tag:30s}: Step {events[-1].step}, Value {events[-1].value:.4f}")
        
if __name__ == '__main__':
    main()
