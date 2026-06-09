import os
from tensorboard.backend.event_processing import event_accumulator

def main():
    log_dir = r"d:\isaac\3D_Drone_RL\logs\rsl_rl\navigation_drone_direct\09-06_01-06"
    if not os.path.exists(log_dir):
        print(f"Log directory not found: {log_dir}")
        return

    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()

    scalars = ea.Tags().get('scalars', [])
    
    print(f"--- Learning Rate and KL divergence ---")
    
    # Let's find learning rate tag
    lr_tag = None
    for tag in ['Loss/learning_rate', 'Learning_Rate/value', 'Loss/learning_rate']:
        if tag in scalars:
            lr_tag = tag
            break
            
    if lr_tag:
        events = ea.Scalars(lr_tag)
        print(f"Total steps: {len(events)}")
        print("Step | Learning Rate")
        print("-" * 30)
        for idx in range(0, len(events), max(1, len(events) // 10)):
            print(f"{events[idx].step:<4d} | {events[idx].value:.6f}")
        print(f"{events[-1].step:<4d} | {events[-1].value:.6f}")
    else:
        print("Learning rate tag not found in scalars:", scalars)

if __name__ == '__main__':
    main()
