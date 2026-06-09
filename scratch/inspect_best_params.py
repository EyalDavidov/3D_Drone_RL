import os

def main():
    best_run_dir = r"d:\isaac\3D_Drone_RL\logs\rsl_rl\navigation_drone_direct\26-05_16-36 BEST MODEL DUCKING FROM CLOSE PILARS"
    params_dir = os.path.join(best_run_dir, "params")
    if not os.path.exists(params_dir):
        print("Params directory not found in best model.")
        return
        
    print("Files in params:")
    for f in os.listdir(params_dir):
        print(f)
        
    # Let's search for collision_penalty or similar inside the python files in params
    import glob
    for py_file in glob.glob(os.path.join(params_dir, "**/*.py"), recursive=True):
        print("\nChecking:", py_file)
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
                for line in content.split('\n'):
                    if 'collision_penalty' in line or 'w_collision' in line:
                        print("  ", line.strip())
        except Exception as e:
            print("  Error reading:", e)

if __name__ == '__main__':
    main()
