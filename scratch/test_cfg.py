import sys
sys.path.append(r"d:\isaac\3D_Drone_RL\source\first_drone")

from first_drone.tasks.direct.navigation_drone.vae_sac_drone_env_cfg import SACDroneEnvCfg

cfg = SACDroneEnvCfg()
print("Has vae_checkpoint_path:", hasattr(cfg, 'vae_checkpoint_path'))
if hasattr(cfg, 'vae_checkpoint_path'):
    print("Value:", cfg.vae_checkpoint_path)
