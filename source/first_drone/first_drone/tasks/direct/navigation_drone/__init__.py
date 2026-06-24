import gymnasium as gym

from . import agents
from .navigation_drone_env import NavigationDroneEnv
from .navigation_drone_env_cfg import NavigationDroneEnvCfg
from .vae_sac_drone_env import SACDroneEnv
from .vae_sac_drone_env_cfg import SACDroneEnvCfg
from .ae_ppo_drone_env import AEPPODroneEnv
from .ae_ppo_drone_env_cfg import AEPPODroneEnvCfg
from .brain_nav_drone_env import BrainNavDroneEnv
from .brain_nav_drone_env_cfg import BrainNavDroneEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="Drone-Navigation-Direct-v0",
    entry_point="first_drone.tasks.direct.navigation_drone.navigation_drone_env:NavigationDroneEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": NavigationDroneEnvCfg,
        "rlgames_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:NavigationPPOCfg",
    },
)

gym.register(
    id="VAE-SAC-Drone-Direct-v0",
    entry_point=f"{__name__}.vae_sac_drone_env:SACDroneEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.vae_sac_drone_env_cfg:SACDroneEnvCfg",
        "sac_cfg_entry_point": f"{agents.__name__}.sac_cfg:SACAgentCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:NavigationPPOCfg",
    },
)

gym.register(
    id="AE-PPO-Drone-Direct-v0",
    entry_point=f"{__name__}.ae_ppo_drone_env:AEPPODroneEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ae_ppo_drone_env_cfg:AEPPODroneEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:NavigationPPOCfg",
    },
)

gym.register(
    id="Brain-Nav-Drone-Direct-v0",
    entry_point=f"{__name__}.brain_nav_drone_env:BrainNavDroneEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.brain_nav_drone_env_cfg:BrainNavDroneEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:NavigationPPOCfg",
    },
)
