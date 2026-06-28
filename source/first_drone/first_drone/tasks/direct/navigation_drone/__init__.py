import gymnasium as gym

from . import agents
from .navigation_drone_env import NavigationDroneEnv
from .navigation_drone_env_cfg import NavigationDroneEnvCfg
from .multilevel_drone_env import MultiLevelDroneEnv
from .multilevel_drone_env_cfg import MultiLevelDroneEnvCfg

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
    id="MultiLevel-Drone-Direct-v0",
    entry_point=f"{__name__}.multilevel_drone_env:MultiLevelDroneEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.multilevel_drone_env_cfg:MultiLevelDroneEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:NavigationPPOCfg",
    },
)

