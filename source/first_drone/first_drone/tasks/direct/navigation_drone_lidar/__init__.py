import gymnasium as gym

from . import agents
from .navigation_drone_lidar_env import NavigationDroneLidarEnv
from .navigation_drone_lidar_env_cfg import NavigationDroneLidarEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="Drone-Navigation-Lidar-Direct-v0",
    entry_point="first_drone.tasks.direct.navigation_drone_lidar.navigation_drone_lidar_env:NavigationDroneLidarEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": NavigationDroneLidarEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:NavigationLidarPPOCfg",
    },
)
