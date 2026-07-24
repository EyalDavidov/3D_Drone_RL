import sys
import os

from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": True})

import torch
from first_drone.tasks.direct.navigation_drone.real_slam.real_slam_env import RealSlamDroneEnv
from first_drone.tasks.direct.navigation_drone.real_slam.real_slam_env_cfg import RealSlamDroneEnvCfg
from scripts.dashboard.live_telemetry import LiveDroneTelemetry

cfg = RealSlamDroneEnvCfg()
cfg.scene.num_envs = 1
env = RealSlamDroneEnv(cfg=cfg)

telemetry = LiveDroneTelemetry(tick_rate=24.0, recording=True, lightweight_recording=True)
print("Testing telemetry push...")
try:
    telemetry.push(env, 0.1)
    print("Push succeeded!")
except Exception as e:
    import traceback
    print("Push failed with exception:")
    traceback.print_exc()

telemetry.close()
env.close()
simulation_app.close()
