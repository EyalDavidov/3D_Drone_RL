from first_drone.robots.cf2x import DRONE_CONFIG

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

@configclass
class FlightControllerDroneEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 10.0
    debug_vis = True

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # terrain — ground plane
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # scene — env_spacing
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=64, env_spacing=3.0, replicate_physics=True
    )

    # robot
    robot_cfg: ArticulationCfg = DRONE_CONFIG.replace(
        prim_path="/World/envs/env_.*/Drone"
    )

    # ---------- Spaces ----------
    action_space = 4          # [thrust, moment_x, moment_y, moment_z]
    observation_space = 13    # policy_obs (4) + imu (9)
    state_space = 13          # Lin_vel(3) + Ang_vel(3) + Gravity(3) + Desired_vel(3) + yaw_err(1)

    # ---------- Physics tuning parameters ----------
    thrust_to_weight = 1.9
    moment_scale = 0.01

    # ---------- Reward scales (reuse from camera config) ----------
 
    died_reward_scale = -50.0
    
    ang_vel_reward_scale = -0.001
    
    action_rate_reward_scale = -0.5

    vel_match_reward_scale = 5.0  # Positive reward (Gaussian)
    yaw_match_reward_scale = 2.0  # Positive reward (Gaussian)