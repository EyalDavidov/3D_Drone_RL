from first_drone.robots.cf2x import DRONE_CONFIG
import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass


@configclass
class CameraVelocityDroneEnvCfg(DirectRLEnvCfg):
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

    # scene — env_spacing must be >= room size (4m) so rooms don't overlap
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=64, env_spacing=6.0, replicate_physics=True
    )

    # robot
    robot_cfg: ArticulationCfg = DRONE_CONFIG.replace(
        prim_path="/World/envs/env_.*/Drone"
    )

    # room USD (we provide a minimal floor USD in the task assets directory)
    room_usd_path: str = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "assets", "floor.usd")
    )

    # ---------- Spaces ----------
    # Actor (policy) action commands: motor-level controls (thrust, moment_x, moment_y, moment_z)
    action_space = 4          # [thrust, moment_x, moment_y, moment_z]
    # Policy (for velocity-controller task) receives desired velocity target: [vx, vy, vz, yaw_rate]
    observation_space = [4]
    # Critic sees the 12-dim body-frame state vector
    state_space = 12

    # ---------- Physics tuning parameters ----------
    thrust_to_weight = 1.9
    moment_scale = 0.01

    # ---------- Reward scales (reuse from camera config) ----------
    goal_radius = 0.1
    drone_radius = 0.0707
    pole_radius = 0.15
    pole_positions = [(0.0, 0.0), (0.5,0.0), (1.0, 0.0), (1.5, 0.0), (-0.5, 00.0), (-1.0, 0.0), (-1.5, 0.0)]

    distance_to_goal_reward_scale = 15.0
    died_reward_scale = -50.0
    hit_pole_reward_scale = -100.0
    progress_reward_scale = 50.0
    ang_vel_reward_scale = -0.1
    lin_vel_reward_scale = -0.05
