from first_drone.robots.cf2x import DRONE_CONFIG

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass


@configclass
class CameraFirstDroneEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 10.0
    # - spaces definition
    action_space = 4          # [thrust, moment_x, moment_y, moment_z]
    observation_space = 12    # [lin_vel_b(3), ang_vel_b(3), projected_gravity_b(3), goal_pos_b(3)]
    state_space = 0
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

    # room with poles — spawned as a static USD prim per-env (no RigidBodyAPI needed)
    room_usd_path: str = "C:\\Isaac\\Assets\\room_with_poles.usd"

    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Drone/body/Camera",
        height=100,
        width=100,
        data_types=["depth"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
        offset=TiledCameraCfg.OffsetCfg(pos=(0.01, 0.0, 0.015), rot=(0.5, -0.5, 0.0, 0.0), convention="ros"),
    )

    # ---------- Physics tuning parameters ----------
    # thrust_to_weight: ratio of max thrust to drone weight.
    #   - 1.0 = can barely hover, >1.0 = can accelerate upward
    #   - Higher values make the drone more agile but harder to control.
    thrust_to_weight = 1.9

    # moment_scale: scale factor for roll/pitch/yaw torques.
    #   - Controls how responsive the drone is to rotational commands.
    #   - Increase for more agile rotation, decrease for smoother flight.
    moment_scale = 0.01

    # ---------- Reward scales ----------
    # These control the relative importance of each reward term.
    # Negative = penalty, Positive = reward.
    #
    # lin_vel_reward_scale: penalizes high linear velocities (encourages smooth flight)
    lin_vel_reward_scale = -0.05
    # ang_vel_reward_scale: penalizes high angular velocities (discourages spinning)
    ang_vel_reward_scale = -0.01
    # distance_to_goal_reward_scale: rewards being close to the goal position
    distance_to_goal_reward_scale = 15.0