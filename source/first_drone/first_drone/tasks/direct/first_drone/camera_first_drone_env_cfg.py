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

    # camera — body-mounted depth sensor
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Drone/body/Camera",
        height=50,
        width=50,
        data_types=["depth"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 5.0)
        ),
        offset=TiledCameraCfg.OffsetCfg(pos=(0.01, 0.0, 0.015), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
    )

    # ---------- Spaces ----------
    # Actor (policy) sees the depth image: (H, W, 1)
    action_space = 4          # [thrust, moment_x, moment_y, moment_z]
    observation_space = [tiled_camera.height, tiled_camera.width, 1]
    # Critic sees the 12-dim body-frame state vector (asymmetric actor-critic)
    state_space = 12

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
    # True if distance to goal is less than this
    goal_radius = 0.1
    # Drone's circumscribed radius (10x10cm square -> ~7.07cm from center)
    drone_radius = 0.0707
    # Virtual bound indicating collision with the poles
    pole_radius = 0.15
    # List of (x, y) coordinates for all poles in the USD room
    pole_positions = [(0.0, 0.0), (0.5,0.0), (1.0, 0.0), (1.5, 0.0), (-0.5, 00.0), (-1.0, 0.0), (-1.5, 0.0)]


    # distance_to_goal_reward_scale: gives smooth continuous reward based on proximity (1 - tanh(dist/1.6))
    distance_to_goal_reward_scale = 15.0
    # died_reward_scale: one-time penalty when the drone crashes into floor/ceiling/walls
    died_reward_scale = -50.0
    # survive_reward_scale: constant positive reward given for staying alive
    survive_reward_scale = 1.0
    # ang_vel_reward_scale: penalizes high angular velocities (discourages spinning/wobbling)
    ang_vel_reward_scale = -0.1
    # lin_vel_reward_scale: penalizes high linear velocities (encourages smooth flight)
    lin_vel_reward_scale = -0.05