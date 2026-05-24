from first_drone.robots.cf2x import DRONE_CONFIG
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.sensors import RayCasterCfg, patterns

@configclass
class NavigationDroneLidarEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2  # Run at 50Hz (Exact same frequency the Low-Level Controller was trained at!)
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

    # scene — spacing
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=64, env_spacing=4.0, replicate_physics=True
    )

    # robot
    robot_cfg: ArticulationCfg = DRONE_CONFIG.replace(
        prim_path="/World/envs/env_.*/Drone"
    )

    # ---------- Lidar Sensor ----------
    lidar = RayCasterCfg(
        prim_path="/World/envs/env_.*/Drone/body",
        update_period=0.02,

        offset=RayCasterCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.1)
        ),

        attach_yaw_only=False,

        pattern_cfg=patterns.LidarPatternCfg(
            channels=1,
            vertical_fov_range=(0.0, 0.0),
            horizontal_fov_range=(-90.0, 90.0),
            horizontal_res=2.0,
        ),

        max_distance=3.0,

        debug_vis=True,

        mesh_prim_paths=[
            "/World/ground",
        ],
    )

    # ---------- Spaces ----------
    # High-Level Agent Action: [vx_body, vy_body, vz_body, target_yaw_rate]
    action_space = 4
    # High-Level Agent Obs: pos_err_b(3) + lin_vel_b(3) + distance(1) + lidar_distances(91)
    # Lidar: horizontal_fov=180deg / horizontal_res=2.0deg + 1 = 91 rays
    num_lidar_rays = 91
    observation_space = 7 + num_lidar_rays  # = 98
    state_space = 0

    # ---------- Reward scales ----------
    progress_reward_scale = 10.0
    reached_goal_reward = 200.0
    died_reward_scale = -50.0
    ang_vel_reward_scale = -0.05

    # ---------- Navigation Settings ----------
    goal_radius = 0.1

    # ---------- Low Level Integration ----------
    llc_checkpoint_path = r"./logs/rsl_rl/flight_controller_drone_direct/Flight_Controller/exported/policy.pt"
    thrust_to_weight = 1.9
    moment_scale = 0.01
