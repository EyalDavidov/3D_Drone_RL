from first_drone.robots.cf2x import DRONE_CONFIG
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

@configclass
class NavigationDroneEnvCfg(DirectRLEnvCfg):
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

    # ---------- Spaces ----------
    # High-Level Agent Action: [vx_body, vy_body, vz_body, target_yaw_rate]
    action_space = 4          
    # High-Level Agent Obs: pos_err_w(3) + lin_vel_w(3) + distance(1)
    observation_space = 7     
    state_space = 0           

    # ---------- Reward scales ----------
    progress_reward_scale = 10.0
    reached_goal_reward = 200.0
    died_reward_scale = -50.0
    ang_vel_reward_scale = -0.05

    # ---------- Navigation Settings ----------
    goal_radius = 0.1

    # ---------- Low Level Integration ----------
    llc_checkpoint_path = r"C:\Isaac\Projects\first_drone\logs\rsl_rl\flight_controller_drone_direct\Flight_Controller\exported\policy.pt"
    thrust_to_weight = 1.9
    moment_scale = 0.01
