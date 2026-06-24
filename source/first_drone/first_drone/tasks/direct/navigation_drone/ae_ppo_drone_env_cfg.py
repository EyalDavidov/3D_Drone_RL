"""Configuration for the PPO+AE drone environment.

Camera is set to 128×72 as specified in the paper.
Observation space is 45-dim flat vector (after AE encoding + state concat).
"""

import os

# Get the repository root directory
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../../"))

from first_drone.robots.cf2x import DRONE_CONFIG

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass


@configclass
class AEPPODroneEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 30.0
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

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=512, env_spacing=6.0, replicate_physics=True
    )

    # robot
    robot_cfg: ArticulationCfg = DRONE_CONFIG.replace(
        prim_path="/World/envs/env_.*/Drone"
    )
    # Enable contact sensors on the drone spawner
    robot_cfg.spawn.activate_contact_sensors = True

    # room — empty room (pillars are now dynamic RigidObjects)
    room_usd_path: str = os.path.join(_REPO_ROOT, "assets", "final_flat.usd")

    # ---------- Dynamic Obstacles (Domain Randomization) ----------
    # 6 diverse obstacle shapes — defined in env._setup_scene() to avoid @configclass serialization issues
    num_pillars: int = 0
    # Zone-based X randomization: 6 tight zones within the room (±2.5m walls)
    pillar_x_zones: tuple = (
        (-1.80, -1.40),  # Zone 0
        (-1.10, -0.70),  # Zone 1
        (-0.40,  0.00),  # Zone 2
        ( 0.20,  0.60),  # Zone 3
        ( 0.80,  1.20),  # Zone 4
        ( 1.40,  1.80),  # Zone 5
    )
    # Y jitter to make straight lines impossible
    pillar_y_range: tuple = (-0.3, 0.3)
    pillar_z: float = 1.25  # obstacle center height

    # ---------- Arena Map Static Obstacles (2D bounding boxes) ----------
    # format: [min_x, max_x, min_y, max_y] in local env frame
    map_obstacles: tuple = (
        (14.012, 19.012, -2.025, 4.975),
        (4.012, 9.012, 6.975, 12.975),
        (-15.988, -10.988, -21.025, -11.025),
        (0.012, 2.012, -8.025, -1.025),
        (8.012, 10.012, -17.025, -10.025),
        (-7.988, -5.988, 1.975, 8.975),
        (-18.988, -16.988, -4.025, 2.975),
        (-21.988, -16.988, 8.975, 15.975),
        (15.012, 17.012, 11.975, 18.975),
        (-11.318, -2.756, 13.145, 20.975),
        (-1.988, 3.012, -22.025, -16.025),
        (17.012, 19.012, -21.025, -14.025),
        (6.012, 7.012, 18.975, 19.975),
        (-18.988, -17.988, 20.975, 21.975),
        (10.012, 11.012, 1.975, 2.975),
        (18.012, 19.012, -8.025, -7.025),
        (-7.988, -6.988, -6.025, -5.025),
        (-20.988, -19.988, -14.025, -13.025),
        # Outer boundary walls to enable LiDAR scans and proximity penalties to see them
        (-24.988, -23.988, -24.025, 24.975),
        (-23.988, 25.012, 23.975, 24.975),
        (24.012, 25.012, -25.025, 23.975),
        (-24.988, 24.012, -25.025, -24.025),
    )
    spawn_obstacle_margin: float = 0.5  # safety margin around obstacles for spawning

    # camera — 128×72 depth as specified in the paper
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Drone/body/Camera",
        height=72,
        width=128,
        data_types=["depth", "rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
        offset=TiledCameraCfg.OffsetCfg(pos=(0.01, 0.0, 0.015), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
    )

    # ---------- Spaces ----------
    # Policy sees the 49-dim flat vector (AE latent + state features + previous actions)
    action_space = 4
    observation_space = 73
    state_space = 0

    # ---------- Autoencoder (AE) ----------
    ae_latent_dim: int = 32
    depth_max: float = 15.0  # max depth clamp in meters — matches 15m arena AE training
    ae_checkpoint_path: str = os.path.join(_REPO_ROOT, "logs", "ae_arena", "ae_final.pt")

    # ---------- Flight controller ----------
    llc_checkpoint_path: str = os.path.join(_REPO_ROOT, "logs", "rsl_rl", "flight_controller_drone_direct", "Flight_Controller", "exported", "policy.pt")
    vel_limit: tuple[float, float, float] = (1.0, 1.0, 0.5)
    yaw_rate_limit: float = 0.6

    # ---------- Visualization ----------
    show_ae_images: bool = False
    ae_image_display_interval: int = 100

    # ---------- Physics tuning ----------
    thrust_to_weight = 1.9
    moment_scale = 0.01

    # =========================================================================
    # REWARDS & CRASH VALUES (Matching Phase 2 navigation_drone config exactly)
    # =========================================================================
    w_progress: float = 10.0
    w_goal: float = 500.0
    w_time: float = -0.90  # Balanced time penalty: prevents reward loops without triggering premature self-collisions
    w_heading: float = 0.50   # Heading alignment reward to prevent the drone from flying sideways / looking away from the goal
    w_vel_align: float = 1.0
    vel_align_max_speed: float = 1.0
    collision_penalty: float = -400.0  # Increased from -200 to heavily penalize crashes compared to timeout
    w_ang_vel: float = -0.02
    w_yaw_rate: float = -0.05
    w_forward_speed: float = 0.5
    w_action: float = -0.005  # Bug #3 fix: -0.03 was 6x too aggressive, suppressing necessary agile maneuvers
    w_action_rate: float = -0.02  # Smoothes out commands, set to match Orange Run settings
    w_sideslip: float = -3.0  # Strong lateral penalty: at 0.5m/s lateral vel, -0.75/step forces head-on flight
    w_proximity: float = 1.5  # Bug #2 fix: 2.0 was too strong with radius=0.5, 1.5 gives proportional gradient
    pillar_proximity_radius: float = 0.5  # Crucial: 0.5m allows a free 60cm corridor in the center for 1.6m scaled corridors
    w_speed_proximity: float = -4.0
    w_tilt: float = -0.08                  # penalty for excessive roll/pitch tilt (matching Orange Run settings)
    w_z_deviation: float = -0.3           # penalty for floor/ceiling deviation (matching Orange Run settings)
    goal_radius: float = 0.25
    pillar_collision_radius: float = 0.11  # Set to 0.11m (Plenty safe for Crazyflie, but allows navigating narrow corridors)

    # ---------- Dynamic Map Scaling & Altitude Corridor ----------
    map_scale: float = 0.8
    z_low: float = 0.7
    z_high: float = 1.5

    spawn_y_offset: float = 1.0
    corner_fine_tune: bool = False
    opposite_wall_fine_tune: bool = False
    corner_margin: float = 0.2
    corner_goal_z: float = 1.0

    initial_curriculum_level: int = 1
    load_run: str = ""
    spawn_person: bool = False




