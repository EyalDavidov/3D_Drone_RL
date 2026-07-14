"""Configuration for the Multi-Level PPO+AE drone navigation environment.

4 levels from final.usd. Each episode trains on a random level.
Level targets are the spawn points of the next level.
Levels 1-2: 10s, Levels 3-4: 25s.
"""

import os

# Get the repository root directory
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../../"))

from first_drone.robots.cf2x import DRONE_CONFIG

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.sensors import TiledCameraCfg, ContactSensorCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass


@configclass
class MultiLevelDroneEnvCfg(DirectRLEnvCfg):
    # env — max duration is 25s (levels 3-4); shorter levels handled in env code
    decimation = 2
    episode_length_s = 25.0
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

    # scene — env_spacing=30 to fit the full multi-level arena (~22m span)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=64, env_spacing=30.0, replicate_physics=True
    )

    # robot — with contact sensors enabled for collision detection
    robot_cfg: ArticulationCfg = DRONE_CONFIG.replace(
        prim_path="/World/envs/env_.*/Drone",
        spawn=DRONE_CONFIG.spawn.replace(
            activate_contact_sensors=True,
        ),
    )

    # room — multi-level arena
    room_usd_path: str = os.path.join(_REPO_ROOT, "assets", "final_roof_flat.usd")

    # camera — 128×72 depth
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Drone/body/Camera",
        height=72,
        width=128,
        data_types=["depth"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=7.5, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
        offset=TiledCameraCfg.OffsetCfg(pos=(0.01, 0.0, 0.015), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
    )
 
    # view camera — for viewport viewing
    from isaaclab.sensors import CameraCfg
    view_camera: CameraCfg = CameraCfg(
        prim_path="/World/envs/env_0/Drone/body/Camera_View",
        update_period=0.0,  # We only need it for the viewport, not for RL tensors
        height=720,
        width=1280,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=15.5, 
            focus_distance=400.0, 
            horizontal_aperture=33.0, 
            clipping_range=(0.1, 100000.0)
        ),
        # Quaternion (W, X, Y, Z) exactly as requested
        offset=CameraCfg.OffsetCfg(pos=(-0.15, 0.0, 0.05), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
    )

    # ---------- Spaces ----------
    action_space = 4
    observation_space = 77
    state_space = 0

    # ---------- AE (Autoencoder) ----------
    ae_latent_dim: int = 64
    depth_max: float = 5.0
    ae_checkpoint_path: str = os.path.join(_REPO_ROOT, "logs", "ae_1_7_latent_64", "ae_final.pt")

    # ---------- Flight controller ----------
    llc_checkpoint_path: str = os.path.join(
        _REPO_ROOT, "logs", "rsl_rl", "flight_controller_drone_direct",
        "Flight_Controller", "exported", "policy.pt"
    )
    vel_limit: tuple[float, float, float] = (1.0, 1.0, 0.5)
    yaw_rate_limit: float = 0.5

    # ---------- Visualization ----------
    show_ae_images: bool = False
    ae_image_display_interval: int = 100

    # ---------- Physics tuning ----------
    thrust_to_weight = 1.9
    moment_scale = 0.01

    # =========================================================================
    # MULTI-LEVEL LAYOUT
    # =========================================================================
    # Spawn and target positions for each level (in local env frame)
    # Target of level N = spawn of level N+1 (except last → finish)
    level_spawns: tuple = (
        (0.0,   1.5,  1.0),   # Level 1 spawn
        (0.0,  -2.5,  1.0),   # Level 2 spawn
        (0.0,  -8.5,  1.0),   # Level 3 spawn
        (0.0, -16.0,  1.0),   # Level 4 spawn
    )
    level_targets: tuple = (
        (0.0,  -2.5,  1.0),   # Level 1 target = Level 2 spawn
        (0.0,  -8.5,  1.0),   # Level 2 target = Level 3 spawn
        (0.0, -16.0,  1.0),   # Level 3 target = Level 4 spawn
        (-5.0, -20.5, 1.0),   # Level 4 target = finish
    )
    level_durations: tuple[float, float, float, float] = (7.0, 10.0, 15.0, 15.0)
    num_levels: int = 4
    force_level: int | None = None
    continuous_mode: bool = False

    # ---------- Random Poles ----------
    num_poles: int = 21
    pole_spawn: sim_utils.UsdFileCfg = sim_utils.UsdFileCfg(
        usd_path=os.path.join(_REPO_ROOT, "assets", "1_Pole.usd"),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
    )

    # ---------- Room 3 Obstacles ----------
    num_room3_walls: int = 0
    num_room3_cones: int = 0
    num_room3_big_gates: int = 0
    num_room3_small_gates: int = 0
    num_room3_poles_triangles: int = 0

    wall_spawn: sim_utils.UsdFileCfg = sim_utils.UsdFileCfg(
        usd_path=os.path.join(_REPO_ROOT, "assets", "obstacles", "wall.usd"),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
    )
    cone_spawn: sim_utils.UsdFileCfg = sim_utils.UsdFileCfg(
        usd_path=os.path.join(_REPO_ROOT, "assets", "obstacles", "cone.usd"),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
    )
    big_gate_spawn: sim_utils.UsdFileCfg = sim_utils.UsdFileCfg(
        usd_path=os.path.join(_REPO_ROOT, "assets", "obstacles", "big_gate.usd"),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
    )
    small_gate_spawn: sim_utils.UsdFileCfg = sim_utils.UsdFileCfg(
        usd_path=os.path.join(_REPO_ROOT, "assets", "obstacles", "small_gate.usd"),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
    )
    poles_triangle_spawn: sim_utils.UsdFileCfg = sim_utils.UsdFileCfg(
        usd_path=os.path.join(_REPO_ROOT, "assets", "obstacles", "poles_triangle.usd"),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
    )
    corr1_spawn: sim_utils.UsdFileCfg = sim_utils.UsdFileCfg(
        usd_path=os.path.join(_REPO_ROOT, "assets", "obstacles", "corr1.usd"),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
    )
    corr2_spawn: sim_utils.UsdFileCfg = sim_utils.UsdFileCfg(
        usd_path=os.path.join(_REPO_ROOT, "assets", "obstacles", "corr2.usd"),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
    )
    num_room4_corr1: int = 0
    num_room4_corr2: int = 0

    # =========================================================================
    # REWARDS & TERMINATION
    # =========================================================================
    w_action: float = -0.005
    w_action_rate: float = -0.1
    w_ang_vel: float = -0.05
    w_distance: float = 3.0
    w_forward: float = 0.0   
    w_goal: float = 400.0 
    w_heading: float = 0.1 
    w_progress: float = 10.0
    w_sideslip: float = -3.0
    w_tilt: float = -0.5    
    w_time: float = -0.05
    w_yaw_rate: float = 0.0

    collision_penalty: float = -300.0
    goal_radius: float = 0.20

    # ---------- Contact sensor (collision detection) ----------
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Drone/body",
        history_length=1
    )
    contact_force_threshold: float = 0.0001  # N — force above this = collision
