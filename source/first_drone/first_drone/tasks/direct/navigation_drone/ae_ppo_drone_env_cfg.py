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

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=64, env_spacing=6.0, replicate_physics=True
    )

    # robot
    robot_cfg: ArticulationCfg = DRONE_CONFIG.replace(
        prim_path="/World/envs/env_.*/Drone"
    )

    # room — empty room (pillars are now dynamic RigidObjects)
    room_usd_path: str = os.path.join(_REPO_ROOT, "assets", "Empty_Room.usd")

    # ---------- Dynamic Pillars (Domain Randomization) ----------
    num_pillars: int = 6
    pillar_spawn: sim_utils.CylinderCfg = sim_utils.CylinderCfg(
        radius=0.05,  # Exactly matching the thickness of poles from the trained map
        height=3.0,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.15, 0.15)),
    )
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
    pillar_z: float = 1.5  # pillar center height

    # camera — 128×72 depth as specified in the paper
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Drone/body/Camera",
        height=72,
        width=128,
        data_types=["depth"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
        offset=TiledCameraCfg.OffsetCfg(pos=(0.01, 0.0, 0.015), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
    )

    # ---------- Spaces ----------
    # Policy sees the 45-dim flat vector (AE latent + state features)
    action_space = 4
    observation_space = 45
    state_space = 0

    # ---------- Autoencoder (AE) ----------
    ae_latent_dim: int = 32
    depth_max: float = 5.0  # max depth clamp in meters (room is ~10m, but 5m gives better contrast)
    ae_checkpoint_path: str = os.path.join(_REPO_ROOT, "logs", "vae", "ae_final.pt")

    # ---------- Flight controller ----------
    llc_checkpoint_path: str = os.path.join(_REPO_ROOT, "logs", "rsl_rl", "flight_controller_drone_direct", "Flight_Controller", "exported", "policy.pt")
    vel_limit: tuple[float, float, float] = (1.0, 1.0, 0.5)
    yaw_rate_limit: float = 0.15

    # ---------- Visualization ----------
    show_ae_images: bool = False
    ae_image_display_interval: int = 100

    # ---------- Physics tuning ----------
    thrust_to_weight = 1.9
    moment_scale = 0.01

    # =========================================================================
    # REWARDS & CRASH VALUES (Matching Phase 2 navigation_drone config exactly)
    # =========================================================================
    w_progress: float = 20.0
    w_goal: float = 300.0
    w_time: float = -0.06
    w_heading: float = 0.15
    w_vel_align: float = 0.5
    vel_align_max_speed: float = 1.0
    collision_penalty: float = -50.0
    w_ang_vel: float = -0.01
    w_yaw_rate: float = -0.1
    w_forward_speed: float = 0.3
    w_action: float = -0.005
    w_action_rate: float = -0.02
    w_sideslip: float = -0.2
    w_proximity: float = 1.5
    pillar_proximity_radius: float = 0.5
    goal_radius: float = 0.25
    pillar_collision_radius: float = 0.15

    spawn_y_offset: float = 1.0
    corner_fine_tune: bool = False
    opposite_wall_fine_tune: bool = False
    corner_margin: float = 0.2
    corner_goal_z: float = 1.0
