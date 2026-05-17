"""Configuration for the SAC+VAE drone environment.

Camera is set to 128×72 as specified in the paper.
Observation space is 45-dim flat vector (after VAE encoding + state concat),
so the RL algorithm sees a simple MLP-friendly input.
"""

import os

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
class SACDroneEnvCfg(DirectRLEnvCfg):
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

    # room
    room_usd_path: str = "C:/Isaac/Projects/first_drone/Assets/room_window.usd"

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
    # Policy sees the 45-dim flat vector (VAE latent + state features)
    # The env returns this as "policy" key after internal VAE encoding
    action_space = 4
    # z_img(32) + target_rel_body(3) + target_dist(1) + lin_vel(3) + ang_vel(3) + gravity(3) = 45
    observation_space = 45
    # No separate state_space needed — SAC uses the same obs for actor and critic
    state_space = 0

    # ---------- VAE ----------
    vae_latent_dim: int = 32
    vae_beta: float = 1e-3
    depth_max: float = 5.0  # max depth clamp in meters (room is ~10m, but 5m gives better contrast)
    vae_checkpoint_path: str = r"Projects\first_drone\logs\vae\vae_final.pt"

    # ---------- Flight controller ----------
    llc_checkpoint_path: str = r"Projects\first_drone\logs\rsl_rl\flight_controller_drone_direct\Flight_Controller\exported\policy.pt"
    vel_limit: tuple[float, float, float] = (1.0, 1.0, 0.5)
    yaw_rate_limit: float = 0.05

    # ---------- Visualization ----------
    show_vae_images: bool = False
    vae_image_display_interval: int = 100

    # ---------- Physics tuning ----------
    thrust_to_weight = 1.9
    moment_scale = 0.01

    # ---------- Reward scales ----------
    # Progress reward: getting closer to goal (DOMINANT — drives navigation)
    w_progress: float = 5.0
    # Goal reached bonus (huge one-time termination reward)
    w_goal: float = 100.0
    # Hover bonus (disabled for now)
    w_hover: float = 0.0
    # Depth clearance (center-vs-mean depth)
    w_clearance: float = 0.5
    # Angular velocity penalty (reduced further — still dominated at -20 in graphs)
    w_ang_vel: float = -0.002
    # Tilt penalty (reduced — was too dominant)
    w_tilt: float = -0.01
    # Action magnitude penalty
    w_action: float = -0.01
    # Collision penalty (increased — must clearly outweigh other penalties)
    collision_penalty: float = -25.0
    # Goal radius (meters) — drone is "at goal" when closer than this
    goal_radius: float = 0.4
