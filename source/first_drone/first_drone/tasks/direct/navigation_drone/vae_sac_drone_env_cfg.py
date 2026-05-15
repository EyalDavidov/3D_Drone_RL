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
    room_usd_path: str = "D:\\isaac\\3D_Drone_RL\\assets\\room_with_poles.usd"

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
    vae_checkpoint_path: str = r"D:\isaac\3D_Drone_RL\logs\vae\vae_final.pt"

    # ---------- Flight controller ----------
    llc_checkpoint_path: str = r"D:\isaac\3D_Drone_RL\logs\rsl_rl\flight_controller_drone_direct\Flight_Controller\exported\policy.pt"
    vel_limit: tuple[float, float, float] = (1.0, 1.0, 0.5)
    yaw_rate_limit: float = 0.15  # was 0.05 — too slow for the agent to reorient

    # ---------- Visualization ----------
    show_vae_images: bool = False
    vae_image_display_interval: int = 100

    # ---------- Physics tuning ----------
    thrust_to_weight = 1.9
    moment_scale = 0.01

    # ---------- Reward scales ----------
    # Progress reward: getting closer to goal (DOMINANT — drives navigation)
    w_progress: float = 10.0
    # Goal reached bonus (one-time terminal reward — must dominate over cumulative hover)
    w_goal: float = 200.0
    # Time penalty (negative per-step cost — forces the drone to reach goal FAST)
    w_time: float = -0.2
    # Hover bonus (slow down inside goal radius — only inside goal_radius)
    w_hover: float = 0.1  # heavily reduced to prevent any hover exploit
    # Depth clearance (center-vs-mean depth — obstacle awareness)
    w_clearance: float = 0.1  # was 0.5 — too noisy, dominates reward signal
    # Angular velocity penalty
    w_ang_vel: float = -0.002
    # Tilt penalty
    w_tilt: float = -0.01
    # Action magnitude penalty
    w_action: float = -0.01
    # Collision penalty (one-time on crash — must hurt enough to deter wall/pillar scraping)
    collision_penalty: float = -200.0  # severely punish crashing
    # Goal radius (meters) — drone is "at goal" when closer than this
    goal_radius: float = 0.4
    # Pillar termination radius (meters) — circular distance from pillar center
    pillar_collision_radius: float = 0.1  # accounts for drone radius (~0.15) + pillar radius (~0.10)
