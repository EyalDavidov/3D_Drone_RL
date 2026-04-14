"""Configuration for the SAC+VAE drone environment.

Camera is set to 128×72 as specified in the paper.
Observation space is 45-dim flat vector (after VAE encoding + state concat),
so the RL algorithm sees a simple MLP-friendly input.
"""

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
    room_usd_path: str = "C:\\Isaac\\Assets\\room_with_poles.usd"

    # camera — 128×72 depth as specified in the paper
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Drone/body/Camera",
        height=72,
        width=128,
        data_types=["depth"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
        offset=TiledCameraCfg.OffsetCfg(pos=(0.01, 0.0, 0.015), rot=(0.5, -0.5, 0.0, 0.0), convention="ros"),
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
    vae_lr: float = 1e-4
    depth_max: float = 10.0  # max depth clamp in meters

    # ---------- SAC ----------
    sac_actor_lr: float = 3e-4
    sac_critic_lr: float = 3e-4
    sac_alpha_lr: float = 3e-4
    sac_gamma: float = 0.99
    sac_tau: float = 0.005
    sac_batch_size: int = 256
    sac_replay_size: int = 200_000
    sac_warmup_steps: int = 10_000
    sac_update_every: int = 64       # update SAC every N env steps
    sac_gradient_steps: int = 4      # number of gradient updates per update phase
    sac_success_ratio: float = 0.25  # fraction of batch from success buffer
    sac_max_iterations: int = 500_000  # total env steps

    # ---------- Physics tuning ----------
    thrust_to_weight = 1.9
    moment_scale = 0.01

    # ---------- Reward scales ----------
    # Progress reward: getting closer to goal
    w_progress: float = 2.0
    # Goal reached bonus (huge one-time termination reward)
    w_goal: float = 50.0
    # Hover bonus (disabled for now)
    w_hover: float = 0.0
    # Depth clearance (center-vs-mean depth)
    w_clearance: float = 0.3
    # Angular velocity penalty
    w_ang_vel: float = 0.02
    # Tilt penalty
    w_tilt: float = 0.05
    # Action magnitude penalty
    w_action: float = 0.01
    # Collision penalty
    collision_penalty: float = -8.0
    # Goal radius (meters) — drone is "at goal" when closer than this
    goal_radius: float = 0.4

    # ---------- Logging ----------
    wandb_project: str = "first_drone"
    experiment_name: str = "sac_drone"
    save_interval: int = 10_000  # save model every N env steps
