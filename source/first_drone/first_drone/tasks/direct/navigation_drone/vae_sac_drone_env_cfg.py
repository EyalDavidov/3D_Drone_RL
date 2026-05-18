"""Configuration for the SAC+VAE drone environment.

Camera is set to 128×72 as specified in the paper.
Observation space is 45-dim flat vector (after VAE encoding + state concat),
so the RL algorithm sees a simple MLP-friendly input.
"""

import os

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

    # room — empty room (pillars are now dynamic RigidObjects)
    room_usd_path: str = "D:\\isaac\\3D_Drone_RL\\assets\\Empty_Room.usd"

    # ---------- Dynamic Pillars (Domain Randomization) ----------
    num_pillars: int = 0
    pillar_spawn: sim_utils.CylinderCfg = sim_utils.CylinderCfg(
        radius=0.05,  # Exactly matching the thickness of poles from the trained map
        height=3.0,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.15, 0.15)),
    )
    # Zone-based X randomization: 6 tight zones within the room (±2.5m walls)
    # With pillar radius 0.05, these zones create dynamic, tight but always passable gaps.
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

    # =========================================================================
    # CURRICULUM LEARNING (Toggle between Phase 1 and Phase 2)
    # =========================================================================
    
    # --- PHASE 1: Forced Forward Flight (Overcoming Hesitation) ---
    # * Physics (in env.py): Action[0] clamped to [0,1] (no reverse). Action[1] zeroed (no strafe).
    # * Goal: Teach the drone that moving forward is better than doing nothing, despite randomized pillars.
    # UNCOMMENT THIS BLOCK FOR PHASE 1:
    # w_progress: float = 15.0
    # w_goal: float = 50.0
    # w_time: float = -0.02
    # w_vel_align: float = 0.25
    # vel_align_max_speed: float = 1.0
    # collision_penalty: float = -20.0
    # w_ang_vel: float = -0.005
    # w_action: float = -0.005
    # w_sideslip: float = -0.0
    # w_proximity: float = 1.0
    # pillar_proximity_radius: float = 0.5
    # goal_radius: float = 0.40
    # pillar_collision_radius: float = 0.15

    # --- PHASE 2: 6-DOF Release (Agile Navigation & Dodging) ---
    # * Physics (in env.py): Full 6-DOF restored (strafe and reverse allowed).
    # * Goal: Full freedom to dodge while highly prioritizing goal arrival. Softer crash penalty to prevent freezing.
    w_progress: float = 3.0               # Dense progress reward toward goal
    w_goal: float = 100.0                  # Strong terminal reward for reaching goal
    collision_penalty: float = -100.0      # Terminal penalty for crashing
    w_vel_align: float = 0.5              # Velocity alignment reward (replaces heading)
    vel_align_max_speed: float = 1.0      # Speed normalization constant (matches vel_limit[0])
    w_proximity: float = 1.0              # Proximity penalty weight (pillar danger zone)
    pillar_proximity_radius: float = 0.5  # Soft-zone outer boundary in meters
    w_time: float = -0.05                 # Per-step time penalty
    w_ang_vel: float = 0.0                # Discourage constant spinning (disabled)
    goal_radius: float = 0.40
    pillar_collision_radius: float = 0.15

    # ---------- Curriculum Spawn (Phase 1) ----------
    # Move drone spawn Y closer to target Y = -1.0. (Distance = |spawn_y_offset - (-1.0)|)
    # Default Phase 1 was 1.0 (2.0 meters away). We set it closer to target for early training.
    spawn_y_offset: float = -0.5
    # =========================================================================
