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

    # =========================================================================
    # HISTORICAL REWARD CONFIGURATIONS (For Documentation & Fine-Tuning Reference)
    # =========================================================================
    #
    # --- PHASE 1: Forced Forward Flight (Overcoming Hesitation) ---
    # * Physics: Action[0] clamped to [0,1] (no reverse). Action[1] zeroed (no strafe).
    # * Goal: To teach the drone that moving forward is better than doing nothing.
    # w_progress = 15.0, w_goal = 50.0, w_time = -0.02, w_heading = 0.25
    # collision = -20.0, goal_radius = 0.4
    # Result: Drone learned to fly forward, but was rigid and couldn't dodge tight pillars.
    #
    # --- PHASE 2: 6-DOF Release & Reward Hacking ---
    # * Physics: Full 6-DOF restored (strafe and reverse allowed).
    # * Goal: Allow agile dodges (strafing) while enforcing forward-facing with heavy heading reward.
    # w_progress = 20.0, w_goal = 100.0, w_time = -0.04, w_heading = 1.0 (DANGER)
    # w_sideslip = -0.2, collision = -30.0, goal_radius = 0.15
    # Result: Reward Hacking! Drone learned to hover right outside the 0.15m goal radius,
    # milking the massive +1.0/step heading reward for 500 steps instead of finishing the episode.
    # =========================================================================

    # ---------- Reward scales (Phase 2.1 — Anti-Hacking Fix / CURRENT) ----------
    #
    # Episode budget (500 steps, ~2.5m to goal):
    #   progress: 20 * 2.5 = +50       (dense drive toward goal)
    #   goal:     300 * 1  = +300       (DOMINANT — entering goal is the ultimate prize)
    #   heading:  0.1 * 500 = +50 max   (guidance only — can't get rich from staring)
    #   time:     -0.04 * 500 = -20     (anti-hesitation)
    #   sideslip: -0.2 * v² ≈ -10      (soft lateral damping)
    #   collision: -30 * 1 = -30        (hurts but doesn't paralyze)
    #
    # Scenarios:
    #   SUCCESS (fast, 100 steps):  +50 +300 +10 -4 -2  = +354  ← BEST by far
    #   HOVER+STARE (500 steps):   +40   +0 +50 -20 -5  = +65   ← can't hack anymore
    #   CRASH (trying, 100 steps): +25   +0 +10 -4 -30  = +1    ← still better than nothing
    #   DO NOTHING (500 steps):     0   +0   0  -20  0  = -20   ← worst
    #
    w_progress: float = 20.0
    w_goal: float = 300.0         # TRIPLED — entering goal is 6x better than staring
    w_time: float = -0.04
    w_heading: float = 0.1        # guidance signal only — 90% cut from Phase 2
    collision_penalty: float = -30.0
    # Stabilization + lateral damping
    w_ang_vel: float = -0.005
    w_action: float = -0.005
    w_sideslip: float = -0.2      # soft lateral penalty — allows quick dodges
    # Goal radius (meters) — reasonable size for catching at speed
    goal_radius: float = 0.25
    # Pillar termination radius (meters) — circular distance from pillar center
    pillar_collision_radius: float = 0.15  # accounts for drone radius (~0.15) + pillar radius (~0.10)
