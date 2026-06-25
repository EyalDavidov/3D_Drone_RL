"""Configuration for the Brain-driven Navigation Drone environment.

This environment embeds a pretrained PPO navigator + Brain state machine internally.
It requires an explicit checkpoint path to the trained navigator policy.
"""

import os

# Get the repository root directory
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../../"))

import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from isaaclab.terrains import TerrainImporterCfg
from .ae_ppo_drone_env_cfg import AEPPODroneEnvCfg
from isaaclab.scene import InteractiveSceneCfg


@configclass
class BrainNavDroneEnvCfg(AEPPODroneEnvCfg):
    """Config for the self-contained Brain navigation environment.

    Inherits all physical setup (drone, camera, AE, LLC, terrain, obstacles)
    from AEPPODroneEnvCfg and adds Brain-specific parameters.
    """

    # ---------- Scene: Single environment for Brain-controlled mission ----------
    # replicate_physics=False avoids cloning/replicating envs (prevents "new stage opened" resets in play).
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1, env_spacing=6.0, replicate_physics=False
    )

    # ---------- Map (final_flat.usd) ----------
    # Use meter-scale map as authored; do not reuse legacy FPS-arena obstacle boxes.
    map_scale: float = 1.0
    # Hard wall envelope for Brain planner — full final_flat.usd world AABB (env-local meters).
    map_bounds: tuple = (-8.55, 4.05, -23.05, 2.05)  # min_x, max_x, min_y, max_y
    # Optional interior no-go boxes: (min_x, max_x, min_y, max_y) each
    map_obstacles: tuple = ()

    room_usd_path: str = os.path.join(_REPO_ROOT, "assets", "final_flat.usd")

    # final_flat.usd is authored in meters (Z-up). Do NOT apply the legacy 0.01-scale
    # + 90°-rotation transform used for the old FPS arena USD.
    use_direct_room_spawn: bool = True
    room_spawn_scale: tuple = (1.0, 1.0, 1.0)
    room_spawn_translation: tuple = (0.0, 0.0, 0.0)
    room_spawn_orientation: tuple = (1.0, 0.0, 0.0, 0.0)

    # ---------- Terrain: no visible default Isaac Lab ground plane ----------
    # final_flat.usd provides floor/walls/collision. Keep terrain only for env_origins bookkeeping.
    terrain: TerrainImporterCfg = TerrainImporterCfg(
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
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.0, 0.0, 0.0),
            opacity=0.0,
        ),
    )

    # ---------- Pretrained Navigator Policy (REQUIRED — no auto-discovery) ----------
    navigator_checkpoint_path: str = ""  # Must be set explicitly (path to exported/policy.pt or model_*.pt)

    # ---------- Perception ----------
    use_mock_perception: bool = False  # False = real YOLO detection; True = mock (no detection)
    yolo_person_conf_threshold: float = 0.95  # Only trigger rescue on high-confidence person (class 0)
    yolo_min_bbox_area_frac: float = 0.008    # Ignore tiny false-positive boxes (<0.8% of image)

    # ---------- Walkable floor (R-shaped map) ----------
    walkable_grid_resolution: float = 0.4   # Meters per occupancy cell when parsing floor meshes from USD
    walkable_min_cells: int = 80            # Reject grid if smaller — fall back to room AABB
    walkable_min_area_m2: float = 20.0      # Reject grid if footprint area is too small

    # ---------- Brain Module Parameters ----------
    brain_step_size: float = 10.0       # Lawnmower corridor spacing in meters
    brain_safety_margin: float = 0.7    # Wall clearance for waypoint generation in meters
    brain_yolo_interval: int = 1        # Run YOLO every N steps (1 = every step, needed during SCAN spin)

    # In play mode, reset the drone after a crash so it can continue the mission.
    brain_reset_on_crash: bool = True
    brain_disable_episode_timeout: bool = True  # Brain play runs until rescue or manual exit
    brain_preserve_mission_on_crash: bool = False  # Sequential mission: crash → restart at spawn1

    # ---------- Sequential SLAM mission (scan each room, then next spawn) ----------
    brain_use_sequential_spawns: bool = True
    brain_auto_room_spawns: bool = True  # Measure room_1..N centers from final_flat.usd at load time
    brain_spawn_sequence: tuple = (
        (0.0, 0.0, 1.0),        # room_1 center (fallback if auto-parse fails)
        (0.0, -5.03, 1.0),      # room_2
        (0.0, -12.01, 1.0),     # room_3
        (-6.50, -20.50, 1.0),   # room_4
        (-6.50, -21.70, 1.0),   # finish (south end of room_4)
    )
    brain_spawn_arrival_radius: float = 1.0   # meters to count as "arrived" at next spawn/finish
    brain_coverage_mark_radius: float = 2.0   # meters to mark occupancy cells as visited after scan

    # ---------- Curriculum: Fixed at Level 5 (fully trained policy) ----------
    initial_curriculum_level: int = 5

    # ---------- Episode ----------
    episode_length_s: float = 3600.0    # Long run for full search missions (timeout disabled in play)
    debug_vis: bool = True
    show_ae_images: bool = False

    # ---------- Person spawning ----------
    spawn_person: bool = True

    # ---------- No dynamic obstacles for brain navigation ----------
    num_pillars: int = 0
