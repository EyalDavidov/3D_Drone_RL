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

    # ---------- Map ----------
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

    # ---------- Brain Module Parameters ----------
    brain_step_size: float = 10.0       # Lawnmower corridor spacing in meters
    brain_safety_margin: float = 0.7    # Wall clearance for waypoint generation in meters
    brain_yolo_interval: int = 5        # Run YOLO every N steps (during non-SCAN states)

    # ---------- Curriculum: Fixed at Level 5 (fully trained policy) ----------
    initial_curriculum_level: int = 5

    # ---------- Episode ----------
    episode_length_s: float = 120.0     # Long episodes for full search & rescue missions
    debug_vis: bool = True
    show_ae_images: bool = False

    # ---------- Person spawning ----------
    spawn_person: bool = True

    # ---------- No dynamic obstacles for brain navigation ----------
    num_pillars: int = 0
