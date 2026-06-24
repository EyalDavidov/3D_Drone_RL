"""Configuration for the Brain-driven Navigation Drone environment.

This environment embeds a pretrained PPO navigator + Brain state machine internally.
It requires an explicit checkpoint path to the trained navigator policy.
"""

import os

# Get the repository root directory
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../../"))

from isaaclab.utils import configclass
from .ae_ppo_drone_env_cfg import AEPPODroneEnvCfg
from isaaclab.scene import InteractiveSceneCfg


@configclass
class BrainNavDroneEnvCfg(AEPPODroneEnvCfg):
    """Config for the self-contained Brain navigation environment.

    Inherits all physical setup (drone, camera, AE, LLC, terrain, obstacles)
    from AEPPODroneEnvCfg and adds Brain-specific parameters.
    """

    # ---------- Scene: Single environment for Brain-controlled mission ----------
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1, env_spacing=6.0, replicate_physics=True
    )

    # ---------- Map ----------
    room_usd_path: str = os.path.join(_REPO_ROOT, "assets", "final_flat.usd")

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
