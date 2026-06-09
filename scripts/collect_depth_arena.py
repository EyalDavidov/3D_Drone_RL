"""Depth data collection script — ARENA MAP version.

Spawns a single drone in the FPS Shooter Game Arena Map,
teleports it to random positions around the map with random
yaw angles, and saves every depth frame to disk.

Usage:
    D:\\Isaac\\IsaacLab\\isaaclab.bat -p scripts/collect_depth_arena.py --enable_cameras --num_images 10000
"""

# ── Launch Isaac Sim first ──────────────────────────────────────────
import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Collect depth images from arena map.")
parser.add_argument("--num_images", type=int, default=10_000,
                    help="Number of depth images to collect.")
parser.add_argument("--output_dir", type=str, default=None,
                    help="Output directory for depth images. Defaults to <project>/data/depth_arena.")

AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
sys.argv = [sys.argv[0]]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Imports (after sim launch) ──────────────────────────────────────
import os
import math
import random
import torch
import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.sensors import TiledCamera, TiledCameraCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from first_drone.robots.cf2x import DRONE_CONFIG

# ═══════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════

ARENA_USD_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..",
    "source", "first_drone", "first_drone", "tasks", "direct",
    "first_drone", "assets", "fps_shooter_game_arena_map_v4.usdz"
))
DEPTH_MAX = 15.0
DT = 1.0 / 100.0
DECIMATION = 2
CAMERA_WIDTH = 128
CAMERA_HEIGHT = 72

# 2D bounding boxes of the arena obstacles (local frame, relative to env origin)
# format: [min_x, max_x, min_y, max_y]
MAP_OBSTACLES = [
    [14.012, 19.012, -2.025, 4.975],
    [4.012, 9.012, 6.975, 12.975],
    [-15.988, -10.988, -21.025, -11.025],
    [0.012, 2.012, -8.025, -1.025],
    [8.012, 10.012, -17.025, -10.025],
    [-7.988, -5.988, 1.975, 8.975],
    [-18.988, -16.988, -4.025, 2.975],
    [-21.988, -16.988, 8.975, 15.975],
    [15.012, 17.012, 11.975, 18.975],
    [-11.318, -2.756, 13.145, 20.975],
    [-1.988, 3.012, -22.025, -16.025],
    [17.012, 19.012, -21.025, -14.025],
    [6.012, 7.012, 18.975, 19.975],
    [-18.988, -17.988, 20.975, 21.975],
    [10.012, 11.012, 1.975, 2.975],
    [18.012, 19.012, -8.025, -7.025],
    [-7.988, -6.988, -6.025, -5.025],
    [-20.988, -19.988, -14.025, -13.025],
]
MARGIN = 0.5  # Safety margin around obstacles


@configclass
class ArenaSceneCfg(InteractiveSceneCfg):
    """Minimal scene: 1 env."""
    num_envs = 1
    env_spacing = 6.0
    replicate_physics = True


# ═══════════════════════════════════════════════════════════════════
#  Helper: check if a point is inside any obstacle
# ═══════════════════════════════════════════════════════════════════

def is_inside_obstacle(x: float, y: float) -> bool:
    """Return True if (x, y) is inside any obstacle's bounding box + margin."""
    for obs in MAP_OBSTACLES:
        if (obs[0] - MARGIN) <= x <= (obs[1] + MARGIN) and \
           (obs[2] - MARGIN) <= y <= (obs[3] + MARGIN):
            return True
    return False


def random_free_position() -> tuple[float, float, float]:
    """Generate a random position NOT inside any obstacle."""
    for _ in range(100):
        x = random.uniform(-20.0, 20.0)
        y = random.uniform(-20.0, 20.0)
        if not is_inside_obstacle(x, y):
            z = random.uniform(0.5, 1.8)
            return x, y, z
    # Fallback: center is always free
    return 0.0, 0.0, 1.0


# ═══════════════════════════════════════════════════════════════════
#  Helper: quaternion from roll and yaw
# ═══════════════════════════════════════════════════════════════════

def quat_from_roll_yaw(roll: float, yaw: float) -> tuple[float, float, float, float]:
    """Return (w, x, y, z) quaternion for the drone's base rotation + roll + yaw."""
    bw, bx, by, bz = 0.70710678, 0.0, 0.0, -0.70710678

    hr = roll / 2.0
    rw, rx, ry, rz = math.cos(hr), math.sin(hr), 0.0, 0.0

    tw = bw * rw - bx * rx - by * ry - bz * rz
    tx = bw * rx + bx * rw + by * rz - bz * ry
    ty = bw * ry - bx * rz + by * rw + bz * rx
    tz = bw * rz + bx * ry - by * rx + bz * rw

    extra_yaw = yaw + math.pi / 2.0
    hy = extra_yaw / 2.0
    yw, yx, yy, yz = math.cos(hy), 0.0, 0.0, math.sin(hy)

    w = yw * tw - yx * tx - yy * ty - yz * tz
    x = yw * tx + yx * tw + yy * tz - yz * ty
    y = yw * ty - yx * tz + yy * tw + yz * tx
    z = yw * tz + yx * ty - yy * tx + yz * tw

    return (w, x, y, z)


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    output_dir = args_cli.output_dir or os.path.join(project_root, "data", "depth_arena")
    os.makedirs(output_dir, exist_ok=True)
    print(f"[INFO] Saving depth images to: {output_dir}")
    print(f"[INFO] Arena USD: {ARENA_USD_PATH}")

    # ── Simulation context ──────────────────────────────────────────
    sim_cfg = SimulationCfg(
        dt=DT,
        render_interval=DECIMATION,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[5.0, 5.0, 5.0], target=[0.0, 0.0, 1.0])

    # ── Scene ───────────────────────────────────────────────────────
    scene_cfg = ArenaSceneCfg()
    scene = InteractiveScene(scene_cfg)

    # ── Robot ───────────────────────────────────────────────────────
    robot_cfg: ArticulationCfg = DRONE_CONFIG.replace(
        prim_path="/World/envs/env_.*/Drone"
    )
    robot = Articulation(robot_cfg)
    scene.articulations["robot"] = robot

    # ── Arena Map ───────────────────────────────────────────────────
    room_cfg = sim_utils.UsdFileCfg(usd_path=ARENA_USD_PATH, scale=(0.01, 0.01, 0.01))
    room_cfg.func(
        "/World/envs/env_0/Room",
        room_cfg,
        translation=(25.0, 25.0, -0.9937),
        orientation=(0.7071, 0.7071, 0.0, 0.0),
    )

    # ── Terrain (ground plane) ──────────────────────────────────────
    terrain_cfg = TerrainImporterCfg(
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
    terrain_cfg.num_envs = scene_cfg.num_envs
    terrain_cfg.env_spacing = scene_cfg.env_spacing
    terrain = terrain_cfg.class_type(terrain_cfg)

    # ── Camera ──────────────────────────────────────────────────────
    camera_cfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Drone/body/Camera",
        height=CAMERA_HEIGHT,
        width=CAMERA_WIDTH,
        data_types=["depth"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5),
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.01, 0.0, 0.015),
            rot=(0.5, -0.5, 0.5, -0.5),
            convention="ros",
        ),
    )
    tiled_camera = TiledCamera(camera_cfg)
    scene.sensors["tiled_camera"] = tiled_camera

    # ── Lighting ────────────────────────────────────────────────────
    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    # ── Clone environments & finalize ───────────────────────────────
    scene.clone_environments(copy_from_source=False)

    # ── Reset simulation ────────────────────────────────────────────
    sim.reset()
    scene.reset()

    # ── Physical constants ──────────────────────────────────────────
    body_id = robot.find_bodies("body")[0]
    robot_mass = robot.root_physx_view.get_masses()[0].sum()
    gravity_mag = torch.tensor(sim.cfg.gravity, device=sim.device).norm()
    robot_weight = (robot_mass * gravity_mag).item()
    hover_thrust = robot_weight

    env_origin = terrain.env_origins[0]  # (3,)

    # ── Wrench buffers ──────────────────────────────────────────────
    thrust = torch.zeros(1, 1, 3, device=sim.device)
    moment = torch.zeros(1, 1, 3, device=sim.device)

    # ── Collection loop ─────────────────────────────────────────────
    num_collected = 0
    target_images = args_cli.num_images
    step_count = 0

    # Collection strategy:
    # - Every few steps, teleport to a new random position in the map
    # - Randomize yaw to see different wall angles
    # - Occasionally add roll for diversity
    teleport_interval = 3  # teleport every N steps for maximum diversity
    current_pos = (0.0, 0.0, 1.0)
    current_yaw = 0.0
    current_roll = 0.0

    print(f"[INFO] Collecting {target_images} depth images from arena map ...")

    while num_collected < target_images and simulation_app.is_running():
        # ── Teleport to a new random position every few steps ────────
        if step_count % teleport_interval == 0:
            current_pos = random_free_position()
            current_yaw = random.uniform(0, 2 * math.pi)
            current_roll = random.uniform(-math.radians(15), math.radians(15))

        # ── Slowly rotate yaw between teleports for more views ──────
        current_yaw += 0.3  # ~17 degrees per step

        # ── Build pose ──────────────────────────────────────────────
        px, py, pz = current_pos
        desired_pos = torch.tensor(
            [px + env_origin[0].item(),
             py + env_origin[1].item(),
             pz + env_origin[2].item()],
            device=sim.device,
        )
        qw, qx, qy, qz = quat_from_roll_yaw(current_roll, current_yaw)
        desired_quat = torch.tensor([qw, qx, qy, qz], device=sim.device)

        root_state = robot.data.default_root_state.clone()
        root_state[0, 0:3] = desired_pos
        root_state[0, 3:7] = desired_quat
        root_state[0, 7:] = 0.0

        robot.write_root_pose_to_sim(root_state[:, :7])
        robot.write_root_velocity_to_sim(root_state[:, 7:])

        # Apply hover thrust
        thrust[:, 0, 2] = hover_thrust
        moment[:, 0, :] = 0.0
        robot.permanent_wrench_composer.set_forces_and_torques(
            body_ids=body_id, forces=thrust, torques=moment
        )

        # ── Step simulation ─────────────────────────────────────────
        for _ in range(DECIMATION):
            sim.step()

        # Re-write pose to undo drift
        robot.write_root_pose_to_sim(root_state[:, :7])
        robot.write_root_velocity_to_sim(root_state[:, 7:])

        scene.update(dt=DT * DECIMATION)

        # ── Grab depth & save ───────────────────────────────────────
        raw_depth = tiled_camera.data.output["depth"].clone()

        if raw_depth.numel() == 0:
            step_count += 1
            continue

        depth = raw_depth[0, :, :, 0]
        depth[depth == float("inf")] = DEPTH_MAX
        depth[depth != depth] = DEPTH_MAX  # NaN
        depth = depth.clamp(0.0, DEPTH_MAX) / DEPTH_MAX

        depth_np = depth.cpu().numpy().astype(np.float32)
        filename = os.path.join(output_dir, f"arena_{num_collected:06d}.npy")
        np.save(filename, depth_np)
        num_collected += 1

        if num_collected % 500 == 0 or num_collected == 1:
            print(f"[INFO] Collected {num_collected}/{target_images} images "
                  f"(x={px:.1f}, y={py:.1f}, z={pz:.1f}, "
                  f"yaw={math.degrees(current_yaw % (2*math.pi)):.0f}°)")

        step_count += 1

    print(f"\n[DONE] Collected {num_collected} depth images in: {output_dir}")


if __name__ == "__main__":
    main()
    simulation_app.close()
