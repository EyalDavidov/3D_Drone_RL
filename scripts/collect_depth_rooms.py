"""Depth data collection — Multi-Level Rooms version.

Spawns a single drone in the multi-level arena, teleports it to
structured positions within each room zone, and saves every depth
frame to disk as a .npy file.

Room definitions (local frame, relative to env origin):
  Room 2:   X ∈ [-2, 2],    Y ∈ [-8, -2]    — radial views toward center
  Room 3:   X ∈ [-4, 4],    Y ∈ [-16, -8]   — radial views toward center
  Room 4.1: X ∈ [-0.5, 0.5], Y ∈ [-22, -16] — corridor, ±Y views only
  Room 4.2: X ∈ [-4.5, 0.5], Y ∈ [-21, -20] — corridor, ±X views only

All images at Z = 1.0 ± 0.5 m.

Usage:
    isaaclab.bat -p scripts/collect_depth_rooms.py --enable_cameras --num_images 10000
    isaaclab.bat -p scripts/collect_depth_rooms.py --enable_cameras --num_images 3000 --room 2
"""

# ── Launch Isaac Sim first ──────────────────────────────────────────
import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Collect depth images from multi-level rooms.")
parser.add_argument("--num_images", type=int, default=10_000,
                    help="Total number of depth images to collect.")
parser.add_argument("--output_dir", type=str, default=None,
                    help="Output directory. Defaults to <project>/data/depth_rooms.")
parser.add_argument("--room", type=str, default="all",
                    help="Which room to collect from: '2', '3', '4.1', '4.2', or 'all' (default).")

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
#  Constants
# ═══════════════════════════════════════════════════════════════════

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROOM_USD_PATH = os.path.join(PROJECT_ROOT, "assets", "rooms", "final_roof_flat.usd")
DEPTH_MAX = 5.0
DT = 1.0 / 100.0
DECIMATION = 2
CAMERA_WIDTH = 128
CAMERA_HEIGHT = 72


@configclass
class RoomSceneCfg(InteractiveSceneCfg):
    """Minimal scene: 1 env."""
    num_envs = 1
    env_spacing = 50.0
    replicate_physics = True


# ═══════════════════════════════════════════════════════════════════
#  Room definitions
# ═══════════════════════════════════════════════════════════════════

class RoomDef:
    """Describes a single room's bounds and sampling strategy."""

    def __init__(self, name, x_min, x_max, y_min, y_max, strategy, **kwargs):
        self.name = name
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.strategy = strategy  # "radial" or "corridor_y" or "corridor_x"
        self.kwargs = kwargs

    @property
    def center(self):
        return ((self.x_min + self.x_max) / 2.0,
                (self.y_min + self.y_max) / 2.0)


ROOMS = {
    "2": RoomDef("Room_2", -2, 2, -8, -2, "radial", r_min=0.5, r_max=1.7),
    "3": RoomDef("Room_3", -4, 4, -16, -8, "radial", r_min=0.5, r_max=3.5),
    "4.1": RoomDef("Room_4.1", -0.5, 0.5, -22, -16, "corridor_y", max_dev_deg=10),
    "4.2": RoomDef("Room_4.2", -4.5, 0.5, -21, -20, "corridor_x", max_dev_deg=10),
}


# ═══════════════════════════════════════════════════════════════════
#  Sampling helpers
# ═══════════════════════════════════════════════════════════════════

def sample_radial(room: RoomDef) -> tuple:
    """Sample a position around the room center at a random radius,
    with the camera pointing toward the center.

    Returns: (x, y, z, yaw)
    """
    cx, cy = room.center
    r_min = room.kwargs["r_min"]
    r_max = room.kwargs["r_max"]

    for _ in range(200):
        # Random angle around center
        angle = random.uniform(0, 2 * math.pi)
        radius = random.uniform(r_min, r_max)
        x = cx + radius * math.cos(angle)
        y = cy + radius * math.sin(angle)

        # Check bounds
        if room.x_min <= x <= room.x_max and room.y_min <= y <= room.y_max:
            # Yaw points toward center: atan2(cy - y, cx - x)
            yaw = math.atan2(cy - y, cx - x)
            # Add small random deviation ±15° for variety
            yaw += random.uniform(math.radians(-15), math.radians(15))
            z = 1.0 + random.uniform(-0.5, 0.5)
            return x, y, z, yaw

    # Fallback: center
    z = 1.0 + random.uniform(-0.5, 0.5)
    return cx, cy, z, 0.0


def sample_corridor_y(room: RoomDef) -> tuple:
    """Sample a position in a narrow Y-aligned corridor.
    Camera looks either +Y (yaw=π/2) or -Y (yaw=-π/2).

    Returns: (x, y, z, yaw)
    """
    max_dev = math.radians(room.kwargs["max_dev_deg"])
    x = random.uniform(room.x_min, room.x_max)
    y = random.uniform(room.y_min, room.y_max)
    z = 1.0 + random.uniform(-0.5, 0.5)

    # Choose +Y or -Y direction
    if random.random() < 0.5:
        yaw = math.pi / 2.0   # looking toward +Y
    else:
        yaw = -math.pi / 2.0  # looking toward -Y

    # Add small deviation
    yaw += random.uniform(-max_dev, max_dev)
    return x, y, z, yaw


def sample_corridor_x(room: RoomDef) -> tuple:
    """Sample a position in a narrow X-aligned corridor.
    Camera looks either +X (yaw=0) or -X (yaw=π).

    Returns: (x, y, z, yaw)
    """
    max_dev = math.radians(room.kwargs["max_dev_deg"])
    x = random.uniform(room.x_min, room.x_max)
    y = random.uniform(room.y_min, room.y_max)
    z = 1.0 + random.uniform(-0.5, 0.5)

    # Choose +X or -X direction
    if random.random() < 0.5:
        yaw = 0.0       # looking toward +X
    else:
        yaw = math.pi   # looking toward -X

    # Add small deviation
    yaw += random.uniform(-max_dev, max_dev)
    return x, y, z, yaw


SAMPLERS = {
    "radial": sample_radial,
    "corridor_y": sample_corridor_y,
    "corridor_x": sample_corridor_x,
}


def sample_pose(room: RoomDef) -> tuple:
    """Sample (x, y, z, yaw) for a given room using its strategy."""
    return SAMPLERS[room.strategy](room)


# ═══════════════════════════════════════════════════════════════════
#  Quaternion helper (same as collect_depth_arena.py)
# ═══════════════════════════════════════════════════════════════════

def quat_from_roll_yaw(roll: float, yaw: float) -> tuple:
    """Return (w, x, y, z) quaternion for the drone's base rotation + roll + yaw."""
    # Base orientation: (0.7071, 0, 0, -0.7071)
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
    output_dir = args_cli.output_dir or os.path.join(PROJECT_ROOT, "data", "depth_rooms")
    os.makedirs(output_dir, exist_ok=True)

    # Determine which rooms to collect from
    if args_cli.room.lower() == "all":
        active_rooms = list(ROOMS.values())
    else:
        key = args_cli.room
        if key not in ROOMS:
            print(f"[ERROR] Unknown room '{key}'. Valid options: {list(ROOMS.keys())} or 'all'")
            return
        active_rooms = [ROOMS[key]]

    room_names = [r.name for r in active_rooms]
    print(f"[INFO] Collecting from rooms: {room_names}")
    print(f"[INFO] Saving depth images to: {output_dir}")
    print(f"[INFO] Room USD: {ROOM_USD_PATH}")

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
    scene_cfg = RoomSceneCfg()
    scene = InteractiveScene(scene_cfg)

    # ── Robot ───────────────────────────────────────────────────────
    robot_cfg: ArticulationCfg = DRONE_CONFIG.replace(
        prim_path="/World/envs/env_.*/Drone"
    )
    robot = Articulation(robot_cfg)
    scene.articulations["robot"] = robot

    # ── Room ────────────────────────────────────────────────────────
    room_cfg = sim_utils.UsdFileCfg(usd_path=ROOM_USD_PATH)
    room_cfg.func("/World/envs/env_0/Room", room_cfg)

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

    # ── Camera (same specs as RL training camera) ───────────────────
    camera_cfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Drone/body/Camera",
        height=CAMERA_HEIGHT,
        width=CAMERA_WIDTH,
        data_types=["depth"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=7.5,
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
    images_per_room = {r.name: 0 for r in active_rooms}

    # How many images per room (distribute evenly)
    per_room_target = target_images // len(active_rooms)
    remainder = target_images % len(active_rooms)

    print(f"\n[INFO] Collecting ~{per_room_target} images per room "
          f"({target_images} total, {len(active_rooms)} rooms)")
    print("=" * 60)

    for room_idx, room in enumerate(active_rooms):
        room_target = per_room_target + (1 if room_idx < remainder else 0)
        room_collected = 0

        print(f"\n[INFO] === {room.name} === "
              f"(strategy={room.strategy}, target={room_target} images)")
        print(f"       Bounds: X=[{room.x_min}, {room.x_max}], "
              f"Y=[{room.y_min}, {room.y_max}]")

        while room_collected < room_target and simulation_app.is_running():
            # ── Sample a pose for this room ──────────────────────────
            px, py, pz, yaw = sample_pose(room)

            # Small random roll for diversity (±5°)
            roll = random.uniform(math.radians(-5), math.radians(5))

            # ── Build pose ──────────────────────────────────────────
            desired_pos = torch.tensor(
                [px + env_origin[0].item(),
                 py + env_origin[1].item(),
                 pz + env_origin[2].item()],
                device=sim.device,
            )
            qw, qx, qy, qz = quat_from_roll_yaw(roll, yaw)
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

            # ── Step simulation ─────────────────────────────────────
            for _ in range(DECIMATION):
                sim.step()

            # Re-write pose to undo physics drift
            robot.write_root_pose_to_sim(root_state[:, :7])
            robot.write_root_velocity_to_sim(root_state[:, 7:])

            scene.update(dt=DT * DECIMATION)

            # ── Grab depth & save ───────────────────────────────────
            raw_depth = tiled_camera.data.output["depth"].clone()

            if raw_depth.numel() == 0:
                step_count += 1
                continue

            depth = raw_depth[0, :, :, 0]
            depth[depth == float("inf")] = DEPTH_MAX
            depth[depth != depth] = DEPTH_MAX  # NaN
            depth = (depth.clamp(0.0, DEPTH_MAX) / DEPTH_MAX) ** 1.7

            depth_np = depth.cpu().numpy().astype(np.float32)
            filename = os.path.join(
                output_dir,
                f"{room.name}_{num_collected:06d}.npy"
            )
            np.save(filename, depth_np)
            num_collected += 1
            room_collected += 1
            images_per_room[room.name] = room_collected

            if room_collected % 500 == 0 or room_collected == 1:
                print(f"  [{room.name}] {room_collected}/{room_target} "
                      f"(x={px:.2f}, y={py:.2f}, z={pz:.2f}, "
                      f"yaw={math.degrees(yaw):.0f}°)")

            step_count += 1

        print(f"  [{room.name}] Done: {room_collected} images collected")

    # ── Summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"[DONE] Total: {num_collected} depth images saved to: {output_dir}")
    print("  Per-room breakdown:")
    for name, count in images_per_room.items():
        print(f"    {name}: {count}")
    print("=" * 60)


if __name__ == "__main__":
    main()
    simulation_app.close()
