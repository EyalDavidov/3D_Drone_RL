"""Depth data collection — Room 2 with randomized poles.

Spawns a single drone + 15 random poles (3 rows × 5 poles) inside Room 2,
re-randomizes the pole X positions every few captures (like a new episode),
and collects depth images with the camera pointing toward the room center.

Room 2 bounds: X ∈ [-2, 2], Y ∈ [-8, -2]
Pole rows:  Y = -4.0, -5.5, -7.0  (5 poles per row, X ∈ [-1.7, 1.7])
Camera:     radial placement, radius 0.5–1.7 from center, facing center
Height:     Z = 1.0 ± 0.5

Usage:
    isaaclab.bat -p scripts/collect_depth_room2.py --enable_cameras --num_images 4000
"""

# ── Launch Isaac Sim first ──────────────────────────────────────────
import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Collect depth images from Room 2 with poles.")
parser.add_argument("--num_images", type=int, default=4000,
                    help="Number of depth images to collect.")
parser.add_argument("--output_dir", type=str, default=None,
                    help="Output directory. Defaults to <project>/data/depth_room2.")

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
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
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
ROOM_USD_PATH = os.path.join(PROJECT_ROOT, "assets", "final_roof_flat.usd")
POLE_USD_PATH = os.path.join(PROJECT_ROOT, "assets", "1_Pole.usd")
DEPTH_MAX = 5.0
DT = 1.0 / 100.0
DECIMATION = 2
CAMERA_WIDTH = 128
CAMERA_HEIGHT = 72

# Room 2 bounds
ROOM2_X_MIN, ROOM2_X_MAX = -2.0, 2.0
ROOM2_Y_MIN, ROOM2_Y_MAX = -8.0, -2.0
ROOM2_CX = (ROOM2_X_MIN + ROOM2_X_MAX) / 2.0  # 0.0
ROOM2_CY = (ROOM2_Y_MIN + ROOM2_Y_MAX) / 2.0  # -5.0
R_MIN, R_MAX = 0.5, 1.7

# Pole layout (same as training: 3 rows × 5 poles in Room 2)
POLE_ROWS_Y = [-4.0, -5.5, -7.0]
POLES_PER_ROW = 5
NUM_POLES = len(POLE_ROWS_Y) * POLES_PER_ROW  # 15
POLE_X_RANGE = (-1.7, 1.7)

# Re-randomize poles every N images (simulates a new episode)
RERANDOMIZE_INTERVAL = 20


@configclass
class Room2SceneCfg(InteractiveSceneCfg):
    """Minimal scene: 1 env."""
    num_envs = 1
    env_spacing = 50.0
    replicate_physics = True


# ═══════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════

def sample_radial_room2():
    """Sample a camera pose inside Room 2, facing the center.
    Returns: (x, y, z, yaw)
    """
    for _ in range(200):
        angle = random.uniform(0, 2 * math.pi)
        radius = random.uniform(R_MIN, R_MAX)
        x = ROOM2_CX + radius * math.cos(angle)
        y = ROOM2_CY + radius * math.sin(angle)
        if ROOM2_X_MIN <= x <= ROOM2_X_MAX and ROOM2_Y_MIN <= y <= ROOM2_Y_MAX:
            yaw = math.atan2(ROOM2_CY - y, ROOM2_CX - x)
            yaw += random.uniform(math.radians(-15), math.radians(15))
            z = 1.0 + random.uniform(-0.5, 0.5)
            return x, y, z, yaw
    return ROOM2_CX, ROOM2_CY, 1.0, 0.0


def quat_from_roll_yaw(roll: float, yaw: float) -> tuple:
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


def randomize_poles(poles, env_origin, device):
    """Randomize pole X positions exactly like training does in _reset_idx."""
    pole_idx = 0
    for row_y in POLE_ROWS_Y:
        for _ in range(POLES_PER_ROW):
            pole = poles[pole_idx]
            state = pole.data.default_root_state.clone()

            rand_x = random.uniform(*POLE_X_RANGE)
            state[0, 0] = rand_x + env_origin[0].item()
            state[0, 1] = row_y + env_origin[1].item()
            state[0, 2] = 1.0 + env_origin[2].item()
            state[0, 3] = 1.0   # qw
            state[0, 4:7] = 0.0
            state[0, 7:] = 0.0

            pole.write_root_pose_to_sim(state[:, :7])
            pole.write_root_velocity_to_sim(state[:, 7:])
            pole_idx += 1


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

def main():
    output_dir = args_cli.output_dir or os.path.join(PROJECT_ROOT, "data", "depth_rooms_1_7_room2")
    os.makedirs(output_dir, exist_ok=True)
    print(f"[INFO] Saving depth images to: {output_dir}")

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
    sim.set_camera_view(eye=[5.0, 5.0, 5.0], target=[0.0, -5.0, 1.0])

    # ── Scene ───────────────────────────────────────────────────────
    scene_cfg = Room2SceneCfg()
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

    # ── Poles (same as training: kinematic rigid objects) ────────────
    pole_spawn_cfg = sim_utils.UsdFileCfg(
        usd_path=POLE_USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        collision_props=sim_utils.CollisionPropertiesCfg(),
    )
    poles = []
    for i in range(NUM_POLES):
        pole_cfg = RigidObjectCfg(
            prim_path=f"/World/envs/env_.*/Pole_{i}",
            spawn=pole_spawn_cfg,
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -100.0)),
        )
        pole = RigidObject(pole_cfg)
        scene.rigid_objects[f"pole_{i}"] = pole
        poles.append(pole)

    # ── Terrain ─────────────────────────────────────────────────────
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

    # ── Camera (same specs as RL training) ──────────────────────────
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

    # ── Finalize ────────────────────────────────────────────────────
    scene.clone_environments(copy_from_source=False)
    sim.reset()
    scene.reset()

    # ── Physical constants ──────────────────────────────────────────
    body_id = robot.find_bodies("body")[0]
    robot_mass = robot.root_physx_view.get_masses()[0].sum()
    gravity_mag = torch.tensor(sim.cfg.gravity, device=sim.device).norm()
    hover_thrust = (robot_mass * gravity_mag).item()

    env_origin = terrain.env_origins[0]

    thrust = torch.zeros(1, 1, 3, device=sim.device)
    moment = torch.zeros(1, 1, 3, device=sim.device)

    # ── Collection loop ─────────────────────────────────────────────
    num_collected = 0
    target_images = args_cli.num_images

    print(f"\n[INFO] Collecting {target_images} depth images from Room 2 with {NUM_POLES} poles")
    print(f"       Pole rows at Y = {POLE_ROWS_Y}, X ∈ {POLE_X_RANGE}")
    print(f"       Re-randomize poles every {RERANDOMIZE_INTERVAL} images")
    print("=" * 60)

    # Initial pole randomization
    randomize_poles(poles, env_origin, sim.device)

    while num_collected < target_images and simulation_app.is_running():
        # Re-randomize poles periodically (like a new episode)
        if num_collected % RERANDOMIZE_INTERVAL == 0 and num_collected > 0:
            randomize_poles(poles, env_origin, sim.device)

        # ── Sample camera pose ──────────────────────────────────────
        px, py, pz, yaw = sample_radial_room2()
        roll = random.uniform(math.radians(-5), math.radians(5))

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

        thrust[:, 0, 2] = hover_thrust
        moment[:, 0, :] = 0.0
        robot.permanent_wrench_composer.set_forces_and_torques(
            body_ids=body_id, forces=thrust, torques=moment
        )

        # ── Step simulation ─────────────────────────────────────────
        for _ in range(DECIMATION):
            sim.step()

        robot.write_root_pose_to_sim(root_state[:, :7])
        robot.write_root_velocity_to_sim(root_state[:, 7:])
        scene.update(dt=DT * DECIMATION)

        # ── Grab depth & save ───────────────────────────────────────
        raw_depth = tiled_camera.data.output["depth"].clone()
        if raw_depth.numel() == 0:
            continue

        depth = raw_depth[0, :, :, 0]
        depth[depth == float("inf")] = DEPTH_MAX
        depth[depth != depth] = DEPTH_MAX
        depth = (depth.clamp(0.0, DEPTH_MAX) / DEPTH_MAX) ** 1.7

        depth_np = depth.cpu().numpy().astype(np.float32)
        filename = os.path.join(output_dir, f"room2_{num_collected:06d}.npy")
        np.save(filename, depth_np)
        num_collected += 1

        if num_collected % 500 == 0 or num_collected == 1:
            print(f"  [Room_2] {num_collected}/{target_images} "
                  f"(x={px:.2f}, y={py:.2f}, z={pz:.2f}, "
                  f"yaw={math.degrees(yaw):.0f}°)")

    print(f"\n[DONE] Collected {num_collected} depth images → {output_dir}")


if __name__ == "__main__":
    main()
    simulation_app.close()
