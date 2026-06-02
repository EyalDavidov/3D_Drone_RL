"""Depth data collection script.

Spawns a single drone in the room, flies it in a circle at the center
facing inward, and saves every depth frame to disk until 10,000 images
are collected.  No RL model is used — the drone follows a scripted
circular trajectory via direct wrench control.

Usage:
    isaaclab.bat -p scripts/collect_depth_data.py --enable_cameras
"""

# ── Launch Isaac Sim first ──────────────────────────────────────────
import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Collect depth images from a circling drone.")
parser.add_argument("--num_images", type=int, default=10_000,
                    help="Number of depth images to collect.")
parser.add_argument("--output_dir", type=str, default=None,
                    help="Output directory for depth images. Defaults to <project>/data/depth_collection.")
parser.add_argument("--circle_radius", type=float, default=1.0,
                    help="Radius of the circular path (meters).")
parser.add_argument("--circle_height", type=float, default=1,
                    help="Height of the circular path (meters).")
parser.add_argument("--circle_speed", type=float, default=2.0,
                    help="Angular speed of the circle (rad/s).")

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
#  Configuration
# ═══════════════════════════════════════════════════════════════════

ROOM_USD_PATH = r"D:\isaac\3D_Drone_RL\assets\Empty_Room.usd"
POLE_USD_PATH = r"D:\isaac\3D_Drone_RL\assets\1_Pole.usd"
MAX_POLES = 10             # max poles spawned; each reset picks 3-10 active
DEPTH_MAX = 5.0            # metres – same as training env (5m gives better contrast in indoor rooms)
DT = 1.0 / 100.0           # physics dt
DECIMATION = 2              # render every 2 physics steps
CAMERA_WIDTH = 128
CAMERA_HEIGHT = 72


@configclass
class DepthCollectionSceneCfg(InteractiveSceneCfg):
    """Minimal scene: 1 env, same spacing as training."""
    num_envs = 1
    env_spacing = 6.0
    replicate_physics = True


# ═══════════════════════════════════════════════════════════════════
#  Helper: quaternion from yaw angle (around world Z)
# ═══════════════════════════════════════════════════════════════════

def quat_from_roll_yaw(roll: float, yaw: float) -> tuple[float, float, float, float]:
    """Return (w, x, y, z) quaternion for a rotation.

    Applies:
    1. Base USD rotation (0.7071, 0, 0, -0.7071)
    2. Roll (around body X)
    3. Yaw (around world Z)
    """
    # 1. Base quat from the USD
    bw, bx, by, bz = 0.70710678, 0.0, 0.0, -0.70710678

    # 2. Roll quat (about body X)
    hr = roll / 2.0
    rw, rx, ry, rz = math.cos(hr), math.sin(hr), 0.0, 0.0

    # q_temp = q_base * q_roll
    tw = bw * rw - bx * rx - by * ry - bz * rz
    tx = bw * rx + bx * rw + by * rz - bz * ry
    ty = bw * ry - bx * rz + by * rw + bz * rx
    tz = bw * rz + bx * ry - by * rx + bz * rw

    # 3. Yaw quat (about world Z)
    # We want the *camera* (body-x) to face 'yaw'.
    # Base rotates body-x to world -y (-90 deg).
    # So we add (yaw + 90) to the extra yaw.
    extra_yaw = yaw + math.pi / 2.0
    hy = extra_yaw / 2.0
    yw, yx, yy, yz = math.cos(hy), 0.0, 0.0, math.sin(hy)

    # q_final = q_yaw * q_temp
    w = yw * tw - yx * tx - yy * ty - yz * tz
    x = yw * tx + yx * tw + yy * tz - yz * ty
    y = yw * ty - yx * tz + yy * tw + yz * tx
    z = yw * tz + yx * ty - yy * tx + yz * tw

    return (w, x, y, z)


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

def main():
    # ── Output directory ────────────────────────────────────────────
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    output_dir = args_cli.output_dir or os.path.join(project_root, "data", "depth_collection")
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
    sim.set_camera_view(eye=[3.0, 3.0, 3.0], target=[0.0, 0.0, 1.0])

    # ── Scene ───────────────────────────────────────────────────────
    scene_cfg = DepthCollectionSceneCfg()
    scene = InteractiveScene(scene_cfg)

    # ── Robot ───────────────────────────────────────────────────────
    robot_cfg: ArticulationCfg = DRONE_CONFIG.replace(
        prim_path="/World/envs/env_.*/Drone"
    )
    robot = Articulation(robot_cfg)
    scene.articulations["robot"] = robot

    # ── Room (empty) ────────────────────────────────────────────────
    room_cfg = sim_utils.UsdFileCfg(usd_path=ROOM_USD_PATH)
    room_cfg.func("/World/envs/env_0/Room", room_cfg)

    # ── Poles (RigidObject, kinematic) ───────────────────────────────
    # Spawn MAX_POLES; each randomization picks 3-10 to place in the
    # room and hides the rest far underground.
    pole_objects: list[RigidObject] = []
    for i in range(MAX_POLES):
        cfg = RigidObjectCfg(
            prim_path=f"/World/envs/env_.*/Pole_{i}",
            spawn=sim_utils.UsdFileCfg(
                usd_path=POLE_USD_PATH,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=True,
                    disable_gravity=True,
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -50.0)),
        )
        obj = RigidObject(cfg)
        scene.rigid_objects[f"pole_{i}"] = obj
        pole_objects.append(obj)
    print(f"[INFO] Created {MAX_POLES} pole slots (kinematic RigidObjects)")

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

    # ── Helper: randomize pole positions within room bounds ──────────
    # Room is ~4m×4m; active poles placed in [-1.5, 1.5] XY.
    # Inactive poles hidden at Z = -50.
    env_origin = terrain.env_origins[0]  # (3,)
    identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=sim.device)

    def randomize_poles():
        """Pick 3-10 active poles, place randomly, hide the rest."""
        num_active = random.randint(3, MAX_POLES)
        for i, obj in enumerate(pole_objects):
            pose = obj.data.default_root_state.clone()  # (1, 13)
            if i < num_active:
                pose[0, 0] = random.uniform(-1.5, 1.5) + env_origin[0].item()
                pose[0, 1] = random.uniform(-1.5, 1.5) + env_origin[1].item()
                pose[0, 2] = 1.0 + env_origin[2].item()
            else:
                pose[0, 0] = env_origin[0].item()
                pose[0, 1] = env_origin[1].item()
                pose[0, 2] = -50.0
            pose[0, 3:7] = identity_quat
            pose[0, 7:] = 0.0
            obj.write_root_pose_to_sim(pose[:, :7])
            obj.write_root_velocity_to_sim(pose[:, 7:])
        print(f"[INFO] Randomized poles: {num_active} active / {MAX_POLES} total")

    # Initial randomization
    randomize_poles()

    # ── Physical constants ──────────────────────────────────────────
    body_id = robot.find_bodies("body")[0]
    robot_mass = robot.root_physx_view.get_masses()[0].sum()
    gravity_mag = torch.tensor(sim.cfg.gravity, device=sim.device).norm()
    robot_weight = (robot_mass * gravity_mag).item()

    # Hover thrust (perfectly cancels gravity)
    hover_thrust = robot_weight

    # ── Circle parameters ───────────────────────────────────────────
    radius = args_cli.circle_radius
    height = args_cli.circle_height
    omega = args_cli.circle_speed   # rad/s
    theta = 0.0                      # current angle on circle

    # ── Diversity parameters (randomized periodically) ──────────────
    current_roll = 0.0       # roll angle in radians
    current_height = height  # flying height
    current_radius = radius  # circle radius
    current_omega = omega    # angular speed
    randomize_angle_interval = 50   # re-roll viewing angle every N steps
    randomize_trajectory_interval = 200  # re-roll trajectory params every N steps

    # ── Wrench buffers ──────────────────────────────────────────────
    thrust = torch.zeros(1, 1, 3, device=sim.device)
    moment = torch.zeros(1, 1, 3, device=sim.device)

    # ── Collection loop ─────────────────────────────────────────────
    num_collected = 0
    target_images = args_cli.num_images
    step_count = 0
    pole_randomize_interval = 3.0  # seconds
    time_since_last_randomize = 0.0

    print(f"[INFO] Collecting {target_images} depth images ...")
    print(f"[INFO] Circle: radius={radius}m, height={height}m, speed={omega} rad/s")
    print(f"[INFO] Diversity: roll/angle randomized every {randomize_angle_interval} steps, "
          f"trajectory every {randomize_trajectory_interval} steps")

    while num_collected < target_images and simulation_app.is_running():
        # ── Randomize viewing angle (roll) every N steps ─────────────
        if step_count % randomize_angle_interval == 0:
            current_roll = random.uniform(-math.radians(20), math.radians(20))

        # ── Randomize trajectory params every M steps ────────────────
        if step_count % randomize_trajectory_interval == 0:
            current_height = random.uniform(0.5, 1.8)   # vary height within room
            current_radius = random.uniform(0.3, 2.0)   # vary circle size
            current_omega = random.uniform(1.0, 4.0)     # vary speed

        # ── Compute desired position & orientation ──────────────────
        theta = current_omega * step_count * DT * DECIMATION
        # Position on circle (center of room = env origin)
        px = current_radius * math.cos(theta)
        py = current_radius * math.sin(theta)
        pz = current_height

        # Yaw so camera faces center
        desired_yaw = theta + math.pi

        # Build pose
        env_origin = terrain.env_origins[0]  # (3,)
        desired_pos = torch.tensor(
            [px + env_origin[0].item(),
             py + env_origin[1].item(),
             pz + env_origin[2].item()],
            device=sim.device,
        )
        qw, qx, qy, qz = quat_from_roll_yaw(current_roll, desired_yaw)
        desired_quat = torch.tensor([qw, qx, qy, qz], device=sim.device)

        # ── Build the desired root state ────────────────────────────
        root_state = robot.data.default_root_state.clone()  # (1, 13)
        root_state[0, 0:3] = desired_pos
        root_state[0, 3:7] = desired_quat
        root_state[0, 7:] = 0.0  # zero velocity

        # Write pose BEFORE physics so forces act on the right body
        robot.write_root_pose_to_sim(root_state[:, :7])
        robot.write_root_velocity_to_sim(root_state[:, 7:])

        # Apply hover thrust to keep the drone stationary
        thrust[:, 0, 2] = hover_thrust
        moment[:, 0, :] = 0.0
        robot.permanent_wrench_composer.set_forces_and_torques(
            body_ids=body_id, forces=thrust, torques=moment
        )

        # ── Step simulation ─────────────────────────────────────────
        for _ in range(DECIMATION):
            sim.step()

        # ── Re-write pose AFTER physics to undo any drift ───────────
        # This ensures the camera (read during scene.update) sees the
        # exact kinematic pose, not a physics-drifted orientation.
        robot.write_root_pose_to_sim(root_state[:, :7])
        robot.write_root_velocity_to_sim(root_state[:, 7:])

        # ── Update scene (reads sensors) ────────────────────────────
        scene.update(dt=DT * DECIMATION)

        # ── Randomize poles every 3 seconds ─────────────────────────
        step_dt = DT * DECIMATION
        time_since_last_randomize += step_dt
        if time_since_last_randomize >= pole_randomize_interval:
            randomize_poles()
            time_since_last_randomize = 0.0

        # ── Grab depth & save ───────────────────────────────────────
        raw_depth = tiled_camera.data.output["depth"].clone()  # (1, H, W, 1)

        # Skip if the camera hasn't started producing valid data
        if raw_depth.numel() == 0:
            step_count += 1
            continue

        # Preprocess: replace inf/nan → clamp → normalize to [0,1]
        depth = raw_depth[0, :, :, 0]  # (H, W)
        depth[depth == float("inf")] = DEPTH_MAX
        depth[depth != depth] = DEPTH_MAX  # NaN
        depth = depth.clamp(0.0, DEPTH_MAX) / DEPTH_MAX

        # Save as numpy
        depth_np = depth.cpu().numpy().astype(np.float32)
        filename = os.path.join(output_dir, f"depth_3{num_collected:06d}.npy")
        np.save(filename, depth_np)
        num_collected += 1

        if num_collected % 500 == 0 or num_collected == 1:
            print(f"[INFO] Collected {num_collected}/{target_images} images "
                  f"(θ={math.degrees(theta % (2 * math.pi)):.1f}°, "
                  f"h={current_height:.1f}m, r={current_radius:.1f}m, "
                  f"roll={math.degrees(current_roll):.1f}°)")

        step_count += 1

    print(f"\n[DONE] Collected {num_collected} depth images in: {output_dir}")


if __name__ == "__main__":
    main()
    simulation_app.close()
