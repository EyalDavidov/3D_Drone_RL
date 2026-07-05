# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run the visual SLAM and Frontier Exploration simulation.

This launcher uses the RealSlamDroneEnv which performs visual 2D mapping,
frontier detection, and path planning directly from the camera depth map.

Usage:
    python scripts/real_slam_play.py --navigator_checkpoint <path_to_checkpoint_dir_or_model.pt>
"""

import argparse
import sys
import os
import time
import traceback
import torch
import numpy as np
import math

try:
    import cv2
except ImportError:
    cv2 = None

from isaaclab.app import AppLauncher

# parse arguments
parser = argparse.ArgumentParser(description="Run the Visual SLAM and Frontier Exploration environment.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments (default: 1).")
parser.add_argument(
    "--navigator_checkpoint", type=str, default=None,
    help="Path to model_1450.pt (or run directory).",
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--real-time", action="store_true", default=True, help="Run in real-time, if possible.")
parser.add_argument(
    "--ae_checkpoint", type=str, default=None,
    help="Path to 64-dim AE checkpoint.",
)
parser.add_argument(
    "--yolo_conf", type=float, default=0.70,
    help="Minimum YOLO confidence (0-1) to accept a person detection.",
)

# append AppLauncher args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# Enable camera rendering
args_cli.enable_cameras = True

# launch omniverse
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Enable debug draw extension
try:
    import omni.kit.app
    ext_manager = omni.kit.app.get_app().get_extension_manager()
    if not ext_manager.is_extension_enabled("omni.isaac.debug_draw"):
        ext_manager.set_extension_enabled_immediate("omni.isaac.debug_draw", True)
except Exception as e:
    pass

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import first_drone.tasks  # noqa: F401
from first_drone.tasks.direct.navigation_drone.brain_nav_drone_env import resolve_navigator_checkpoint
from first_drone.tasks.direct.navigation_drone.real_slam.real_slam_env import RealSlamDroneEnv

def draw_slam_visualizer(mapper, drone_pos_w, drone_yaw, slam_state, frontiers, active_frontier, astar_path_world, start_time, rescued_people=None):
    """Render a premium Sci-Fi Visual SLAM Dashboard using OpenCV."""
    if cv2 is None:
        return
        
    prob = mapper.get_occupancy_grid()
    inflated = mapper.get_inflated_grid()
    
    h, w = prob.shape
    # Left Map Area is 450x600 (aspect ratio 3:4, matching 24m x 32m bounds), Right HUD Panel is 250x600
    map_w, map_h = 450, 600
    hud_w = 250
    total_w = map_w + hud_w
    total_h = 600
    
    # 1. Render base map (BGR) with Sci-Fi color scheme
    # Unknown (prob == 0.5): Deep Obsidian Navy (25, 18, 12)
    # Free space (prob < 0.35): Slate Dark Grey (45, 38, 30)
    # Occupied space (prob > 0.65): Neon Cyan Outline (240, 240, 50)
    # Inflated obstacles: Soft Crimson Haze (30, 20, 75)
    
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[(prob >= 0.35) & (prob <= 0.65)] = [25, 18, 12]
    canvas[prob < 0.35] = [45, 38, 30]
    canvas[inflated == 1] = [30, 20, 75]
    canvas[prob > 0.65] = [255, 230, 80] # Neon Cyan
    
    # Resize map canvas with bilinear filtering for smooth boundaries
    canvas_large = cv2.resize(canvas, (map_w, map_h), interpolation=cv2.INTER_LINEAR)
    # NO vertical flip: raw grid already has Room 4 (Y=-21.5) near row 0 = TOP of screen,
    # and Room 1 (Y=+1.5) near last row = BOTTOM of screen.
    # Drone starts at bottom and flies upward — matching the user's desired orientation.
    
    # Draw soft coordinate grid overlay lines on the map (radar feel)
    for x in range(0, map_w, 40):
        cv2.line(canvas_large, (x, 0), (x, map_h), (35, 28, 20), 1)
    for y in range(0, map_h, 40):
        cv2.line(canvas_large, (0, y), (map_w, y), (35, 28, 20), 1)

    def to_disp(wx, wy):
        """Scale world point to display pixel coords (no flip)."""
        r, c = mapper.world_to_grid(wx, wy)
        # col → cx directly (x right = right on screen)
        # row → cy directly (small row = small y world = top of screen)
        cx = int(np.clip(c * (map_w / w), 0, map_w - 1))
        cy = int(np.clip(r * (map_h / h), 0, map_h - 1))
        return cx, cy

    # Draw planned A* path in thick glowing electric green
    if astar_path_world and len(astar_path_world) > 1:
        for idx in range(len(astar_path_world) - 1):
            p0 = to_disp(astar_path_world[idx][0], astar_path_world[idx][1])
            p1 = to_disp(astar_path_world[idx+1][0], astar_path_world[idx+1][1])
            cv2.line(canvas_large, p0, p1, (100, 255, 100), 4, cv2.LINE_AA)
            cv2.line(canvas_large, p0, p1, (0, 255, 0), 2, cv2.LINE_AA)

    # Draw non-active frontiers as bright neon blue circles with glow
    for f in frontiers:
        if active_frontier is None or np.linalg.norm(np.array(f["centroid_world"]) - np.array(active_frontier["centroid_world"])) > 0.1:
            cx, cy = to_disp(f["centroid_world"][0], f["centroid_world"][1])
            cv2.circle(canvas_large, (cx, cy), 7, (255, 180, 50), -1, cv2.LINE_AA) # outer cyan glow
            cv2.circle(canvas_large, (cx, cy), 3, (255, 255, 255), -1, cv2.LINE_AA) # center white
            
    # Draw active target frontier as a glowing gold crosshair target
    if active_frontier is not None:
        cx, cy = to_disp(active_frontier["centroid_world"][0], active_frontier["centroid_world"][1])
        cv2.circle(canvas_large, (cx, cy), 12, (0, 200, 255), 1, cv2.LINE_AA)
        cv2.line(canvas_large, (cx - 15, cy), (cx + 15, cy), (0, 200, 255), 1, cv2.LINE_AA)
        cv2.line(canvas_large, (cx, cy - 15), (cx, cy + 15), (0, 200, 255), 1, cv2.LINE_AA)
        cv2.circle(canvas_large, (cx, cy), 4, (0, 0, 255), -1, cv2.LINE_AA)

    # Draw drone position as a neon green triangle pointing in drone's yaw direction
    d_cx, d_cy = to_disp(drone_pos_w[0], drone_pos_w[1])
    
    # No canvas flip: world X right = screen right (cos unchanged),
    # world +Y = larger row = downward on screen (sin unchanged — standard OpenCV).
    # Drone facing -Y (from Room 1 toward Room 2) → yaw≈-π/2 → nose at (d_cx, d_cy-14) = UP ✓
    p_nose = (
        int(d_cx + 14 * math.cos(drone_yaw)),
        int(d_cy + 14 * math.sin(drone_yaw))
    )
    p_l = (
        int(d_cx + 8 * math.cos(drone_yaw + 2.4)),
        int(d_cy + 8 * math.sin(drone_yaw + 2.4))
    )
    p_r = (
        int(d_cx + 8 * math.cos(drone_yaw - 2.4)),
        int(d_cy + 8 * math.sin(drone_yaw - 2.4))
    )
    
    # Draw triangle fill and border
    pts = np.array([p_nose, p_l, p_r], np.int32)
    cv2.fillPoly(canvas_large, [pts], (0, 230, 0))
    cv2.polylines(canvas_large, [pts], True, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Draw radar scan cone indicator around the drone heading
    fov_half = 0.7
    p_cone_l = (
        int(d_cx + 35 * math.cos(drone_yaw + fov_half)),
        int(d_cy + 35 * math.sin(drone_yaw + fov_half))
    )
    p_cone_r = (
        int(d_cx + 35 * math.cos(drone_yaw - fov_half)),
        int(d_cy + 35 * math.sin(drone_yaw - fov_half))
    )
    cv2.line(canvas_large, (d_cx, d_cy), p_cone_l, (0, 200, 0), 1, cv2.LINE_AA)
    cv2.line(canvas_large, (d_cx, d_cy), p_cone_r, (0, 200, 0), 1, cv2.LINE_AA)
    
    # 2. Render HUD Panel (Obsidian Tech Theme)
    hud_panel = np.zeros((total_h, hud_w, 3), dtype=np.uint8) + 12
    # Borders
    cv2.line(hud_panel, (0, 0), (0, total_h), (45, 38, 30), 2)
    
    # Header Banner
    cv2.rectangle(hud_panel, (10, 15), (hud_w - 10, 50), (25, 18, 12), -1)
    cv2.rectangle(hud_panel, (10, 15), (hud_w - 10, 50), (45, 38, 30), 1)
    cv2.putText(hud_panel, "SLAM HUD CONTROL", (25, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1, cv2.LINE_AA)
    
    # Status Card
    cv2.rectangle(hud_panel, (10, 65), (hud_w - 10, 140), (25, 20, 15), -1)
    cv2.rectangle(hud_panel, (10, 65), (hud_w - 10, 140), (40, 30, 80) if slam_state == "SCAN" else (45, 38, 30), 1)
    
    cv2.putText(hud_panel, "MISSION STATE:", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1, cv2.LINE_AA)
    state_color = (0, 255, 255) if slam_state == "SCAN" else (100, 255, 100) if slam_state == "EXPLORE" else (0, 255, 0)
    cv2.putText(hud_panel, slam_state, (20, 107), cv2.FONT_HERSHEY_SIMPLEX, 0.55, state_color, 2, cv2.LINE_AA)
    
    # Telemetry Card
    cv2.rectangle(hud_panel, (10, 155), (hud_w - 10, 340), (20, 15, 12), -1)
    cv2.rectangle(hud_panel, (10, 155), (hud_w - 10, 340), (45, 38, 30), 1)
    
    cv2.putText(hud_panel, "TELEMETRY DATA", (20, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1, cv2.LINE_AA)
    
    cv2.putText(hud_panel, f"X-POS: {drone_pos_w[0]:.2f} m", (20, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(hud_panel, f"Y-POS: {drone_pos_w[1]:.2f} m", (20, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(hud_panel, f"ALTITUDE: {drone_pos_w[2]:.2f} m", (20, 255), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(hud_panel, f"HEADING: {math.degrees(drone_yaw):.1f} deg", (20, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
    
    elapsed = time.time() - start_time
    cv2.putText(hud_panel, f"TIME RUNNING: {elapsed:.1f} s", (20, 315), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 200, 255), 1, cv2.LINE_AA)
    
    # Targets Info Card
    cv2.rectangle(hud_panel, (10, 355), (hud_w - 10, 520), (20, 15, 12), -1)
    cv2.rectangle(hud_panel, (10, 355), (hud_w - 10, 520), (45, 38, 30), 1)
    
    cv2.putText(hud_panel, "MAPPED TARGETS", (20, 375), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1, cv2.LINE_AA)
    n_detected = len(rescued_people) if rescued_people else 0
    cv2.putText(hud_panel, f"People Found: {n_detected}", (20, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255) if n_detected > 0 else (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(hud_panel, f"Frontiers (Blue): {len(frontiers)}", (20, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
    
    target_str = "None"
    if active_frontier is not None:
        target_str = f"({active_frontier['centroid_world'][0]:.1f}, {active_frontier['centroid_world'][1]:.1f})"
    cv2.putText(hud_panel, f"Active Goal: {target_str}", (20, 435), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
 
    path_len = len(astar_path_world) if astar_path_world else 0
    cv2.putText(hud_panel, f"A* Path length: {path_len} nodes", (20, 465), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
 
    if hasattr(mapper, "walkable_mask") and mapper.walkable_mask is not None:
        explored_pct = (np.sum(((prob < 0.35) | (prob > 0.65)) & mapper.walkable_mask)) / np.sum(mapper.walkable_mask) * 100.0
    else:
        explored_pct = (np.sum(prob < 0.35) + np.sum(prob > 0.65)) / (h * w) * 100.0
    cv2.putText(hud_panel, f"Map Explored: {explored_pct:.1f}%", (20, 495), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
    
    # Bottom watermark logo
    cv2.putText(hud_panel, "ANTIGRAVITY SYSTEMS V2", (35, 570), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1, cv2.LINE_AA)
    
    # Assemble panels side-by-side
    frame = np.hstack([canvas_large, hud_panel])
    
    cv2.imshow("Brain Nav - SLAM Map", frame)

def main():
    # 1. Parse config using the default task name
    env_cfg = parse_env_cfg(
        "Brain-Nav-Drone-Direct-v0", device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=True
    )
    
    # Override seed
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed
    else:
        env_cfg.seed = int(time.time()) % 100000
        
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.navigator_checkpoint_path = resolve_navigator_checkpoint(
        args_cli.navigator_checkpoint or env_cfg.navigator_checkpoint_path
    )
    
    if args_cli.ae_checkpoint is not None:
        env_cfg.ae_checkpoint_path = args_cli.ae_checkpoint
        
    env_cfg.yolo_person_conf_threshold = args_cli.yolo_conf
    env_cfg.debug_vis = False
    env_cfg.show_ae_images = False
    
    # 2. Instantiate RealSlamDroneEnv directly (bypassing gym registry wrapper)
    print("[SLAM Launcher] Initializing RealSlamDroneEnv...")
    env = RealSlamDroneEnv(cfg=env_cfg)
    
    # Set viewport camera to behind-drone view
    try:
        import omni.kit.viewport.utility
        viewport_api = omni.kit.viewport.utility.get_active_viewport()
        if viewport_api is not None:
            viewport_api.camera_path = "/World/envs/env_0/Drone/body/Camera_View"
            print(f"[INFO] Viewport camera: {viewport_api.camera_path} (chase view)")
    except Exception as e:
        pass
        
    print("[SLAM Launcher] Resetting environment...")
    obs, info = env.reset()
    
    dt = env.step_dt
    dummy_action = torch.zeros((args_cli.num_envs, 4), device=env.device)
    
    print("[SLAM Launcher] Loop running. Press Q in windows or Ctrl+C to exit.")
    
    start_run_time = time.time()
    
    try:
        while simulation_app.is_running():
            start_time = time.time()
            
            # Step environment (internally runs Visual SLAM, A*, frontier exploration, and YOLO)
            obs, rewards, terminated, truncated, info = env.step(dummy_action)
            
            # Retrieve SLAM stats and draw OpenCV map visualizer
            mapper = env.mapper
            d_pos = env._robot.data.root_pos_w[0].cpu().numpy()
            
            # Get yaw of drone
            d_quat = env._robot.data.root_quat_w[0].cpu().numpy() # [w, x, y, z]
            qw, qx, qy, qz = d_quat
            drone_yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
            
            frontiers = mapper.detect_frontiers()
            
            rescued = getattr(env._brain, "rescued_people", []) if hasattr(env, "_brain") else []
            draw_slam_visualizer(
                mapper, d_pos, drone_yaw, env.slam_state, frontiers,
                env.active_frontier, env.astar_path_world, start_run_time,
                rescued_people=rescued
            )
            
            # Handle OpenCV window inputs
            if cv2 is not None and cv2.waitKey(1) & 0xFF == ord('q'):
                print("[SLAM Launcher] Exiting on user request.")
                break
                
            if env.slam_state == "COMPLETE":
                print("[SLAM Launcher] Mission finished successfully.")
                time.sleep(2.0)
                break
                
            # Real-time synchronization
            sleep_time = dt - (time.time() - start_time)
            if args_cli.real_time and sleep_time > 0:
                time.sleep(sleep_time)
                
    except KeyboardInterrupt:
        print("[SLAM Launcher] Interrupted by user.")
    except Exception as e:
        print(f"[SLAM Launcher] Crash in loop: {e}")
        traceback.print_exc()
    finally:
        env.close()
        simulation_app.close()

if __name__ == "__main__":
    main()
