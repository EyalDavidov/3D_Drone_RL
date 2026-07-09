# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run the Visual SLAM and Frontier Exploration simulation.

This launcher uses RealSlamDroneEnv which performs visual 2D mapping,
frontier detection, and A* path planning directly from the camera depth map.

Usage:
    python scripts/real_slam_play.py --navigator_checkpoint <path>
    python scripts/real_slam_play.py --navigator_checkpoint <path> --no-dashboard
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

# ---- CLI arguments --------------------------------------------------------
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
parser.add_argument(
    "--no-dashboard", action="store_true", default=False,
    help="Disable the web dashboard.",
)
parser.add_argument(
    "--no-slam-window", action="store_true", default=False,
    help="Disable the OpenCV SLAM visualiser window.",
)
parser.add_argument(
    "--opencv", action="store_true", default=False,
    help="Force OpenCV SLAM + YOLO windows (off by default when dashboard is on).",
)

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True

# ---- Launch Omniverse -----------------------------------------------------
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

try:
    import omni.kit.app
    ext_manager = omni.kit.app.get_app().get_extension_manager()
    if not ext_manager.is_extension_enabled("omni.isaac.debug_draw"):
        ext_manager.set_extension_enabled_immediate("omni.isaac.debug_draw", True)
except Exception:
    pass

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import first_drone.tasks  # noqa: F401
from first_drone.tasks.direct.navigation_drone.brain_nav_drone_env import resolve_navigator_checkpoint
from first_drone.tasks.direct.navigation_drone.real_slam.real_slam_env import RealSlamDroneEnv

# ---- Dashboard (optional) -------------------------------------------------
_DASHBOARD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard")
if _DASHBOARD_DIR not in sys.path:
    sys.path.insert(0, _DASHBOARD_DIR)

_DASHBOARD_AVAILABLE = False
if not args_cli.no_dashboard:
    try:
        from live_telemetry import LiveDroneTelemetry
        from server import start_dashboard_server
        _DASHBOARD_AVAILABLE = True
    except ImportError as _e:
        print(f"[Dashboard] Not available: {_e}")


# ---------------------------------------------------------------------------
# SLAM map visualiser
# ---------------------------------------------------------------------------

def draw_slam_visualizer(
    mapper,
    drone_pos_w,
    drone_yaw: float,
    slam_state: str,
    frontiers: list,
    active_frontier,
    astar_path_world: list,
    start_time: float,
    person_found: bool = False,
    person_pos_w=None,
):
    """Render the Visual SLAM dashboard using OpenCV."""
    if cv2 is None:
        return

    prob     = mapper.get_occupancy_grid()
    grid_h, grid_w = prob.shape

    if hasattr(mapper, "get_wall_obstacle_masks"):
        wall_mask, obstacle_mask = mapper.get_wall_obstacle_masks(use_walkable=False)
        danger = mapper.get_planning_grid()
    else:
        wall_mask = (prob > 0.65).astype(np.uint8)
        obstacle_mask = np.zeros_like(wall_mask)
        danger = mapper.get_inflated_grid()

    # ---- Base map (sci-fi colour scheme) ----------------------------------
    canvas = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    canvas[(prob >= 0.35) & (prob <= 0.65)] = [25, 18, 12]   # unknown — deep navy
    canvas[prob < 0.35]                      = [45, 38, 30]   # free — dark slate
    canvas[danger == 1]                      = [30, 20, 75]   # danger halo (walls only)
    canvas[obstacle_mask == 1]               = [140, 156, 31] # teal — dodgeable props
    canvas[wall_mask == 1]                   = [255, 230, 80] # amber — structural walls

    map_w, map_h = 450, 600
    hud_w  = 250
    total_h = 600

    canvas_large = cv2.resize(canvas, (map_w, map_h), interpolation=cv2.INTER_LINEAR)
    # Flip vertically: Room 1 (Y≈+2) at TOP, Room 4 (Y≈-20) at BOTTOM.
    # This is the standard "north-up" top-down convention and fixes the rotation
    # direction bug: left turns in the sim now look like left turns on the map.
    canvas_large = cv2.flip(canvas_large, 0)

    # Grid overlay (radar feel)
    for x in range(0, map_w, 40):
        cv2.line(canvas_large, (x, 0), (x, map_h), (35, 28, 20), 1)
    for y in range(0, map_h, 40):
        cv2.line(canvas_large, (0, y), (map_w, y), (35, 28, 20), 1)

    def to_disp(wx, wy):
        """World coords → display pixel with Y flipped (Room 1 at top)."""
        r, c = mapper.world_to_grid(wx, wy)
        cx = int(np.clip(c * (map_w / grid_w), 0, map_w - 1))
        cy_raw = int(np.clip(r * (map_h / grid_h), 0, map_h - 1))
        cy = map_h - 1 - cy_raw  # flip: large world-Y row → top of screen
        return cx, cy

    # ---- A* path ----------------------------------------------------------
    if astar_path_world and len(astar_path_world) > 1:
        for i in range(len(astar_path_world) - 1):
            p0 = to_disp(astar_path_world[i][0],   astar_path_world[i][1])
            p1 = to_disp(astar_path_world[i+1][0], astar_path_world[i+1][1])
            cv2.line(canvas_large, p0, p1, (100, 255, 100), 4, cv2.LINE_AA)
            cv2.line(canvas_large, p0, p1, (0,   255, 0),   2, cv2.LINE_AA)

    # ---- Non-active frontiers (cyan circles) ------------------------------
    for f in frontiers:
        if active_frontier is None or np.linalg.norm(
            np.array(f["centroid_world"]) - np.array(active_frontier["centroid_world"])
        ) > 0.1:
            cx, cy = to_disp(f["centroid_world"][0], f["centroid_world"][1])
            cv2.circle(canvas_large, (cx, cy), 7, (255, 180, 50), -1, cv2.LINE_AA)
            cv2.circle(canvas_large, (cx, cy), 3, (255, 255, 255), -1, cv2.LINE_AA)

    # ---- Active frontier (gold crosshair) ---------------------------------
    if active_frontier is not None:
        cx, cy = to_disp(active_frontier["centroid_world"][0], active_frontier["centroid_world"][1])
        cv2.circle(canvas_large, (cx, cy), 12, (0, 200, 255), 1, cv2.LINE_AA)
        cv2.line(canvas_large, (cx - 15, cy), (cx + 15, cy), (0, 200, 255), 1, cv2.LINE_AA)
        cv2.line(canvas_large, (cx, cy - 15), (cx, cy + 15), (0, 200, 255), 1, cv2.LINE_AA)
        cv2.circle(canvas_large, (cx, cy), 4, (0, 0, 255), -1, cv2.LINE_AA)

    # ---- Detected person (magenta star) -----------------------------------
    if person_found and person_pos_w is not None:
        try:
            px_disp, py_disp = to_disp(float(person_pos_w[0]), float(person_pos_w[1]))
            cv2.drawMarker(canvas_large, (px_disp, py_disp), (255, 0, 255),
                           cv2.MARKER_STAR, 18, 2, cv2.LINE_AA)
            cv2.putText(canvas_large, "PERSON", (px_disp + 10, py_disp - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 255), 1, cv2.LINE_AA)
        except Exception:
            pass

    # ---- Drone (neon-green triangle) --------------------------------------
    # With the flipped Y display, negate yaw so left turns look like left turns visually.
    d_cx, d_cy = to_disp(drone_pos_w[0], drone_pos_w[1])
    dy = -drone_yaw  # display yaw: negate because screen Y is flipped relative to world Y
    p_nose = (int(d_cx + 14 * math.cos(dy)),       int(d_cy + 14 * math.sin(dy)))
    p_l    = (int(d_cx + 8  * math.cos(dy + 2.4)), int(d_cy + 8  * math.sin(dy + 2.4)))
    p_r    = (int(d_cx + 8  * math.cos(dy - 2.4)), int(d_cy + 8  * math.sin(dy - 2.4)))
    pts = np.array([p_nose, p_l, p_r], np.int32)
    cv2.fillPoly(canvas_large, [pts], (0, 230, 0))
    cv2.polylines(canvas_large, [pts], True, (255, 255, 255), 1, cv2.LINE_AA)

    # Radar FOV cone
    fov_half = 0.7
    p_cl = (int(d_cx + 35 * math.cos(dy + fov_half)), int(d_cy + 35 * math.sin(dy + fov_half)))
    p_cr = (int(d_cx + 35 * math.cos(dy - fov_half)), int(d_cy + 35 * math.sin(dy - fov_half)))
    cv2.line(canvas_large, (d_cx, d_cy), p_cl, (0, 200, 0), 1, cv2.LINE_AA)
    cv2.line(canvas_large, (d_cx, d_cy), p_cr, (0, 200, 0), 1, cv2.LINE_AA)

    # ---- HUD panel --------------------------------------------------------
    hud_panel = np.full((total_h, hud_w, 3), 12, dtype=np.uint8)
    cv2.line(hud_panel, (0, 0), (0, total_h), (45, 38, 30), 2)

    # Header
    cv2.rectangle(hud_panel, (10, 15), (hud_w - 10, 50), (25, 18, 12), -1)
    cv2.rectangle(hud_panel, (10, 15), (hud_w - 10, 50), (45, 38, 30), 1)
    cv2.putText(hud_panel, "SLAM HUD CONTROL", (25, 37),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1, cv2.LINE_AA)

    # Mission state card
    border_col = (40, 30, 80) if slam_state == "SCAN" else (45, 38, 30)
    cv2.rectangle(hud_panel, (10, 65), (hud_w - 10, 140), (25, 20, 15), -1)
    cv2.rectangle(hud_panel, (10, 65), (hud_w - 10, 140), border_col, 1)
    cv2.putText(hud_panel, "MISSION STATE:", (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1, cv2.LINE_AA)
    state_color = (
        (0, 255, 255) if slam_state == "SCAN"
        else (100, 255, 100) if slam_state == "EXPLORE"
        else (0, 255, 0)
    )
    cv2.putText(hud_panel, slam_state, (20, 107),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, state_color, 2, cv2.LINE_AA)

    # Telemetry card
    cv2.rectangle(hud_panel, (10, 155), (hud_w - 10, 340), (20, 15, 12), -1)
    cv2.rectangle(hud_panel, (10, 155), (hud_w - 10, 340), (45, 38, 30), 1)
    cv2.putText(hud_panel, "TELEMETRY DATA", (20, 175),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1, cv2.LINE_AA)
    cv2.putText(hud_panel, f"X-POS: {drone_pos_w[0]:.2f} m", (20, 205),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(hud_panel, f"Y-POS: {drone_pos_w[1]:.2f} m", (20, 230),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(hud_panel, f"ALT:   {drone_pos_w[2]:.2f} m", (20, 255),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(hud_panel, f"HDG:   {math.degrees(drone_yaw):.1f} deg", (20, 280),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
    elapsed = time.time() - start_time
    cv2.putText(hud_panel, f"TIME:  {elapsed:.1f} s", (20, 315),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 200, 255), 1, cv2.LINE_AA)

    # Targets / coverage card
    cv2.rectangle(hud_panel, (10, 355), (hud_w - 10, 520), (20, 15, 12), -1)
    cv2.rectangle(hud_panel, (10, 355), (hud_w - 10, 520), (45, 38, 30), 1)
    cv2.putText(hud_panel, "MAPPED TARGETS", (20, 375),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1, cv2.LINE_AA)

    ppl_color = (0, 200, 255) if person_found else (200, 200, 200)
    ppl_label = "FOUND" if person_found else "Not found"
    cv2.putText(hud_panel, f"Person: {ppl_label}", (20, 400),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, ppl_color, 1, cv2.LINE_AA)
    cv2.putText(hud_panel, f"Frontiers: {len(frontiers)}", (20, 420),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

    target_str = "None"
    if active_frontier is not None:
        target_str = (f"({active_frontier['centroid_world'][0]:.1f},"
                      f" {active_frontier['centroid_world'][1]:.1f})")
    cv2.putText(hud_panel, f"Goal: {target_str}", (20, 440),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA)

    path_len = len(astar_path_world) if astar_path_world else 0
    cv2.putText(hud_panel, f"A* nodes: {path_len}", (20, 465),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

    # Map coverage (pure SLAM — known cells / bbox of mapped region)
    if hasattr(mapper, "coverage_stats"):
        visited, total = mapper.coverage_stats()
        explored_pct = visited / max(1, total) * 100.0
    else:
        explored_pct = (np.sum(prob < 0.35) + np.sum(prob > 0.65)) / (grid_h * grid_w) * 100.0
    cv2.putText(hud_panel, f"Explored: {explored_pct:.1f}%", (20, 495),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)

    # Watermark
    cv2.putText(hud_panel, "ANTIGRAVITY SYSTEMS V2", (35, 570),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1, cv2.LINE_AA)

    # ---- Compose and display -------------------------------------------
    frame = np.hstack([canvas_large, hud_panel])
    cv2.imshow("Brain Nav - SLAM Map", frame)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Clear and recreate dashboard static saves folder on startup
    try:
        import shutil
        saves_dir = Path(r"D:\isaac\3D_Drone_RL\scripts\dashboard\static\yolo_saves")
        if saves_dir.exists():
            try:
                shutil.rmtree(saves_dir)
            except Exception as e_rm:
                print(f"[SLAM Launcher] Warning: could not delete old saves: {e_rm}")
        saves_dir.mkdir(parents=True, exist_ok=True)
        print(f"[SLAM Launcher] Cleared and initialized dashboard YOLO saves: {saves_dir}")
    except Exception as e:
        print(f"[SLAM Launcher] Failed to clear dashboard saves folder: {e}")

    # 1. Parse config
    env_cfg = parse_env_cfg(
        "Brain-Nav-Drone-Direct-v0",
        device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=True,
    )

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
    env_cfg.debug_vis   = False
    env_cfg.show_ae_images = False

    _web_dashboard = _DASHBOARD_AVAILABLE and (not args_cli.no_dashboard)
    _use_opencv = args_cli.opencv or (not _web_dashboard and not args_cli.no_slam_window)
    env_cfg.yolo_show_opencv = _use_opencv and (not args_cli.no_slam_window)
    if _web_dashboard:
        # Lighter YOLO for web-only runs — 1280@3x upscale was blocking the sim loop on CPU
        # env_cfg.yolo_camera_upscale = 2
        # env_cfg.yolo_imgsz = 640
        # env_cfg.yolo_sharpen = False
        env_cfg.yolo_noted_conf_threshold = 0.35
        env_cfg.yolo_noted_confirm_frames = 1
        print("[SLAM Launcher] YOLO high-res config kept active for GPU performance")

    # 2. Instantiate RealSlamDroneEnv
    print("[SLAM Launcher] Initializing RealSlamDroneEnv…")
    env = RealSlamDroneEnv(cfg=env_cfg)

    # Set viewport to chase-cam
    try:
        import omni.kit.viewport.utility
        viewport_api = omni.kit.viewport.utility.get_active_viewport()
        if viewport_api is not None:
            viewport_api.camera_path = "/World/envs/env_0/Drone/body/Camera_View"
            print(f"[INFO] Viewport camera set to chase view")
    except Exception:
        pass

    print("[SLAM Launcher] Resetting environment…")
    obs, info = env.reset()

    # 3. Start live dashboard (if enabled and available)
    _telemetry = None
    if _web_dashboard:
        try:
            _telemetry = LiveDroneTelemetry(tick_rate=24.0, perf_mode=False)
            start_dashboard_server(
                http_port=8000, ws_port=8001,
                telemetry_source=_telemetry,
                open_browser=True,
                blocking=False,
            )
            print("[Dashboard] Live dashboard at http://localhost:8000 (perf mode: OpenCV off)")
        except Exception as de:
            print(f"[Dashboard] Failed to start: {de}")
            _telemetry = None
    elif _DASHBOARD_AVAILABLE and args_cli.no_dashboard:
        print("[Dashboard] Disabled via --no-dashboard")

    dt            = env.step_dt
    dummy_action  = torch.zeros((args_cli.num_envs, 4), device=env.device)
    show_slam_win = (cv2 is not None) and bool(env_cfg.yolo_show_opencv)

    if not show_slam_win and not env_cfg.yolo_show_opencv:
        print("[SLAM Launcher] OpenCV windows disabled (use web dashboard or pass --opencv)")
    elif show_slam_win:
        print("[SLAM Launcher] OpenCV SLAM window enabled (Q to quit)")

    print("[SLAM Launcher] Loop running. Ctrl+C to exit.")
    start_run_time = time.time()

    try:
        while simulation_app.is_running():
            t0 = time.time()

            obs, rewards, terminated, truncated, info = env.step(dummy_action)

            # ---- SLAM visualiser data ---------------------------------
            brain       = getattr(env, "_brain", None)
            person_found = bool(getattr(brain, "found_person", False)) if brain else False
            person_pos   = getattr(brain, "target_person_pos", None) if brain else None

            if show_slam_win:
                mapper   = env.mapper
                d_pos    = env._robot.data.root_pos_w[0].cpu().numpy()
                d_quat   = env._robot.data.root_quat_w[0].cpu().numpy()
                qw, qx, qy, qz = d_quat
                drone_yaw = math.atan2(
                    2.0 * (qw * qz + qx * qy),
                    1.0 - 2.0 * (qy * qy + qz * qz),
                )
                # Same robust BFS-reachable frontiers the brain selects from, then
                # the brain's direction-gated visited filter — so the OpenCV overlay
                # matches exactly what the drone can target.
                _brain = getattr(env, "_brain", None)
                _explorable = getattr(_brain, "is_explorable_frontier", None)
                if hasattr(mapper, "find_reachable_frontiers"):
                    _sr, _sc = mapper.world_to_grid(float(d_pos[0]), float(d_pos[1]))
                    frontiers, _cf = mapper.find_reachable_frontiers(_sr, _sc, min_size=3)
                    # Hide occlusion-shadow pockets (tiny unknown gain) so the overlay
                    # matches the picker's actual candidates.
                    _subst = [f for f in frontiers if int(f.get("unknown_gain", 0)) >= 20]
                    if _subst:
                        frontiers = _subst
                    if callable(_explorable):
                        _dxy = np.array(d_pos[:2], dtype=np.float64)
                        frontiers = [
                            f for f in frontiers
                            if _explorable(f, _dxy, came_from=_cf)
                        ]
                else:
                    frontiers = mapper.detect_frontiers()
                    _ahead = getattr(_brain, "is_frontier_ahead", None)
                    if callable(_ahead):
                        frontiers = [f for f in frontiers if _ahead(f["centroid_world"])]
                draw_slam_visualizer(
                    mapper, d_pos, drone_yaw, env.slam_state, frontiers,
                    env.active_frontier, env.astar_path_world, start_run_time,
                    person_found=person_found, person_pos_w=person_pos,
                )

            # ---- Push to web dashboard --------------------------------
            if _telemetry is not None:
                _telemetry.push(env, time.time() - start_run_time)

            # ---- Key handling -----------------------------------------
            if show_slam_win and cv2.waitKey(1) & 0xFF == ord("q"):
                print("[SLAM Launcher] Exiting on user request.")
                break

            if env.slam_state == "COMPLETE":
                print("[SLAM Launcher] Mission complete!")
                time.sleep(2.0)
                break

            # ---- Real-time pacing -------------------------------------
            sleep_time = dt - (time.time() - t0)
            if args_cli.real_time and sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("[SLAM Launcher] Interrupted by user.")
    except Exception as e:
        print(f"[SLAM Launcher] Crash: {e}")
        traceback.print_exc()
    finally:
        if show_slam_win and cv2 is not None:
            cv2.destroyAllWindows()
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
