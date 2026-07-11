"""live_telemetry.py — Real telemetry source for the RL Drone Dashboard.

Provides LiveDroneTelemetry, a drop-in replacement for MockDroneTelemetry
that reads live data from a running RealSlamDroneEnv (or BrainNavDroneEnv).

Usage (from real_slam_play.py):
    from live_telemetry import LiveDroneTelemetry
    telemetry = LiveDroneTelemetry(tick_rate=10.0)
    start_dashboard_server(telemetry_source=telemetry, blocking=False)
    # After each env.step():
    telemetry.push(env, elapsed_secs)
"""
from __future__ import annotations

import base64
import math
import struct
import threading
import time
import zlib
from typing import Any


# ---------------------------------------------------------------------------
# Map layout from assets/rooms/final_flat.usd (env-local = world coords)
# ---------------------------------------------------------------------------
MAP_ZONES = {
    "room_1":        {"bounds": [-2.05,  2.05,  -2.05,  2.05]},
    "room_2":        {"bounds": [-2.05,  2.05,  -8.05, -2.00]},
    "room_3":        {"bounds": [-4.05,  4.05, -16.05, -7.95]},
    "room_4":        {"bounds": [-8.55, -4.45, -23.05, -17.95]},
    "corridor":      {"bounds": [-4.50,  0.55, -22.05, -16.00]},
    "side_coridors": {"bounds": [-2.70,  2.70, -18.05, -16.00]},
}

# Legacy list form (x_min, x_max, y_min, y_max, z_min, z_max) for older consumers
ROOM_BOUNDS = [
    (*MAP_ZONES["room_1"]["bounds"],        0.0, 2.0),
    (*MAP_ZONES["room_2"]["bounds"],        0.0, 2.0),
    (*MAP_ZONES["room_3"]["bounds"],        0.0, 2.0),
    (*MAP_ZONES["room_4"]["bounds"],        0.0, 2.0),
    (*MAP_ZONES["corridor"]["bounds"],      0.0, 2.0),
    (*MAP_ZONES["side_coridors"]["bounds"], 0.0, 2.0),
]


# ---------------------------------------------------------------------------
# Image encoding helpers — vectorised with NumPy + cv2 when available
# ---------------------------------------------------------------------------
def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    c = chunk_type + data
    crc = zlib.crc32(c) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + c + struct.pack(">I", crc)


def _make_gray_png(width: int, height: int, values) -> bytes:
    """Pure-Python fallback: encode a 2-D list of 0-255 ints as grayscale PNG."""
    raw = bytearray()
    for row in values:
        raw.append(0)
        for v in row:
            raw.append(int(max(0, min(255, v))))
    header = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    out = b"\x89PNG\r\n\x1a\n"
    out += _png_chunk(b"IHDR", header)
    out += _png_chunk(b"IDAT", zlib.compress(bytes(raw), 4))
    out += _png_chunk(b"IEND", b"")
    return out


def _make_rgb_png(width: int, height: int, pixels) -> bytes:
    """Pure-Python fallback: encode a 2-D list of (R, G, B) tuples as RGB PNG."""
    raw = bytearray()
    for row in pixels:
        raw.append(0)
        for r, g, b in row:
            raw.append(int(r)); raw.append(int(g)); raw.append(int(b))
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    out = b"\x89PNG\r\n\x1a\n"
    out += _png_chunk(b"IHDR", header)
    out += _png_chunk(b"IDAT", zlib.compress(bytes(raw), 4))
    out += _png_chunk(b"IEND", b"")
    return out


def _auto_brighten(bgr, target: float = 110.0, max_gain: float = 3.5):
    """Automatic camera 'light': lift dark frames toward a target brightness.

    Mimics auto-exposure/auto-gain — computes the frame's mean luma and applies a
    clamped gain plus CLAHE for local contrast, so the drone's dim indoor camera
    feeds are visible on the dashboard without blowing out already-bright frames.
    """
    try:
        import cv2
        import numpy as np

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        mean = float(gray.mean()) + 1e-3
        gain = float(np.clip(target / mean, 1.0, max_gain))
        out = bgr
        if gain > 1.02:
            out = np.clip(bgr.astype(np.float32) * gain, 0, 255).astype(np.uint8)
        # Local contrast so shadow detail returns, not just a global scale.
        lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab = cv2.merge((clahe.apply(l), a, b))
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    except Exception:
        return bgr


def _ndarray_to_jpeg_b64(arr_bgr, quality: int = 80) -> str:
    """Encode a BGR uint8 ndarray as JPEG, return base64 string."""
    try:
        import cv2
        ok, buf = cv2.imencode(".jpg", arr_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok:
            return base64.b64encode(buf.tobytes()).decode("ascii")
    except Exception:
        pass
    return ""


def _ndarray_to_png_b64(arr_bgr) -> str:
    """Encode a BGR uint8 ndarray as PNG, return base64 string."""
    try:
        import cv2
        ok, buf = cv2.imencode(".png", arr_bgr)
        if ok:
            return base64.b64encode(buf.tobytes()).decode("ascii")
    except Exception:
        pass
    return ""


def _depth_jet_b64(depth_np, near: float = 0.05, far: float = 10.0) -> str:
    """Jet-colourmap depth → JPEG base64."""
    try:
        import cv2
        import numpy as np
        norm = np.clip((depth_np - near) / (far - near), 0.0, 1.0)
        # OpenCV COLORMAP_JET uses BGR order — correct for our BGR pipeline
        gray = (norm * 255).astype("uint8")
        jet = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
        return _ndarray_to_jpeg_b64(jet)
    except Exception:
        return ""


def _depth_gray_b64(depth_np, near: float = 0.05, far: float = 10.0) -> str:
    """Inverted-gray depth → JPEG base64."""
    try:
        import cv2
        import numpy as np
        norm = np.clip(1.0 - (depth_np - near) / (far - near), 0.0, 1.0)
        gray = (norm * 255).astype("uint8")
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return _ndarray_to_jpeg_b64(bgr)
    except Exception:
        return ""


def _norm_depth_gray_b64(depth_norm) -> str:
    """Display env-normalized depth [0,1] (close=0) as inverted grayscale JPEG."""
    try:
        import cv2
        import numpy as np

        norm = np.clip(1.0 - depth_norm, 0.0, 1.0)
        gray = (norm * 255).astype("uint8")
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        bgr = cv2.resize(bgr, (512, 288), interpolation=cv2.INTER_CUBIC)
        return _ndarray_to_jpeg_b64(bgr, quality=88)
    except Exception:
        return ""


def _upscale_ae_bgr(bgr) -> "object":
    """Upscale native 72×128 AE frames for dashboard canvas (512×288)."""
    try:
        import cv2
        return cv2.resize(bgr, (512, 288), interpolation=cv2.INTER_CUBIC)
    except Exception:
        return bgr


def _ensure_ae_depth_batch(depth_t):
    """Force depth tensor to native AE shape (1, 1, 72, 128)."""
    import torch.nn.functional as F

    if depth_t.shape[-2] == 72 and depth_t.shape[-1] == 128:
        return depth_t
    return F.interpolate(depth_t, size=(72, 128), mode="bilinear", align_corners=False)


# ---------------------------------------------------------------------------
# Live telemetry source
# ---------------------------------------------------------------------------
class LiveDroneTelemetry:
    """Live data source for the dashboard WebSocket server.

    Call ``push(env, elapsed_secs)`` after every ``env.step()`` to feed fresh
    data into the server.  The WebSocket server calls ``tick()`` at tick_rate
    Hz to get the latest snapshot.
    """

    def __init__(self, tick_rate: float = 10.0, perf_mode: bool = False):
        self.tick_rate = tick_rate
        self.force_level: int | None = None  # required by server.py interface
        self.pending_spawn_count: int | None = None
        self._perf_mode = bool(perf_mode)

        self._lock = threading.Lock()
        self._state: dict[str, Any] = self._empty_state()
        self._tick_count = 0
        self._start_time = time.time()

        # Image cache — regenerated every N pushes to keep CPU load low
        self._image_cache: dict = {}
        self._slam3d_cache: dict = {}
        self._yolo_hud_cache: str = ""
        self._yolo_hud_left_cache: str = ""
        self._yolo_hud_right_cache: str = ""
        self._img_push_counter = 0
        self._img_regen_serial = 0
        self._slam3d_push_counter = 0
        self._img_regen_interval = 10 if perf_mode else 1
        self._slam3d_regen_interval = 3 if perf_mode else 1
        # Saliency is expensive (15 backprop samples); throttle in perf mode but
        # keep it live (every ~4th image regen) rather than disabling it entirely.
        self._saliency_regen_interval = 1
        self._yolo_upscale = 1 if perf_mode else 3
        self._yolo_jpeg_quality = 78 if perf_mode else 90
        # Always read body-mounted rear/left/right cameras — skipping them made every
        # dashboard panel show the front-camera fallback (all four views identical).
        self._skip_angle_cams = False
        self._cam_err_logged: set[str] = set()
        # Auto "camera light" — auto-exposure/gain + local contrast on RGB feeds so
        # the dim indoor drone cameras are visible on the dashboard.
        self._cam_auto_light = True

    # ------------------------------------------------------------------ #
    #  Interface expected by server.py                                     #
    # ------------------------------------------------------------------ #

    def tick(self) -> dict:
        """Return the latest telemetry snapshot (called by WS server thread)."""
        with self._lock:
            state = dict(self._state)
        self._tick_count += 1
        state["timestamp"] = time.time() - self._start_time
        state["tick"] = self._tick_count
        return state

    def _reset_to_level(self, lv: int):
        """No-op — level is determined automatically from drone position."""
        pass

    # ------------------------------------------------------------------ #
    #  Data extraction from env                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _get_map_zones(env) -> dict:
        """Return USD zone outlines (rooms + corridors) in env-local coordinates."""
        zones = getattr(env, "_map_zones", None)
        if zones:
            out: dict[str, dict] = {}
            for name, zone in sorted(zones.items()):
                lx0, lx1, ly0, ly1 = zone["bounds"]
                entry: dict = {
                    "bounds": [
                        round(float(lx0), 2), round(float(lx1), 2),
                        round(float(ly0), 2), round(float(ly1), 2),
                    ],
                }
                center = zone.get("center")
                if center is not None:
                    entry["center"] = [round(float(center[0]), 2), round(float(center[1]), 2)]
                out[name] = entry
            return out
        return MAP_ZONES

    @staticmethod
    def _get_blueprint(env) -> list:
        """Extract a 1:1 wall boundary blueprint from the USD walkable grid."""
        if getattr(env, "_blueprint_cache", None) is not None:
            return env._blueprint_cache

        walkable = getattr(env, "_walkable_grid", None)
        if walkable is None:
            return []

        try:
            import numpy as np
            ox, oy = env._walkable_grid_origin
            res = env._walkable_grid_res
            w = walkable.astype(bool)
            
            # Find boundary cells (False cells adjacent to at least one True cell)
            neighbors = np.zeros_like(w)
            neighbors[1:, :] |= w[:-1, :]   # shift down
            neighbors[:-1, :] |= w[1:, :]   # shift up
            neighbors[:, 1:] |= w[:, :-1]   # shift right
            neighbors[:, :-1] |= w[:, 1:]   # shift left
            boundary = neighbors & ~w
            
            bx, by = np.where(boundary)
            coords = []
            for ix, iy in zip(bx, by):
                wx = float(ox + (ix + 0.5) * res)
                wy = float(oy + (iy + 0.5) * res)
                coords.append([round(wx, 2), round(wy, 2)])
                
            env._blueprint_cache = coords
            print(f"[LiveTelemetry] Extracted 1:1 blueprint wall count: {len(coords)}")
            return coords
        except Exception as e:
            print(f"[LiveTelemetry] Failed to extract blueprint: {e}")
            return []

    def push(self, env, elapsed_secs: float) -> None:
        """Extract live data from a running env instance and update snapshot."""
        import numpy as np

        try:
            robot = getattr(env, "_robot", None)
            if robot is None:
                return

            brain = getattr(env, "_brain", None)
            mapper = getattr(env, "mapper", None)  # only on RealSlamDroneEnv

            # ---- Position & orientation --------------------------------
            pos_w = robot.data.root_pos_w[0].cpu().numpy()
            quat_w = robot.data.root_quat_w[0].cpu().numpy()  # [w, x, y, z]
            qw, qx, qy, qz = quat_w
            yaw   = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
            pitch = math.asin(max(-1.0, min(1.0, 2.0 * (qw * qy - qz * qx))))
            roll  = math.atan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))

            # ---- Velocities --------------------------------------------
            lin_vel = robot.data.root_lin_vel_w[0].cpu().numpy().tolist()
            ang_vel = robot.data.root_ang_vel_w[0].cpu().numpy().tolist()

            # ---- Navigation goal ---------------------------------------
            # Prefer active_frontier centroid (SLAM mode), else _desired_pos_w
            active_frontier = getattr(env, "active_frontier", None)
            if active_frontier is not None:
                cw = active_frontier.get("centroid_world", [pos_w[0], pos_w[1]])
                goal_pos = [float(cw[0]), float(cw[1]), 1.0]
            else:
                desired = getattr(env, "_desired_pos_w", None)
                if desired is not None:
                    dp = desired[0].cpu().numpy()
                    goal_pos = [float(dp[0]), float(dp[1]), float(dp[2])]
                else:
                    goal_pos = list(pos_w)

            dist_to_goal = float(np.linalg.norm(np.array(pos_w[:2]) - np.array(goal_pos[:2])))

            # OpenCV HUD parity: goal = active frontier centroid only; A* path length
            astar_path = getattr(env, "astar_path_world", []) or []
            astar_nodes = len(astar_path)
            slam_goal = None
            if active_frontier is not None:
                try:
                    cw = active_frontier["centroid_world"]
                    slam_goal = [round(float(cw[0]), 2), round(float(cw[1]), 2)]
                except Exception:
                    pass

            # ---- SLAM state & stats ------------------------------------
            slam_state = getattr(env, "slam_state", "EXPLORE")

            # people_found: brain.found_person is a bool (SlamBrainModule and BrainModule)
            people_found = 1 if (brain and getattr(brain, "found_person", False)) else 0

            # Map coverage — pure SLAM (known cells / bbox of mapped region)
            map_explored_pct = 0.0
            if mapper is not None and hasattr(mapper, "coverage_stats"):
                try:
                    visited, total = mapper.coverage_stats()
                    if total > 0:
                        map_explored_pct = visited / total * 100.0
                except Exception:
                    pass
            elif brain and hasattr(brain, "coverage_stats"):
                try:
                    visited, total = brain.coverage_stats()
                    if total > 0:
                        map_explored_pct = visited / total * 100.0
                except Exception:
                    pass

            # Frontiers — single detect_frontiers() call shared by 2D map, 3D map, and stats
            frontiers_raw = self._collect_slam_frontiers(
                mapper, (float(pos_w[0]), float(pos_w[1])), getattr(env, "_brain", None)
            )
            frontier_count = len(frontiers_raw)

            # ---- Current room from Y position --------------------------
            drone_y = float(pos_w[1])
            if drone_y > -2.5:
                level = 1
            elif drone_y > -8.5:
                level = 2
            elif drone_y > -16.5:
                level = 3
            else:
                level = 4

            # ---- PPO actions -------------------------------------------
            # BrainNavDroneEnv composes navigator output in env._actions each step.
            ppo_vx = ppo_vy = ppo_vz = ppo_yaw_rate = 0.0
            hl_actions = getattr(env, "_actions", None)
            if hl_actions is None:
                hl_actions = getattr(env, "_previous_actions", None)
            if hl_actions is not None:
                try:
                    pa = hl_actions[0].cpu().numpy()
                    ppo_vx, ppo_vy, ppo_vz, ppo_yaw_rate = (
                        float(pa[0]), float(pa[1]), float(pa[2]), float(pa[3])
                    )
                except Exception:
                    pass

            # ---- LLC motor wrench (from frozen flight controller) -------
            thrust = moment_x = moment_y = moment_z = 0.0
            thrust_buf = getattr(env, "_thrust", None)
            moment_buf = getattr(env, "_moment", None)
            if thrust_buf is not None and moment_buf is not None:
                try:
                    thrust = float(thrust_buf[0, 0, 2].item())
                    moment_x = float(moment_buf[0, 0, 0].item())
                    moment_y = float(moment_buf[0, 0, 1].item())
                    moment_z = float(moment_buf[0, 0, 2].item())
                except Exception:
                    pass

            # ---- Camera images (rate-limited) ----------------------------
            self._img_push_counter += 1
            if self._img_push_counter % self._img_regen_interval == 0 or not self._image_cache:
                self._img_regen_serial += 1
                compute_saliency = (
                    not self._perf_mode
                    or self._img_regen_serial % self._saliency_regen_interval == 0
                    or not self._image_cache.get("depth_saliency")
                )
                self._image_cache = self._grab_images(env, compute_saliency=compute_saliency)
                yolo_img = self._grab_yolo_frame_by_key(env, "_web_frame_bgr", "_yolo_hud_cache")
                if yolo_img:
                    self._image_cache["yolo_frame"] = yolo_img
                yolo_left = self._grab_yolo_frame_by_key(env, "_web_frame_left_bgr", "_yolo_hud_left_cache")
                if yolo_left:
                    self._image_cache["yolo_frame_left"] = yolo_left
                yolo_right = self._grab_yolo_frame_by_key(env, "_web_frame_right_bgr", "_yolo_hud_right_cache")
                if yolo_right:
                    self._image_cache["yolo_frame_right"] = yolo_right

            # SLAM 3D grid for native 2D/3D browser maps (rate-limited in perf mode)
            self._slam3d_push_counter += 1
            if (
                self._slam3d_push_counter % self._slam3d_regen_interval == 0
                or not self._slam3d_cache
            ):
                self._slam3d_cache = self._get_slam_3d(env, frontiers_raw=frontiers_raw)
                self._slam3d_cache["blueprint"] = self._get_blueprint(env)
                self._slam3d_cache["res"] = float(env._walkable_grid_res) if getattr(env, "_walkable_grid_res", None) is not None else 0.4

            # YOLO native-HUD payload (boxes + intel + rescue log + status)
            perception = getattr(env, "_perception", None)
            yolo_stats = self._build_yolo_stats(perception, brain)

            # ---- Spawn mission status (operator-only markers) ------------
            spawn_detected, spawn_total = (0, 0)
            if hasattr(env, "count_spawned_targets_detected"):
                spawn_detected, spawn_total = env.count_spawned_targets_detected()
            spawn_info = {
                "active": bool(getattr(env, "dynamic_spawn_active", False)),
                "total": int(spawn_total),
                "detected": int(spawn_detected),
                "pending": getattr(self, "pending_spawn_count", None),
                "coverage_required": 95.0,
            }

            # ---- Assemble state dict -----------------------------------
            state: dict[str, Any] = {
                "pos":        [round(float(v), 4) for v in pos_w],
                "roll":       round(roll, 4),
                "pitch":      round(pitch, 4),
                "yaw":        round(yaw, 4),
                "lin_vel":    [round(float(v), 4) for v in lin_vel],
                "ang_vel":    [round(float(v), 4) for v in ang_vel],
                "goal_pos":   [round(float(v), 4) for v in goal_pos],
                "dist_to_goal": round(dist_to_goal, 4),
                "slam_goal":    slam_goal,
                "astar_nodes":  astar_nodes,
                "ppo_actions": {
                    "vx": round(ppo_vx, 4), "vy": round(ppo_vy, 4),
                    "vz": round(ppo_vz, 4), "yaw_rate": round(ppo_yaw_rate, 4),
                },
                "llc_outputs": {
                    "thrust":   round(thrust, 4),
                    "moment_x": round(moment_x, 6),
                    "moment_y": round(moment_y, 6),
                    "moment_z": round(moment_z, 6),
                },
                "slam_state":       slam_state,
                "map_explored_pct": round(map_explored_pct, 2),
                "people_found":     people_found,
                "frontier_count":   frontier_count,
                "images":           self._image_cache,
                "slam_3d":          self._slam3d_cache,
                "yolo_stats":       yolo_stats,
                "level":            level,
                "level_time":       round(elapsed_secs, 2),
                "level_duration":   999.0,
                "level_mode":       "auto",
                "status":           "running",
                "room_bounds":      ROOM_BOUNDS,
                "map_zones":        self._get_map_zones(env),
                "poles":            [],
                "sim_running":      True,
                "spawn_info":       spawn_info,
            }

            with self._lock:
                self._state = state

        except Exception as exc:
            import traceback
            print(f"[LiveTelemetry] push() error: {exc}")
            traceback.print_exc()

    # ------------------------------------------------------------------ #
    #  Image extraction                                                    #
    # ------------------------------------------------------------------ #

    def _grab_images(self, env, *, compute_saliency: bool = True) -> dict:
        """Extract camera frames from the tiled camera and angle cameras.

        Feeds:
          rgb_first_person  – forward RGB (main body-mounted camera)
          rgb_third_1       – chase camera (2 m behind, 0.8 m above)
          rgb_third_2       – left-side camera (2.5 m to the left)
          rgb_third_3       – top-down camera (3 m above)
          depth             – inverted-grey depth, small
          ae_recon          – AE reconstruction
          depth_saliency    – PPO policy saliency heatmap (play_saliency.py)
          slam_map          – SLAM map for nav-tab mini panel
        """
        import numpy as np
        images: dict[str, str] = {}
        fallback_rgb_b64: str = ""
        try:
            import cv2

            tiled_cam = getattr(env, "_tiled_camera", None)
            if tiled_cam is None or tiled_cam.data.output is None:
                return images

            # --- Depth / AE / saliency (use env pipeline: depth_max=5, 72×128) ---
            depth_tensor = tiled_cam.data.output.get("depth")
            depth_proc = getattr(env, "_last_depth_processed", None)
            if depth_proc is not None and depth_proc.numel() > 0:
                depth_batch = _ensure_ae_depth_batch(depth_proc[0:1].clone())
                depth_norm = depth_batch[0, 0].detach().cpu().numpy()
                images["depth"] = _norm_depth_gray_b64(depth_norm)
                ae_recon = self._grab_ae_recon(env, depth_batch)
                if ae_recon:
                    images["ae_recon"] = ae_recon
                elif self._image_cache.get("ae_recon"):
                    images["ae_recon"] = self._image_cache["ae_recon"]

                saliency = ""
                if compute_saliency:
                    saliency = self._grab_policy_saliency(env, depth_batch)
                if saliency:
                    images["depth_saliency"] = saliency
                elif self._image_cache.get("depth_saliency"):
                    images["depth_saliency"] = self._image_cache["depth_saliency"]
            elif depth_tensor is not None:
                # Fallback when obs not built yet (first frames after reset)
                depth_np = depth_tensor[0].squeeze().detach().cpu().numpy().astype("float32")
                depth_np = np.nan_to_num(depth_np, nan=10.0, posinf=10.0, neginf=0.0)
                depth_max = float(getattr(getattr(env, "cfg", None), "depth_max", 5.0))
                depth_small = cv2.resize(depth_np, (128, 72), interpolation=cv2.INTER_AREA)
                depth_norm = np.clip(depth_small / depth_max, 0.0, 1.0)
                images["depth"] = _norm_depth_gray_b64(depth_norm)
                images["ae_recon"] = images["depth"]
                images["depth_saliency"] = images["depth"]

            # --- Forward RGB (first-person / nav camera) ---
            rgb_tensor = tiled_cam.data.output.get("rgb")
            if rgb_tensor is not None:
                rgb_np = rgb_tensor[0].cpu().numpy()[:, :, :3]
                bgr = cv2.cvtColor(rgb_np.astype("uint8"), cv2.COLOR_RGB2BGR)
                bgr_small = cv2.resize(bgr, (320, 180), interpolation=cv2.INTER_AREA)
                if getattr(self, "_cam_auto_light", True):
                    bgr_small = _auto_brighten(bgr_small)
                fallback_rgb_b64 = _ndarray_to_jpeg_b64(bgr_small)
                images["rgb_first_person"] = fallback_rgb_b64

            # --- Real drone-mounted cameras: rear (behind), left side, right side ---
            # These are physically attached to the drone (rear = Camera_View, sides =
            # SLAM mapping cameras), so they always move with it — unlike the old
            # world-anchored follow cameras which sat outside the map.
            if not getattr(self, "_skip_angle_cams", False):
                _angle_cams = [
                    ("_view_camera",       "rgb_third_1"),  # rear / behind
                    ("_view_left_camera",  "rgb_third_2"),  # left side
                    ("_view_right_camera", "rgb_third_3"),  # right side
                ]
                for cam_attr, img_key in _angle_cams:
                    cam = getattr(env, cam_attr, None)
                    if cam is None:
                        cam = getattr(getattr(env, "unwrapped", env), cam_attr, None)
                    captured = False
                    if cam is not None:
                        try:
                            out = cam.data.output
                            if out is None:
                                raise ValueError("camera output is None")
                            rgb_out = out.get("rgb")
                            if rgb_out is None:
                                rgb_out = out.get("rgba")
                            if rgb_out is not None and rgb_out.numel() > 0:
                                rgb_np = rgb_out[0].detach().cpu().numpy()
                                if rgb_np.ndim == 3 and rgb_np.shape[-1] >= 3:
                                    rgb_np = rgb_np[:, :, :3]
                                bgr = cv2.cvtColor(rgb_np.astype("uint8"), cv2.COLOR_RGB2BGR)
                                bgr_small = cv2.resize(bgr, (320, 180), interpolation=cv2.INTER_AREA)
                                if getattr(self, "_cam_auto_light", True):
                                    bgr_small = _auto_brighten(bgr_small)
                                images[img_key] = _ndarray_to_jpeg_b64(bgr_small)
                                captured = True
                        except Exception as exc:
                            if cam_attr not in self._cam_err_logged:
                                self._cam_err_logged.add(cam_attr)
                                print(f"[LiveTelemetry] {cam_attr} capture failed: {exc}")
                    if not captured and fallback_rgb_b64:
                        images[img_key] = fallback_rgb_b64
            elif fallback_rgb_b64:
                for key in ("rgb_third_1", "rgb_third_2", "rgb_third_3"):
                    images[key] = fallback_rgb_b64

        except Exception as exc:
            print(f"[LiveTelemetry] _grab_images() error: {exc}")

        return images

    @staticmethod
    def _collect_slam_frontiers(mapper, drone_xy=None, brain=None) -> list:
        """Return frontier dicts from the mapper, filtered by pure-SLAM reachability.

        Only frontiers the drone can actually reach through observed free space are
        shown. This drops "blue targets" that leaked outside the rooms or got walled
        off — using the drone's own occupancy grid, NOT the ground-truth/USD map.
        """
        if mapper is None:
            return []
        try:
            import numpy as np
            # Match the drone's own selection: BFS-reachable frontiers only, then the
            # brain's direction-gated visited filter. Falls back to plain detection if
            # we don't have the drone position or the newer mapper method.
            if drone_xy is not None and hasattr(mapper, "find_reachable_frontiers"):
                start_r, start_c = mapper.world_to_grid(float(drone_xy[0]), float(drone_xy[1]))
                frontiers, came_from = mapper.find_reachable_frontiers(start_r, start_c, min_size=3)
                # Match the picker: hide occlusion-shadow pockets (tiny unknown gain)
                # so displayed blue targets are the ones the drone would actually chase.
                substantial = [f for f in frontiers if int(f.get("unknown_gain", 0)) >= 20]
                if substantial:
                    frontiers = substantial
                explorable = getattr(brain, "is_explorable_frontier", None)
                if callable(explorable) and drone_xy is not None:
                    dxy = np.array(drone_xy[:2])
                    frontiers = [
                        f for f in frontiers
                        if explorable(f, dxy, came_from=came_from)
                    ]
                return frontiers
            else:
                frontiers = mapper.detect_frontiers()
            ahead = getattr(brain, "is_frontier_ahead", None)
            if callable(ahead):
                frontiers = [f for f in frontiers if ahead(f["centroid_world"])]
            return frontiers
        except Exception:
            return []

    def _render_slam_map(self, env, frontiers_raw: list | None = None) -> str:
        """Render the SLAM occupancy grid — same visual logic as OpenCV draw_slam_visualizer."""
        try:
            import cv2
            import numpy as np

            mapper = getattr(env, "mapper", None)
            if mapper is None:
                return ""

            robot = getattr(env, "_robot", None)
            if robot is None:
                return ""

            brain    = getattr(env, "_brain", None)
            pos_w    = robot.data.root_pos_w[0].cpu().numpy()
            quat_w   = robot.data.root_quat_w[0].cpu().numpy()
            qw, qx, qy, qz = quat_w
            yaw      = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))

            person_found = bool(getattr(brain, "found_person", False)) if brain else False
            person_pos   = getattr(brain, "target_person_pos", None) if brain else None
            active_frontier = getattr(env, "active_frontier", None)
            astar_path      = getattr(env, "astar_path_world", [])

            prob     = mapper.get_occupancy_grid()
            grid_h, grid_w = prob.shape

            # Split walls vs. dodgeable props (SLAM-only) so the 2D map paints
            # them differently — same classification the planner uses.
            if hasattr(mapper, "get_wall_obstacle_masks"):
                wall_mask, obstacle_mask = mapper.get_wall_obstacle_masks(use_walkable=False)
                danger = mapper.get_planning_grid()
            else:
                wall_mask = (prob > 0.65).astype(np.uint8)
                obstacle_mask = np.zeros_like(wall_mask)
                danger = mapper.get_inflated_grid()

            # HD output (2× OpenCV) — PNG lossless for sharp walls/overlays
            map_w, map_h = 900, 1200

            canvas = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
            canvas[(prob >= 0.35) & (prob <= 0.65)] = [25, 18, 12]
            canvas[prob < 0.35]                      = [45, 38, 30]
            canvas[danger == 1]                      = [30, 20, 75]
            canvas[obstacle_mask == 1]               = [140, 156, 31]   # teal — dodgeable props
            canvas[wall_mask == 1]                   = [255, 230, 80]    # amber — structural walls

            # NEAREST keeps occupancy cells crisp before overlay drawing
            canvas_s = cv2.resize(canvas, (map_w, map_h), interpolation=cv2.INTER_NEAREST)
            # North-up: Room 1 at top, Room 4 at bottom (same as OpenCV)
            canvas_s = cv2.flip(canvas_s, 0)

            # Radar grid overlay (scaled for HD)
            grid_step = 60
            for x in range(0, map_w, grid_step):
                cv2.line(canvas_s, (x, 0), (x, map_h), (35, 28, 20), 1, cv2.LINE_AA)
            for y in range(0, map_h, grid_step):
                cv2.line(canvas_s, (0, y), (map_w, y), (35, 28, 20), 1, cv2.LINE_AA)

            def to_disp(wx, wy):
                r, c = mapper.world_to_grid(wx, wy)
                cx = int(np.clip(c * (map_w / grid_w), 0, map_w - 1))
                cy_raw = int(np.clip(r * (map_h / grid_h), 0, map_h - 1))
                cy = map_h - 1 - cy_raw
                return cx, cy

            # A* path (dual stroke like OpenCV, scaled for HD)
            if astar_path and len(astar_path) > 1:
                for i in range(len(astar_path) - 1):
                    p0 = to_disp(astar_path[i][0],   astar_path[i][1])
                    p1 = to_disp(astar_path[i + 1][0], astar_path[i + 1][1])
                    cv2.line(canvas_s, p0, p1, (100, 255, 100), 6, cv2.LINE_AA)
                    cv2.line(canvas_s, p0, p1, (0, 255, 0), 3, cv2.LINE_AA)

            if frontiers_raw is None:
                _dxy = env._robot.data.root_pos_w[0]
                frontiers_raw = self._collect_slam_frontiers(
                    mapper, (float(_dxy[0]), float(_dxy[1])), getattr(env, "_brain", None)
                )
            try:
                for f in frontiers_raw:
                    if active_frontier is None or np.linalg.norm(
                        np.array(f["centroid_world"]) - np.array(active_frontier["centroid_world"])
                    ) > 0.1:
                        cx, cy = to_disp(f["centroid_world"][0], f["centroid_world"][1])
                        cv2.circle(canvas_s, (cx, cy), 10, (255, 180, 50), -1, cv2.LINE_AA)
                        cv2.circle(canvas_s, (cx, cy), 4, (255, 255, 255), -1, cv2.LINE_AA)
            except Exception:
                pass

            # Active frontier (gold crosshair)
            if active_frontier is not None:
                cx, cy = to_disp(active_frontier["centroid_world"][0], active_frontier["centroid_world"][1])
                cv2.circle(canvas_s, (cx, cy), 18, (0, 200, 255), 2, cv2.LINE_AA)
                cv2.line(canvas_s, (cx - 22, cy), (cx + 22, cy), (0, 200, 255), 2, cv2.LINE_AA)
                cv2.line(canvas_s, (cx, cy - 22), (cx, cy + 22), (0, 200, 255), 2, cv2.LINE_AA)
                cv2.circle(canvas_s, (cx, cy), 6, (0, 0, 255), -1, cv2.LINE_AA)

            # Person marker
            if person_found and person_pos is not None:
                try:
                    px, py = to_disp(float(person_pos[0]), float(person_pos[1]))
                    cv2.drawMarker(canvas_s, (px, py), (255, 0, 255),
                                   cv2.MARKER_STAR, 24, 3, cv2.LINE_AA)
                except Exception:
                    pass

            # Drone triangle + FOV cone
            d_cx, d_cy = to_disp(pos_w[0], pos_w[1])
            dy = -yaw
            p_nose = (int(d_cx + 20 * math.cos(dy)),       int(d_cy + 20 * math.sin(dy)))
            p_l    = (int(d_cx + 12 * math.cos(dy + 2.4)), int(d_cy + 12 * math.sin(dy + 2.4)))
            p_r    = (int(d_cx + 12 * math.cos(dy - 2.4)), int(d_cy + 12 * math.sin(dy - 2.4)))
            pts = np.array([p_nose, p_l, p_r], np.int32)
            cv2.fillPoly(canvas_s, [pts], (0, 230, 0))
            cv2.polylines(canvas_s, [pts], True, (255, 255, 255), 2, cv2.LINE_AA)

            fov_half = 0.7
            p_cl = (int(d_cx + 50 * math.cos(dy + fov_half)), int(d_cy + 50 * math.sin(dy + fov_half)))
            p_cr = (int(d_cx + 50 * math.cos(dy - fov_half)), int(d_cy + 50 * math.sin(dy - fov_half)))
            cv2.line(canvas_s, (d_cx, d_cy), p_cl, (0, 200, 0), 2, cv2.LINE_AA)
            cv2.line(canvas_s, (d_cx, d_cy), p_cr, (0, 200, 0), 2, cv2.LINE_AA)

            return _ndarray_to_png_b64(canvas_s)

        except Exception as exc:
            print(f"[LiveTelemetry] _render_slam_map() error: {exc}")
            return ""

    def _get_slam_3d(self, env, frontiers_raw: list | None = None) -> dict:
        """Build compact 3D SLAM telemetry dict for the Three.js SLAM scene.

        Returns a dict with:
          grid   – base64 Uint8 flat array (row-major, downsampled)
          H, W   – grid dimensions after downsampling
          min_x/max_x/min_y/max_y – world bounds of the grid
          cell_w, cell_d – world-space cell size after downsampling
          drone  – {x, y, z, yaw}
          frontiers – list of [x, y] centroids (all, same set as 2D map)
          active – active frontier [x, y] or null
          path   – A* path [[x,y], …] every 4th point
          person – detected person [x, y] or null
        """
        try:
            import numpy as np

            mapper = getattr(env, "mapper", None)
            if mapper is None:
                return {}

            robot = getattr(env, "_robot", None)
            if robot is None:
                return {}

            brain = getattr(env, "_brain", None)

            # ---- Grid: quantise at FULL resolution, then max-pool downsample ----
            # High fidelity: quantise every native cell first so no occupied
            # cell is ever thrown away (stride-sampling used to drop walls,
            # which is why the 3D map looked incomplete / "not the full map").
            prob     = mapper.get_occupancy_grid()   # (H, W) float32 in [0,1]
            H_orig, W_orig = prob.shape

            # Split occupied cells into structural walls vs. dodgeable props so the
            # 3D scene can paint them differently (props are not walls the drone
            # must route around). Danger = inflation around WALLS only.
            if hasattr(mapper, "get_wall_obstacle_masks"):
                wall_mask, obstacle_mask = mapper.get_wall_obstacle_masks(use_walkable=True)
                danger = mapper.get_planning_grid()
            else:
                inflated = mapper.get_inflated_grid()
                wall_mask = (prob > 0.65).astype(np.uint8)
                obstacle_mask = np.zeros_like(wall_mask)
                danger = inflated

            # Quantise full-res: 0=unknown, 1=free, 2=danger(wall inflation),
            # 3=dodgeable obstacle, 4=structural wall.
            # Ordering matters — higher value wins so max-pool preserves walls.
            full = np.zeros((H_orig, W_orig), dtype=np.uint8)
            full[prob < 0.35]     = 1
            full[danger == 1]     = 2
            full[obstacle_mask == 1] = 3
            full[wall_mask == 1]  = 4

            # Downsample by block-MAX so every wall cell survives. Target a
            # larger max dimension (200) than before for a fuller, denser map.
            TARGET_MAX = 200
            ds = max(1, int(np.ceil(max(H_orig, W_orig) / TARGET_MAX)))
            if ds > 1:
                Hc = (H_orig // ds) * ds
                Wc = (W_orig // ds) * ds
                cropped = full[:Hc, :Wc]
                grid = cropped.reshape(Hc // ds, ds, Wc // ds, ds).max(axis=(1, 3))
            else:
                grid = full
            grid = np.ascontiguousarray(grid, dtype=np.uint8)
            H, W = grid.shape

            grid_bytes = grid.tobytes()
            grid_b64 = base64.b64encode(grid_bytes).decode("ascii")
            # Monotonic version so the client rebuilds whenever the map changes
            # anywhere (previously it only compared the last few bytes → static).
            occupied_count = int(((grid == 3) | (grid == 4)).sum())
            grid_ver = int(zlib.adler32(grid_bytes) & 0xFFFFFFFF)

            # ---- Drone pose ----
            pos_w  = robot.data.root_pos_w[0].cpu().numpy()
            quat_w = robot.data.root_quat_w[0].cpu().numpy()
            qw, qx, qy, qz = quat_w
            yaw = math.atan2(2.0 * (qw * qz + qx * qy),
                             1.0 - 2.0 * (qy * qy + qz * qz))

            # ---- Frontiers (same list as 2D map / OpenCV visualizer) ----
            frontiers_list: list = []
            active_frontier = getattr(env, "active_frontier", None)
            if frontiers_raw is None:
                _dxy = env._robot.data.root_pos_w[0]
                frontiers_raw = self._collect_slam_frontiers(
                    mapper, (float(_dxy[0]), float(_dxy[1])), getattr(env, "_brain", None)
                )
            for f in frontiers_raw:
                try:
                    cw = f["centroid_world"]
                    frontiers_list.append([round(float(cw[0]), 3), round(float(cw[1]), 3)])
                except Exception:
                    pass

            # ---- Active frontier ----
            active_data = None
            if active_frontier is not None:
                try:
                    cw = active_frontier["centroid_world"]
                    active_data = [round(float(cw[0]), 3), round(float(cw[1]), 3)]
                except Exception:
                    pass

            # ---- A* path (every 4th point to limit size) ----
            astar_path = getattr(env, "astar_path_world", [])
            path_list  = [[round(float(p[0]), 3), round(float(p[1]), 3)]
                          for p in astar_path[::4]]

            # ---- Person ----
            person_found = False
            person_pos = None
            rescued_list = []
            yolo_thresh = float(getattr(env.cfg, "yolo_person_conf_threshold", 0.70))
            if brain is not None:
                rescued = getattr(brain, "rescued_people", None)
                rescued_conf = getattr(brain, "rescued_people_conf", []) or []
                if rescued:
                    for idx, p in enumerate(rescued):
                        conf = float(rescued_conf[idx]) if idx < len(rescued_conf) else yolo_thresh
                        if conf < yolo_thresh:
                            continue
                        rescued_list.append([round(float(p[0]), 3), round(float(p[1]), 3)])
                if rescued_list:
                    person_found = True
                    person_pos = rescued_list[0]
                elif getattr(brain, "found_person", False):
                    person_found = True
                    person_pos = getattr(brain, "target_person_pos", None)

            person_data  = None
            if person_found and person_pos is not None:
                try:
                    person_data = [round(float(person_pos[0]), 3),
                                   round(float(person_pos[1]), 3)]
                except Exception:
                    pass

            # ---- Spawned Targets ----
            spawned_targets = []
            origin = env._terrain.env_origins[0].cpu().numpy() if hasattr(env, "_terrain") else np.zeros(3)
            local_targets = getattr(env.unwrapped, "spawned_targets_local", []) or []
            if not local_targets:
                cfg = getattr(env.unwrapped, "cfg", None)
                if cfg is not None:
                    room3 = getattr(cfg, "brain_room3_person_local", None)
                    final_a = getattr(cfg, "brain_final_person_local", None)
                    if room3 is not None and final_a is not None:
                        local_targets = [room3, final_a]
            for t in local_targets:
                spawned_targets.append([round(float(t[0] + origin[0]), 3), round(float(t[1] + origin[1]), 3)])

            cell_w = (mapper.max_x - mapper.min_x) / W
            cell_d = (mapper.max_y - mapper.min_y) / H

            return {
                "grid":   grid_b64,
                "gver":   grid_ver,
                "occupied": occupied_count,
                "H": int(H), "W": int(W),
                "min_x": float(mapper.min_x), "max_x": float(mapper.max_x),
                "min_y": float(mapper.min_y), "max_y": float(mapper.max_y),
                "cell_w": round(float(cell_w), 4),
                "cell_d": round(float(cell_d), 4),
                "drone": {
                    "x":   round(float(pos_w[0]), 3),
                    "y":   round(float(pos_w[1]), 3),
                    "z":   round(float(pos_w[2]), 3),
                    "yaw": round(float(yaw), 4),
                },
                "frontiers": frontiers_list,
                "active":    active_data,
                "path":      path_list,
                "person":    person_data,
                "persons":   rescued_list,
                "spawned_targets": spawned_targets,
            }

        except Exception as exc:
            print(f"[LiveTelemetry] _get_slam_3d() error: {exc}")
            return {}

    def _grab_yolo_frame(self, env) -> str:
        return self._grab_yolo_frame_by_key(env, "_web_frame_bgr", "_yolo_hud_cache")

    def _grab_yolo_frame_by_key(self, env, key: str, cache_attr: str) -> str:
        """Return base64 JPEG of the *clean* camera frame perception ran YOLO on.

        This is the exact frame the normalized boxes in ``yolo_stats`` were
        computed from, so the native web HUD overlays them in perfect sync.
        Upscaled 2x (Lanczos) for a crisp HD backdrop, and auto-brightened.
        """
        try:
            import cv2

            perception = getattr(env, "_perception", None)
            if perception is None:
                return getattr(self, cache_attr, "")

            frame = getattr(perception, key, None)
            if frame is None:
                return getattr(self, cache_attr, "")

            h, w = frame.shape[:2]
            scale = max(1, int(getattr(self, "_yolo_upscale", 2)))
            if scale > 1:
                frame = cv2.resize(frame, (w * scale, h * scale), interpolation=cv2.INTER_LINEAR)
            
            # Always apply the auto-brightener so it matches the Cameras page light level
            frame = _auto_brighten(frame)

            quality = int(getattr(self, "_yolo_jpeg_quality", 92))
            b64 = _ndarray_to_jpeg_b64(frame, quality=quality)
            setattr(self, cache_attr, b64)
            return b64
        except Exception as exc:
            print(f"[LiveTelemetry] _grab_yolo_frame_by_key({key}) error: {exc}")
            return getattr(self, cache_attr, "")

    @staticmethod
    def _is_generic_camera_view(label: str | None = None, person_key: str | None = None) -> bool:
        """True only for the placeholder row (no room / scan / rescue slot)."""
        lab = str(label or "").strip().upper().replace("_", " ")
        pk = str(person_key or "").strip().lower().replace(" ", "_")
        return lab == "CAMERA VIEW" and pk in ("camera_view", "cameraview", "")

    @staticmethod
    def _normalize_rescue_entry(label: str, key: str) -> tuple[str, str]:
        """Map legacy placeholder rows to a meaningful rescue-log label."""
        if LiveDroneTelemetry._is_generic_camera_view(label, key):
            return "PERSON DETECTED", "person_detected"
        return label, key

    @staticmethod
    def _build_yolo_stats(perception, brain) -> dict:
        """Assemble the full native-HUD payload from the perception module."""
        if perception is None:
            return {}

        # ---- Confidence (persistent peak vs instantaneous) ----
        current_conf = float(getattr(perception, "last_best_person_conf", 0.0))
        state = getattr(perception, "_web_state", None) or {}
        try:
            current_conf = max(current_conf, float(state.get("display_conf", 0.0)))
        except (TypeError, ValueError):
            pass
        peak_conf = current_conf
        try:
            peak_conf = max(
                peak_conf,
                float(state.get("display_conf", 0.0)),
                float(state.get("alert_conf", 0.0)),
            )
        except (TypeError, ValueError):
            pass
        pbc = getattr(perception, "_person_best_conf", None)
        if isinstance(pbc, dict) and pbc:
            peak_conf = max(peak_conf, max(float(v) for v in pbc.values()))

        intel = getattr(perception, "_last_intel", None)
        if isinstance(intel, dict):
            try:
                peak_conf = max(peak_conf, float(intel.get("conf", 0.0)))
            except (TypeError, ValueError):
                pass

        # ---- Rescue log snapshot (same source as OpenCV sidebar: _detection_log) ----
        rescue_log = []
        for entry in (getattr(perception, "_detection_log", None) or [])[:30]:
            try:
                label = str(entry.get("label", ""))
                key = str(entry.get("person_key", ""))
                label, key = LiveDroneTelemetry._normalize_rescue_entry(label, key)
                rescue_log.append({
                    "label":   label,
                    "conf":    round(float(entry.get("conf", 0.0)), 4),
                    "gps_lat": entry.get("gps_lat"),
                    "gps_lon": entry.get("gps_lon"),
                    "xyz":     entry.get("xyz"),
                    "key":     key,
                    "frame":   int(entry.get("frame", 0)),
                })
            except Exception:
                continue

        # ---- Intel panel ----
        intel_out = None
        if isinstance(intel, dict):
            ilabel, _ = LiveDroneTelemetry._normalize_rescue_entry(
                str(intel.get("label", "")), "person_detected"
            )
            intel_out = {
                "label":   ilabel.upper(),
                "conf":    round(float(intel.get("conf", 0.0)), 4),
                "gps_lat": intel.get("gps_lat"),
                "gps_lon": intel.get("gps_lon"),
                "dist":    intel.get("dist"),
            }

        brain_rescued = bool(getattr(brain, "found_person", False)) if brain else False
        yolo_seen = bool(getattr(perception, "person_ever_detected", False))
        person_ui = (
            brain_rescued
            or bool(state.get("has_confirmed"))
            or yolo_seen
            or bool(state.get("operator_alert"))
        )
        thresh = float(getattr(perception, "person_conf_threshold", 0.7))

        # ---- Canonical detection state (single source of truth for the UI) ----
        if brain_rescued or state.get("has_confirmed"):
            status, status_label = "confirmed", "TARGET CONFIRMED"
        elif peak_conf >= thresh or state.get("operator_alert"):
            status, status_label = "detected", "HUMAN DETECTED"
        elif peak_conf > 0.0:
            status, status_label = "seen", "CONTACT · TRACKING"
        else:
            status, status_label = "idle", "SCANNING"

        # Get list of captured yolo frames in static/yolo_saves directory (newest first)
        captured_frames = []
        try:
            from pathlib import Path
            dash_saves = Path(r"D:\isaac\3D_Drone_RL\scripts\dashboard\static\yolo_saves")
            if dash_saves.exists():
                captured_frames = [f.name for f in sorted(dash_saves.glob("*.jpg"), reverse=True)]
        except Exception:
            pass

        return {
            "conf_threshold":  round(thresh, 3),
            "best_conf":       round(peak_conf, 4),
            "current_conf":    round(current_conf, 4),
            "detection_count": int(getattr(perception, "detection_count", 0)),
            "person_found":    person_ui,
            "person_seen":     yolo_seen,
            "rescue_log_count": len(rescue_log),
            "status":          status,
            "status_label":    status_label,
            "scan_label":      state.get("scan_label"),
            "operator_alert":  bool(state.get("operator_alert", False)),
            "alert_conf":      round(float(state.get("alert_conf", 0.0)), 4),
            "boxes":           list(getattr(perception, "_web_boxes", []) or []),
            "boxes_left":      list(getattr(perception, "_web_boxes_left", []) or []),
            "boxes_right":     list(getattr(perception, "_web_boxes_right", []) or []),
            "intel":           intel_out,
            "rescue_log":      rescue_log,
            "captured_frames":  captured_frames,
        }

    def _grab_ae_recon(self, env, depth_t) -> str:
        """Run AE encode→decode on env-normalized depth (1,1,72,128) in [0,1]."""
        try:
            import cv2
            import numpy as np
            import torch

            ae = getattr(env, "ae", None)
            if ae is None:
                return ""

            device = next(ae.parameters()).device
            depth_t = _ensure_ae_depth_batch(depth_t.to(device))

            with torch.no_grad():
                z     = ae.encode(depth_t)
                recon = ae.decode(z)

            recon_np = recon[0, 0].cpu().numpy()
            recon_u8 = (np.clip(recon_np, 0.0, 1.0) * 255).astype("uint8")
            bgr      = cv2.cvtColor(recon_u8, cv2.COLOR_GRAY2BGR)
            bgr      = _upscale_ae_bgr(bgr)
            return _ndarray_to_jpeg_b64(bgr, quality=88)
        except Exception as exc:
            if not getattr(self, "_ae_recon_err_logged", False):
                print(f"[LiveTelemetry] AE recon failed: {exc}")
                self._ae_recon_err_logged = True
            return ""

    def _grab_policy_saliency(self, env, depth_t) -> str:
        """PPO actor saliency w.r.t. depth pixels — matches scripts/play_saliency.py."""
        try:
            import cv2
            import numpy as np
            import torch

            actor = getattr(env, "_navigator_actor", None)
            ae    = getattr(env, "ae", None)
            if actor is None or ae is None:
                return ""

            device = next(ae.parameters()).device
            depth_t = _ensure_ae_depth_batch(depth_t.to(device))

            # Non-image state features from the same 77-dim policy obs the navigator uses
            obs_dict = env._get_observations()
            policy_obs = obs_dict["policy"][0:1].to(device)
            latent_dim = int(getattr(getattr(env, "cfg", None), "ae_latent_dim", 64))
            state_features = policy_obs[:, latent_dim:].clone().detach()

            ae_encoder = ae.encoder
            ae_fc_z    = ae.fc_z
            # LayerNorm applied to the latent — the policy consumes ae.encode(x) =
            # ln_z(fc_z(encoder(x))), so the saliency graph must include it too or
            # the gradients won't reflect the real 64-dim latent the navigator sees.
            ae_ln_z    = getattr(ae, "ln_z", None)
            actor_mlp  = actor.mlp
            actor_norm = actor.obs_normalizer
            actor_det  = None
            if actor.distribution is not None:
                actor_det = actor.distribution.as_deterministic_output_module()

            n_samples   = 15
            noise_level = 0.10
            total_grad  = torch.zeros(72, 128, device=device)

            ae.eval()
            actor.eval()
            for _ in range(n_samples):
                noise       = torch.randn_like(depth_t) * noise_level
                noisy_depth = (depth_t + noise).clamp(0.0, 1.0)
                depth_input = noisy_depth.clone().detach().requires_grad_(True)

                h     = ae_encoder(depth_input)
                z_img = ae_fc_z(h)
                if ae_ln_z is not None:
                    z_img = ae_ln_z(z_img)
                obs   = torch.cat([z_img, state_features], dim=-1)
                obs   = actor_norm(obs)
                out   = actor_mlp(obs)
                if actor_det is not None:
                    out = actor_det(out)
                loss = out.abs().sum()
                loss.backward()

                if depth_input.grad is not None:
                    total_grad += depth_input.grad.abs().squeeze(0).squeeze(0)

            saliency = total_grad / n_samples
            depth_np = depth_t[0, 0].detach().cpu().numpy()

            proximity = 1.0 - depth_np
            sal_np    = saliency.detach().cpu().numpy() * proximity
            s_min, s_max = sal_np.min(), sal_np.max()
            if s_max - s_min > 1e-8:
                sal_np = (sal_np - s_min) / (s_max - s_min)
            else:
                sal_np = np.zeros_like(sal_np)

            depth_u8  = (np.clip(depth_np, 0.0, 1.0) * 255).astype("uint8")
            depth_bgr = cv2.cvtColor(depth_u8, cv2.COLOR_GRAY2BGR)
            heatmap   = cv2.applyColorMap((sal_np * 255).astype("uint8"), cv2.COLORMAP_JET)
            overlay   = cv2.addWeighted(depth_bgr, 0.4, heatmap, 0.6, 0)
            overlay   = _upscale_ae_bgr(overlay)
            return _ndarray_to_jpeg_b64(overlay, quality=88)
        except Exception as exc:
            if not getattr(self, "_saliency_err_logged", False):
                print(f"[LiveTelemetry] Policy saliency failed: {exc}")
                self._saliency_err_logged = True
            return ""

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _empty_state() -> dict:
        return {
            "pos": [0.0, 0.0, 1.0],
            "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
            "lin_vel": [0.0, 0.0, 0.0],
            "ang_vel": [0.0, 0.0, 0.0],
            "goal_pos": [0.0, 0.0, 1.0],
            "dist_to_goal": 0.0,
            "slam_goal": None,
            "astar_nodes": 0,
            "ppo_actions": {"vx": 0.0, "vy": 0.0, "vz": 0.0, "yaw_rate": 0.0},
            "llc_outputs": {"thrust": 0.0, "moment_x": 0.0, "moment_y": 0.0, "moment_z": 0.0},
            "slam_state": "INIT",
            "map_explored_pct": 0.0,
            "people_found": 0,
            "frontier_count": 0,
            "images": {},
            "slam_3d": {},
            "yolo_stats": {},
            "level": 1,
            "level_time": 0.0,
            "level_duration": 999.0,
            "level_mode": "auto",
            "status": "waiting",
            "room_bounds": ROOM_BOUNDS,
            "map_zones":   MAP_ZONES,
            "poles": [],
        }
