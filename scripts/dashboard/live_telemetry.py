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


# ---------------------------------------------------------------------------
# Live telemetry source
# ---------------------------------------------------------------------------
class LiveDroneTelemetry:
    """Live data source for the dashboard WebSocket server.

    Call ``push(env, elapsed_secs)`` after every ``env.step()`` to feed fresh
    data into the server.  The WebSocket server calls ``tick()`` at tick_rate
    Hz to get the latest snapshot.
    """

    def __init__(self, tick_rate: float = 10.0):
        self.tick_rate = tick_rate
        self.force_level: int | None = None  # required by server.py interface

        self._lock = threading.Lock()
        self._state: dict[str, Any] = self._empty_state()
        self._tick_count = 0
        self._start_time = time.time()

        # Image cache — regenerated every N pushes to keep CPU load low
        self._image_cache: dict = {}
        self._slam3d_cache: dict = {}
        self._yolo_hud_cache: str = ""
        self._img_push_counter = 0
        self._img_regen_interval = 4   # regenerate every 4 env steps (~15 Hz at 60 fps)

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

            # ---- SLAM state & stats ------------------------------------
            slam_state = getattr(env, "slam_state", "EXPLORE")

            # people_found: brain.found_person is a bool (SlamBrainModule and BrainModule)
            people_found = 1 if (brain and getattr(brain, "found_person", False)) else 0

            # Map coverage from SLAM occupancy grid (preferred) or sequential coverage grid
            map_explored_pct = 0.0
            if mapper is not None:
                try:
                    prob = mapper.get_occupancy_grid()
                    # Cells that are known (free or occupied) vs total
                    known = int(np.sum(prob < 0.35) + np.sum(prob > 0.65))
                    total = prob.size
                    if total > 0:
                        map_explored_pct = known / total * 100.0
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
            frontiers_raw = self._collect_slam_frontiers(mapper)
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
            # Try to read the last applied actions; fall back to position-delta proxy
            ppo_vx = ppo_vy = ppo_vz = ppo_yaw_rate = 0.0
            prev_actions = getattr(env, "_previous_actions", None)
            if prev_actions is not None:
                try:
                    pa = prev_actions[0].cpu().numpy()
                    ppo_vx, ppo_vy, ppo_vz, ppo_yaw_rate = float(pa[0]), float(pa[1]), float(pa[2]), float(pa[3])
                except Exception:
                    pass
            else:
                desired = getattr(env, "_desired_pos_w", None)
                if desired is not None:
                    dp = desired[0].cpu().numpy()
                    ppo_vx = float(np.clip((dp[0] - pos_w[0]) / 1.0, -1, 1))
                    ppo_vy = float(np.clip((dp[1] - pos_w[1]) / 1.0, -1, 1))
                    ppo_vz = float(np.clip((dp[2] - pos_w[2]) / 0.5, -1, 1))
                    ppo_yaw_rate = float(np.clip(ang_vel[2] / 3.0, -1, 1))

            # ---- LLC / thrust estimation --------------------------------
            # Isaac Lab CF2X uses dummy joint actuators — real thrust isn't exposed.
            # Estimate from vertical acceleration proxy: thrust ≈ mass × (g + az)
            thrust = 0.027 * 9.81  # hover thrust for 27 g CF2X
            try:
                az = float(robot.data.root_lin_vel_b[0, 2].item())
                thrust = max(0.0, 0.027 * (9.81 + az * 3.0))
            except Exception:
                pass
            moment_x = float(ang_vel[0]) * 0.0005
            moment_y = float(ang_vel[1]) * 0.0005
            moment_z = float(ang_vel[2]) * 0.0005

            # ---- Camera images (rate-limited) ----------------------------
            self._img_push_counter += 1
            if self._img_push_counter % self._img_regen_interval == 0 or not self._image_cache:
                self._image_cache = self._grab_images(env)

            # SLAM 3D grid + YOLO frame: refresh every push (zero-delay web sync).
            # The 2D SLAM radar is now rendered natively in the browser from this
            # same slam_3d grid, so we no longer encode a heavy SLAM PNG here.
            self._slam3d_cache = self._get_slam_3d(env, frontiers_raw=frontiers_raw)
            yolo_img = self._grab_yolo_frame(env)
            if yolo_img:
                if not self._image_cache:
                    self._image_cache = {}
                # Clean camera frame — native component overlays boxes itself.
                self._image_cache["yolo_frame"] = yolo_img

            # YOLO native-HUD payload (boxes + intel + rescue log + status)
            perception = getattr(env, "_perception", None)
            yolo_stats = self._build_yolo_stats(perception, brain)

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

    def _grab_images(self, env) -> dict:
        """Extract camera frames from the tiled camera and angle cameras.

        Feeds:
          rgb_first_person  – forward RGB (main body-mounted camera)
          rgb_third_1       – chase camera (2 m behind, 0.8 m above)
          rgb_third_2       – left-side camera (2.5 m to the left)
          rgb_third_3       – top-down camera (3 m above)
          depth             – inverted-grey depth, small
          ae_recon          – AE reconstruction
          depth_saliency    – AE gradient saliency heatmap
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

            # --- Depth image ---
            depth_tensor = tiled_cam.data.output.get("depth")
            if depth_tensor is not None:
                depth_np = depth_tensor[0].squeeze().detach().cpu().numpy().astype("float32")
                depth_np = np.nan_to_num(depth_np, nan=10.0, posinf=10.0, neginf=0.0)
                depth_small = cv2.resize(depth_np, (128, 72), interpolation=cv2.INTER_AREA)

                images["depth"] = _depth_gray_b64(depth_small)

                # AE reconstruction: proper normalised encode→decode
                ae_recon = self._grab_ae_recon(env, depth_small)
                images["ae_recon"] = ae_recon if ae_recon else images["depth"]

                # Saliency: AE latent gradient (SmoothGrad), same as play_saliency.py
                saliency = self._grab_ae_saliency(env, depth_small)
                images["depth_saliency"] = saliency if saliency else _depth_jet_b64(depth_small)

            # --- Forward RGB (first-person / nav camera) ---
            rgb_tensor = tiled_cam.data.output.get("rgb")
            if rgb_tensor is not None:
                rgb_np = rgb_tensor[0].cpu().numpy()[:, :, :3]
                bgr = cv2.cvtColor(rgb_np.astype("uint8"), cv2.COLOR_RGB2BGR)
                bgr_small = cv2.resize(bgr, (320, 180), interpolation=cv2.INTER_AREA)
                fallback_rgb_b64 = _ndarray_to_jpeg_b64(bgr_small)
                images["rgb_first_person"] = fallback_rgb_b64

            # --- Angle cameras: chase (behind), left-side, top-down ---
            _angle_cams = [
                ("_chase_camera", "rgb_third_1"),
                ("_left_camera",  "rgb_third_2"),
                ("_top_camera",   "rgb_third_3"),
            ]
            for cam_attr, img_key in _angle_cams:
                cam = getattr(env, cam_attr, None)
                captured = False
                if cam is not None:
                    try:
                        rgb_out = cam.data.output.get("rgb")
                        if rgb_out is not None and rgb_out.numel() > 0:
                            rgb_np = rgb_out[0].cpu().numpy()[:, :, :3]
                            bgr    = cv2.cvtColor(rgb_np.astype("uint8"), cv2.COLOR_RGB2BGR)
                            images[img_key] = _ndarray_to_jpeg_b64(bgr)
                            captured = True
                    except Exception:
                        pass
                if not captured:
                    # Fallback to forward camera if angle cam not ready
                    if fallback_rgb_b64:
                        images[img_key] = fallback_rgb_b64

        except Exception as exc:
            print(f"[LiveTelemetry] _grab_images() error: {exc}")

        return images

    @staticmethod
    def _collect_slam_frontiers(mapper) -> list:
        """Return frontier dicts from the mapper (same call as OpenCV visualizer)."""
        if mapper is None:
            return []
        try:
            return mapper.detect_frontiers()
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
            inflated = mapper.get_inflated_grid()
            grid_h, grid_w = prob.shape

            # HD output (2× OpenCV) — PNG lossless for sharp walls/overlays
            map_w, map_h = 900, 1200

            canvas = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
            canvas[(prob >= 0.35) & (prob <= 0.65)] = [25, 18, 12]
            canvas[prob < 0.35]                      = [45, 38, 30]
            canvas[inflated == 1]                    = [30, 20, 75]
            canvas[prob > 0.65]                      = [255, 230, 80]

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
                frontiers_raw = self._collect_slam_frontiers(mapper)
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
            inflated = mapper.get_inflated_grid()    # (H, W) int/bool
            H_orig, W_orig = prob.shape

            # Quantise full-res: 0=unknown, 1=free, 2=inflated(danger), 3=wall
            # Ordering matters — higher value wins so max-pool preserves walls.
            full = np.zeros((H_orig, W_orig), dtype=np.uint8)
            full[prob < 0.35]  = 1
            full[inflated == 1] = 2
            full[prob > 0.65]  = 3   # occupied (OccupancyGrid > 65)

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
            occupied_count = int((grid == 3).sum())
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
                frontiers_raw = self._collect_slam_frontiers(mapper)
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
            person_found = bool(getattr(brain, "found_person", False)) if brain else False
            person_pos   = getattr(brain, "target_person_pos", None) if brain else None
            person_data  = None
            if person_found and person_pos is not None:
                try:
                    person_data = [round(float(person_pos[0]), 3),
                                   round(float(person_pos[1]), 3)]
                except Exception:
                    pass

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
            }

        except Exception as exc:
            print(f"[LiveTelemetry] _get_slam_3d() error: {exc}")
            return {}

    def _grab_yolo_frame(self, env) -> str:
        """Return base64 JPEG of the *clean* camera frame perception ran YOLO on.

        This is the exact frame the normalized boxes in ``yolo_stats`` were
        computed from, so the native web HUD overlays them in perfect sync.
        Upscaled 2× (Lanczos) for a crisp HD backdrop.
        """
        try:
            import cv2

            perception = getattr(env, "_perception", None)
            if perception is None:
                return self._yolo_hud_cache

            frame = getattr(perception, "_web_frame_bgr", None)
            if frame is None:
                return self._yolo_hud_cache

            h, w = frame.shape[:2]
            hd = cv2.resize(frame, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)
            b64 = _ndarray_to_jpeg_b64(hd, quality=92)
            self._yolo_hud_cache = b64
            return b64
        except Exception as exc:
            print(f"[LiveTelemetry] _grab_yolo_frame() error: {exc}")
            return self._yolo_hud_cache

    @staticmethod
    def _build_yolo_stats(perception, brain) -> dict:
        """Assemble the full native-HUD payload from the perception module."""
        if perception is None:
            return {}

        # ---- Confidence (persistent peak vs instantaneous) ----
        current_conf = float(getattr(perception, "last_best_person_conf", 0.0))
        peak_conf = current_conf
        pbc = getattr(perception, "_person_best_conf", None)
        if isinstance(pbc, dict) and pbc:
            peak_conf = max(peak_conf, max(float(v) for v in pbc.values()))

        state = getattr(perception, "_web_state", None) or {}
        intel = getattr(perception, "_last_intel", None)
        if isinstance(intel, dict):
            try:
                peak_conf = max(peak_conf, float(intel.get("conf", 0.0)))
            except (TypeError, ValueError):
                pass

        # ---- Rescue log snapshot ----
        rescue_log = []
        for entry in (getattr(perception, "_detection_log", None) or [])[:12]:
            try:
                rescue_log.append({
                    "label":   str(entry.get("label", "")),
                    "conf":    round(float(entry.get("conf", 0.0)), 4),
                    "gps_lat": entry.get("gps_lat"),
                    "gps_lon": entry.get("gps_lon"),
                    "key":     str(entry.get("person_key", "")),
                    "frame":   int(entry.get("frame", 0)),
                })
            except Exception:
                continue

        # ---- Intel panel ----
        intel_out = None
        if isinstance(intel, dict):
            intel_out = {
                "label":   str(intel.get("label", "")).upper(),
                "conf":    round(float(intel.get("conf", 0.0)), 4),
                "gps_lat": intel.get("gps_lat"),
                "gps_lon": intel.get("gps_lon"),
                "dist":    intel.get("dist"),
            }

        person_found = bool(getattr(brain, "found_person", False)) if brain else False
        thresh = float(getattr(perception, "person_conf_threshold", 0.7))

        # ---- Canonical detection state (single source of truth for the UI) ----
        if person_found or state.get("has_confirmed"):
            status, status_label = "confirmed", "TARGET CONFIRMED"
        elif peak_conf >= thresh or state.get("operator_alert"):
            status, status_label = "detected", "HUMAN DETECTED"
        elif peak_conf > 0.0:
            status, status_label = "seen", "CONTACT · TRACKING"
        else:
            status, status_label = "idle", "SCANNING"

        return {
            "conf_threshold":  round(thresh, 3),
            "best_conf":       round(peak_conf, 4),
            "current_conf":    round(current_conf, 4),
            "detection_count": int(getattr(perception, "detection_count", 0)),
            "person_found":    person_found,
            "person_seen":     bool(getattr(perception, "person_ever_detected", False)),
            "status":          status,
            "status_label":    status_label,
            "scan_label":      state.get("scan_label"),
            "operator_alert":  bool(state.get("operator_alert", False)),
            "alert_conf":      round(float(state.get("alert_conf", 0.0)), 4),
            "boxes":           list(getattr(perception, "_web_boxes", []) or []),
            "intel":           intel_out,
            "rescue_log":      rescue_log,
        }

    def _grab_ae_recon(self, env, depth_small) -> str:
        """Run the AE encoder+decoder on the current depth frame and return JPEG b64.

        depth_small is in raw metres (0.05…10).  The AE was trained on depth
        images normalised to [0, 1] using near=0.05, far=10.0, so we apply the
        same normalisation here before encoding.
        """
        try:
            import cv2
            import numpy as np
            import torch

            ae = getattr(env, "ae", None)
            if ae is None:
                return ""

            # Normalise depth metres → [0, 1] exactly as the training pipeline does
            near, far = 0.05, 10.0
            depth_norm = np.clip((depth_small - near) / (far - near), 0.0, 1.0).astype("float32")

            depth_t = torch.from_numpy(depth_norm).unsqueeze(0).unsqueeze(0)  # (1,1,72,128)
            device = next(ae.parameters()).device
            depth_t = depth_t.to(device)

            with torch.no_grad():
                z     = ae.encode(depth_t)          # (1, latent_dim)
                recon = ae.decode(z)                # (1, 1, 72, 128) in [0, 1]

            # Convert AE output [0,1] → uint8 grayscale → BGR JPEG
            recon_np  = recon[0, 0].cpu().numpy()                     # (72, 128)
            recon_u8  = (np.clip(recon_np, 0.0, 1.0) * 255).astype("uint8")
            bgr       = cv2.cvtColor(recon_u8, cv2.COLOR_GRAY2BGR)
            return _ndarray_to_jpeg_b64(bgr)
        except Exception:
            return ""

    def _grab_ae_saliency(self, env, depth_small) -> str:
        """Compute AE latent gradient saliency map — identical to play_saliency.py method.

        Saliency = |∂‖z‖₁ / ∂x| weighted by obstacle proximity (1 − depth_norm).
        This shows which pixels most influence the AE's latent representation.
        Uses SmoothGrad (N=8) for a cleaner, less noisy result.
        """
        try:
            import cv2
            import numpy as np
            import torch
            import torch.nn.functional as F

            ae = getattr(env, "ae", None)
            if ae is None:
                return ""

            near, far = 0.05, 10.0
            depth_norm = np.clip((depth_small - near) / (far - near), 0.0, 1.0).astype("float32")
            device = next(ae.parameters()).device

            depth_t = torch.from_numpy(depth_norm).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,72,128)

            # SmoothGrad: average |grad| over N noisy copies of the input
            n_samples   = 8
            noise_level = 0.05
            total_grad  = torch.zeros(72, 128, device=device)

            ae.eval()
            for _ in range(n_samples):
                noise    = torch.randn_like(depth_t) * noise_level
                noisy    = (depth_t + noise).clamp(0.0, 1.0)
                inp      = noisy.clone().detach().requires_grad_(True)

                h = ae.encoder(inp)           # CNN → flatten
                z = ae.fc_z(h)               # linear bottleneck
                if hasattr(ae, "ln_z") and ae.ln_z is not None:
                    z = ae.ln_z(z)

                loss = z.abs().sum()          # |∂‖z‖₁/∂x|
                loss.backward()

                if inp.grad is not None:
                    total_grad += inp.grad.abs().squeeze(0).squeeze(0)

            saliency = (total_grad / n_samples).detach().cpu().numpy()  # (72, 128)

            # Weight by proximity: nearby obstacles (small depth) contribute more
            proximity = 1.0 - depth_norm        # (72, 128) in [0, 1]
            saliency  = saliency * proximity

            # Normalise to [0, 1]
            s_min, s_max = saliency.min(), saliency.max()
            if s_max - s_min > 1e-8:
                saliency = (saliency - s_min) / (s_max - s_min)
            else:
                saliency = np.zeros_like(saliency)

            # Jet colormap heatmap
            sal_u8  = (saliency * 255).astype("uint8")
            heatmap = cv2.applyColorMap(sal_u8, cv2.COLORMAP_JET)

            # Blend depth (grey) + heatmap (40/60) — same weights as play_saliency.py
            depth_u8  = (np.clip(depth_norm, 0.0, 1.0) * 255).astype("uint8")
            depth_bgr = cv2.cvtColor(depth_u8, cv2.COLOR_GRAY2BGR)
            overlay   = cv2.addWeighted(depth_bgr, 0.4, heatmap, 0.6, 0)

            return _ndarray_to_jpeg_b64(overlay)
        except Exception:
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
