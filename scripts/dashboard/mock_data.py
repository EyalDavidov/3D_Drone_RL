"""Mock telemetry data generator for the RL Drone Dashboard.

Simulates realistic drone physics through the 4-level room layout
so the dashboard can be tested without launching Isaac Sim.
"""

from __future__ import annotations

import base64
import math
import random
import struct
import zlib


# ---------------------------------------------------------------------------
# Level layout (mirrors multilevel_drone_env_cfg.py)
# ---------------------------------------------------------------------------
LEVEL_SPAWNS = [
    (0.0,   1.5,  1.0),
    (0.0,  -2.5,  1.0),
    (0.0,  -8.5,  1.0),
    (0.0, -16.5,  1.0),
]

LEVEL_TARGETS = [
    (0.0,  -2.5,  1.0),
    (0.0,  -8.5,  1.0),
    (0.0, -16.5,  1.0),
    (-5.0, -20.5, 1.0),
]

LEVEL_DURATIONS = [7.0, 10.0, 12.0, 12.0]

# Map layout from assets/rooms/final_flat.usd (env-local = world coords)
MAP_ZONES = {
    "room_1":        {"bounds": [-2.05,  2.05,  -2.05,  2.05]},
    "room_2":        {"bounds": [-2.05,  2.05,  -8.05, -2.00]},
    "room_3":        {"bounds": [-4.05,  4.05, -16.05, -7.95]},
    "room_4":        {"bounds": [-8.55, -4.45, -23.05, -17.95]},
    "corridor":      {"bounds": [-4.50,  0.55, -22.05, -16.00]},
    "side_coridors": {"bounds": [-2.70,  2.70, -18.05, -16.00]},
}

# Room bounding boxes (x_min, x_max, y_min, y_max, z_min, z_max)
ROOM_BOUNDS = [
    (*MAP_ZONES["room_1"]["bounds"],        0.0, 2.0),
    (*MAP_ZONES["room_2"]["bounds"],        0.0, 2.0),
    (*MAP_ZONES["room_3"]["bounds"],        0.0, 2.0),
    (*MAP_ZONES["room_4"]["bounds"],        0.0, 2.0),
    (*MAP_ZONES["corridor"]["bounds"],      0.0, 2.0),
    (*MAP_ZONES["side_coridors"]["bounds"], 0.0, 2.0),
]

POLE_POSITIONS_Y = {
    "level1": [0.0] * 6,
    "level2_row1": [-4.0] * 5,
    "level2_row2": [-5.5] * 5,
    "level2_row3": [-7.0] * 5,
}


# ---------------------------------------------------------------------------
# Fast in-memory PNG encoder (no Pillow dependency)
# ---------------------------------------------------------------------------
def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    c = chunk_type + data
    crc = zlib.crc32(c) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + c + struct.pack(">I", crc)


def _make_rgb_png(width: int, height: int, pixels: list[list[tuple[int, int, int]]]) -> bytes:
    """Create a minimal RGB PNG. pixels: height × width of (R,G,B)."""
    raw = bytearray()
    for row in pixels:
        raw.append(0)  # filter byte
        for r, g, b in row:
            raw.append(r)
            raw.append(g)
            raw.append(b)
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    out = b"\x89PNG\r\n\x1a\n"
    out += _png_chunk(b"IHDR", header)
    out += _png_chunk(b"IDAT", zlib.compress(bytes(raw), 4))
    out += _png_chunk(b"IEND", b"")
    return out


def _make_gray_png(width: int, height: int, values: list[list[int]]) -> bytes:
    """Create a minimal grayscale PNG. values: height × width of 0-255."""
    raw = bytearray()
    for row in values:
        raw.append(0)
        for v in row:
            raw.append(max(0, min(255, v)))
    header = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    out = b"\x89PNG\r\n\x1a\n"
    out += _png_chunk(b"IHDR", header)
    out += _png_chunk(b"IDAT", zlib.compress(bytes(raw), 4))
    out += _png_chunk(b"IEND", b"")
    return out


def _jet_color(v: float) -> tuple[int, int, int]:
    """Simple jet colormap. v in [0, 1] -> (R, G, B)."""
    if v < 0.25:
        r, g, b = 0, v / 0.25, 1.0
    elif v < 0.5:
        r, g, b = 0, 1.0, 1.0 - (v - 0.25) / 0.25
    elif v < 0.75:
        r, g, b = (v - 0.5) / 0.25, 1.0, 0
    else:
        r, g, b = 1.0, 1.0 - (v - 0.75) / 0.25, 0
    return (int(r * 255), int(g * 255), int(b * 255))


# ---------------------------------------------------------------------------
# MockDroneTelemetry
# ---------------------------------------------------------------------------
class MockDroneTelemetry:
    """Generates realistic mock drone telemetry data."""

    def __init__(self, tick_rate: float = 20.0):
        self.tick_rate = tick_rate
        self.dt = 1.0 / tick_rate
        self.t = 0.0
        self.tick_count = 0

        # Level state
        self.level = 0
        self.level_time = 0.0
        self.force_level: int | None = None  # None = auto-cycle

        # Drone state
        self.pos = list(LEVEL_SPAWNS[0])
        self.vel = [0.0, 0.0, 0.0]
        self.ang_vel = [0.0, 0.0, 0.0]
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = -math.pi / 2

        # PPO / LLC state
        self.ppo_vx = 0.0
        self.ppo_vy = 0.0
        self.ppo_vz = 0.0
        self.ppo_yaw_rate = 0.0
        self.thrust = 0.27 * 9.81
        self.moment_x = 0.0
        self.moment_y = 0.0
        self.moment_z = 0.0
        self._mock_yaw_err = 0.0
        self._mock_z_err = 0.0
        self._mock_explore = 0

        # Image caches
        self._img_counter = 0
        self._image_cache: dict | None = None

        # Poles
        self.pole_positions: list[tuple[float, float, float]] = []
        self._randomize_poles()

    def _randomize_poles(self):
        self.pole_positions = []
        for y_val in POLE_POSITIONS_Y["level1"]:
            self.pole_positions.append((random.uniform(-1.7, 1.7), y_val, 1.0))
        for key in ["level2_row1", "level2_row2", "level2_row3"]:
            for y_val in POLE_POSITIONS_Y[key]:
                self.pole_positions.append((random.uniform(-1.7, 1.7), y_val, 1.0))

    def _reset_to_level(self, level: int):
        self.level = level % 4
        self.level_time = 0.0
        spawn = LEVEL_SPAWNS[self.level]
        self.pos = list(spawn)
        self.vel = [0.0, 0.0, 0.0]
        self.ang_vel = [0.0, 0.0, 0.0]
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = -math.pi / 2
        if self.level == 0:
            self._randomize_poles()

    # ---- Image Generation ----

    def _gen_base_brightness(self, w: int, h: int, phase_offset: float = 0.0) -> list[list[int]]:
        """Generate a base brightness map used for RGB cameras."""
        phase = self.t * 0.5 + phase_offset
        pattern = []
        for y in range(h):
            row = []
            ny = y / h
            for x in range(w):
                if ny < 0.55:
                    v = int(50 + 40 * ny + 15 * math.sin(phase + x * 0.04))
                else:
                    grid = ((x // 16) + (y // 16)) % 2
                    v = int(40 + grid * 25 + 10 * math.sin(phase * 2 + x * 0.03))
                row.append(max(0, min(255, v)))
            pattern.append(row)
        return pattern

    def _colorize(self, pattern: list[list[int]], rs: float, ro: int,
                  gs: float, go: int, bs: float, bo: int) -> list[list[tuple[int, int, int]]]:
        """Apply color palette to brightness pattern."""
        pixels = []
        for row in pattern:
            prow = []
            for v in row:
                r = max(0, min(255, int(v * rs + ro)))
                g = max(0, min(255, int(v * gs + go)))
                b = max(0, min(255, int(v * bs + bo)))
                prow.append((r, g, b))
            pixels.append(prow)
        return pixels

    def _generate_all_images(self) -> dict[str, str]:
        """Generate all camera images and return as base64 PNG dict."""
        W_RGB, H_RGB = 160, 90
        W_D, H_D = 128, 72

        # --- RGB cameras (shared base, different palettes) ---
        base_fp = self._gen_base_brightness(W_RGB, H_RGB, 0.0)
        base_t1 = self._gen_base_brightness(W_RGB, H_RGB, 1.5)
        base_t2 = self._gen_base_brightness(W_RGB, H_RGB, 3.0)
        base_t3 = self._gen_base_brightness(W_RGB, H_RGB, 4.5)

        rgb_fp = self._colorize(base_fp, 0.7, 10, 0.9, 15, 0.6, 5)    # neutral green (FPV)
        rgb_t1 = self._colorize(base_t1, 0.5, 15, 0.6, 20, 1.0, 30)   # blue (behind)
        rgb_t2 = self._colorize(base_t2, 0.8, 20, 0.5, 10, 0.9, 25)   # purple (side)
        rgb_t3 = self._colorize(base_t3, 1.0, 20, 0.7, 15, 0.4, 5)    # warm (above)

        # --- Depth ---
        phase = self.t * 0.8
        depth_vals: list[list[int]] = []
        for y in range(H_D):
            row = []
            for x in range(W_D):
                cx = (x - W_D / 2) / (W_D / 2)
                cy = (y - H_D / 2) / (H_D / 2)
                dist = math.sqrt(cx * cx + cy * cy)
                base = int(40 + 180 * dist)
                bar_x = (x + int(phase * 30)) % 40
                if bar_x < 4:
                    base = int(30 + 20 * random.random())
                base += int(random.gauss(0, 3))
                row.append(max(0, min(255, base)))
            depth_vals.append(row)

        # --- AE reconstruction (blurred depth) ---
        ae_vals: list[list[int]] = []
        for y in range(H_D):
            row = []
            for x in range(W_D):
                total = 0
                count = 0
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        ny_, nx_ = y + dy, x + dx
                        if 0 <= ny_ < H_D and 0 <= nx_ < W_D:
                            total += depth_vals[ny_][nx_]
                            count += 1
                row.append(total // count)
            ae_vals.append(row)

        # --- Saliency heatmap (jet colormap on depth + hotspots) ---
        saliency_pixels: list[list[tuple[int, int, int]]] = []
        # Create a few random hotspots
        hx1 = int(W_D * (0.4 + 0.2 * math.sin(self.t * 0.7)))
        hy1 = int(H_D * (0.3 + 0.2 * math.cos(self.t * 0.9)))
        hx2 = int(W_D * (0.6 + 0.15 * math.cos(self.t * 1.1)))
        hy2 = int(H_D * (0.6 + 0.15 * math.sin(self.t * 0.8)))
        for y in range(H_D):
            row = []
            for x in range(W_D):
                v = depth_vals[y][x] / 255.0
                # Add hotspot influence
                d1 = math.sqrt((x - hx1) ** 2 + (y - hy1) ** 2) / 30.0
                d2 = math.sqrt((x - hx2) ** 2 + (y - hy2) ** 2) / 25.0
                hot = max(0, 1.0 - d1) * 0.5 + max(0, 1.0 - d2) * 0.4
                v = min(1.0, v * 0.6 + hot)
                row.append(_jet_color(v))
            saliency_pixels.append(row)

        # Encode all to base64 PNG
        def _b64_rgb(px):
            return base64.b64encode(_make_rgb_png(W_RGB, H_RGB, px)).decode("ascii")

        def _b64_gray(vals):
            return base64.b64encode(_make_gray_png(W_D, H_D, vals)).decode("ascii")

        return {
            "rgb_first_person": _b64_rgb(rgb_fp),
            "rgb_third_1": _b64_rgb(rgb_t1),
            "rgb_third_2": _b64_rgb(rgb_t2),
            "rgb_third_3": _b64_rgb(rgb_t3),
            "depth": _b64_gray(depth_vals),
            "depth_saliency": base64.b64encode(
                _make_rgb_png(W_D, H_D, saliency_pixels)
            ).decode("ascii"),
            "ae_recon": _b64_gray(ae_vals),
        }

    # ---- Main Tick ----

    def tick(self) -> dict:
        target = LEVEL_TARGETS[self.level]
        spawn = LEVEL_SPAWNS[self.level]
        duration = LEVEL_DURATIONS[self.level]

        # Progress along path
        progress = min(self.level_time / duration, 1.0)
        smooth_p = 0.5 - 0.5 * math.cos(progress * math.pi)

        target_x = spawn[0] + (target[0] - spawn[0]) * smooth_p
        target_y = spawn[1] + (target[1] - spawn[1]) * smooth_p
        target_z = spawn[2] + (target[2] - spawn[2]) * smooth_p

        osc_x = 0.6 * math.sin(self.t * 1.8 + self.level * 2.0)
        osc_z = 0.2 * math.sin(self.t * 2.5 + 0.7)

        des_x = target_x + osc_x
        des_y = target_y
        des_z = target_z + osc_z

        old_pos = self.pos[:]
        alpha = 0.15
        self.pos[0] += alpha * (des_x - self.pos[0])
        self.pos[1] += alpha * (des_y - self.pos[1])
        self.pos[2] += alpha * (des_z - self.pos[2])

        self.vel[0] = (self.pos[0] - old_pos[0]) / self.dt
        self.vel[1] = (self.pos[1] - old_pos[1]) / self.dt
        self.vel[2] = (self.pos[2] - old_pos[2]) / self.dt

        self.pitch = -0.1 * self.vel[1]
        self.roll = 0.15 * self.vel[0]
        target_yaw = math.atan2(target[1] - self.pos[1], target[0] - self.pos[0])
        yaw_err = target_yaw - self.yaw
        while yaw_err > math.pi:
            yaw_err -= 2 * math.pi
        while yaw_err < -math.pi:
            yaw_err += 2 * math.pi
        self.yaw += 0.05 * yaw_err

        self.ang_vel = [
            0.3 * math.sin(self.t * 3.0),
            0.2 * math.cos(self.t * 2.5),
            0.05 * yaw_err / self.dt,
        ]

        self.ppo_vx = max(-1, min(1, self.vel[0] / 1.0))
        self.ppo_vy = max(-1, min(1, self.vel[1] / 1.0))
        self.ppo_vz = max(-1, min(1, self.vel[2] / 0.5))
        self.ppo_yaw_rate = max(-1, min(1, yaw_err * 2.0))
        dx = target[0] - self.pos[0]
        dy = target[1] - self.pos[1]
        dz = target[2] - self.pos[2]
        self._mock_yaw_err = yaw_err
        self._mock_z_err = dz
        self._mock_explore += 1

        hover_thrust = 0.27 * 9.81
        self.thrust = hover_thrust + 0.3 * self.vel[2] + 0.1 * math.sin(self.t * 4.0)
        self.moment_x = 0.005 * self.roll + 0.002 * math.sin(self.t * 5.0)
        self.moment_y = 0.005 * self.pitch + 0.002 * math.cos(self.t * 4.5)
        self.moment_z = 0.003 * yaw_err

        dist_to_goal = math.sqrt(dx * dx + dy * dy + dz * dz)

        # Generate images every 8 ticks
        self._img_counter += 1
        if self._img_counter % 8 == 0 or self._image_cache is None:
            self._image_cache = self._generate_all_images()

        data = {
            "timestamp": self.t,
            "tick": self.tick_count,
            "level": self.level + 1,
            "level_time": round(self.level_time, 2),
            "level_duration": duration,
            "level_mode": "auto" if self.force_level is None else "forced",
            "status": "running",

            "pos": [round(v, 4) for v in self.pos],
            "roll": round(self.roll, 4),
            "pitch": round(self.pitch, 4),
            "yaw": round(self.yaw, 4),

            "lin_vel": [round(v, 4) for v in self.vel],
            "ang_vel": [round(v, 4) for v in self.ang_vel],

            "goal_pos": list(target),
            "dist_to_goal": round(dist_to_goal, 4),

            "ppo_actions": {
                "vx": round(self.ppo_vx, 4),
                "vy": round(self.ppo_vy, 4),
                "vz": round(self.ppo_vz, 4),
                "yaw_rate": round(self.ppo_yaw_rate, 4),
            },

            "llc_outputs": {
                "thrust": round(self.thrust, 4),
                "moment_x": round(self.moment_x, 6),
                "moment_y": round(self.moment_y, 6),
                "moment_z": round(self.moment_z, 6),
            },

            "images": self._image_cache,

            "room_bounds": ROOM_BOUNDS,
            "map_zones":   MAP_ZONES,
            "poles": [(round(p[0], 2), round(p[1], 2), round(p[2], 2))
                      for p in self.pole_positions],

            "flight_control": {
                "yaw_error_deg": round(math.degrees(self._mock_yaw_err), 3),
                "z_error_m": round(self._mock_z_err, 4),
                "xy_pos_err_m": round(math.hypot(dx, dy), 4),
                "pos_err_b": [round(dx, 4), round(dy, 4), round(dz, 4)],
                "vel_err_b": [
                    round(self.ppo_vx * 1.0 - self.vel[0], 4),
                    round(self.ppo_vy * 1.0 - self.vel[1], 4),
                    round(self.ppo_vz * 0.5 - self.vel[2], 4),
                ],
                "desired_vel_b": [
                    round(self.ppo_vx * 1.0, 4),
                    round(self.ppo_vy * 1.0, 4),
                    round(self.ppo_vz * 0.5, 4),
                ],
                "lin_vel_b": [round(v, 4) for v in self.vel],
                "ang_vel_b": [round(v, 4) for v in self.ang_vel],
                "ll_actions": [
                    round(self.thrust / (0.27 * 9.81 * 2) - 0.5, 4),
                    round(self.moment_x * 100, 4),
                    round(self.moment_y * 100, 4),
                    round(self.moment_z * 100, 4),
                ],
                "ll_obs": [0.0] * 13,
            },
            "brain_telemetry": {
                "state": "EXPLORE",
                "segment_idx": self.level,
                "segment_label": f"Level {self.level + 1}",
                "nav_target": [round(target[0], 3), round(target[1], 3), round(target[2], 3)],
                "waypoint_idx": 0,
                "waypoint_total": 0,
                "explore_steps": self._mock_explore,
                "stuck_steps": 0,
                "stuck_ticks": 0,
                "mission_finished": False,
                "path_nodes": 0,
            },
            "mission_status": {
                "status": "EXPLORE",
                "brain_state": "EXPLORE",
                "targets_found": "0/2",
                "detected": 0,
                "total": 2,
                "detail": "",
                "crashed": False,
                "crash_reason": "",
            },
            "spawn_info": {
                "active": False,
                "total": 2,
                "detected": 0,
                "pending": None,
                "coverage_required": 95.0,
            },
            "slam_state": "EXPLORE",
            "map_explored_pct": min(99.0, self.t * 2.5),
            "frontier_count": 3,
            "people_found": 0,
            "sim_running": False,
        }

        self.t += self.dt
        self.tick_count += 1
        self.level_time += self.dt

        # Level completion: reset to forced level or advance
        if dist_to_goal < 0.3 or self.level_time >= duration:
            if self.force_level is not None:
                self._reset_to_level(self.force_level)
            else:
                self._reset_to_level(self.level + 1)

        return data
