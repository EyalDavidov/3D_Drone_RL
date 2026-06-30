"""Mock telemetry data generator for the RL Drone Dashboard.

Simulates realistic drone physics through the 4-level room layout
so the dashboard can be tested without launching Isaac Sim.
"""

from __future__ import annotations

import base64
import io
import math
import random
import struct
import time
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

# Room bounding boxes per level (x_min, x_max, y_min, y_max, z_min, z_max)
ROOM_BOUNDS = [
    (-2.0, 2.0,  -3.0,  2.0,  0.0, 2.0),
    (-2.0, 2.0,  -9.0, -2.0,  0.0, 2.0),
    (-2.0, 2.0, -17.0, -8.0,  0.0, 2.0),
    (-6.0, 2.0, -21.0, -16.0, 0.0, 2.0),
]

# Pole layout (y positions per group from the env code)
POLE_POSITIONS_Y = {
    "level1": [0.0] * 6,
    "level2_row1": [-4.0] * 5,
    "level2_row2": [-5.5] * 5,
    "level2_row3": [-7.0] * 5,
}


# ---------------------------------------------------------------------------
# Tiny in-memory PNG encoder (no Pillow dependency)
# ---------------------------------------------------------------------------
def _make_png(width: int, height: int, pixels: list[list[tuple[int, int, int]]]) -> bytes:
    """Create a minimal RGB PNG from pixel data.
    
    pixels: height × width array of (R, G, B) tuples.
    """
    def _chunk(chunk_type: bytes, data: bytes) -> bytes:
        c = chunk_type + data
        crc = zlib.crc32(c) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + c + struct.pack(">I", crc)

    raw_rows = b""
    for row in pixels:
        raw_rows += b"\x00"  # filter byte (None)
        for r, g, b in row:
            raw_rows += struct.pack("BBB", r, g, b)

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    png = b"\x89PNG\r\n\x1a\n"
    png += _chunk(b"IHDR", header)
    png += _chunk(b"IDAT", zlib.compress(raw_rows, 6))
    png += _chunk(b"IEND", b"")
    return png


def _make_grayscale_png(width: int, height: int, values: list[list[int]]) -> bytes:
    """Create a minimal grayscale PNG from 0-255 values."""
    def _chunk(chunk_type: bytes, data: bytes) -> bytes:
        c = chunk_type + data
        crc = zlib.crc32(c) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + c + struct.pack(">I", crc)

    raw_rows = b""
    for row in values:
        raw_rows += b"\x00"
        for v in row:
            raw_rows += struct.pack("B", max(0, min(255, v)))

    header = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    png = b"\x89PNG\r\n\x1a\n"
    png += _chunk(b"IHDR", header)
    png += _chunk(b"IDAT", zlib.compress(raw_rows, 6))
    png += _chunk(b"IEND", b"")
    return png


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

        # Current level
        self.level = 0
        self.level_time = 0.0

        # Drone state
        self.pos = list(LEVEL_SPAWNS[0])
        self.vel = [0.0, 0.0, 0.0]
        self.ang_vel = [0.0, 0.0, 0.0]
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = -math.pi / 2  # facing -Y (towards targets)

        # PPO actions
        self.ppo_vx = 0.0
        self.ppo_vy = 0.0
        self.ppo_vz = 0.0
        self.ppo_yaw_rate = 0.0

        # LLC outputs
        self.thrust = 0.27 * 9.81  # hover thrust (mass * g)
        self.moment_x = 0.0
        self.moment_y = 0.0
        self.moment_z = 0.0

        # Synthetic image caches (regenerated periodically)
        self._img_counter = 0
        self._rgb_cache: str | None = None
        self._depth_cache: str | None = None
        self._ae_cache: str | None = None

        # Pole x-positions (randomized once)
        self.pole_positions: list[tuple[float, float, float]] = []
        self._randomize_poles()

    def _randomize_poles(self):
        """Generate random pole positions matching the env layout."""
        self.pole_positions = []
        for y_val in POLE_POSITIONS_Y["level1"]:
            self.pole_positions.append((random.uniform(-1.7, 1.7), y_val, 1.0))
        for key in ["level2_row1", "level2_row2", "level2_row3"]:
            for y_val in POLE_POSITIONS_Y[key]:
                self.pole_positions.append((random.uniform(-1.7, 1.7), y_val, 1.0))

    def _reset_to_level(self, level: int):
        """Reset drone to a specific level's spawn."""
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

    def _generate_synthetic_rgb(self) -> str:
        """Generate a synthetic RGB camera image as base64 PNG."""
        w, h = 160, 90
        pixels = []
        phase = self.t * 0.5
        for y in range(h):
            row = []
            for x in range(w):
                # Gradient sky + ground with moving elements
                ny = y / h
                if ny < 0.6:
                    # Sky gradient
                    r = int(30 + 20 * ny + 10 * math.sin(phase + x * 0.05))
                    g = int(35 + 25 * ny + 8 * math.cos(phase + x * 0.03))
                    b = int(60 + 40 * ny)
                else:
                    # Ground / walls
                    grid = ((x // 16) + (y // 16)) % 2
                    base = 40 + grid * 20
                    r = base + int(10 * math.sin(phase * 2))
                    g = base + 5
                    b = base
                row.append((max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b))))
            pixels.append(row)
        png_bytes = _make_png(w, h, pixels)
        return base64.b64encode(png_bytes).decode("ascii")

    def _generate_synthetic_depth(self) -> tuple[str, str]:
        """Generate synthetic depth and AE reconstruction as base64 PNGs."""
        w, h = 128, 72
        depth_vals: list[list[int]] = []
        phase = self.t * 0.8

        for y in range(h):
            row = []
            for x in range(w):
                # Simulate depth: near objects in center, far at edges
                cx = (x - w / 2) / (w / 2)
                cy = (y - h / 2) / (h / 2)
                dist = math.sqrt(cx * cx + cy * cy)

                # Base depth increases with distance from center
                base = int(40 + 180 * dist)

                # Add "obstacles" — vertical bars that move
                bar_x = (x + int(phase * 30)) % 40
                if bar_x < 4:
                    base = int(30 + 20 * random.random())

                # Add noise
                base += int(random.gauss(0, 3))
                row.append(max(0, min(255, base)))
            depth_vals.append(row)

        # AE reconstruction = smoothed version of depth (simulates lossy AE)
        ae_vals: list[list[int]] = []
        for y in range(h):
            row = []
            for x in range(w):
                # Simple box blur (3x3)
                total = 0
                count = 0
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        ny_, nx_ = y + dy, x + dx
                        if 0 <= ny_ < h and 0 <= nx_ < w:
                            total += depth_vals[ny_][nx_]
                            count += 1
                row.append(total // count)
            ae_vals.append(row)

        depth_png = base64.b64encode(_make_grayscale_png(w, h, depth_vals)).decode("ascii")
        ae_png = base64.b64encode(_make_grayscale_png(w, h, ae_vals)).decode("ascii")
        return depth_png, ae_png

    def tick(self) -> dict:
        """Advance simulation by one tick and return telemetry dict."""
        target = LEVEL_TARGETS[self.level]
        spawn = LEVEL_SPAWNS[self.level]
        duration = LEVEL_DURATIONS[self.level]

        # Progress along the path (0 → 1)
        progress = min(self.level_time / duration, 1.0)
        smooth_p = 0.5 - 0.5 * math.cos(progress * math.pi)  # smooth ease

        # Compute desired position along path
        target_x = spawn[0] + (target[0] - spawn[0]) * smooth_p
        target_y = spawn[1] + (target[1] - spawn[1]) * smooth_p
        target_z = spawn[2] + (target[2] - spawn[2]) * smooth_p

        # Add lateral oscillation (obstacle avoidance simulation)
        osc_x = 0.6 * math.sin(self.t * 1.8 + self.level * 2.0)
        osc_z = 0.2 * math.sin(self.t * 2.5 + 0.7)

        # Desired position
        des_x = target_x + osc_x
        des_y = target_y
        des_z = target_z + osc_z

        # Compute velocity from position change
        old_pos = self.pos[:]
        alpha = 0.15  # smoothing factor
        self.pos[0] += alpha * (des_x - self.pos[0])
        self.pos[1] += alpha * (des_y - self.pos[1])
        self.pos[2] += alpha * (des_z - self.pos[2])

        self.vel[0] = (self.pos[0] - old_pos[0]) / self.dt
        self.vel[1] = (self.pos[1] - old_pos[1]) / self.dt
        self.vel[2] = (self.pos[2] - old_pos[2]) / self.dt

        # Attitude from velocity (realistic tilt)
        speed = math.sqrt(self.vel[0] ** 2 + self.vel[1] ** 2)
        self.pitch = -0.1 * self.vel[1]  # pitch forward when moving -Y
        self.roll = 0.15 * self.vel[0]   # roll into turns
        target_yaw = math.atan2(
            target[1] - self.pos[1],
            target[0] - self.pos[0]
        )
        # Smooth yaw tracking
        yaw_err = target_yaw - self.yaw
        while yaw_err > math.pi:
            yaw_err -= 2 * math.pi
        while yaw_err < -math.pi:
            yaw_err += 2 * math.pi
        self.yaw += 0.05 * yaw_err

        # Angular velocity
        self.ang_vel = [
            0.3 * math.sin(self.t * 3.0),
            0.2 * math.cos(self.t * 2.5),
            0.05 * yaw_err / self.dt,
        ]

        # PPO actions (normalized -1 to 1)
        self.ppo_vx = max(-1, min(1, self.vel[0] / 1.0))
        self.ppo_vy = max(-1, min(1, self.vel[1] / 1.0))
        self.ppo_vz = max(-1, min(1, self.vel[2] / 0.5))
        self.ppo_yaw_rate = max(-1, min(1, yaw_err * 2.0))

        # LLC outputs
        hover_thrust = 0.27 * 9.81
        self.thrust = hover_thrust + 0.3 * self.vel[2] + 0.1 * math.sin(self.t * 4.0)
        self.moment_x = 0.005 * self.roll + 0.002 * math.sin(self.t * 5.0)
        self.moment_y = 0.005 * self.pitch + 0.002 * math.cos(self.t * 4.5)
        self.moment_z = 0.003 * yaw_err

        # Distance to goal
        dx = target[0] - self.pos[0]
        dy = target[1] - self.pos[1]
        dz = target[2] - self.pos[2]
        dist_to_goal = math.sqrt(dx * dx + dy * dy + dz * dz)

        # Generate images every 5 ticks to reduce CPU load
        self._img_counter += 1
        if self._img_counter % 5 == 0 or self._rgb_cache is None:
            self._rgb_cache = self._generate_synthetic_rgb()
            self._depth_cache, self._ae_cache = self._generate_synthetic_depth()

        # Build telemetry payload
        data = {
            "timestamp": self.t,
            "tick": self.tick_count,
            "level": self.level + 1,  # 1-indexed for display
            "level_time": round(self.level_time, 2),
            "level_duration": duration,
            "status": "running",

            # Position & orientation
            "pos": [round(v, 4) for v in self.pos],
            "roll": round(self.roll, 4),
            "pitch": round(self.pitch, 4),
            "yaw": round(self.yaw, 4),

            # Velocities
            "lin_vel": [round(v, 4) for v in self.vel],
            "ang_vel": [round(v, 4) for v in self.ang_vel],

            # Goal
            "goal_pos": list(target),
            "dist_to_goal": round(dist_to_goal, 4),

            # PPO actions
            "ppo_actions": {
                "vx": round(self.ppo_vx, 4),
                "vy": round(self.ppo_vy, 4),
                "vz": round(self.ppo_vz, 4),
                "yaw_rate": round(self.ppo_yaw_rate, 4),
            },

            # LLC outputs
            "llc_outputs": {
                "thrust": round(self.thrust, 4),
                "moment_x": round(self.moment_x, 6),
                "moment_y": round(self.moment_y, 6),
                "moment_z": round(self.moment_z, 6),
            },

            # Camera images (base64 PNG)
            "images": {
                "rgb": self._rgb_cache,
                "depth": self._depth_cache,
                "ae_recon": self._ae_cache,
            },

            # Room layout for 3D scene (sent once, client caches)
            "room_bounds": ROOM_BOUNDS,
            "poles": [(round(p[0], 2), round(p[1], 2), round(p[2], 2)) for p in self.pole_positions],
        }

        # Advance time
        self.t += self.dt
        self.tick_count += 1
        self.level_time += self.dt

        # Check level completion
        if dist_to_goal < 0.3 or self.level_time >= duration:
            self._reset_to_level(self.level + 1)

        return data
