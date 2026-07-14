import numpy as np
import torch
import cv2
import math
import os
from first_drone.tasks.direct.navigation_drone.brain_nav_drone_env import BrainNavDroneEnv
from first_drone.models.brain import BrainModule
from .occupancy_grid import OccupancyGridMapper

def plan_astar(grid, start_grid, goal_grid):
    """A* path planning on a binary grid (0 = free, 1 = occupied)."""
    import heapq
    h, w = grid.shape
    r0, c0 = start_grid
    r1, c1 = goal_grid

    if not (0 <= r0 < h and 0 <= c0 < w) or not (0 <= r1 < h and 0 <= c1 < w):
        return None

    def heuristic(r, c):
        return np.sqrt((r - r1) ** 2 + (c - c1) ** 2)

    open_set = []
    heapq.heappush(open_set, (heuristic(r0, c0), 0, (r0, c0)))

    came_from = {}
    g_score = {(r0, c0): 0}
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    max_iterations = 8000
    iterations = 0

    while open_set and iterations < max_iterations:
        iterations += 1
        _, current_g, current = heapq.heappop(open_set)

        if current == (r1, c1):
            path = []
            curr = current
            while curr in came_from:
                path.append(curr)
                curr = came_from[curr]
            path.append((r0, c0))
            return path[::-1]

        r, c = current
        for dr, dc in neighbors:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                if grid[nr, nc] == 1:
                    continue

                step_cost = 1.414 if (dr != 0 and dc != 0) else 1.0
                tentative_g = current_g + step_cost

                if (nr, nc) not in g_score or tentative_g < g_score[(nr, nc)]:
                    came_from[(nr, nc)] = current
                    g_score[(nr, nc)] = tentative_g
                    f_score = tentative_g + heuristic(nr, nc)
                    heapq.heappush(open_set, (f_score, tentative_g, (nr, nc)))

    return None


class SlamBrainModule(BrainModule):
    def __init__(self, env):
        super().__init__(env, step_size=env.cfg.brain_step_size, safety_margin=env.cfg.brain_safety_margin)
        self.env = env

        self.mapper = OccupancyGridMapper(
            min_x=-12.0, max_x=12.0,
            min_y=-27.0, max_y=5.0,
            cell_size=0.10,
            # 0.15m ≈ drone radius + a small buffer. 0.12 planned dangerously close to
            # walls for the stricter OBB collision check; 0.15 keeps narrow corridors
            # passable (>0.5m gaps) while leaving ~1 extra cell of wall clearance.
            safety_margin=0.15,
        )

        # PURE SLAM: never load USD / ground-truth walkable geometry into the mapper.
        self.mapper.walkable_mask = None
        print(
            "[SLAM Brain] Pure-SLAM mode: frontiers, paths, and targets use the "
            "depth-built occupancy grid only (no USD walkable / ground-truth map)."
        )
        
        # Calculate expected total cells dynamically based on the USD map zones
        self.mapper.expected_total_cells = self.calculate_expected_total_cells()
        print(
            f"[SLAM Brain] Dynamically calculated expected total cells from USD zones: {self.mapper.expected_total_cells}"
        )

        self.state = "EXPLORE"
        self.segment_idx = 0
        self.mission_finished = False

        self.active_frontier = None
        self.astar_path_world = []
        self.waypoints = []
        self.current_wp_idx = 0
        self.last_drone_yaw = None
        self.explore_step_count = 0
        self.last_scan_pos = None
        self.rescued_people = []
        self.rescued_people_conf = []
        self.blacklisted_frontiers = []
        self.visited_frontier_centroids = []
        self.active_frontier_ticks = 0
        self.scanned_rooms = set()  # kept for snapshot compat; no scans are triggered
        # Highest room-checkpoint index reached (bookkeeping only; frontier choice is
        # now pure-SLAM via the heading-biased scorer, no USD room gating).
        self.max_segment_reached = 0
        self.dynamic_room_nodes = []
        self._start_pos_xy = None
        # PURE-SLAM anti-backtrack: grid of cells the drone has physically flown
        # near. Frontiers inside this mask are rejected so the drone never turns
        # around to re-target a room/corridor it already crossed. Uses only the
        # drone's own trajectory — no USD/ground-truth map.
        self.visited_mask = None
        self._prev_pos_xy = None
        self._prev_stamp_xy = None
        self._travel_dir = None
        # A frontier BEHIND the travel direction can only be chosen if it's within
        # this range (a branch right off the current spot). Prevents the drone from
        # flying all the way back across the map into rooms it already explored.
        self.BACKTRACK_MAX_M = 5.0
        # Minimum unknown-region area touching a frontier (filters shadow pockets).
        self.MIN_UNKNOWN_GAIN = 4
        self._hold_log_ticks = 0
        self._frontier_lock_ticks = 0
        self._stuck_ref_pos = None
        self._stuck_ticks = 0
        self._corridor_gate_log_ticks = 0
        self._corridor_context_ticks = 0
        self._mission_assist_idx = 0
        self._mission_assist_active = False
        self._forced_corridor_route_active = False
        self._forced_corridor_route_idx = 0
        self._forced_corridor_route_logged = False

    def calculate_expected_total_cells(self) -> int:
        """Estimate the expected total floor and wall cells of the track dynamically by unioning USD zones."""
        zones = getattr(self.env, "_map_zones", None)
        if not zones:
            return 11400  # Fallback

        h, w = self.mapper.h, self.mapper.w
        mask = np.zeros((h, w), dtype=bool)

        for name, zone in zones.items():
            bounds = zone.get("bounds")
            if not bounds:
                continue
            lx0, lx1, ly0, ly1 = bounds

            # Convert world bounds to grid indices
            r0, c0 = self.mapper.world_to_grid(lx0, ly0)
            r1, c1 = self.mapper.world_to_grid(lx1, ly1)

            # Row/col can be inverted based on coordinate orientations
            min_r, max_r = min(r0, r1), max(r0, r1)
            min_c, max_c = min(c0, c1), max(c0, c1)

            # Clamp to grid size
            min_r = max(0, min(min_r, h - 1))
            max_r = max(0, min(max_r, h - 1))
            min_c = max(0, min(min_c, w - 1))
            max_c = max(0, min(max_c, w - 1))

            mask[min_r : max_r + 1, min_c : max_c + 1] = True

        expected_cells = int(mask.sum())
        return expected_cells if expected_cells > 0 else 11400

    def reset_mission_from_start(self) -> None:
        """Keep room-1 spawn from sequential config but start in SLAM EXPLORE (not SCAN)."""
        super().reset_mission_from_start()
        self.state = "EXPLORE"
        self.segment_idx = 0
        self.max_segment_reached = 0
        self.active_frontier = None
        self.astar_path_world = []
        self.waypoints = []
        self.current_wp_idx = 0
        self.explore_step_count = 0
        self.last_scan_pos = None
        self.rescued_people = []
        self.rescued_people_conf = []
        self.blacklisted_frontiers = []
        self.visited_frontier_centroids = []
        self.active_frontier_ticks = 0
        self.scanned_rooms = set()
        self.max_segment_reached = 0
        self.dynamic_room_nodes = []
        self._start_pos_xy = None
        self.visited_mask = None  # fresh trajectory on a full restart
        self._prev_stamp_xy = None
        self._prev_pos_xy = None
        self._travel_dir = None
        self._hold_log_ticks = 0
        self._frontier_lock_ticks = 0

    def capture_mission_snapshot(self):
        # Decouple from parent sequential checks, always create a valid snapshot dictionary
        snap = {}
        snap["segment_idx"] = int(self.segment_idx)
        snap["max_segment_reached"] = int(getattr(self, "max_segment_reached", 0))
        snap["rescued_people"] = [p.copy() for p in self.rescued_people]
        snap["rescued_people_conf"] = [float(c) for c in getattr(self, "rescued_people_conf", [])]
        snap["scanned_rooms"] = set(self.scanned_rooms)
        blk = [list(b) for b in getattr(self, "blacklisted_frontiers", [])]
        af = getattr(self, "active_frontier", None)
        if af is not None and af.get("centroid_world") is not None:
            # Smart Blacklist: Do not blacklist a major pathway (large unknown gain)
            # if the crash occurred far from it. Mid-corridor collisions/control failures en-route
            # should not block the pathway; only targets where the drone crashed/got stuck directly at
            # the target location (within 1.5m) should be blacklisted.
            d_pos_w = self.env._robot.data.root_pos_w[0].cpu().numpy()
            centroid = np.array(af["centroid_world"])
            dist = np.linalg.norm(d_pos_w[:2] - centroid[:2])
            unknown_gain = int(af.get("unknown_gain", 0))

            if dist >= 1.5 and unknown_gain >= 12:
                print(
                    f"\n[SLAM Brain] Drone crashed en-route to major target ({dist:.2f}m away, gain {unknown_gain}). "
                    "Skipping blacklist to preserve this pathway."
                )
            else:
                print(
                    f"\n[SLAM Brain] Blacklisting unreachable/dead-end target centroid {af['centroid_world']} "
                    f"({dist:.2f}m away, gain {unknown_gain})."
                )
                blk.append(list(af["centroid_world"]))
        snap["blacklisted_frontiers"] = blk
        snap["forced_corridor_route_active"] = bool(
            getattr(self, "_forced_corridor_route_active", False)
        )
        snap["forced_corridor_route_idx"] = int(
            getattr(self, "_forced_corridor_route_idx", 0)
        )
        if self.last_scan_pos is not None:
            snap["last_scan_pos"] = self.last_scan_pos.copy()
        if getattr(self, "visited_mask", None) is not None:
            snap["visited_mask"] = self.visited_mask.copy()
            
        snap["dynamic_room_nodes"] = [node.tolist() for node in getattr(self, "dynamic_room_nodes", [])]
        snap["visited_frontier_centroids"] = [c.tolist() for c in getattr(self, "visited_frontier_centroids", [])]
        snap["start_pos_xy"] = self._start_pos_xy.tolist() if getattr(self, "_start_pos_xy", None) is not None else None

        return snap

    def restore_mission_snapshot(self, snap):
        if not snap:
            return
        self.segment_idx = snap.get("segment_idx", 0)
        # Preserve forward progress across crash recovery — never backtrack.
        self.max_segment_reached = max(
            int(getattr(self, "max_segment_reached", 0)),
            int(snap.get("max_segment_reached", 0)),
        )
        self.rescued_people = [np.array(p) for p in snap.get("rescued_people", [])]
        self.rescued_people_conf = [float(c) for c in snap.get("rescued_people_conf", [])]
        if len(self.rescued_people_conf) < len(self.rescued_people):
            thresh = float(getattr(self.env.cfg, "yolo_person_conf_threshold", 0.70))
            missing = len(self.rescued_people) - len(self.rescued_people_conf)
            self.rescued_people_conf.extend([thresh] * missing)
        self.scanned_rooms = set(snap.get("scanned_rooms", set()))
        if "last_scan_pos" in snap and snap["last_scan_pos"] is not None:
            self.last_scan_pos = snap["last_scan_pos"].copy()
        
        # Clear state/target so that on recovery we force immediate safe path planning
        self.state = "EXPLORE"
        self.active_frontier = None
        self.astar_path_world = []
        self.waypoints = []
        self.current_wp_idx = 0
        self.explore_step_count = 50  # force immediate path generation
        # Keep the blacklist (incl. the frontier we just crashed on) so recovery
        # doesn't re-select it and loop.
        self.blacklisted_frontiers = [np.array(b) for b in snap.get("blacklisted_frontiers", [])]
        self.visited_frontier_centroids = [np.array(c) for c in snap.get("visited_frontier_centroids", [])]
        if snap.get("visited_mask") is not None:
            self.visited_mask = np.array(snap["visited_mask"], dtype=bool)
        if "dynamic_room_nodes" in snap:
            self.dynamic_room_nodes = [np.array(node) for node in snap["dynamic_room_nodes"]]
        if "start_pos_xy" in snap and snap["start_pos_xy"] is not None:
            self._start_pos_xy = np.array(snap["start_pos_xy"])
        self._forced_corridor_route_active = bool(
            snap.get("forced_corridor_route_active", False)
        )
        self._forced_corridor_route_idx = int(
            snap.get("forced_corridor_route_idx", 0)
        )
        if self._forced_corridor_route_active:
            self._mission_assist_active = False
            self._mission_assist_idx = 0
        self.active_frontier_ticks = 0
        self._stuck_ref_pos = None
        self._stuck_ticks = 0
        self._prev_stamp_xy = None

    def _ensure_visited_mask(self) -> None:
        """Allocate or resize visited_mask to match the current occupancy grid."""
        if self.visited_mask is None:
            self.visited_mask = np.zeros((self.mapper.h, self.mapper.w), dtype=bool)
        elif self.visited_mask.shape != (self.mapper.h, self.mapper.w):
            old = self.visited_mask
            self.visited_mask = np.zeros((self.mapper.h, self.mapper.w), dtype=bool)
            oh, ow = old.shape
            self.visited_mask[: min(oh, self.mapper.h), : min(ow, self.mapper.w)] = old[
                : min(oh, self.mapper.h), : min(ow, self.mapper.w)
            ]

    def _stamp_trajectory_step(self, x0, y0, x1, y1) -> None:
        """Stamp visited cells along the segment the drone just flew."""
        self._ensure_visited_mask()
        r0, c0 = self.mapper.world_to_grid(x0, y0)
        r1, c1 = self.mapper.world_to_grid(x1, y1)
        n = int(max(abs(r1 - r0), abs(c1 - c0))) + 1
        rad = max(1, int(round(1.2 / self.mapper.cell_size)))
        rs = np.linspace(r0, r1, max(n, 2)).round().astype(int)
        cs = np.linspace(c0, c1, max(n, 2)).round().astype(int)
        for r, c in zip(rs, cs):
            if not self.mapper.is_in_bounds(r, c):
                continue
            r0b, r1b = max(0, r - rad), min(self.mapper.h, r + rad + 1)
            c0b, c1b = max(0, c - rad), min(self.mapper.w, c + rad + 1)
            ys, xs = np.ogrid[r0b:r1b, c0b:c1b]
            disk = (ys - r) ** 2 + (xs - c) ** 2 <= rad * rad
            self.visited_mask[r0b:r1b, c0b:c1b] |= disk

    def coverage_stats(self) -> tuple[int, int]:
        """SLAM-only coverage: known cells / bounding box of mapped region."""
        return self.mapper.coverage_stats()

    def get_segment_label(self, idx=0):
        return f"SLAM {self.state}"

    def _ray_probe_in_direction(self, d_pos_w, came_from, direction_xy, max_dist=6.0):
        """Ray-march through known-free space along any unit direction (pure SLAM).

        Works for hallways at ANY angle — not tied to travel heading or ±90°.
        Returns a probe target dict or None.
        """
        d = np.asarray(direction_xy[:2], dtype=np.float64)
        n = float(np.linalg.norm(d))
        if n < 1e-6:
            return None
        d = d / n
        known_free = self.mapper.get_traversable_free()
        cell = self.mapper.cell_size
        best = None
        step = 0.4
        while step <= max_dist:
            wx = float(d_pos_w[0] + d[0] * step)
            wy = float(d_pos_w[1] + d[1] * step)
            r, c = self.mapper.world_to_grid(wx, wy)
            if not self.mapper.is_in_bounds(r, c):
                break
            if known_free[r, c]:
                best = (wx, wy, r, c)
            else:
                break
            step += cell
        if best is None:
            return None
        wx, wy, r, c = best
        if float(np.hypot(wx - d_pos_w[0], wy - d_pos_w[1])) < 1.0:
            return None
        if not self.mapper.is_cell_frontier((r, c), radius=1):
            return None
        path = self.mapper.reconstruct_path(came_from, (r, c))
        if not path or len(path) < 2:
            return None
        unk = self.mapper.unknown_touch_count(r, c)
        if unk < self.MIN_UNKNOWN_GAIN:
            return None
        probe = {
            "centroid_world": (wx, wy),
            "centroid_grid": (r, c),
            "goal_grid": (r, c),
            "size": 0,
            "unknown_gain": unk,
            "probe": True,
            "probe_score": float(unk),
        }
        if self._is_backtrack_target(probe, d_pos_w):
            return None
        return probe

    def _discover_opening_probes(self, d_pos_w, came_from, bfs_frontiers):
        """Last-resort probes toward bearings of known frontiers that failed commit.

        No 360° fan — that overfit arbitrary angles and spawned targets outside the
        mapped envelope. Only march along directions where BFS already found frontier
        clusters but path planning could not commit (e.g. tight turns).
        """
        if not bfs_frontiers:
            return []

        ranked = sorted(
            bfs_frontiers,
            key=lambda f: -int(f.get("unknown_gain", 0)),
        )[:6]
        probes = []
        seen_goals = set()
        for f in ranked:
            to_f = np.array(f["centroid_world"][:2], dtype=np.float64) - d_pos_w[:2]
            dist = float(np.linalg.norm(to_f))
            if dist < 1.0:
                continue
            probe = self._ray_probe_in_direction(d_pos_w, came_from, to_f / dist)
            if probe is None:
                continue
            g = probe["goal_grid"]
            if g in seen_goals:
                continue
            seen_goals.add(g)
            probe["probe_score"] = float(probe.get("unknown_gain", 0))
            probes.append(probe)

        probes.sort(key=lambda p: -float(p.get("probe_score", 0)))
        return probes

    def _get_closest_path_index(self, d_pos_w) -> int:
        """Find the index of the closest node on the A* path that is not obstructed by a wall."""
        if not getattr(self, "astar_path_world", None):
            return 0
        d_pos_2d = d_pos_w[:2]

        valid_indices = []
        for idx, node in enumerate(self.astar_path_world):
            if not self.mapper.segment_hits_wall(d_pos_w[0], d_pos_w[1], node[0], node[1]):
                valid_indices.append(idx)

        if valid_indices:
            sub_distances = [np.linalg.norm(d_pos_2d - np.array(self.astar_path_world[idx])) for idx in valid_indices]
            return valid_indices[int(np.argmin(sub_distances))]

        # Fallback if all are somehow obstructed
        distances = [np.linalg.norm(d_pos_2d - np.array(node)) for node in self.astar_path_world]
        return int(np.argmin(distances))

    def _stamp_visited(self, d_pos_w) -> None:
        """Mark a disk around the drone's current cell as 'visited' (pure SLAM).

        Radius 0.9 m — comfortably smaller than the 1.3 m minimum frontier distance,
        so a frontier the drone is currently APPROACHING (still ahead in unexplored
        space) is never falsely flagged visited; only ground the drone has actually
        flown over gets marked.
        """
        self._ensure_visited_mask()
        r, c = self.mapper.world_to_grid(d_pos_w[0], d_pos_w[1])
        if not self.mapper.is_in_bounds(r, c):
            return
        # 1.8 m disk: wide enough that flying through a room/corridor marks the whole
        # width as 'explored', so lingering shadow-pocket frontiers behind the drone
        # get rejected instead of luring it back into a room it already crossed.
        rad = max(1, int(round(1.8 / self.mapper.cell_size)))
        r0, r1 = max(0, r - rad), min(self.mapper.h, r + rad + 1)
        c0, c1 = max(0, c - rad), min(self.mapper.w, c + rad + 1)
        ys, xs = np.ogrid[r0:r1, c0:c1]
        disk = (ys - r) ** 2 + (xs - c) ** 2 <= rad * rad
        self.visited_mask[r0:r1, c0:c1] |= disk

    def _stamp_path_visited(self, path_world) -> None:
        """Mark cells along a path segment the drone has already flown."""
        if path_world is None or len(path_world) < 2:
            return
        self._ensure_visited_mask()
        rad = max(1, int(round(1.2 / self.mapper.cell_size)))
        for wx, wy in path_world:
            r, c = self.mapper.world_to_grid(float(wx), float(wy))
            if not self.mapper.is_in_bounds(r, c):
                continue
            r0, r1 = max(0, r - rad), min(self.mapper.h, r + rad + 1)
            c0, c1 = max(0, c - rad), min(self.mapper.w, c + rad + 1)
            ys, xs = np.ogrid[r0:r1, c0:c1]
            disk = (ys - r) ** 2 + (xs - c) ** 2 <= rad * rad
            self.visited_mask[r0:r1, c0:c1] |= disk

    def _path_visited_fraction(self, path_world, grid_path=None) -> float:
        """Fraction of path cells that lie in already-visited ground."""
        vis = getattr(self, "visited_mask", None)
        if vis is None:
            return 0.0
        cells = []
        if grid_path is not None:
            cells = list(grid_path)
        elif path_world is not None:
            for wx, wy in path_world:
                cells.append(self.mapper.world_to_grid(float(wx), float(wy)))
        if len(cells) < 2:
            return 0.0
        rev = sum(
            1 for r, c in cells
            if self.mapper.is_in_bounds(int(r), int(c)) and vis[int(r), int(c)]
        )
        return rev / len(cells)

    def _segment_mostly_visited(self, x0, y0, x1, y1, threshold=0.45) -> bool:
        """True if the straight segment drone→frontier mostly crosses visited cells."""
        vis = getattr(self, "visited_mask", None)
        if vis is None:
            return False
        r0, c0 = self.mapper.world_to_grid(x0, y0)
        r1, c1 = self.mapper.world_to_grid(x1, y1)
        n = int(max(abs(r1 - r0), abs(c1 - c0))) + 1
        if n < 2:
            return False
        rs = np.linspace(r0, r1, n).round().astype(int)
        cs = np.linspace(c0, c1, n).round().astype(int)
        rev = sum(
            1 for r, c in zip(rs, cs)
            if self.mapper.is_in_bounds(r, c) and vis[r, c]
        )
        return (rev / n) >= threshold

    def _frontier_fwd_dot(self, frontier, d_pos_w) -> tuple[float, float]:
        """Return (forward_dot, distance) from drone to frontier centroid."""
        cw = frontier["centroid_world"]
        to_f = np.array(cw[:2], dtype=np.float64) - np.array(d_pos_w[:2], dtype=np.float64)
        dist = float(np.linalg.norm(to_f))
        travel = getattr(self, "_travel_dir", None)
        if travel is None:
            travel = getattr(self, "_last_heading", None)
        if travel is None or dist < 1e-3:
            return 1.0, dist
        return float(np.dot(to_f / dist, travel)), dist

    def _frontier_fwd_dot_heading(self, frontier, d_pos_w) -> tuple[float, float]:
        """Forward dot using instantaneous yaw (reliable after turns)."""
        cw = frontier["centroid_world"]
        to_f = np.array(cw[:2], dtype=np.float64) - np.array(d_pos_w[:2], dtype=np.float64)
        dist = float(np.linalg.norm(to_f))
        hdg = getattr(self, "_last_heading", None)
        if hdg is None or dist < 1e-3:
            return 1.0, dist
        return float(np.dot(to_f / dist, hdg)), dist

    def _corridor_progress_dir(self) -> np.ndarray | None:
        """Stable forward direction inside corridors, based on motion before yaw."""
        for attr in ("_travel_dir", "_last_heading"):
            d = getattr(self, attr, None)
            if d is None:
                continue
            d = np.asarray(d[:2], dtype=np.float64)
            n = float(np.linalg.norm(d))
            if n > 1e-6:
                return d / n
        return None

    def _corridor_axis_from_local_free(self, d_pos_w, radius_m=2.2) -> np.ndarray | None:
        """Infer whether local SLAM free space is corridor-shaped, without USD hints."""
        r0, c0 = self.mapper.world_to_grid(d_pos_w[0], d_pos_w[1])
        if not self.mapper.is_in_bounds(r0, c0):
            return None
        free = self.mapper.get_traversable_free()
        rad = max(3, int(round(radius_m / self.mapper.cell_size)))
        r_min, r_max = max(0, r0 - rad), min(self.mapper.h, r0 + rad + 1)
        c_min, c_max = max(0, c0 - rad), min(self.mapper.w, c0 + rad + 1)
        pts = []
        for rr in range(r_min, r_max):
            for cc in range(c_min, c_max):
                if not free[rr, cc]:
                    continue
                wx, wy = self.mapper.grid_to_world(rr, cc)
                dx, dy = float(wx - d_pos_w[0]), float(wy - d_pos_w[1])
                if dx * dx + dy * dy <= radius_m * radius_m:
                    pts.append((dx, dy))
        if len(pts) < 12:
            return None
        arr = np.asarray(pts, dtype=np.float64)
        cov = np.cov(arr.T)
        vals, vecs = np.linalg.eigh(cov)
        small = max(float(vals[0]), 1e-6)
        large = float(vals[1])
        if large / small < 2.2:
            return None
        axis = vecs[:, 1]
        axis_n = float(np.linalg.norm(axis))
        if axis_n < 1e-6:
            return None
        return axis / axis_n

    def _is_corridor_context(self, d_pos_w) -> bool:
        return self._corridor_axis_from_local_free(d_pos_w) is not None

    def _update_corridor_progress_context(self, d_pos_w) -> None:
        """Keep corridor behavior active briefly through junctions/open doorways."""
        if self._is_corridor_context(d_pos_w):
            self._corridor_context_ticks = 80
        else:
            self._corridor_context_ticks = max(
                0, int(getattr(self, "_corridor_context_ticks", 0)) - 1
            )

    def _in_corridor_progress_context(self, d_pos_w) -> bool:
        return (
            int(getattr(self, "_corridor_context_ticks", 0)) > 0
            or self._is_corridor_context(d_pos_w)
        )

    def _frontier_progress_dot(self, frontier, d_pos_w) -> tuple[float, float]:
        """Dot against stable travel direction; side openings are near zero, rear is negative."""
        cw = frontier["centroid_world"]
        to_f = np.array(cw[:2], dtype=np.float64) - np.array(d_pos_w[:2], dtype=np.float64)
        dist = float(np.linalg.norm(to_f))
        progress = self._corridor_progress_dir()
        if progress is None or dist < 1e-3:
            return 1.0, dist
        return float(np.dot(to_f / dist, progress)), dist

    def _is_forward_corridor_frontier(self, frontier, d_pos_w) -> bool:
        """True for a corridor target that is ahead or a side opening, not behind."""
        if not self._in_corridor_progress_context(d_pos_w):
            return False
        dot, _ = self._frontier_progress_dot(frontier, d_pos_w)
        return dot >= -0.10

    def _corridor_frontier_gate(self, frontiers, d_pos_w, stage_label=""):
        """In corridors, keep only forward/lateral new openings; reject rear side-camera steals."""
        if not frontiers or not self._in_corridor_progress_context(d_pos_w):
            return frontiers
        front_or_side = []
        rear = 0
        for f in frontiers:
            dot, dist = self._frontier_progress_dot(f, d_pos_w)
            if dot >= -0.10 or dist < 1.25:
                front_or_side.append(f)
            else:
                rear += 1
        if rear > 0:
            self._corridor_gate_log_ticks = int(getattr(self, "_corridor_gate_log_ticks", 0)) + 1
            if self._corridor_gate_log_ticks % 20 == 1:
                print(
                    f"[SLAM Brain] Corridor gate{stage_label}: keeping "
                    f"{len(front_or_side)} forward/side frontier(s), ignoring {rear} rear frontier(s)."
                )
        return front_or_side

    def _mission_assist_route_local(self) -> list[tuple[float, float, float]]:
        """Cheat-mode route through the corridor/side corridor/final room."""
        route = []
        use_usd_corridor = bool(
            getattr(self.env.cfg, "brain_use_usd_corridor_waypoints", False)
        )
        if not use_usd_corridor:
            route.append(
                tuple(
                    getattr(
                        self.env.cfg,
                        "brain_room4_corr1_waypoint",
                        (0.0, -20.5, 1.0),
                    )
                )
            )
            if not bool(getattr(self.env.cfg, "brain_single_corridor_to_final", True)):
                corr2 = tuple(
                    getattr(
                        self.env.cfg,
                        "brain_room4_corr2_waypoint",
                        (0.0, -20.5, 1.0),
                    )
                )
                if np.linalg.norm(np.asarray(corr2[:2]) - np.asarray(route[-1][:2])) > 0.5:
                    route.append(corr2)
            route.append(
                tuple(
                    getattr(
                        self.env.cfg,
                        "brain_final_room_waypoint",
                        (-6.0, -21.5, 1.0),
                    )
                )
            )
            return [
                (float(p[0]), float(p[1]), float(p[2]))
                for p in route
                if len(p) >= 3
            ]

        zones = getattr(self.env, "_map_zones", None) or {}
        corridor = zones.get("corridor")
        side = zones.get("side_coridors")

        corridor_point = None
        if corridor and corridor.get("bounds"):
            lx0, lx1, ly0, ly1 = [float(v) for v in corridor["bounds"]]
            corridor_point = (0.5 * (lx0 + lx1), min(ly0, ly1) + 0.75, 1.0)
        else:
            corridor_point = tuple(getattr(self.env.cfg, "brain_room4_corr1_waypoint", (0.0, -20.5, 1.0)))

        if corridor_point is not None:
            route.append(corridor_point)

        if side and side.get("bounds"):
            lx0, lx1, ly0, ly1 = [float(v) for v in side["bounds"]]
            # After the long corridor exposes the side corridor, walk the short corridor.
            route.append((max(lx0, lx1) - 0.45, 0.5 * (ly0 + ly1), 1.0))
            route.append((min(lx0, lx1) + 0.45, 0.5 * (ly0 + ly1), 1.0))
        else:
            route.append(tuple(getattr(self.env.cfg, "brain_room4_corr2_waypoint", (0.0, -20.5, 1.0))))

        final = tuple(getattr(self.env.cfg, "brain_final_room_waypoint", (-6.0, -21.5, 1.0)))
        route.append(final)

        cleaned = []
        for p in route:
            if len(p) >= 3:
                cleaned.append((float(p[0]), float(p[1]), float(p[2])))
        return cleaned

    def _forced_corridor_route_local(self) -> list[tuple[float, float, float]]:
        """Hard late-mission route: corridor entrance -> long corridor -> side corridor -> final room."""
        seq = list(getattr(self.env.cfg, "brain_spawn_sequence", ()) or ())
        corridor_start = tuple(seq[3]) if len(seq) > 3 else (0.0, -16.5, 1.0)
        corr1 = tuple(getattr(self.env.cfg, "brain_room4_corr1_waypoint", (0.0, -20.5, 1.0)))
        corr2 = tuple(getattr(self.env.cfg, "brain_room4_corr2_waypoint", (-3.2, -20.5, 1.0)))
        final = tuple(getattr(self.env.cfg, "brain_final_room_waypoint", (-6.0, -21.5, 1.0)))
        return [
            (float(corridor_start[0]), float(corridor_start[1]), 1.0),
            (float(corr1[0]), float(corr1[1]), 1.0),
            (float(corr2[0]), float(corr2[1]), 1.0),
            (float(final[0]), float(final[1]), 1.0),
        ]

    def _coverage_fraction(self) -> float:
        visited, total = self.coverage_stats()
        return float(visited) / max(float(total), 1.0)

    def _forced_corridor_route_index_from_local_xy(self, local_xy) -> int:
        """Pick the next forced-route waypoint from the drone's real local position."""
        route = self._forced_corridor_route_local()
        if not route:
            return 0
        pts = [np.asarray(p[:2], dtype=np.float64) for p in route]
        local_xy = np.asarray(local_xy[:2], dtype=np.float64)

        if len(pts) == 1:
            return 0

        cum = [0.0]
        for i in range(1, len(pts)):
            cum.append(cum[-1] + float(np.linalg.norm(pts[i] - pts[i - 1])))

        best_arc = 0.0
        best_d2 = float("inf")
        for i in range(len(pts) - 1):
            a, b = pts[i], pts[i + 1]
            ab = b - a
            l2 = float(ab @ ab)
            t = 0.0 if l2 < 1e-9 else float(np.clip(((local_xy - a) @ ab) / l2, 0.0, 1.0))
            proj = a + t * ab
            d2 = float(np.linalg.norm(local_xy - proj) ** 2)
            if d2 < best_d2:
                best_d2 = d2
                best_arc = cum[i] + t * float(np.linalg.norm(ab))

        next_idx = 0
        for i, arc in enumerate(cum):
            if arc <= best_arc + 0.35:
                next_idx = min(i + 1, len(pts) - 1)
            else:
                break

        for i, p in enumerate(pts):
            arrival_radius = 0.95 if i == 0 else 0.75
            if float(np.linalg.norm(local_xy - p)) <= arrival_radius:
                next_idx = min(i + 1, len(pts) - 1)

        # Before the corridor entrance, always restart the forced section from there.
        if float(np.linalg.norm(local_xy - pts[0])) > 2.25 and best_arc < cum[0] + 0.5:
            next_idx = 0
        return int(max(0, min(next_idx, len(pts) - 1)))

    def _maybe_start_forced_corridor_route(self) -> bool:
        if bool(getattr(self, "_forced_corridor_route_active", False)):
            return True
        threshold = float(
            getattr(self.env.cfg, "brain_forced_corridor_route_coverage", 0.68)
        )
        if self._coverage_fraction() < threshold:
            return False
        self._forced_corridor_route_active = True
        self._forced_corridor_route_idx = 0
        self._mission_assist_active = False
        self._mission_assist_idx = 0
        self.active_frontier = None
        self.astar_path_world = []
        if not getattr(self, "_forced_corridor_route_logged", False):
            print(
                f"[SLAM Brain] Forced corridor route activated at "
                f"{self._coverage_fraction() * 100.0:.1f}% coverage."
            )
            self._forced_corridor_route_logged = True
        return True

    def _commit_forced_corridor_route(self, d_pos_w) -> bool:
        route = self._forced_corridor_route_local()
        if not route:
            return False
        env_origin = self.env._terrain.env_origins[0].cpu().numpy()
        local_xy = np.asarray(d_pos_w[:2], dtype=np.float64) - env_origin[:2]
        saved_idx = min(int(getattr(self, "_forced_corridor_route_idx", 0)), len(route) - 1)
        pos_idx = self._forced_corridor_route_index_from_local_xy(local_xy)
        if pos_idx < saved_idx or saved_idx == 0:
            idx = pos_idx
        else:
            idx = saved_idx

        while idx < len(route) - 1:
            wp = np.asarray(route[idx][:2], dtype=np.float64)
            arrival_radius = 0.95 if idx == 0 else 0.75
            if float(np.linalg.norm(local_xy - wp)) > arrival_radius:
                break
            idx += 1

        self._forced_corridor_route_idx = idx
        target = route[idx]
        target_world = (
            float(target[0] + env_origin[0]),
            float(target[1] + env_origin[1]),
        )
        self.active_frontier = {
            "centroid_world": target_world,
            "centroid_grid": self.mapper.world_to_grid(target_world[0], target_world[1]),
            "goal_grid": self.mapper.world_to_grid(target_world[0], target_world[1]),
            "size": 1,
            "unknown_gain": 600,
            "mission_assist": True,
            "forced_corridor_route": True,
            "forced_route_idx": idx,
        }
        self.astar_path_world = [
            (float(d_pos_w[0]), float(d_pos_w[1])),
            target_world,
        ]
        self._frontier_lock_ticks = 320
        self._corridor_context_ticks = max(int(getattr(self, "_corridor_context_ticks", 0)), 120)
        if int(getattr(self, "_forced_route_log_tick", 0)) % 20 == 0:
            print(
                f"[SLAM Brain] Forced corridor route target {idx + 1}/{len(route)}: "
                f"({target[0]:.2f}, {target[1]:.2f})"
            )
        self._forced_route_log_tick = int(getattr(self, "_forced_route_log_tick", 0)) + 1
        return True

    def _should_use_mission_assist(self, d_pos_w) -> bool:
        if bool(getattr(self, "_forced_corridor_route_active", False)):
            return False
        if bool(getattr(self, "_mission_assist_active", False)):
            return True
        env_origin = self.env._terrain.env_origins[0].cpu().numpy()
        local_xy = np.asarray(d_pos_w[:2], dtype=np.float64) - env_origin[:2]
        zones = getattr(self.env, "_map_zones", None) or {}
        for key in ("corridor", "side_coridors", "room_4"):
            zone = zones.get(key)
            if not zone or not zone.get("bounds"):
                continue
            x0, x1, y0, y1 = [float(v) for v in zone["bounds"]]
            margin = 1.5
            if (
                min(x0, x1) - margin <= local_xy[0] <= max(x0, x1) + margin
                and min(y0, y1) - margin <= local_xy[1] <= max(y0, y1) + margin
            ):
                return True
        return int(getattr(self, "max_segment_reached", 0)) >= 3

    def _mission_assist_target(self, d_pos_w, came_from, start_grid):
        """Pick the reachable mapped-free cell closest to the next mission route point."""
        if not self._should_use_mission_assist(d_pos_w):
            return None
        route = self._mission_assist_route_local()
        if not route:
            return None

        env_origin = self.env._terrain.env_origins[0].cpu().numpy()
        current_local = np.asarray(d_pos_w[:2], dtype=np.float64) - env_origin[:2]
        idx = min(int(getattr(self, "_mission_assist_idx", 0)), len(route) - 1)
        while idx < len(route) - 1:
            wp = np.asarray(route[idx][:2], dtype=np.float64)
            if float(np.linalg.norm(current_local - wp)) >= 1.1:
                break
            idx += 1
        self._mission_assist_idx = idx

        target_local = np.asarray(route[idx][:2], dtype=np.float64)
        target_world = target_local + env_origin[:2]
        displayed_route_point = route[idx]

        # The first assist leg is "continue down this corridor". Its configured
        # x value is only a nominal centerline. If the mapped corridor lane is
        # offset, forcing the drone toward that nominal x pulls it out of the
        # corridor. Keep the current lane and advance only along corridor Y.
        if idx == 0 and abs(float(target_local[1] - current_local[1])) > abs(float(target_local[0] - current_local[0])) * 1.8:
            target_world = np.array(
                [float(d_pos_w[0]), float(target_world[1])], dtype=np.float64
            )
            displayed_route_point = (
                float(current_local[0]),
                float(route[idx][1]),
                float(route[idx][2]),
            )
        free = self.mapper.get_traversable_free()
        prob = self.mapper.get_occupancy_grid()
        reached = (came_from[:, :, 0] >= 0) | (
            (np.indices((self.mapper.h, self.mapper.w))[0] == int(start_grid[0]))
            & (np.indices((self.mapper.h, self.mapper.w))[1] == int(start_grid[1]))
        )

        candidates = []
        progress_dir = target_world - np.asarray(d_pos_w[:2], dtype=np.float64)
        progress_norm = float(np.linalg.norm(progress_dir))
        if progress_norm > 1e-6:
            progress_dir = progress_dir / progress_norm

        route_is_final_leg = idx >= len(route) - 1
        max_lateral = 2.35 if route_is_final_leg else 1.15
        min_forward = 0.35

        rows, cols = np.where(reached & free & (prob < 0.35))
        for r, c in zip(rows, cols):
            if self.mapper.get_clearance_at_grid(int(r), int(c)) < 0.15:
                continue
            wx, wy = self.mapper.grid_to_world(int(r), int(c))
            if not self._clear_of_live_dynamic_obstacles(wx, wy, margin=0.35):
                continue
            from_drone = np.array([wx - d_pos_w[0], wy - d_pos_w[1]], dtype=np.float64)
            dist_from_drone = float(np.linalg.norm(from_drone))
            if dist_from_drone < 0.9:
                continue
            dist_to_target = float(np.linalg.norm(np.array([wx, wy]) - target_world))
            forward_bonus = 0.0
            forward_progress = 0.0
            lateral_error = 0.0
            if progress_norm > 1e-6 and dist_from_drone > 1e-6:
                forward_bonus = float(np.dot(from_drone / dist_from_drone, progress_dir))
                forward_progress = float(np.dot(from_drone, progress_dir))
                lateral_vec = from_drone - forward_progress * progress_dir
                lateral_error = float(np.linalg.norm(lateral_vec))
            if forward_progress < min_forward:
                continue
            if lateral_error > max_lateral:
                continue
            # Mission assist should pull the drone deeper along the route, not
            # sideways to arbitrary reachable cells near the corridor mouth.
            score = (
                -2.25 * forward_progress
                + 1.65 * lateral_error
                + 0.04 * dist_to_target
                - 0.25 * forward_bonus
            )
            candidates.append((score, int(r), int(c), float(wx), float(wy)))

        if not candidates and progress_norm > 1e-6:
            # If the corridor is still barely mapped, allow a tiny tube expansion
            # before giving up. This keeps the drone moving forward while the side
            # depth cameras reveal more cells, but still blocks rear/side-room steals.
            relaxed_lateral = max_lateral + 0.65
            for r, c in zip(rows, cols):
                wx, wy = self.mapper.grid_to_world(int(r), int(c))
                from_drone = np.array([wx - d_pos_w[0], wy - d_pos_w[1]], dtype=np.float64)
                dist_from_drone = float(np.linalg.norm(from_drone))
                if dist_from_drone < 0.8:
                    continue
                forward_progress = float(np.dot(from_drone, progress_dir))
                if forward_progress < 0.15:
                    continue
                lateral_vec = from_drone - forward_progress * progress_dir
                lateral_error = float(np.linalg.norm(lateral_vec))
                if lateral_error > relaxed_lateral:
                    continue
                if self.mapper.get_clearance_at_grid(int(r), int(c)) < 0.10:
                    continue
                if not self._clear_of_live_dynamic_obstacles(wx, wy, margin=0.25):
                    continue
                dist_to_target = float(np.linalg.norm(np.array([wx, wy]) - target_world))
                score = -1.6 * forward_progress + 1.9 * lateral_error + 0.05 * dist_to_target
                candidates.append((score, int(r), int(c), float(wx), float(wy)))

        if not candidates:
            return None

        wall_mask, obstacle_mask = self.mapper.get_wall_obstacle_masks(use_walkable=False)
        best = None
        for _, r, c, wx, wy in sorted(candidates, key=lambda item: item[0])[:350]:
            grid_path = self.mapper.reconstruct_path(came_from, (r, c))
            if not grid_path or len(grid_path) < 2:
                continue
            path_safe = True
            for pr, pc in grid_path[1:]:
                if not self.mapper.is_in_bounds(pr, pc):
                    path_safe = False
                    break
                if wall_mask[pr, pc] == 1 or obstacle_mask[pr, pc] == 1 or prob[pr, pc] > 0.65:
                    path_safe = False
                    break
                pwx, pwy = self.mapper.grid_to_world(pr, pc)
                if not self._clear_of_live_dynamic_obstacles(pwx, pwy, margin=0.25):
                    path_safe = False
                    break
            if path_safe:
                best = (r, c, wx, wy, grid_path)
                break

        if best is None:
            return None
        r, c, wx, wy, grid_path = best
        world_path = [self.mapper.grid_to_world(pr, pc) for pr, pc in grid_path]
        frontier = {
            "centroid_world": (wx, wy),
            "centroid_grid": (r, c),
            "goal_grid": (r, c),
            "size": 1,
            "unknown_gain": 600,
            "mission_assist": True,
            "mission_assist_idx": idx,
        }
        return frontier, world_path, displayed_route_point

    def _commit_mission_assist(self, d_pos_w, came_from, start_grid, reason="fallback") -> bool:
        assist = self._mission_assist_target(d_pos_w, came_from, start_grid)
        if assist is None:
            return False
        frontier, world_path, route_point = assist
        self.active_frontier = frontier
        self.astar_path_world = world_path
        self._mission_assist_active = True
        self._hold_log_ticks = 0
        self._frontier_lock_ticks = max(220, min(520, len(world_path) * 4))
        self._corridor_context_ticks = max(int(getattr(self, "_corridor_context_ticks", 0)), 80)
        print(
            f"[SLAM Brain] Target frontier (mission assist/{reason}): "
            f"{frontier['centroid_world']} -> route "
            f"({route_point[0]:.2f}, {route_point[1]:.2f}) "
            f"({len(world_path)} waypoints)"
        )
        return True

    def _active_target_is_mission_assist(self) -> bool:
        af = getattr(self, "active_frontier", None)
        return bool(isinstance(af, dict) and af.get("mission_assist", False))

    def _active_target_is_forced_corridor_route(self) -> bool:
        af = getattr(self, "active_frontier", None)
        return bool(isinstance(af, dict) and af.get("forced_corridor_route", False))

    def _clear_of_live_dynamic_obstacles(self, x: float, y: float, margin: float = 0.25) -> bool:
        """Mission-assist guard: do not place/route cheat targets through live randomized props."""
        obstacles = getattr(self.env, "_live_dynamic_obstacle_clearance_xyr", None) or []
        if not obstacles:
            return True
        for ox, oy, radius in obstacles:
            dx = float(x) - float(ox)
            dy = float(y) - float(oy)
            effective_radius = max(0.25, float(radius) * 0.55)
            if dx * dx + dy * dy < (effective_radius + margin) ** 2:
                return False
        return True

    def _goal_region_mostly_visited(self, frontier, radius=3, threshold=0.45) -> bool:
        """True when the free-space portion of the frontier goal's neighborhood has been explored."""
        vis = getattr(self, "visited_mask", None)
        if vis is None:
            return False
        goal = frontier.get("goal_grid")
        if goal is None:
            cw = frontier["centroid_world"]
            goal = self.mapper.world_to_grid(cw[0], cw[1])
        r, c = int(goal[0]), int(goal[1])
        
        prob = self.mapper.get_occupancy_grid()
        free_total, rev = 0, 0
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                rr, cc = r + dr, c + dc
                if self.mapper.is_in_bounds(rr, cc):
                    # Only count known free cells (prob < 0.35)
                    if prob[rr, cc] < 0.35:
                        free_total += 1
                        if vis[rr, cc]:
                            rev += 1
        return free_total > 0 and (rev / free_total) >= threshold

    def _is_live_opening(self, frontier) -> bool:
        """True if the goal still borders unexplored space (active corridor/room mouth)."""
        goal = frontier.get("goal_grid")
        if goal is None:
            cw = frontier.get("centroid_world")
            if cw is None:
                return False
            goal = self.mapper.world_to_grid(cw[0], cw[1])
        return self.mapper.is_cell_frontier(goal, radius=1)

    def _unknown_ahead(self, frontier, radius=6) -> int:
        """Unknown cells near the goal — high means real new space, not a shadow pocket."""
        goal = frontier.get("goal_grid")
        if goal is None:
            cw = frontier.get("centroid_world")
            if cw is None:
                return 0
            goal = self.mapper.world_to_grid(cw[0], cw[1])
        return int(self.mapper.unknown_touch_count(int(goal[0]), int(goal[1]), radius=radius))

    def _path_ends_in_visited(self, came_from, goal, tail_frac=0.30, threshold=0.50) -> bool:
        """True if the last portion of the BFS route ends in explored territory."""
        vis = getattr(self, "visited_mask", None)
        if vis is None or came_from is None or goal is None:
            return False
        path = self.mapper.reconstruct_path(came_from, goal)
        if not path or len(path) < 3:
            return False
        n_tail = max(2, int(len(path) * tail_frac))
        tail = path[-n_tail:]
        rev = sum(
            1 for r, c in tail
            if self.mapper.is_in_bounds(int(r), int(c)) and vis[int(r), int(c)]
        )
        return (rev / len(tail)) >= threshold

    def _is_corridor_frontier(self, frontier) -> bool:
        """True if this frontier goal sits in a narrow corridor passage."""
        goal = frontier.get("goal_grid")
        if goal is None:
            return False
        return self.mapper.is_narrow_frontier(goal)

    def _is_backtrack_target(self, frontier, d_pos_w, came_from=None, ignore_heading_backtrack=False) -> bool:
        """True if committing to this frontier means revisiting explored ground."""
        if frontier.get("mission_assist", False):
            return False
        # 1. Segment-based backtracking: block targets in previously cleared rooms/areas
        max_reached = int(getattr(self, "max_segment_reached", 0))
        if hasattr(self, "dynamic_room_nodes") and len(self.dynamic_room_nodes) > 0 and max_reached > 0:
            env_origin = self.env._terrain.env_origins[0].cpu().numpy()
            c_world = frontier.get("centroid_world")
            if c_world is not None:
                # Get frontier coordinates relative to environment origin to match our dynamic nodes
                f_xy = np.array([float(c_world[0]) - float(env_origin[0]), float(c_world[1]) - float(env_origin[1])])
                distances = [np.linalg.norm(f_xy - node) for node in self.dynamic_room_nodes]
                f_segment_idx = int(np.argmin(distances))
                
                if f_segment_idx < max_reached:
                    # Map drone current relative position
                    drone_xy = d_pos_w[:2] - env_origin[:2]
                    dist_to_frontier = np.linalg.norm(f_xy - drone_xy)
                    
                    # Local backtrack: allow targets that are nearby (within 6.0m)
                    if dist_to_frontier > 6.0 and f_segment_idx < max_reached:
                        # Global backtrack: block if the frontier is in the opposite direction of the overall exploration path
                        if getattr(self, "_start_pos_xy", None) is not None:
                            expl_vec = drone_xy - self._start_pos_xy
                            to_frontier_vec = f_xy - drone_xy
                            norm_expl = float(np.linalg.norm(expl_vec))
                            norm_tf = float(np.linalg.norm(to_frontier_vec))
                            if norm_expl > 1.0 and norm_tf > 1.0:
                                dot_val = float(np.dot(expl_vec / norm_expl, to_frontier_vec / norm_tf))
                                # If the target is clearly behind the general exploration direction, block it
                                if dot_val < -0.2:
                                    return True
                                else:
                                    return False  # Forward exploration! Allow it.
                        # If start_pos_xy is not set or direction check is inconclusive, block anyway if segment is too old
                        return True

        if ignore_heading_backtrack:
            return False

        goal = frontier.get("goal_grid")
        fwd_h, dist = self._frontier_fwd_dot_heading(frontier, d_pos_w)
        unk = self._unknown_ahead(frontier)

        is_visited_region = self._goal_region_mostly_visited(frontier)
        if came_from is not None and goal is not None:
            is_visited_region = is_visited_region or self._path_ends_in_visited(came_from, goal)

        # Only block targets clearly behind the drone (not lateral corridor openings) in visited territory.
        if fwd_h < -0.35 and dist > 2.0 and is_visited_region:
            return True
        if fwd_h < -0.50 and dist > 1.0 and is_visited_region:
            return True

        if is_visited_region:
            if fwd_h > 0.15:
                return False
            if unk >= 20 and fwd_h > -0.20:
                return False
            return True

        fwd_t, _ = self._frontier_fwd_dot(frontier, d_pos_w)
        if fwd_t < -0.15 and fwd_h < -0.15 and dist > self.BACKTRACK_MAX_M and is_visited_region:
            return True
        return False

    def is_explorable_frontier(self, frontier, d_pos_w, came_from=None, ignore_heading_backtrack=False) -> bool:
        """True if this frontier is not a genuine backtrack into explored rooms."""
        return not self._is_backtrack_target(frontier, d_pos_w, came_from=came_from, ignore_heading_backtrack=ignore_heading_backtrack)

    def _prepare_commit_frontier(self, frontier, came_from, start_grid):
        """Copy frontier dict and push the goal to the deepest corridor dead-end."""
        f = dict(frontier)
        goal = f.get("goal_grid")
        if goal is None:
            cw = f["centroid_world"]
            goal = self.mapper.world_to_grid(cw[0], cw[1])
            f["goal_grid"] = goal
        if self.mapper.is_cell_frontier(goal, radius=1):
            vis = getattr(self, "visited_mask", None)
            deeper = self.mapper.deepen_frontier_goal(
                goal, came_from, start_grid, visited_mask=vis
            )
            if deeper is not None:
                f["goal_grid"] = deeper
                wx, wy = self.mapper.grid_to_world(deeper[0], deeper[1])
                f["centroid_world"] = (wx, wy)
                f["centroid_grid"] = deeper
        return f

    def _has_arrived_at_frontier(self, d_pos_w, frontier, dist_to_f) -> bool:
        """True only when the drone is within arrival tolerance."""
        if frontier.get("forced_corridor_route", False):
            idx = int(frontier.get("forced_route_idx", 0))
            return dist_to_f < (1.0 if idx == 0 else 0.75)
        if self._is_forward_corridor_frontier(frontier, d_pos_w):
            return dist_to_f < 0.25
        return dist_to_f < 0.50

    def _try_extend_active_goal(self, d_pos_w, came_from) -> bool:
        """Push the active corridor target deeper as the drone maps forward."""
        af = self.active_frontier
        if af is None:
            return False
        goal = af.get("goal_grid")
        if goal is None or not self.mapper.is_cell_frontier(goal):
            return False
        sr, sc = self.mapper.world_to_grid(d_pos_w[0], d_pos_w[1])
        vis = getattr(self, "visited_mask", None)
        deeper = self.mapper.deepen_frontier_goal(
            goal, came_from, (sr, sc), visited_mask=vis
        )
        if deeper is None or deeper == goal:
            return False
        wx, wy = self.mapper.grid_to_world(deeper[0], deeper[1])
        af["goal_grid"] = deeper
        af["centroid_world"] = (wx, wy)
        af["centroid_grid"] = deeper
        world_path = self.mapper.plan_path_centered((sr, sc), deeper)
        if world_path is None or len(world_path) < 2:
            grid_path = self.mapper.reconstruct_path(came_from, deeper)
            if grid_path and len(grid_path) >= 2:
                is_safe = True
                prob = self.mapper.get_occupancy_grid()
                wall_mask, _ = self.mapper.get_wall_obstacle_masks(use_walkable=False)
                for r, c in grid_path[:-1]:
                    if not self.mapper.is_in_bounds(r, c) or wall_mask[r, c] == 1 or prob[r, c] >= 0.40:
                        is_safe = False
                        break
                if is_safe:
                    world_path = [
                        self.mapper.grid_to_world(r, c) for r, c in grid_path
                    ]
        if world_path and len(world_path) >= 2:
            self.astar_path_world = world_path
            self._frontier_lock_ticks = max(
                int(getattr(self, "_frontier_lock_ticks", 0)),
                min(500, len(world_path) * 3),
            )
            print(
                f"[SLAM Brain] Extended corridor goal to {af['centroid_world']} "
                f"({len(world_path)} waypoints)"
            )
            return True
        return False

    def is_frontier_ahead(self, centroid_world) -> bool:
        """Pure-SLAM, DIRECTION-GATED anti-backtrack (no USD/ground-truth map).

        Rejects a frontier ONLY when it is both:
          (a) clearly BEHIND the drone's current heading (a fly-back), AND
          (b) sitting in ground the drone has already flown over (visited mask).

        Forward and side openings are ALWAYS allowed — even if the drone's visited
        disk grazes them — which is the fix for "there's clear space ahead but no
        target". Only genuine turn-arounds into already-mapped rooms are blocked.
        Used by both the brain's selection and the dashboard for consistency.
        """
        mask = getattr(self, "visited_mask", None)
        pos = getattr(self, "_last_drone_xy", None)
        # Prefer the smoothed net-travel direction; fall back to instantaneous yaw.
        hdg = getattr(self, "_travel_dir", None)
        if hdg is None:
            hdg = getattr(self, "_last_heading", None)
        if mask is None or pos is None or hdg is None:
            return True

        to_f = np.array(centroid_world[:2], dtype=np.float64) - pos
        dist = float(np.linalg.norm(to_f))
        fwd = float(np.dot(to_f / dist, hdg)) if dist > 1e-3 else 1.0
        if fwd > -0.2:
            return True  # ahead / to the side → always a valid exploration target

        # Behind the drone. NEVER fly a long way back across the map (that's how it
        # ended up crossing everything to re-enter the first room). A behind frontier
        # is only acceptable if it's NEARBY (a branch just off the current spot)...
        if dist > self.BACKTRACK_MAX_M:
            return False
        # ...and even then, not if it's re-treading ground already flown over.
        r, c = self.mapper.world_to_grid(centroid_world[0], centroid_world[1])
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if self.mapper.is_in_bounds(rr, cc) and mask[rr, cc]:
                    return False  # behind AND already visited → backtrack, reject
        return True

    def _fuse_depth_camera(self, camera, cam_cfg, d_pos_w, d_quat, yaw_offset_deg) -> bool:
        """Project a side camera's depth into the grid using the mapper's convention.

        The mapper assumes the camera optical axis points along the DRONE BODY +X
        (that's why the front cam is fed the drone root quaternion, not the camera's
        optical quaternion). For a side camera we therefore feed the drone quaternion
        rotated by ±90 deg yaw so its 'forward' aligns with the left/right view.
        Passing the raw optical quaternion instead scatters the depth at wrong angles
        and shreds the map into ghost walls.
        """
        if camera is None or cam_cfg is None:
            return False
        depth_tensor = camera.data.output.get("depth")
        if depth_tensor is None:
            return False
        depth_np = torch.squeeze(depth_tensor[0]).detach().cpu().numpy()

        half = math.radians(yaw_offset_deg) * 0.5
        yaw_q = np.array([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=np.float64)
        w0, x0, y0, z0 = (float(d_quat[0]), float(d_quat[1]), float(d_quat[2]), float(d_quat[3]))
        w1, x1, y1, z1 = yaw_q
        eff_quat = np.array([
            w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
            w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
            w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
            w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
        ], dtype=np.float64)

        self.mapper.update_from_depth(
            depth_np, d_pos_w, eff_quat,
            focal_length=float(cam_cfg.spawn.focal_length),
            horizontal_aperture=float(cam_cfg.spawn.horizontal_aperture),
        )
        return True

    def _yolo_rescue_conf_threshold(self) -> float:
        return float(getattr(self.env.cfg, "yolo_person_conf_threshold", 0.70))

    def _current_yolo_confidence(self) -> float:
        perception = getattr(self.env, "_perception", None)
        if perception is None:
            return 0.0
        state = getattr(perception, "_web_state", None) or {}
        if not state.get("has_confirmed"):
            return 0.0
        intel = getattr(perception, "_last_intel", None)
        if isinstance(intel, dict) and intel.get("conf") is not None:
            return float(intel["conf"])
        return float(getattr(perception, "last_best_person_conf", 0.0))

    def update(self, person_found, person_world_xyz, drone_pos, drone_quat):
        """SLAM-driven high-level brain update logic."""
        env_origin = self.env._terrain.env_origins[0].cpu().numpy()
        d_pos_w = drone_pos[0].cpu().numpy()
        d_quat = drone_quat[0].cpu().numpy()

        qw, qx, qy, qz = d_quat
        drone_yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        # Cache pose so is_frontier_ahead() (also called by the dashboard) can apply
        # the direction-gated visited check without extra arguments.
        self._last_drone_xy = np.array(d_pos_w[:2], dtype=np.float64)
        self._last_heading = np.array([math.cos(drone_yaw), math.sin(drone_yaw)])
        # Smoothed NET travel direction (from actual position deltas). More stable
        # than instantaneous yaw for anti-backtrack: a momentary rotation doesn't
        # flip it, so "behind" keeps meaning "where the drone came from".
        prev = getattr(self, "_prev_pos_xy", None)
        if prev is not None:
            step = self._last_drone_xy - prev
            sn = float(np.linalg.norm(step))
            if sn > 0.02:
                sd = step / sn
                td = getattr(self, "_travel_dir", None)
                if td is None:
                    self._travel_dir = sd
                else:
                    self._travel_dir = 0.9 * td + 0.1 * sd
                    n = float(np.linalg.norm(self._travel_dir))
                    if n > 1e-6:
                        self._travel_dir = self._travel_dir / n
        self._prev_pos_xy = self._last_drone_xy.copy()

        # Update segment_idx dynamically to match the current room of the drone
        env_origin = self.env._terrain.env_origins[0].cpu().numpy()
        drone_xy = d_pos_w[:2] - env_origin[:2]
        
        if getattr(self, "_start_pos_xy", None) is None:
            self._start_pos_xy = drone_xy.copy()
            
        if not hasattr(self, "dynamic_room_nodes") or self.dynamic_room_nodes is None:
            self.dynamic_room_nodes = []

        if len(self.dynamic_room_nodes) == 0:
            self.dynamic_room_nodes.append(drone_xy.copy())
            self.segment_idx = 0
            self.max_segment_reached = 0
        else:
            distances = [np.linalg.norm(drone_xy - node) for node in self.dynamic_room_nodes]
            closest_idx = int(np.argmin(distances))
            min_d = distances[closest_idx]
            
            # If the drone moves at least 6.0 meters from all existing nodes,
            # spawn a new dynamic node.
            if min_d > 6.0:
                self.dynamic_room_nodes.append(drone_xy.copy())
                closest_idx = len(self.dynamic_room_nodes) - 1
                
            self.segment_idx = closest_idx
            self.max_segment_reached = max(
                int(getattr(self, "max_segment_reached", 0)), self.segment_idx
            )
        self._env_origin_xy = env_origin[:2]

        # Fuse the front depth camera into the occupancy grid.
        depth_tensor = self.env._tiled_camera.data.output.get("depth")
        if depth_tensor is not None:
            depth_np = torch.squeeze(depth_tensor[0]).detach().cpu().numpy()
            self.mapper.update_from_depth(
                depth_np, d_pos_w, d_quat,
                focal_length=float(self.env.cfg.tiled_camera.spawn.focal_length),
                horizontal_aperture=float(self.env.cfg.tiled_camera.spawn.horizontal_aperture),
            )

        # Left/right body cameras — SLAM mapping only (policy still uses front cam).
        # Left cam looks toward drone +Y (yaw +90), right cam toward -Y (yaw -90).
        if getattr(self.env.cfg, "brain_slam_side_map_cameras", False):
            side_fused = 0
            for cam_attr, cfg_attr, yaw_off in (
                ("_view_left_camera", "view_left_camera", 90.0),
                ("_view_right_camera", "view_right_camera", -90.0),
            ):
                cam = getattr(self.env, cam_attr, None)
                cam_cfg = getattr(self.env.cfg, cfg_attr, None)
                if self._fuse_depth_camera(cam, cam_cfg, d_pos_w, d_quat, yaw_off):
                    side_fused += 1
            if side_fused and not getattr(self, "_side_map_logged", False):
                self._side_map_logged = True
                print(
                    f"[SLAM Brain] Side depth cameras active — fusing {side_fused} "
                    "side view(s) into the occupancy grid."
                )

        self._update_corridor_progress_context(d_pos_w)

        if self.state != "COMPLETE":
            perception = getattr(self.env.unwrapped, "_perception", None)
            confirmed = (
                list(getattr(perception, "frame_confirmed_persons", None) or [])
                if perception is not None
                else []
            )
            thresh = self._yolo_rescue_conf_threshold()

            if confirmed:
                for det in confirmed:
                    conf = float(det.get("conf", 0.0))
                    if conf < thresh:
                        continue
                    xyz = det.get("world_xyz")
                    if xyz is None:
                        continue
                    p_pos_w = np.asarray(xyz, dtype=np.float64)

                    already_detected = False
                    for detected in self.rescued_people:
                        if np.linalg.norm(p_pos_w - detected) < 2.0:
                            already_detected = True
                            break

                    if not already_detected:
                        print(
                            f"\n[SLAM Brain] YOLO CONFIRMED ({conf:.0%}) NEW HUMAN AT WORLD: "
                            f"X:{p_pos_w[0]:.2f} Y:{p_pos_w[1]:.2f} Z:{p_pos_w[2]:.2f}"
                        )
                        self.rescued_people.append(p_pos_w.copy())
                        self.rescued_people_conf.append(conf)
            elif person_found[0].item():
                conf = self._current_yolo_confidence()
                if conf >= thresh:
                    p_pos_w = person_world_xyz[0].cpu().numpy()

                    already_detected = False
                    for detected in self.rescued_people:
                        if np.linalg.norm(p_pos_w - detected) < 2.0:
                            already_detected = True
                            break

                    if not already_detected:
                        print(
                            f"\n[SLAM Brain] YOLO CONFIRMED ({conf:.0%}) NEW HUMAN AT WORLD: "
                            f"X:{p_pos_w[0]:.2f} Y:{p_pos_w[1]:.2f} Z:{p_pos_w[2]:.2f}"
                        )
                        self.rescued_people.append(p_pos_w.copy())
                        self.rescued_people_conf.append(conf)

        if self.state not in ("EXPLORE", "COMPLETE"):
            self.state = "EXPLORE"

        # No SCAN mode: the front + left + right depth cameras already build the map
        # dynamically while the drone flies, so spinning in place is unnecessary.

        desired_pos_w = np.zeros(3, dtype=np.float32)
        target_yaw = drone_yaw
        cruise_z = 1.0

        if self.state == "EXPLORE":
            self.explore_step_count += 1
            # Record where the drone has physically been (pure-SLAM anti-backtrack).
            prev_stamp = getattr(self, "_prev_stamp_xy", None)
            if prev_stamp is not None:
                self._stamp_trajectory_step(
                    prev_stamp[0], prev_stamp[1], d_pos_w[0], d_pos_w[1]
                )
            self._stamp_visited(d_pos_w)
            self._prev_stamp_xy = np.array(d_pos_w[:2], dtype=np.float64)
            forced_route_mode = self._maybe_start_forced_corridor_route()
            if forced_route_mode:
                self._commit_forced_corridor_route(d_pos_w)
            if getattr(self, "astar_path_world", None):
                closest_idx = self._get_closest_path_index(d_pos_w)
                if closest_idx > 0:
                    self._stamp_path_visited(self.astar_path_world[: closest_idx + 1])

            if self.active_frontier is not None:
                self.active_frontier_ticks += 1

                # Do NOT invalidate the active target mid-corridor. Only drop it once
                # the drone has actually arrived, or is genuinely stuck trying to
                # reach it. (Early is_cell_frontier checks were clearing valid corridor
                # targets while the drone was still en route, causing mid-way switches
                # to side rooms.)
            if self.active_frontier is not None:
                # Fast stuck detector: if the drone has barely moved for a while
                # while chasing this frontier, it's wedged against a wall trying to
                # reach a dead-end target. Blacklist it NOW instead of ramming for
                # the full 150-step timeout.
                cur_xy = np.array(d_pos_w[:2], dtype=np.float64)
                prog_pos = getattr(self, "_stuck_ref_pos", None)
                if prog_pos is None:
                    self._stuck_ref_pos = cur_xy
                    self._stuck_ticks = 0
                else:
                    if float(np.linalg.norm(cur_xy - prog_pos)) > 0.35:
                        self._stuck_ref_pos = cur_xy
                        self._stuck_ticks = 0
                    else:
                        self._stuck_ticks = int(getattr(self, "_stuck_ticks", 0)) + 1

                stuck = int(getattr(self, "_stuck_ticks", 0)) >= 90
                lock = int(getattr(self, "_frontier_lock_ticks", 0))
                dist_now = float(np.linalg.norm(
                    d_pos_w[:2] - np.array(self.active_frontier["centroid_world"])
                ))
                path_len = len(getattr(self, "astar_path_world", None) or [])
                blacklist_timeout = max(350, path_len * 4)
                # Never blacklist mid-corridor while the commitment lock is active, or
                # while still far from the goal (slow progress in a tight turn ≠ stuck).
                may_blacklist = (
                    not self._active_target_is_forced_corridor_route()
                    and lock <= 0
                    and (
                        self.active_frontier_ticks > blacklist_timeout
                        or (stuck and dist_now < 2.0)
                    )
                )
                if may_blacklist:
                    centroid = self.active_frontier["centroid_world"]
                    why = "STUCK (no progress)" if stuck else f"UNREACHABLE after {self.active_frontier_ticks} steps"
                    print(
                        f"\n[SLAM Brain] Active frontier at X:{centroid[0]:.2f} Y:{centroid[1]:.2f} "
                        f"is {why}. Blacklisting it to prevent loop."
                    )
                    self.blacklisted_frontiers.append(centroid)
                    self.active_frontier = None
                    self.astar_path_world = []
                    self.active_frontier_ticks = 0
                    self._stuck_ref_pos = None
                    self._stuck_ticks = 0
                    self.explore_step_count = 50  # force immediate replanning in this frame
            else:
                self.active_frontier_ticks = 0
                self._stuck_ref_pos = None
                self._stuck_ticks = 0

            dist_to_f = (
                float(np.linalg.norm(d_pos_w[:2] - np.array(self.active_frontier["centroid_world"])))
                if self.active_frontier is not None
                else float("inf")
            )

            if self.active_frontier is not None and not self._active_target_is_forced_corridor_route():
                # If the goal is no longer a frontier (fully mapped) or is occupied (blocked by wall), clear it
                # immediately to prevent the drone from flying into closed/mapped walls.
                goal = self.active_frontier.get("goal_grid")
                if goal is not None:
                    prob = self.mapper.get_occupancy_grid()
                    r, c = int(goal[0]), int(goal[1])
                    is_occupied = False
                    if self.mapper.is_in_bounds(r, c):
                        is_occupied = (prob[r, c] > 0.65)
                    
                    # Dropped only if:
                    # The goal cell is occupied (blocked by wall) -> drop immediately to avoid crash.
                    if is_occupied:
                        why = "occupied (wall/obstacle)"
                        print(
                            f"[SLAM Brain] Active target {self.active_frontier['centroid_world']} "
                            f"is {why}. Clearing it."
                        )
                        self.active_frontier = None
                        self.astar_path_world = []
                        self.active_frontier_ticks = 0
                        self._frontier_lock_ticks = 0
                        self.explore_step_count = 50  # force replan

            if self.active_frontier is not None and not self._active_target_is_forced_corridor_route():
                # Check if the active path is blocked by a newly mapped wall/obstacle
                path_blocked = False
                if self.astar_path_world:
                    prob = self.mapper.get_occupancy_grid()
                    wall_mask, obstacle_mask = self.mapper.get_wall_obstacle_masks(use_walkable=False)
                    closest_idx = self._get_closest_path_index(d_pos_w)
                    for node in self.astar_path_world[closest_idx : closest_idx + 15]:
                        r, c = self.mapper.world_to_grid(node[0], node[1])
                        if (
                            self.mapper.is_in_bounds(r, c)
                            and (
                                wall_mask[r, c] == 1
                                or obstacle_mask[r, c] == 1
                                or prob[r, c] > 0.65
                            )
                        ):
                            path_blocked = True
                            break
                if path_blocked:
                    print(
                        f"[SLAM Brain] Active path to {self.active_frontier['centroid_world']} "
                        f"is blocked by a mapped wall/obstacle. Clearing path to force immediate replan."
                    )
                    self.astar_path_world = []
                    self.explore_step_count = 80  # force immediate replan

            if self.active_frontier is not None:
                lock = int(getattr(self, "_frontier_lock_ticks", 0))
                if lock > 0:
                    self._frontier_lock_ticks = lock - 1
                    lock -= 1

                # While chasing a corridor, push the goal deeper as the map grows.
                if (
                    self._is_live_opening(self.active_frontier)
                    and self.explore_step_count % 10 == 0
                ):
                    sr, sc = self.mapper.world_to_grid(d_pos_w[0], d_pos_w[1])
                    _, cf_extend = self.mapper.find_reachable_frontiers(
                        sr, sc, min_size=3
                    )
                    self._try_extend_active_goal(d_pos_w, cf_extend)

                # Only clear on true arrival, or when hard-stuck far from goal AFTER
                # the commitment lock expires — never mid-corridor because a nearer
                # frontier appeared elsewhere.
                is_close = self._has_arrived_at_frontier(
                    d_pos_w, self.active_frontier, dist_to_f
                )
                is_stuck_far = (
                    lock <= 0
                    and int(getattr(self, "_stuck_ticks", 0)) >= 90
                    and dist_to_f > 2.0
                )
                if is_close or is_stuck_far:
                    if is_close:
                        sr, sc = self.mapper.world_to_grid(d_pos_w[0], d_pos_w[1])
                        if not self._active_target_is_forced_corridor_route():
                            _, cf_arrive = self.mapper.find_reachable_frontiers(
                                sr, sc, min_size=3
                            )
                            if self._try_extend_active_goal(d_pos_w, cf_arrive):
                                is_close = False
                    if is_close or is_stuck_far:
                        reason = "arrived" if is_close else "hard-stuck (no progress)"
                        print(
                            f"[SLAM Brain] Cleared frontier at {dist_to_f:.2f}m ({reason})."
                        )
                        c = self.active_frontier["centroid_world"]
                        corridor_arrival = (
                            reason == "arrived"
                            and self._is_forward_corridor_frontier(
                                self.active_frontier, d_pos_w
                            )
                        )
                        if (
                            not corridor_arrival
                            and (
                                reason == "arrived"
                                or self._goal_region_mostly_visited(
                                    self.active_frontier
                                )
                            )
                        ):
                            self.blacklisted_frontiers.append(
                                np.array(c, dtype=np.float64)
                            )
                            self.visited_frontier_centroids.append(
                                np.array(c, dtype=np.float64)
                            )
                        elif corridor_arrival:
                            print(
                                "[SLAM Brain] Corridor arrival kept open; not marking "
                                "nearby forward frontiers as completed."
                            )
                        self.active_frontier = None
                        self.astar_path_world = []
                        self.active_frontier_ticks = 0
                        self._frontier_lock_ticks = 0
                    # Fall through to pick the NEXT unvisited frontier (not a revisit).
                    pass

            need_target = self.active_frontier is None
            # Refresh path to the SAME goal only — never re-pick a different target.
            periodic_replan = (
                self.active_frontier is not None and self.explore_step_count >= 80
            )

            if need_target and bool(getattr(self, "_forced_corridor_route_active", False)):
                self.explore_step_count = 0
                self.active_frontier_ticks = 0
                if self._commit_forced_corridor_route(d_pos_w):
                    need_target = False

            if need_target:
                self.explore_step_count = 0
                self.active_frontier_ticks = 0
                # ROBUST PURE-SLAM FRONTIER SEARCH (Yamauchi frontier exploration):
                # a single BFS over the drone's own observed-free space returns only
                # frontiers it actually walked to, each with a guaranteed path. This
                # is the fix for "clear opening ahead but no target": the old chain
                # (detect -> reachability -> A*) could reject a genuinely reachable
                # opening at any stage. No USD/ground-truth map is ever consulted.
                start_r, start_c = self.mapper.world_to_grid(d_pos_w[0], d_pos_w[1])
                bfs_frontiers, came_from = self.mapper.find_reachable_frontiers(
                    start_r, start_c, min_size=1
                )

                def _not_blacklisted(f):
                    return not any(
                        np.linalg.norm(np.array(f["centroid_world"]) - np.array(b)) < 1.5
                        for b in self.blacklisted_frontiers
                    )

                def _not_visited_centroid(f):
                    c = np.array(f["centroid_world"])
                    revisit_radius = (
                        0.75 if self._is_forward_corridor_frontier(f, d_pos_w) else 3.0
                    )
                    for vc in getattr(self, "visited_frontier_centroids", []):
                        if np.linalg.norm(c[:2] - vc[:2]) < revisit_radius:
                            return False
                    return True

                def _substantial(f):
                    fwd, _ = self._frontier_fwd_dot_heading(f, d_pos_w)
                    limit = 4 if fwd > 0.15 else self.MIN_UNKNOWN_GAIN
                    return int(f.get("unknown_gain", 0)) >= limit

                def _is_real_frontier(f):
                    goal = f.get("goal_grid")
                    if goal is None:
                        cw = f["centroid_world"]
                        goal = self.mapper.world_to_grid(cw[0], cw[1])
                    prob = self.mapper.get_occupancy_grid()
                    gr, gc = int(goal[0]), int(goal[1])
                    if not self.mapper.is_in_bounds(gr, gc) or prob[gr, gc] >= 0.35:
                        return False
                    if self.mapper.get_clearance_at_grid(gr, gc) < 0.10:
                        return False
                    return self.mapper.is_cell_frontier(goal, radius=1)

                candidates = [
                    f for f in bfs_frontiers
                    if np.linalg.norm(d_pos_w[:2] - np.array(f["centroid_world"])) > 0.35
                    and _not_blacklisted(f)
                    and _not_visited_centroid(f)
                    and _substantial(f)
                    and _is_real_frontier(f)
                    and not self._is_backtrack_target(
                        f, d_pos_w, came_from=came_from
                    )
                ]
                candidates = self._corridor_frontier_gate(candidates, d_pos_w)

                def _frontier_score(f):
                    """Yamauchi-style: lower = better (high unknown gain, short path)."""
                    dist = float(
                        np.linalg.norm(
                            np.array(f["centroid_world"]) - d_pos_w[:2]
                        )
                    )
                    gain = max(float(f.get("unknown_gain", 1)), 1.0)
                    score = dist / gain
                    if self._in_corridor_progress_context(d_pos_w):
                        dot, _ = self._frontier_progress_dot(f, d_pos_w)
                        unknown = float(self._unknown_ahead(f))
                        if dot < -0.10:
                            score *= 100.0
                        elif dot <= 0.55 and unknown >= self.MIN_UNKNOWN_GAIN:
                            # Prefer newly opened side branches at corridor junctions.
                            score *= 0.35
                        elif dot > 0.55:
                            score *= 0.75
                    return score

                def _commit(frontier, label, ignore_heading_backtrack=False, ignore_substantial=False):
                    frontier = self._prepare_commit_frontier(
                        frontier, came_from, (start_r, start_c)
                    )
                    goal = frontier["goal_grid"]
                    prob = self.mapper.get_occupancy_grid()
                    gr, gc = int(goal[0]), int(goal[1])
                    if not self.mapper.is_in_bounds(gr, gc) or prob[gr, gc] >= 0.35:
                        return False
                    if self.mapper.get_clearance_at_grid(gr, gc) < 0.10:
                        return False
                    if not self.mapper.is_cell_frontier(goal, radius=1):
                        return False
                    if not ignore_substantial and not _substantial(frontier):
                        return False
                    
                    # Strictly prevent backtracking to previous rooms/segments at all times.
                    if self._is_backtrack_target(frontier, d_pos_w, came_from=came_from, ignore_heading_backtrack=True):
                        return False

                    if not ignore_heading_backtrack and not self.is_explorable_frontier(
                        frontier, d_pos_w, came_from=came_from
                    ):
                        return False
                    if (
                        not self._is_forward_corridor_frontier(frontier, d_pos_w)
                        and not ignore_heading_backtrack
                        and (
                        self._path_ends_in_visited(came_from, goal)
                        and self._unknown_ahead(frontier) < 30
                        )
                    ):
                        return False
                    is_back = self._is_backtrack_target(
                        frontier, d_pos_w, came_from=came_from
                    )
                    REVISIT_MAX = 0.40 if not ignore_heading_backtrack else 1.0
                    world_path = self.mapper.plan_path_centered(
                        (start_r, start_c), goal
                    )
                    if world_path and len(world_path) >= 2:
                        prev = (float(d_pos_w[0]), float(d_pos_w[1]))
                        for node in world_path:
                            if self.mapper.segment_hits_wall(prev[0], prev[1], node[0], node[1]):
                                world_path = None
                                break
                            prev = node
                    if is_back and world_path and self._path_visited_fraction(
                        world_path
                    ) > REVISIT_MAX:
                        world_path = None

                    if world_path is None or len(world_path) < 2:
                        grid_path = self.mapper.reconstruct_path(came_from, goal)
                        if is_back and grid_path and self._path_visited_fraction(
                            None, grid_path=grid_path
                        ) > REVISIT_MAX:
                            grid_path = None
                        if grid_path and len(grid_path) >= 2:
                            is_safe = True
                            prob = self.mapper.get_occupancy_grid()
                            wall_mask, obstacle_mask = self.mapper.get_wall_obstacle_masks(use_walkable=False)
                            for r, c in grid_path[:-1]:
                                if (
                                    not self.mapper.is_in_bounds(r, c)
                                    or wall_mask[r, c] == 1
                                    or obstacle_mask[r, c] == 1
                                    or prob[r, c] > 0.65
                                ):
                                    is_safe = False
                                    break
                            if is_safe:
                                world_path = [
                                    self.mapper.grid_to_world(r, c) for r, c in grid_path
                                ]

                    if world_path and len(world_path) >= 2:
                        self.active_frontier = frontier
                        self.astar_path_world = world_path
                        self._hold_log_ticks = 0
                        self._frontier_lock_ticks = max(
                            200, min(500, len(world_path) * 3)
                        )
                        print(
                            f"[SLAM Brain] {label}: {frontier['centroid_world']} "
                            f"(gain {frontier.get('unknown_gain', 0)}, {len(world_path)} waypoints)"
                        )
                        return True
                    return False

                committed = False
                assist_allowed = self._should_use_mission_assist(d_pos_w)
                assist_required = bool(getattr(self, "_mission_assist_active", False)) or assist_allowed
                if assist_allowed:
                    committed = self._commit_mission_assist(
                        d_pos_w, came_from, (start_r, start_c), reason="priority"
                    )
                if assist_required and not committed:
                    self._hold_log_ticks = int(getattr(self, "_hold_log_ticks", 0)) + 1
                    if self._hold_log_ticks % 30 == 1:
                        print(
                            "[SLAM Brain] Mission assist active but next route cell "
                            "is not reachable yet. Holding / mapping instead of "
                            "falling back to old-room frontiers."
                        )
                    self.active_frontier = None
                    self.astar_path_world = []
                if not committed and not assist_required:
                    for f in sorted(candidates, key=_frontier_score):
                        if _commit(f, "Target frontier"):
                            committed = True
                            break

                # Last resort: ray-march toward high-gain frontiers path planning missed.
                if not committed and not assist_required and candidates:
                    for probe in self._discover_opening_probes(
                        d_pos_w, came_from, candidates
                    ):
                        if _commit(probe, "Target frontier (corridor probe)"):
                            committed = True
                            break

                if not committed and not assist_required and self.blacklisted_frontiers:
                    # If we couldn't commit to any candidate, but we have blacklisted frontiers,
                    # clear the blacklist and retry selection once. This prevents the drone
                    # from getting permanently locked out of corridors after a crash reset.
                    print("[SLAM Brain] No candidates found but blacklist is active. Clearing blacklist to retry.")
                    self.blacklisted_frontiers = []
                    candidates = [
                        f for f in bfs_frontiers
                        if np.linalg.norm(d_pos_w[:2] - np.array(f["centroid_world"])) > 0.35
                        and _not_visited_centroid(f)
                        and _substantial(f)
                        and _is_real_frontier(f)
                        and not self._is_backtrack_target(
                            f, d_pos_w, came_from=came_from
                        )
                    ]
                    candidates = self._corridor_frontier_gate(
                        candidates, d_pos_w, " blacklist recovery"
                    )
                    for f in sorted(candidates, key=_frontier_score):
                        if _commit(f, "Target frontier (blacklist recovery)"):
                            committed = True
                            break
                    if not committed and candidates:
                        for probe in self._discover_opening_probes(
                            d_pos_w, came_from, candidates
                        ):
                            if _commit(probe, "Target frontier (corridor probe recovery)"):
                                committed = True
                                break

                if not committed and not assist_required:
                    committed = self._commit_mission_assist(
                        d_pos_w, came_from, (start_r, start_c), reason="frontier fallback"
                    )

                if not committed and not assist_required:
                    # Third resort: allow backtrack targets (heading backtracking only) if no other choice exists
                    candidates = [
                        f for f in bfs_frontiers
                        if np.linalg.norm(d_pos_w[:2] - np.array(f["centroid_world"])) > 0.35
                        and _not_visited_centroid(f)
                        and _substantial(f)
                        and _is_real_frontier(f)
                    ]
                    candidates = self._corridor_frontier_gate(
                        candidates, d_pos_w, " backtrack recovery"
                    )
                    for f in sorted(candidates, key=_frontier_score):
                        if _commit(f, "Target frontier (backtrack recovery)", ignore_heading_backtrack=True):
                            committed = True
                            break

                if not committed and not assist_required:
                    # Final resort: allow any real reachable frontier regardless of gain or backtracking (heading backtracking only)
                    candidates = [
                        f for f in bfs_frontiers
                        if np.linalg.norm(d_pos_w[:2] - np.array(f["centroid_world"])) > 0.35
                        and _not_visited_centroid(f)
                        and _is_real_frontier(f)
                    ]
                    candidates = self._corridor_frontier_gate(
                        candidates, d_pos_w, " low-gain recovery"
                    )
                    for f in sorted(candidates, key=_frontier_score):
                        if _commit(f, "Target frontier (low-gain/backtrack recovery)", ignore_heading_backtrack=True, ignore_substantial=True):
                            committed = True
                            break

                if not committed and not assist_required:
                    visited, total = self.coverage_stats()
                    coverage_pct = visited / max(total, 1) * 100.0

                    has_substantial_frontiers = any(
                        int(f.get("unknown_gain", 0)) >= self.MIN_UNKNOWN_GAIN
                        for f in candidates
                    )

                    if total >= 10000 and coverage_pct >= 95.0 and not has_substantial_frontiers:
                        spawned_active = bool(
                            getattr(self.env, "dynamic_spawn_active", False)
                        )
                        all_spawned_found = True
                        if spawned_active and hasattr(
                            self.env, "count_spawned_targets_detected"
                        ):
                            det, tot = self.env.count_spawned_targets_detected()
                            all_spawned_found = tot > 0 and det >= tot
                        if not spawned_active or all_spawned_found:
                            print(
                                f"[SLAM Brain] All frontiers cleared. "
                                f"Exploration COMPLETE ({visited}/{total} cells = {coverage_pct:.1f}%)."
                            )
                            self.state = "COMPLETE"
                            self.mission_finished = True
                        elif self._hold_log_ticks % 30 == 1:
                            det, tot = self.env.count_spawned_targets_detected()
                            print(
                                f"[SLAM Brain] Coverage {coverage_pct:.1f}% but waiting for "
                                f"spawned targets ({det}/{tot} detected by YOLO)."
                            )
                    else:
                        self._hold_log_ticks = int(getattr(self, "_hold_log_ticks", 0)) + 1
                        if self._hold_log_ticks % 30 == 1:
                            n_bfs = len(bfs_frontiers)
                            print(
                                f"[SLAM Brain] No routable frontier ({n_bfs} detected), "
                                f"coverage {coverage_pct:.1f}%. Holding / mapping."
                            )
                        self.active_frontier = None
                        self.astar_path_world = []

            elif periodic_replan:
                self.explore_step_count = 0
                start_r, start_c = self.mapper.world_to_grid(d_pos_w[0], d_pos_w[1])
                goal = self.active_frontier.get("goal_grid")
                if goal is None:
                    gr, gc = self.mapper.world_to_grid(
                        self.active_frontier["centroid_world"][0],
                        self.active_frontier["centroid_world"][1],
                    )
                    goal = (gr, gc)
                _, came_from = self.mapper.find_reachable_frontiers(
                    start_r, start_c, min_size=1
                )
                world_path = self.mapper.plan_path_centered((start_r, start_c), goal)
                if world_path is None or len(world_path) < 2:
                    grid_path = self.mapper.reconstruct_path(came_from, goal)
                    if grid_path and len(grid_path) >= 2:
                        is_safe = True
                        prob = self.mapper.get_occupancy_grid()
                        wall_mask, obstacle_mask = self.mapper.get_wall_obstacle_masks(use_walkable=False)
                        for r, c in grid_path[:-1]:
                            if (
                                not self.mapper.is_in_bounds(r, c)
                                or wall_mask[r, c] == 1
                                or obstacle_mask[r, c] == 1
                                or prob[r, c] > 0.65
                            ):
                                is_safe = False
                                break
                        if is_safe:
                            world_path = [
                                self.mapper.grid_to_world(r, c) for r, c in grid_path
                            ]
                REVISIT_MAX = 0.40
                is_back = self._is_backtrack_target(
                    self.active_frontier, d_pos_w, came_from=came_from
                )
                path_ok = (
                    world_path
                    and len(world_path) >= 2
                    and (
                        not is_back
                        or self._path_visited_fraction(world_path) <= REVISIT_MAX
                    )
                )
                if path_ok:
                    self.astar_path_world = world_path
                else:
                    why = "stale backtrack" if is_back else "no safe path"
                    if self._active_target_is_mission_assist():
                        if self._commit_mission_assist(
                            d_pos_w, came_from, (start_r, start_c), reason="repath"
                        ):
                            pass
                        else:
                            if self._hold_log_ticks % 30 == 1:
                                print(
                                    f"[SLAM Brain] Mission assist replan failed ({why}); "
                                    "holding instead of falling back to old-room targets."
                                )
                            self.astar_path_world = []
                    else:
                        print(
                            f"[SLAM Brain] Replan dropped active frontier ({why}). "
                            f"Clearing target to force replan."
                        )
                        self.active_frontier = None
                        self.astar_path_world = []

            if self.astar_path_world:
                d_pos_2d = d_pos_w[:2]
                closest_idx = self._get_closest_path_index(d_pos_w)
                self.current_wp_idx = closest_idx
                self.waypoints = self.astar_path_world

                # Look ahead along the path, but make sure we do not beeline through a wall corner!
                next_target = self.astar_path_world[closest_idx]
                for node in self.astar_path_world[closest_idx:]:
                    if (
                        not self._active_target_is_forced_corridor_route()
                        and self.mapper.segment_hits_wall(d_pos_w[0], d_pos_w[1], node[0], node[1])
                    ):
                        break
                    next_target = node
                    if np.linalg.norm(d_pos_2d - np.array(node)) > 0.6:
                        break

                desired_pos_w[0] = float(next_target[0])
                desired_pos_w[1] = float(next_target[1])
                desired_pos_w[2] = cruise_z
                target_yaw = math.atan2(
                    desired_pos_w[1] - d_pos_w[1], desired_pos_w[0] - d_pos_w[0]
                )
            else:
                desired_pos_w[:] = d_pos_w
                self.current_wp_idx = 0
                self.waypoints = []
                if need_target and self.state != "COMPLETE":
                    self.active_frontier = None
                    # In corridors, do not slowly spin until side cameras discover rear targets.
                    target_yaw = drone_yaw if self._is_corridor_context(d_pos_w) else drone_yaw + 0.15

        elif self.state == "COMPLETE":
            desired_pos_w[:] = d_pos_w

        desired_pos_w_tensor = torch.tensor(desired_pos_w, device=drone_pos.device).repeat(self.env.num_envs, 1)
        target_yaw_tensor = torch.tensor(target_yaw, device=drone_pos.device).repeat(self.env.num_envs)
        return desired_pos_w_tensor, target_yaw_tensor


class RealSlamDroneEnv(BrainNavDroneEnv):
    def __init__(self, cfg, **kwargs):
        self._allow_obstacle_randomization = True
        cfg.brain_real_slam_mode = True
        cfg.brain_slam_side_map_cameras = True
        # Mission assist is only a late-stage rescue rail for the room-4
        # corridor sequence: long corridor -> side corridor -> final room.
        cfg.brain_single_corridor_to_final = False
        cfg.brain_room4_corr1_waypoint = (0.0, -20.5, 1.0)
        cfg.brain_room4_corr2_waypoint = (-3.2, -20.5, 1.0)
        cfg.brain_final_room_waypoint = (-6.0, -21.5, 1.0)
        repo_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../../../../../../")
        )
        cfg.room_usd_path = os.path.join(
            repo_root, "assets", "rooms", "final_no_obstacles.usd"
        )
        # Keep the base USD obstacle-free; live props are spawned as kinematic
        # objects and randomized once per run.
        cfg.num_room3_walls = 4
        cfg.num_room3_cones = 3
        cfg.num_room3_big_gates = 1
        cfg.num_room3_small_gates = 2
        cfg.num_room3_poles_triangles = 2
        cfg.brain_slam_room3_max_obstacles = 6
        cfg.num_room4_corr1 = 5
        cfg.num_room4_corr2 = 5
        super().__init__(cfg, **kwargs)

        self._brain = SlamBrainModule(self)
        self.mapper = self._brain.mapper

        self.slam_state = self._brain.state
        self.active_frontier = None
        self.astar_path_world = []

    def step(self, action):
        obs, rewards, terminated, truncated, info = super().step(action)

        self.slam_state = self._brain.state
        self.active_frontier = self._brain.active_frontier
        self.astar_path_world = self._brain.astar_path_world

        return obs, rewards, terminated, truncated, info

    def _capture_brain_mission(self):
        if not hasattr(self, "_brain") or getattr(self._brain, "visited_mask", None) is None:
            return None
        # Always capture and preserve the SLAM mission progress on crash reset
        snap = self._brain.capture_mission_snapshot()
        if snap is not None:
            pos = self._robot.data.root_pos_w[0] - self._terrain.env_origins[0]
            snap["crash_local_xyz"] = (
                float(pos[0].item()),
                float(pos[1].item()),
                max(1.0, float(pos[2].item())),
            )
        return snap

    def _sample_brain_spawn_xyz(self, env_count, crash_local=None, force_checkpoint=False):
        device = self.device
        debug_start = getattr(self.cfg, "brain_debug_start_local", None)
        if debug_start is not None and crash_local is None and not force_checkpoint:
            sx, sy, sz = (
                float(debug_start[0]),
                float(debug_start[1]),
                float(debug_start[2] if len(debug_start) > 2 else 1.0),
            )
            seq = getattr(self.cfg, "brain_spawn_sequence", None)
            if seq and hasattr(self, "_brain"):
                debug_xy = np.array([sx, sy], dtype=np.float64)
                pts = [np.array(p[:2], dtype=np.float64) for p in seq]
                nearest_idx = int(np.argmin([float(np.linalg.norm(debug_xy - p)) for p in pts]))
                self._brain.segment_idx = nearest_idx
                self._brain.max_segment_reached = max(
                    int(getattr(self._brain, "max_segment_reached", 0)), nearest_idx
                )
                self._brain.state = "EXPLORE"
                if nearest_idx + 1 < len(seq):
                    route_dir = np.array(seq[nearest_idx + 1][:2], dtype=np.float64) - debug_xy
                elif nearest_idx > 0:
                    route_dir = debug_xy - np.array(seq[nearest_idx - 1][:2], dtype=np.float64)
                else:
                    route_dir = np.zeros(2, dtype=np.float64)
                route_n = float(np.linalg.norm(route_dir))
                if route_n > 1e-6:
                    self._brain._travel_dir = route_dir / route_n
            label = getattr(self.cfg, "brain_debug_start_label", "debug start")
            if not getattr(self, "_debug_start_logged", False):
                print(
                    f"[SLAM Environment] Debug start ({label}): "
                    f"({sx:.2f}, {sy:.2f}, {sz:.2f})"
                )
                self._debug_start_logged = True
            spawn_x = torch.full((env_count,), sx, device=device)
            spawn_y = torch.full((env_count,), sy, device=device)
            spawn_z = torch.full((env_count,), sz, device=device)
            return spawn_x, spawn_y, spawn_z
        # Respawn at the LAST checkpoint the drone actually passed — never one ahead
        # of the crash. Nearest-by-distance could snap forward (e.g. crash in the
        # first corridor → respawn at the second corridor), which skips progress and
        # feels like cheating. We project the crash onto the route polyline and pick
        # the most recent checkpoint behind it.
        seq = getattr(self.cfg, "brain_spawn_sequence", None)
        if seq and len(seq) > 0 and crash_local is not None:
            pts = [np.array(p[:2], dtype=np.float64) for p in seq]
            crash_xy = np.array(crash_local[:2], dtype=np.float64)

            # Cumulative arc-length at each checkpoint.
            cum = [0.0]
            for i in range(1, len(pts)):
                cum.append(cum[-1] + float(np.linalg.norm(pts[i] - pts[i - 1])))

            # Project crash onto each segment; find its arc-length along the route.
            best_seg, best_d2, best_arc = 0, float("inf"), 0.0
            for i in range(len(pts) - 1):
                a, b = pts[i], pts[i + 1]
                ab = b - a
                L2 = float(ab @ ab)
                t = 0.0 if L2 < 1e-9 else float(np.clip(((crash_xy - a) @ ab) / L2, 0.0, 1.0))
                proj = a + t * ab
                d2 = float(np.linalg.norm(crash_xy - proj) ** 2)
                if d2 < best_d2:
                    best_d2, best_seg = d2, i
                    best_arc = cum[i] + t * float(np.linalg.norm(ab))
            if len(pts) == 1:
                best_arc = 0.0

            # Last checkpoint at or behind the crash arc-length.
            spawn_idx = 0
            for i in range(len(pts)):
                if cum[i] <= best_arc + 1e-6:
                    spawn_idx = i
                else:
                    break

            if getattr(self, "max_segment_reached", 0) >= 3:
                spawn_idx = max(3, spawn_idx)

            sx, sy, sz = seq[spawn_idx]

            if hasattr(self, "_brain"):
                self._brain.segment_idx = spawn_idx
                if bool(getattr(self._brain, "_forced_corridor_route_active", False)):
                    self._brain._forced_corridor_route_idx = (
                        self._brain._forced_corridor_route_index_from_local_xy((sx, sy))
                    )
                    self._brain._mission_assist_active = False
                    self._brain._mission_assist_idx = 0

            label = ""
            labels = getattr(self, "_brain_spawn_labels", None)
            if labels and spawn_idx < len(labels):
                label = f" ({labels[spawn_idx]})"
            print(
                f"[SLAM Environment] Crash recovery — Respawning at last checkpoint "
                f"passed {spawn_idx + 1}/{len(seq)}{label}: ({sx:.2f}, {sy:.2f}, {sz:.2f})"
            )

            spawn_x = torch.full((env_count,), sx, device=device)
            spawn_y = torch.full((env_count,), sy, device=device)
            spawn_z = torch.full((env_count,), sz, device=device)
            return spawn_x, spawn_y, spawn_z

        return super()._sample_brain_spawn_xyz(env_count, crash_local=crash_local, force_checkpoint=force_checkpoint)

    def _randomize_obstacles(self, env_ids: torch.Tensor):
        if not getattr(self, "_allow_obstacle_randomization", True):
            return
        super()._randomize_obstacles(env_ids)
        self._allow_obstacle_randomization = False

    def _reset_idx(self, env_ids: torch.Tensor | None = None):
        self._allow_obstacle_randomization = not bool(
            getattr(self, "_real_slam_obstacles_placed", False)
        )
        if hasattr(self, "_brain") and self._brain is not None:
            self._brain.blacklisted_frontiers = []
        super()._reset_idx(env_ids)

    def _obstacle_half_extents(self, obstacle) -> tuple[float, float]:
        """Return each obstacle's local XY half-extents (cached), from its USD geometry.

        A single center-to-center radius can't catch long/wide obstacles (walls,
        gates) — the drone clips their edge without tripping. We read the real
        geometry once so the collision test uses the true footprint.
        """
        cache = getattr(self, "_obs_half_cache", None)
        if cache is None:
            cache = {}
            self._obs_half_cache = cache
        key = obstacle.cfg.prim_path
        if key in cache:
            return cache[key]
        hx, hy = 0.20, 0.20  # safe fallback if geometry can't be read
        try:
            from pxr import Usd, UsdGeom
            path = key.replace("env_.*", "env_0")
            prim = self.sim.stage.GetPrimAtPath(path)
            if prim.IsValid():
                bbc = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
                rng = bbc.ComputeLocalBound(prim).ComputeAlignedRange()
                mn, mx = rng.GetMin(), rng.GetMax()
                hx = max(0.03, 0.5 * float(mx[0] - mn[0]))
                hy = max(0.03, 0.5 * float(mx[1] - mn[1]))
        except Exception:
            pass
        cache[key] = (hx, hy)
        return cache[key]

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated, truncated = super()._get_dones()

        # Analytical failsafe collision check for the map obstacles.
        # By default this only LOGS (does not terminate): the drone often clears a
        # pole cleanly and the physical contact sensor (in the parent _get_dones)
        # already catches real crashes. The extra OBB margin was terminating on
        # near-misses ("crossed the pole but still terminates"). Set
        # brain_slam_obb_terminate=True to make it disqualify again.
        obb_terminate = bool(getattr(self.cfg, "brain_slam_obb_terminate", False))
        # Only count genuine penetration: no drone-radius padding, plus a small
        # negative tolerance so the center must be clearly inside the footprint.
        pen_tol = 0.03

        hit_extra_obs = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        pos_local = self._robot.data.root_pos_w[:, :3] - self._terrain.env_origins

        for obstacle in self.scene.rigid_objects.values():
            obs_pos_w = obstacle.data.root_pos_w[:, :3] - self._terrain.env_origins
            active = obs_pos_w[:, 2] > -10.0
            if not bool(active.any()):
                continue

            hx, hy = self._obstacle_half_extents(obstacle)

            q = obstacle.data.root_quat_w
            qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
            yaw = torch.atan2(2.0 * (qw * qz + qx * qy),
                              1.0 - 2.0 * (qy * qy + qz * qz))
            cy, sy = torch.cos(yaw), torch.sin(yaw)

            rel_x = pos_local[:, 0] - obs_pos_w[:, 0]
            rel_y = pos_local[:, 1] - obs_pos_w[:, 1]
            local_x = cy * rel_x + sy * rel_y
            local_y = -sy * rel_x + cy * rel_y

            thr_x = torch.clamp(torch.as_tensor(hx - pen_tol), min=0.02)
            thr_y = torch.clamp(torch.as_tensor(hy - pen_tol), min=0.02)
            inside_xy = (local_x.abs() < thr_x) & (local_y.abs() < thr_y)
            z_overlap = (pos_local[:, 2] >= 0.2) & (pos_local[:, 2] <= 1.7)
            hit_extra_obs = hit_extra_obs | (active & z_overlap & inside_xy)

        if hit_extra_obs.any():
            now = int(getattr(self, "_timestep", 0))
            if now - int(getattr(self, "_last_obb_log_step", -1000)) > 50:
                self._last_obb_log_step = now
                verdict = "DISQUALIFIED" if obb_terminate else "logged only (loop)"
                print(f"[COLLISION] Analytical OBB overlap with a map obstacle — {verdict}.")
            if obb_terminate:
                terminated = terminated | hit_extra_obs

        return terminated, truncated
