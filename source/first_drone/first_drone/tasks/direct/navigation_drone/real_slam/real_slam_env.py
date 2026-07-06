import numpy as np
import torch
import cv2
import math
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

        # Precompute walkable mask for coverage statistics matching the house boundaries
        h, w = self.mapper.h, self.mapper.w
        r_indices, c_indices = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        
        # Convert all grid indices to world coordinates
        wx_grid = self.mapper.min_x + (c_indices + 0.5) * self.mapper.cell_size
        wy_grid = self.mapper.min_y + (r_indices + 0.5) * self.mapper.cell_size
        
        tx = torch.tensor(wx_grid.flatten(), device=env.device, dtype=torch.float32)
        ty = torch.tensor(wy_grid.flatten(), device=env.device, dtype=torch.float32)
        
        if hasattr(env, "_is_on_navigable_floor"):
            # Query the navigable floor and map obstacles with small margins to identify exactly what is in the rooms
            on_floor = env._is_on_navigable_floor(tx, ty, margin=0.1)
            if hasattr(env, "_is_inside_map_obstacle"):
                inside_obstacle = env._is_inside_map_obstacle(tx, ty, margin=0.05)
                walkable = on_floor & (~inside_obstacle)
            else:
                walkable = on_floor
            self.mapper.walkable_mask = walkable.cpu().numpy().reshape((h, w))
        else:
            self.mapper.walkable_mask = np.ones((h, w), dtype=bool)

        self.state = "EXPLORE"
        self.segment_idx = 0
        self.mission_finished = False

        self.active_frontier = None
        self.astar_path_world = []
        self.last_drone_yaw = None
        self.explore_step_count = 0
        self.last_scan_pos = None
        self.rescued_people = []
        self.blacklisted_frontiers = []
        self.active_frontier_ticks = 0
        # Room-entry 360 scan bookkeeping: each spawn-sequence checkpoint gets one
        # slow full-turn scan the first time the drone arrives, then it explores.
        self.scanned_rooms = set()
        # Forward-only progression: highest room-checkpoint index the drone has
        # reached. Frontiers in earlier rooms are rejected so the drone never
        # backtracks — it always pushes forward through the corridors.
        self.max_segment_reached = 0
        # Where the last discovery 360-scan happened (prevents scan loops).
        self._last_explore_scan_pos = None

    def reset_mission_from_start(self) -> None:
        """Keep room-1 spawn from sequential config but start in SLAM EXPLORE (not SCAN)."""
        super().reset_mission_from_start()
        self.state = "EXPLORE"
        self.segment_idx = 0
        self.active_frontier = None
        self.astar_path_world = []
        self.explore_step_count = 0
        self.last_scan_pos = None
        self.rescued_people = []
        self.blacklisted_frontiers = []
        self.active_frontier_ticks = 0
        self.scanned_rooms = set()
        self.max_segment_reached = 0

    def capture_mission_snapshot(self):
        # Decouple from parent sequential checks, always create a valid snapshot dictionary
        snap = {}
        snap["segment_idx"] = int(self.segment_idx)
        snap["max_segment_reached"] = int(getattr(self, "max_segment_reached", 0))
        snap["rescued_people"] = [p.copy() for p in self.rescued_people]
        snap["scanned_rooms"] = set(self.scanned_rooms)
        # Preserve the frontier blacklist across the crash, and blacklist the target
        # the drone was chasing when it crashed — otherwise it re-picks the same
        # unreachable frontier after respawn and crashes again (reset loop).
        blk = [list(b) for b in getattr(self, "blacklisted_frontiers", [])]
        af = getattr(self, "active_frontier", None)
        if af is not None and af.get("centroid_world") is not None:
            blk.append(list(af["centroid_world"]))
        snap["blacklisted_frontiers"] = blk
        if self.last_scan_pos is not None:
            snap["last_scan_pos"] = self.last_scan_pos.copy()
            
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
        self.scanned_rooms = set(snap.get("scanned_rooms", set()))
        if "last_scan_pos" in snap and snap["last_scan_pos"] is not None:
            self.last_scan_pos = snap["last_scan_pos"].copy()
        
        # Clear state/target so that on recovery we force immediate safe path planning
        self.state = "EXPLORE"
        self.active_frontier = None
        self.astar_path_world = []
        self.explore_step_count = 50  # force immediate path generation
        # Keep the blacklist (incl. the frontier we just crashed on) so recovery
        # doesn't re-select it and loop.
        self.blacklisted_frontiers = [np.array(b) for b in snap.get("blacklisted_frontiers", [])]
        self.active_frontier_ticks = 0
        self._stuck_ref_pos = None
        self._stuck_ticks = 0

    def coverage_stats(self) -> tuple[int, int]:
        """SLAM-only coverage: known cells / bounding box of mapped region."""
        return self.mapper.coverage_stats()

    def get_segment_label(self, idx=0):
        return f"SLAM {self.state}"

    def _is_world_xy_walkable(self, wx: float, wy: float) -> bool:
        """True if a world point falls inside the precomputed walkable house mask.

        Used to reject frontiers/targets that land outside the rooms (e.g. cells
        the depth rays leaked through a window/door gap into unreachable space).
        """
        mask = getattr(self.mapper, "walkable_mask", None)
        if mask is None:
            return True
        row, col = self.mapper.world_to_grid(wx, wy)
        if not self.mapper.is_in_bounds(row, col):
            return False
        return bool(mask[row, col])

    def _frontier_room_idx(self, centroid_world) -> int | None:
        """Nearest spawn-sequence checkpoint (room index) for a world-XY frontier."""
        seq = getattr(self.env.cfg, "brain_spawn_sequence", None)
        if not seq or len(seq) == 0:
            return None
        origin = getattr(self, "_env_origin_xy", np.zeros(2))
        cw_local = np.array(centroid_world[:2], dtype=np.float64) - np.array(origin[:2])
        dists = [np.linalg.norm(cw_local - np.array(pt[:2])) for pt in seq]
        return int(np.argmin(dists))

    def is_frontier_ahead(self, centroid_world) -> bool:
        """True if a frontier is in the current room or a later one (never behind).

        Enforces forward-only exploration: once the drone has advanced to a room,
        frontiers that belong to earlier rooms are rejected so it pushes through the
        corridors instead of turning around to re-map rooms it already passed.
        """
        idx = self._frontier_room_idx(centroid_world)
        if idx is None:
            return True
        return idx >= int(getattr(self, "max_segment_reached", 0))

    def _trigger_explore_scan(self, d_pos_w) -> bool:
        """Do a 360 SCAN in place to DISCOVER new frontiers with the drone's own
        cameras (front + sides) — the honest alternative to jumping to a USD map
        checkpoint. Returns True if a scan was started.

        Guarded so it can't loop forever: it only rescans after the drone has
        moved at least ~1.5 m from where the last exploration scan happened.
        """
        last = getattr(self, "_last_explore_scan_pos", None)
        if last is not None and float(np.linalg.norm(d_pos_w[:2] - last[:2])) < 1.5:
            return False  # already scanned near here; don't spin in place forever
        self._last_explore_scan_pos = np.array(d_pos_w, dtype=np.float64)
        self.state = "SCAN"
        self.active_frontier = None
        self.astar_path_world = []
        self.active_frontier_ticks = 0
        self.last_scan_pos = d_pos_w.copy()
        self.env._scan_step_count = 0
        self.env._scan_rot_accum = 0.0
        self.env._scan_prev_yaw = None
        print(
            "[SLAM Brain] No reachable frontier — 360 scan to DISCOVER openings "
            "with the drone's own cameras (no map checkpoints)."
        )
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

    def update(self, person_found, person_world_xyz, drone_pos, drone_quat):
        """SLAM-driven high-level brain update logic."""
        env_origin = self.env._terrain.env_origins[0].cpu().numpy()
        d_pos_w = drone_pos[0].cpu().numpy()
        d_quat = drone_quat[0].cpu().numpy()

        qw, qx, qy, qz = d_quat
        drone_yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))

        # Update segment_idx dynamically to match the current room of the drone
        seq = getattr(self.env.cfg, "brain_spawn_sequence", None)
        if seq and len(seq) > 0:
            drone_xy = d_pos_w[:2] - env_origin[:2]
            distances = [np.linalg.norm(drone_xy - np.array(pt[:2])) for pt in seq]
            self.segment_idx = int(np.argmin(distances))
            # Never let the "furthest room reached" go down → no backtracking.
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

        if person_found[0].item() and self.state != "COMPLETE":
            p_pos_w = person_world_xyz[0].cpu().numpy()
            
            # Check if this person is close to any already detected person
            already_detected = False
            for detected in self.rescued_people:
                if np.linalg.norm(p_pos_w - detected) < 2.0:
                    already_detected = True
                    break
                    
            if not already_detected:
                print(
                    f"\n[SLAM Brain] YOLO DETECTED NEW HUMAN AT WORLD: "
                    f"X:{p_pos_w[0]:.2f} Y:{p_pos_w[1]:.2f} Z:{p_pos_w[2]:.2f}"
                )
                self.rescued_people.append(p_pos_w.copy())

        if self.state not in ("EXPLORE", "SCAN", "COMPLETE"):
            self.state = "EXPLORE"

        # --- Room-entry 360 scan ---------------------------------------------
        # The first time the drone reaches each room checkpoint, do one slow full
        # turn so the front + side cameras sweep the whole room into the map before
        # picking the next frontier. This makes exploration faster overall (fewer
        # frontier hops) and gives the SLAM map a clean 360 view of each room.
        if (
            self.state == "EXPLORE"
            and getattr(self.env.cfg, "brain_room_entry_scan", True)
            and seq and len(seq) > 0
        ):
            trigger_r = float(getattr(self.env.cfg, "brain_scan_trigger_radius", 2.0))
            drone_xy_local = d_pos_w[:2] - env_origin[:2]
            labels = (
                getattr(self.env, "_brain_spawn_labels", None)
                or getattr(self.env.cfg, "brain_spawn_labels", None)
                or []
            )
            for idx, pt in enumerate(seq):
                if idx in self.scanned_rooms:
                    continue
                # Only 360-scan in actual ROOMS, never in corridors — the corridor
                # checkpoints ("corridor pass"/"corr...") are transit points, and a
                # mid-corridor spin wastes time and confuses the map.
                label = str(labels[idx]).lower() if idx < len(labels) else ""
                if "corr" in label:
                    self.scanned_rooms.add(idx)  # mark done so it never triggers here
                    continue
                if np.linalg.norm(drone_xy_local - np.array(pt[:2])) < trigger_r:
                    self.scanned_rooms.add(idx)
                    self.state = "SCAN"
                    self.active_frontier = None
                    self.astar_path_world = []
                    self.active_frontier_ticks = 0
                    self.last_scan_pos = d_pos_w.copy()
                    self.env._scan_step_count = 0
                    self.env._scan_rot_accum = 0.0
                    self.env._scan_prev_yaw = None
                    print(
                        f"[SLAM Brain] Entered room checkpoint {idx + 1}/{len(seq)} — "
                        f"starting slow 360 scan to map the room."
                    )
                    break

        desired_pos_w = np.zeros(3, dtype=np.float32)
        target_yaw = drone_yaw
        cruise_z = 1.0

        if self.state == "EXPLORE":
            self.explore_step_count += 1

            if self.active_frontier is not None:
                self.active_frontier_ticks += 1

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

                stuck = int(getattr(self, "_stuck_ticks", 0)) >= 45
                # If we've been trying to reach the active frontier for more than 15 seconds (150 steps), blacklist it!
                if self.active_frontier_ticks > 150 or stuck:
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

            if self.active_frontier is not None:
                is_close = dist_to_f < 1.3
                is_near_and_blocked = (dist_to_f < 1.6 and self.active_frontier_ticks > 40)
                if is_close or is_near_and_blocked:
                    reason = "direct arrival" if is_close else "proximity timeout (blocked/dead-end)"
                    print(f"[SLAM Brain] Cleared frontier at distance {dist_to_f:.2f}m via {reason}.")
                    self.active_frontier = None
                    self.active_frontier_ticks = 0
                    # Fall through to planning block to plan new path in this very frame!

            need_target = self.active_frontier is None
            # Replan every ~100 steps (was 150): keeps the A* path fresh as the map
            # fills in, so the drone re-routes around newly seen walls sooner and
            # cuts corners less on long legs.
            periodic_replan = (
                self.active_frontier is not None and self.explore_step_count >= 100
            )

            if need_target:
                self.explore_step_count = 0
                self.active_frontier_ticks = 0
                # min_size=12: small enough to catch narrow side-corridor mouths
                # (now that the side cameras map them) while the walkable filter
                # below rejects noise clusters outside the rooms.
                frontiers = self.mapper.detect_frontiers(min_size=12)

                # Reject frontiers using ONLY the SLAM map (no ground-truth/USD
                # "cheat"): a frontier is valid only if the drone can reach it
                # through observed free space. This drops "blue targets" that leaked
                # outside the rooms or got walled off by mapped obstacles, without
                # ever consulting the true map.
                start_r, start_c = self.mapper.world_to_grid(d_pos_w[0], d_pos_w[1])
                reachable_mask = self.mapper.compute_reachable_mask(start_r, start_c)
                frontiers = [
                    f for f in frontiers
                    if np.linalg.norm(d_pos_w[:2] - np.array(f["centroid_world"])) > 1.8
                    and self.mapper.is_frontier_reachable(reachable_mask, f["centroid_grid"])
                    and self.is_frontier_ahead(f["centroid_world"])
                    and not any(np.linalg.norm(np.array(f["centroid_world"]) - np.array(b)) < 1.5 for b in self.blacklisted_frontiers)
                ]

                if not frontiers:
                    visited, total = self.coverage_stats()
                    coverage_pct = visited / max(total, 1) * 100.0
                    
                    # Generic completion: all frontiers gone AND >70% of walkable area mapped
                    if coverage_pct > 70.0:
                        print(
                            f"[SLAM Brain] All frontiers cleared. "
                            f"Exploration COMPLETE ({visited}/{total} cells = {coverage_pct:.1f}%)."
                        )
                        self.state = "COMPLETE"
                        self.mission_finished = True
                    else:
                        # Coverage below threshold but no reachable frontier right now.
                        # DO NOT jump to a USD map checkpoint (that's cheating — it
                        # tells the drone where rooms are). Instead spin a 360 so the
                        # front + side cameras discover the next opening on their own.
                        # If we already scanned near here, hover and keep mapping.
                        if not self._trigger_explore_scan(d_pos_w):
                            print(
                                f"[SLAM Brain] No frontiers, coverage {coverage_pct:.1f}%. "
                                f"Already scanned here — hovering to map more."
                            )
                            self.active_frontier = None
                            self.astar_path_world = []
                else:
                    # Frontier scoring: standard nearest-first with information-gain weighting
                    # (Yamauchi 1997, "A Frontier-Based Approach for Autonomous Exploration")
                    # Score = distance - alpha * log(size)
                    # Larger frontiers (more information gain) are preferred over tiny noise clusters
                    heading = np.array([math.cos(drone_yaw), math.sin(drone_yaw)])

                    def _frontier_score(f):
                        cw   = np.array(f["centroid_world"])
                        to_f = cw - d_pos_w[:2]
                        dist = float(np.linalg.norm(to_f))
                        info_gain = float(np.log1p(f.get("size", 1)))
                        # Forward preference (drone's OWN heading, not USD): frontiers
                        # ahead score better; ones behind (dead-end pockets requiring a
                        # fly-back) are penalised. Keeps exploration pushing forward and
                        # lets the side cameras reveal branch corridors naturally.
                        fwd = 0.0
                        if dist > 1e-3:
                            fwd = float(np.dot(to_f / dist, heading))  # +1 ahead, -1 behind
                        back_penalty = 3.0 * max(0.0, -fwd)  # only penalise behind
                        return dist - 2.0 * info_gain + back_penalty

                    sorted_frontiers = sorted(frontiers, key=_frontier_score)
                    grid_path = None
                    selected_f = None
                    # Walls-only grid: A* routes through dodgeable prop fields (PPO
                    # weaves) but never through structural walls.
                    inflated = self.mapper.get_planning_grid()
                    start_r, start_c = self.mapper.world_to_grid(d_pos_w[0], d_pos_w[1])

                    for f in sorted_frontiers:
                        goal_r, goal_c = self.mapper.world_to_grid(
                            f["centroid_world"][0], f["centroid_world"][1]
                        )
                        inflated_temp = inflated.copy()
                        # Clear only a 3x3 region around start and the exact goal cell to prevent overlapping wall erasure
                        for r_off in range(-1, 2):
                            for c_off in range(-1, 2):
                                if self.mapper.is_in_bounds(start_r + r_off, start_c + c_off):
                                    inflated_temp[start_r + r_off, start_c + c_off] = 0
                        inflated_temp[goal_r, goal_c] = 0

                        path = plan_astar(inflated_temp, (start_r, start_c), (goal_r, goal_c))
                        if path is not None:
                            grid_path = path
                            selected_f = f
                            break

                    if grid_path is not None:
                        self.active_frontier = selected_f
                        self.astar_path_world = [
                            self.mapper.grid_to_world(r, c) for r, c in grid_path
                        ]
                        print(
                            f"[SLAM Brain] Selected new target frontier: "
                            f"{self.active_frontier['centroid_world']}"
                        )
                    elif sorted_frontiers:
                        # A* couldn't route to any frontier this frame. A straight
                        # line is only safe through OBSERVED-FREE space and only for
                        # a short hop — beelining across UNKNOWN space is what drove
                        # the drone through unmapped walls (crash) and let it chase
                        # far rooms it never really mapped. Require known-free line +
                        # a distance cap; otherwise hover and re-map.
                        STRAIGHT_LINE_MAX_M = 4.0
                        clear_f = None
                        for f in sorted_frontiers:
                            cw = f["centroid_world"]
                            dist = float(np.linalg.norm(d_pos_w[:2] - np.array(cw)))
                            if dist > STRAIGHT_LINE_MAX_M:
                                continue  # too far to trust without an A* route
                            if self.mapper.segment_is_known_free(
                                d_pos_w[0], d_pos_w[1], cw[0], cw[1]
                            ):
                                clear_f = f
                                break

                        if clear_f is not None:
                            self.active_frontier = clear_f
                            self.astar_path_world = [
                                [float(d_pos_w[0]), float(d_pos_w[1])],
                                [float(clear_f["centroid_world"][0]), float(clear_f["centroid_world"][1])],
                            ]
                            print(
                                "[SLAM Brain] No A* route yet — short straight-line through "
                                f"known-free space toward {clear_f['centroid_world']}"
                            )
                        else:
                            # No safe near frontier — hover and let the map fill in
                            # instead of guessing through unknown/blocked space.
                            print(
                                "[SLAM Brain] No A*-reachable frontier and no safe near "
                                "straight-line — hovering to re-map."
                            )
                            self.active_frontier = None
                            self.astar_path_world = []
                    else:
                        # No frontier this frame — scan to discover more (no USD
                        # checkpoints). Guard prevents spinning forever in one spot.
                        if not self._trigger_explore_scan(d_pos_w):
                            print("[SLAM Brain] No route and already scanned here — hovering to map.")
                            self.astar_path_world = []
                            self.active_frontier = None

            elif periodic_replan:
                self.explore_step_count = 0
                inflated = self.mapper.get_planning_grid()
                start_r, start_c = self.mapper.world_to_grid(d_pos_w[0], d_pos_w[1])
                goal_r, goal_c = self.mapper.world_to_grid(
                    self.active_frontier["centroid_world"][0],
                    self.active_frontier["centroid_world"][1],
                )
                inflated_temp = inflated.copy()
                # Clear only a 3x3 region around start and the exact goal cell to prevent overlapping wall erasure
                for r_off in range(-1, 2):
                    for c_off in range(-1, 2):
                        if self.mapper.is_in_bounds(start_r + r_off, start_c + c_off):
                            inflated_temp[start_r + r_off, start_c + c_off] = 0
                inflated_temp[goal_r, goal_c] = 0
                path = plan_astar(inflated_temp, (start_r, start_c), (goal_r, goal_c))
                if path is not None:
                    self.astar_path_world = [
                        self.mapper.grid_to_world(r, c) for r, c in path
                    ]

            if self.astar_path_world:
                # Find closest index on the A* path to the drone's current 2D position
                d_pos_2d = d_pos_w[:2]
                distances = [np.linalg.norm(d_pos_2d - np.array(node)) for node in self.astar_path_world]
                closest_idx = int(np.argmin(distances))
                
                # Look ahead ~0.6m along the path (was 1.0m): 1.0 was smooth but let
                # the drone aim too far ahead and clip inside corners; 0.6 hugs the
                # planned route through turns while staying smoother than the old 0.4.
                next_target = self.astar_path_world[-1]
                for node in self.astar_path_world[closest_idx:]:
                    if np.linalg.norm(d_pos_2d - np.array(node)) > 0.6:
                        next_target = node
                        break
                        
                desired_pos_w[0] = float(next_target[0])
                desired_pos_w[1] = float(next_target[1])
                desired_pos_w[2] = cruise_z
                target_yaw = math.atan2(
                    desired_pos_w[1] - d_pos_w[1], desired_pos_w[0] - d_pos_w[0]
                )
            else:
                desired_pos_w[:] = d_pos_w
                if need_target and self.state != "COMPLETE":
                    self.active_frontier = None

        elif self.state == "SCAN":
            desired_pos_w[:] = d_pos_w

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
        # Room 3: only four simple props at fixed corners (easier to read / navigate).
        cfg.num_room3_walls = 1
        cfg.num_room3_cones = 2
        cfg.num_room3_big_gates = 0
        cfg.num_room3_small_gates = 1
        cfg.num_room3_poles_triangles = 0
        cfg.brain_slam_room3_max_obstacles = 4
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

        if self.slam_state == "SCAN":
            # Exit on a completed full turn rather than a fixed step count. The old
            # 130-step cutoff only covered ~180 deg at the gentle scan yaw rate, so
            # the drone never saw the room behind it. Accumulate the actual yaw
            # travelled and stop once we've swept a full 360 (with a hard cap).
            qw = self._robot.data.root_quat_w[0]
            w, x, y, z = (float(qw[0]), float(qw[1]), float(qw[2]), float(qw[3]))
            cur_yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

            steps_in_scan = int(getattr(self, "_scan_step_count", 0))
            prev_yaw = getattr(self, "_scan_prev_yaw", None)
            if prev_yaw is None or steps_in_scan <= 1:
                self._scan_rot_accum = 0.0
            else:
                dyaw = (cur_yaw - prev_yaw + math.pi) % (2.0 * math.pi) - math.pi
                self._scan_rot_accum = getattr(self, "_scan_rot_accum", 0.0) + abs(dyaw)
            self._scan_prev_yaw = cur_yaw

            full_turn = 2.0 * math.pi
            if self._scan_rot_accum >= full_turn or steps_in_scan >= 500:
                done_deg = math.degrees(self._scan_rot_accum)
                print(
                    f"[SLAM Environment] 360 scan complete ({done_deg:.0f} deg swept). "
                    f"Returning to EXPLORE."
                )
                self._brain.active_frontier = None
                self._brain.astar_path_world = []
                self._brain.state = "EXPLORE"
                self._scan_step_count = 0
                self._scan_rot_accum = 0.0
                self._scan_prev_yaw = None

        return obs, rewards, terminated, truncated, info

    def _capture_brain_mission(self):
        if not hasattr(self, "_brain"):
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

            sx, sy, sz = seq[spawn_idx]

            if hasattr(self, "_brain"):
                self._brain.segment_idx = spawn_idx

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
        self._allow_obstacle_randomization = True
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
