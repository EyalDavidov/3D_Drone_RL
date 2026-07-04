import numpy as np
import torch
import math

class BrainModule:
    def __init__(self, env, step_size=6.0, safety_margin=0.8):
        """
        Initialize the high-level Brain.
        :param env: The gymnasium/IsaacLab environment instance.
        :param step_size: Distance between lawnmower passes (meters).
        :param safety_margin: Safety clearance from walls and obstacles (meters).
        """
        self.env = env
        self.device = env.unwrapped.device
        self.step_size = step_size
        self.safety_margin = safety_margin
        self.curriculum_min_distance, self.curriculum_max_distance = self._get_curriculum_goal_distance()
        self.waypoint_clearance = self._compute_waypoint_clearance()

        # States: "SCAN", "GOTO_WAYPOINT", "APPROACH_TARGET", "COMPLETE"
        self.state = "SCAN"
        
        # State variables
        self.current_wp_idx = 0
        self.target_person_pos = None
        self.scan_yaw_accumulated = 0.0
        self.last_drone_yaw = None
        self.found_person = False
        self.scan_lock_pos = None
        self._pending_scan_snap_pos = None
        self.person_noted_anywhere = False
        self._pending_rescue_pos = None
        self._arrival_dwell_steps = 0
        self.segment_idx = 0
        self.nav_target = None
        self.mission_finished = False
        self._scan_announced_segment = -1
        
        # Extract boundaries and obstacles dynamically from USD stage
        self.min_x, self.max_x, self.min_y, self.max_y, self.obstacles = self._extract_map_data()
        
        self._uses_sequential = self._uses_sequential_mission()
        if self._uses_sequential:
            self.waypoints = []
            self._init_coverage_grid()
            seq = self._get_spawn_sequence()
            print(
                f"[Brain] Sequential SLAM mission: {len(seq)} points "
                f"(scan at spawns 1–{len(seq) - 1}, finish at end)."
            )
        else:
            # Generate the search waypoints
            self.waypoints = self._generate_search_waypoints()
            print(
                f"[Brain] Generated {len(self.waypoints)} search waypoints "
                f"(clearance >= {self.waypoint_clearance:.2f}m, "
                f"goal step <= {self.curriculum_max_distance:.1f}m)."
            )

    def _get_curriculum_goal_distance(self):
        """Read the same target-distance range used by the active training curriculum."""
        # For play/demo scripts, we force Level 5 target distances
        # (9.0m to 12.0m) to match the final trained PPO policy model's distribution.
        if getattr(self.env.unwrapped, "is_brain_play", False):
            return (9.0, 12.0)

        env = self.env.unwrapped
        level = int(getattr(env, "curriculum_level", getattr(env.cfg, "initial_curriculum_level", 5)))
        distances = getattr(env, "curriculum_distances", None)
        if distances is not None and level in distances:
            return distances[level]

        # Keep this fallback in sync with AEPPODroneEnv's level-5 curriculum.
        return (9.0, 12.0)

    def _compute_waypoint_clearance(self):
        """Choose a conservative clearance that the trained policy can actually reach."""
        cfg = self.env.unwrapped.cfg
        return max(
            float(self.safety_margin),
            float(getattr(cfg, "spawn_obstacle_margin", 0.5)),
            float(getattr(cfg, "pillar_proximity_radius", 0.5)) + 0.35,
            float(getattr(cfg, "pillar_collision_radius", 0.25)) + 0.45,
        )

    def _uses_sequential_mission(self) -> bool:
        cfg = self.env.unwrapped.cfg
        return bool(getattr(cfg, "brain_use_sequential_spawns", False)) and len(
            getattr(cfg, "brain_spawn_sequence", ())
        ) >= 2

    def _get_spawn_sequence(self) -> list:
        return [tuple(p) for p in self.env.unwrapped.cfg.brain_spawn_sequence]

    def reset_mission_from_start(self) -> None:
        """Crash recovery: restart entire mission at spawn1."""
        self.state = "SCAN"
        self.segment_idx = 0
        self.current_wp_idx = 0
        self.found_person = False
        self.target_person_pos = None
        self._pending_rescue_pos = None
        self.scan_yaw_accumulated = 0.0
        self.scan_lock_pos = None
        self._pending_scan_snap_pos = None
        self._arrival_dwell_steps = 0
        self.last_drone_yaw = None
        self.nav_target = None
        self.mission_finished = False
        self._scan_announced_segment = -1
        self.person_noted_anywhere = False
        self._pending_rescue_pos = None
        if self._uses_sequential:
            self._init_coverage_grid()

    def _checkpoint_index_for_target(self, target, seq) -> int | None:
        """Return spawn-sequence index nearest to a navigation target."""
        if target is None:
            return None
        best_i, best_d = None, float("inf")
        for i, pt in enumerate(seq):
            d = math.hypot(float(pt[0]) - float(target[0]), float(pt[1]) - float(target[1]))
            if d < best_d:
                best_d, best_i = d, i
        return best_i if best_d < 1.5 else None

    def _arrival_radius_for_target(self, tgt_idx: int | None, seq: list) -> float:
        """Per-waypoint arrival radius — rooms/final need to be reached, not approximated."""
        cfg = self.env.unwrapped.cfg
        skip_scan = set(getattr(cfg, "brain_skip_scan_segment_indices", ()))
        if tgt_idx is None:
            return float(getattr(cfg, "brain_scan_arrival_radius", 1.0))
        if tgt_idx in skip_scan:
            return float(getattr(cfg, "brain_corridor_arrival_radius", 0.55))
        if tgt_idx == len(seq) - 1:
            return float(getattr(cfg, "brain_final_room_arrival_radius", 1.2))
        return float(getattr(cfg, "brain_scan_arrival_radius", 1.0))

    def _can_count_arrival(self, d_pos, tgt_idx: int, seq: list, arrive_r: float) -> bool:
        """Reject fake arrivals when the previous checkpoint is only centimetres away."""
        if tgt_idx is None or tgt_idx <= 0:
            return True
        prev = seq[tgt_idx - 1]
        curr = seq[tgt_idx]
        sep = math.hypot(float(curr[0]) - float(prev[0]), float(curr[1]) - float(prev[1]))
        if sep >= 1.5:
            return True
        dist_from_prev = math.hypot(d_pos[0] - float(prev[0]), d_pos[1] - float(prev[1]))
        dist_to_tgt = math.hypot(d_pos[0] - float(curr[0]), d_pos[1] - float(curr[1]))
        if dist_to_tgt >= arrive_r:
            return False
        # For tightly spaced pairs (room4→corr1), require meaningful travel from previous point.
        return dist_from_prev >= max(0.55, sep * 0.75)

    def get_segment_label(self, segment_idx: int | None = None) -> str:
        """Human-readable label for the current mission segment (room 4, corr1, etc.)."""
        idx = self.segment_idx if segment_idx is None else int(segment_idx)
        labels = getattr(self.env.unwrapped, "_brain_spawn_labels", None)
        if labels and 0 <= idx < len(labels):
            return labels[idx]
        return f"segment {idx + 1}"

    def _init_coverage_grid(self) -> None:
        """Occupancy grid for visited-zone tracking during sequential SLAM."""
        env = self.env.unwrapped
        cfg = env.cfg
        grid = getattr(env, "_walkable_grid", None)
        if grid is not None and not getattr(env, "_walkable_grid_unreliable", False):
            self._cov_origin = env._walkable_grid_origin
            self._cov_res = float(env._walkable_grid_res)
            self._explorable = grid.astype(np.uint8).copy()
            self._visited = np.zeros_like(self._explorable, dtype=np.uint8)
        else:
            res = float(getattr(cfg, "walkable_grid_resolution", 0.4))
            self._cov_origin = (self.min_x, self.min_y)
            self._cov_res = res
            nx = max(1, int(math.ceil((self.max_x - self.min_x) / res)))
            ny = max(1, int(math.ceil((self.max_y - self.min_y) / res)))
            self._explorable = np.ones((nx, ny), dtype=np.uint8)
            self._visited = np.zeros((nx, ny), dtype=np.uint8)

    def _mark_visited_at(self, x: float, y: float, radius_m: float | None = None) -> None:
        if not self._uses_sequential or not hasattr(self, "_visited"):
            return
        if radius_m is None:
            radius_m = float(getattr(self.env.unwrapped.cfg, "brain_coverage_mark_radius", 2.0))
        ox, oy = self._cov_origin
        res = self._cov_res
        cx = int((x - ox) / res)
        cy = int((y - oy) / res)
        r_cells = max(1, int(radius_m / res))
        nx, ny = self._visited.shape
        for ix in range(max(0, cx - r_cells), min(nx, cx + r_cells + 1)):
            for iy in range(max(0, cy - r_cells), min(ny, cy + r_cells + 1)):
                if (ix - cx) ** 2 + (iy - cy) ** 2 <= r_cells ** 2:
                    self._visited[ix, iy] = 1

    def coverage_stats(self) -> tuple[int, int]:
        """Return (visited_cells, explorable_cells)."""
        if not hasattr(self, "_visited"):
            return 0, 0
        explorable = int(self._explorable.sum())
        visited = int((self._visited & self._explorable).sum())
        return visited, explorable

    def _extract_map_data(self):
        """
        Dynamically calculate room boundaries and extract internal obstacles.
        Prioritizes map_bounds / map_obstacles in config, then env/USD fallbacks.
        """
        cfg = self.env.unwrapped.cfg

        # 0. Explicit map_bounds from config (final_flat.usd envelope)
        try:
            map_bounds = getattr(cfg, "map_bounds", None)
            if map_bounds is not None and len(map_bounds) == 4:
                min_x, max_x, min_y, max_y = map_bounds
                obstacles = []
                if hasattr(cfg, "map_obstacles") and len(cfg.map_obstacles) > 0:
                    room_wx, room_wy = max_x - min_x, max_y - min_y
                    for obs in cfg.map_obstacles:
                        ox0, ox1, oy0, oy1 = obs
                        sx, sy = ox1 - ox0, oy1 - oy0
                        if sx < 0.8 * room_wx and sy < 0.8 * room_wy:
                            obstacles.append(obs)
                else:
                    obstacles = self._extract_usd_internal_obstacles(min_x, max_x, min_y, max_y)
                print(f"[Brain] Using map_bounds from config:")
                print(f"  • Bounds -> X: [{min_x:.1f}, {max_x:.1f}] | Y: [{min_y:.1f}, {max_y:.1f}]")
                print(f"  • Internal obstacles: {len(obstacles)}")
                return min_x, max_x, min_y, max_y, obstacles
        except Exception as e:
            print(f"[Brain] Warning: Could not parse map_bounds ({e}).")

        # 1. Prioritize map_obstacles defined in environment config (representing actual active arena boundaries)
        try:
            if hasattr(self.env.unwrapped.cfg, "map_obstacles") and len(self.env.unwrapped.cfg.map_obstacles) > 0:
                map_obstacles = self.env.unwrapped.cfg.map_obstacles
                all_min_x = [obs[0] for obs in map_obstacles]
                all_max_x = [obs[1] for obs in map_obstacles]
                all_min_y = [obs[2] for obs in map_obstacles]
                all_max_y = [obs[3] for obs in map_obstacles]
                
                min_x = min(all_min_x)
                max_x = max(all_max_x)
                min_y = min(all_min_y)
                max_y = max(all_max_y)
                
                # Extract internal obstacles (not covering 80% of width/height)
                obstacles = []
                room_width_x = max_x - min_x
                room_width_y = max_y - min_y
                for obs in map_obstacles:
                    obs_min_x, obs_max_x, obs_min_y, obs_max_y = obs
                    size_x = obs_max_x - obs_min_x
                    size_y = obs_max_y - obs_min_y
                    if size_x < 0.8 * room_width_x and size_y < 0.8 * room_width_y:
                        obstacles.append((obs_min_x, obs_max_x, obs_min_y, obs_max_y))
                        
                print(f"[Brain] Dynamically analyzed map_obstacles config:")
                print(f"  • Estimated Bounds -> X: [{min_x:.1f}, {max_x:.1f}] | Y: [{min_y:.1f}, {max_y:.1f}]")
                print(f"  • Extracted {len(obstacles)} internal obstacles.")
                return min_x, max_x, min_y, max_y, obstacles
        except Exception as e:
            print(f"[Brain] Warning: Could not parse map_obstacles config ({e}). Trying walkable footprint.")

        # 2. Use env room bounds (+ optional walkable footprint for R-shaped maps)
        try:
            env = self.env.unwrapped
            if getattr(env, "_room_bounds_local", None) is not None:
                min_x, max_x, min_y, max_y, _ = env._room_bounds_local
                obstacles = self._extract_usd_internal_obstacles(min_x, max_x, min_y, max_y)
                label = "room bounds"
                if getattr(env, "_walkable_grid", None) is not None and not getattr(
                    env, "_walkable_grid_unreliable", False
                ):
                    label = "walkable floor footprint"
                print(f"[Brain] Using {label} from env:")
                print(f"  • Bounds -> X: [{min_x:.1f}, {max_x:.1f}] | Y: [{min_y:.1f}, {max_y:.1f}]")
                print(f"  • Extracted {len(obstacles)} internal obstacles.")
                return min_x, max_x, min_y, max_y, obstacles
        except Exception as e:
            print(f"[Brain] Warning: Could not read walkable footprint ({e}). Trying USD stage.")

        # 3. Fallback to USD stage parsing (full AABB — less accurate for R-shaped maps)
        try:
            from pxr import Usd, UsdGeom
            stage = self.env.unwrapped.sim.stage
            # Get environment 0 origin to compute local coordinates
            env_origin = self.env.unwrapped._terrain.env_origins[0].cpu().numpy()
            
            bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"])
            room_prim_path = "/World/envs/env_0/Room"
            room_prim = stage.GetPrimAtPath(room_prim_path)
            
            if room_prim.IsValid():
                # 1. Bounding box of the entire room to determine bounds
                bbox = bbox_cache.ComputeWorldBound(room_prim)
                r = bbox.GetRange()
                min_pt = r.GetMin()
                max_pt = r.GetMax()
                
                min_x = min_pt[0] - env_origin[0]
                max_x = max_pt[0] - env_origin[0]
                min_y = min_pt[1] - env_origin[1]
                max_y = max_pt[1] - env_origin[1]
                
                # 2. Bounding boxes of internal obstacles (skip floor/ceiling and outer boundaries)
                obstacles = []
                room_width_x = max_x - min_x
                room_width_y = max_y - min_y
                
                for prim in Usd.PrimRange(room_prim):
                    if prim.IsA(UsdGeom.Mesh):
                        prim_name = prim.GetPath().name.lower()
                        prim_path_str = prim.GetPath().pathString.lower()
                        
                        # Ignore outer walls, ceiling, floor, or drone/ground geometries
                        if any(w in prim_path_str or w in prim_name for w in ["ground", "floor", "ceiling", "drone", "terrain"]):
                            continue
                            
                        bbox_p = bbox_cache.ComputeWorldBound(prim)
                        rp = bbox_p.GetRange()
                        if rp.IsEmpty():
                            continue
                            
                        p_min = rp.GetMin() - env_origin
                        p_max = rp.GetMax() - env_origin
                        
                        size_x = p_max[0] - p_min[0]
                        size_y = p_max[1] - p_min[1]
                        
                        # If mesh covers more than 80% of room in either axis, it's likely outer walls/boundaries
                        if size_x > 0.8 * room_width_x or size_y > 0.8 * room_width_y:
                            continue
                            
                        obstacles.append((p_min[0], p_max[0], p_min[1], p_max[1]))
                        
                print(f"[Brain] Dynamically analyzed USD stage:")
                print(f"  • Room Bounds -> X: [{min_x:.1f}, {max_x:.1f}] | Y: [{min_y:.1f}, {max_y:.1f}]")
                print(f"  • Found {len(obstacles)} internal obstacles:")
                for idx, obs in enumerate(obstacles):
                    print(f"    - Obstacle {idx}: X: [{obs[0]:.2f}, {obs[1]:.2f}] | Y: [{obs[2]:.2f}, {obs[3]:.2f}]")
                    
                return min_x, max_x, min_y, max_y, obstacles
                
        except Exception as e:
            print(f"[Brain] Warning parsing USD stage ({e}).")

        # Hardcoded fallback
        print("[Brain] Falling back to default spacing-based bounds.")
        spacing = float(getattr(self.env.unwrapped.scene.cfg, "env_spacing", 6.0))
        return -spacing/2.0, spacing/2.0, -spacing/2.0, spacing/2.0, []

    def _extract_usd_internal_obstacles(self, min_x, max_x, min_y, max_y):
        """Extract internal obstacle boxes from USD meshes within the given room bounds."""
        from pxr import Usd, UsdGeom

        obstacles = []
        room_width_x = max_x - min_x
        room_width_y = max_y - min_y
        if room_width_x <= 0.0 or room_width_y <= 0.0:
            return obstacles

        stage = self.env.unwrapped.sim.stage
        env_origin = self.env.unwrapped._terrain.env_origins[0].cpu().numpy()
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"])
        room_prim = stage.GetPrimAtPath("/World/envs/env_0/Room")
        if not room_prim.IsValid():
            return obstacles

        skip_keywords = (
            "character", "person", "human", "worker", "reallusion", "cc_base",
            "hair", "body", "cloth", "ground", "floor", "ceiling", "drone", "terrain",
        )
        for prim in Usd.PrimRange(room_prim):
            if not prim.IsA(UsdGeom.Mesh):
                continue
            prim_name = prim.GetPath().name.lower()
            prim_path_str = prim.GetPath().pathString.lower()
            if any(w in prim_path_str or w in prim_name for w in skip_keywords):
                continue

            bbox_p = bbox_cache.ComputeWorldBound(prim)
            rp = bbox_p.GetRange()
            if rp.IsEmpty():
                continue

            p_min = rp.GetMin() - env_origin
            p_max = rp.GetMax() - env_origin
            size_x = p_max[0] - p_min[0]
            size_y = p_max[1] - p_min[1]
            if size_x > 0.8 * room_width_x or size_y > 0.8 * room_width_y:
                continue
            if p_max[2] < 0.5:
                continue
            obstacles.append((p_min[0], p_max[0], p_min[1], p_max[1]))
        return obstacles

    def _is_inside_obstacle(self, x, y, margin):
        """
        Check if a 2D coordinate is inside any static obstacle or outside room walls.
        """
        # 1. Wall check (room boundaries)
        if (x < self.min_x + margin) or (x > self.max_x - margin):
            return True
        if (y < self.min_y + margin) or (y > self.max_y - margin):
            return True
            
        # 2. Obstacles check
        for obs in self.obstacles:
            min_x, max_x, min_y, max_y = obs
            if (x >= min_x - margin) and (x <= max_x + margin) and \
               (y >= min_y - margin) and (y <= max_y + margin):
                return True
                
        # 3. Environment class built-in checks (if available)
        if hasattr(self.env.unwrapped, "_is_inside_map_obstacle"):
            # Call env helper directly (inputs as PyTorch tensors)
            x_t = torch.tensor([x], device=self.device)
            y_t = torch.tensor([y], device=self.device)
            if self.env.unwrapped._is_inside_map_obstacle(x_t, y_t, margin=margin).item():
                return True

        # 4. Walkable floor footprint (R-shaped maps — voids inside the AABB are not navigable)
        if hasattr(self.env.unwrapped, "_is_on_navigable_floor") and (
            getattr(self.env.unwrapped, "_floor_tris_xy", None)
            or (
                getattr(self.env.unwrapped, "_walkable_grid", None) is not None
                and not getattr(self.env.unwrapped, "_walkable_grid_unreliable", False)
            )
        ):
            x_t = torch.tensor([x], device=self.device)
            y_t = torch.tensor([y], device=self.device)
            if not self.env.unwrapped._is_on_navigable_floor(x_t, y_t, margin=margin).item():
                return True
                
        return False

    def _nearest_safe_point(self, x, y):
        """
        Move an unsafe candidate waypoint to the nearest nearby safe point.

        The lawnmower grid is intentionally coarse. If a grid point falls on a
        static obstacle or too close to a wall, search a small spiral around it
        and keep the closest reachable location instead of using the bad point.
        """
        if not self._is_inside_obstacle(x, y, self.waypoint_clearance):
            return (x, y, 1.0)

        search_radius = max(self.step_size * 0.75, self.waypoint_clearance * 2.0)
        radius_values = np.arange(0.25, search_radius + 0.25, 0.25)
        angle_values = np.linspace(0.0, 2.0 * math.pi, 32, endpoint=False)

        best = None
        best_dist = float("inf")
        for radius in radius_values:
            for angle in angle_values:
                candidate_x = x + radius * math.cos(angle)
                candidate_y = y + radius * math.sin(angle)
                if self._is_inside_obstacle(candidate_x, candidate_y, self.waypoint_clearance):
                    continue

                dist = math.hypot(candidate_x - x, candidate_y - y)
                if dist < best_dist:
                    best = (candidate_x, candidate_y, 1.0)
                    best_dist = dist

            if best is not None:
                return best

        return None

    def _resolve_rescue_target(self, p_pos: np.ndarray) -> tuple[np.ndarray | None, str | None]:
        """Validate a YOLO person position and return a safe approach point."""
        if not np.all(np.isfinite(p_pos)):
            return None, "non-finite position"

        reject_reason = None
        if hasattr(self.env.unwrapped, "_is_on_navigable_floor") and (
            getattr(self.env.unwrapped, "_floor_tris_xy", None)
            or (
                getattr(self.env.unwrapped, "_walkable_grid", None) is not None
                and not getattr(self.env.unwrapped, "_walkable_grid_unreliable", False)
            )
        ):
            on_floor = self.env.unwrapped._is_on_navigable_floor(
                torch.tensor([p_pos[0]], device=self.device),
                torch.tensor([p_pos[1]], device=self.device),
                margin=self.waypoint_clearance,
            ).item()
            if not on_floor:
                reject_reason = (
                    f"detected position X:{p_pos[0]:.2f} Y:{p_pos[1]:.2f} is outside walkable floor"
                )

        safe_target = None
        if reject_reason is None:
            safe_target = self._nearest_safe_point(float(p_pos[0]), float(p_pos[1]))
            if safe_target is None:
                reject_reason = (
                    f"no safe approach point near X:{p_pos[0]:.2f} Y:{p_pos[1]:.2f}"
                )

        if safe_target is None:
            return None, reject_reason

        safe_target = np.array(safe_target, dtype=float)
        if np.linalg.norm(safe_target[:2] - p_pos[:2]) > 0.05:
            print(
                f"[Brain] Target is too close to an obstacle. "
                f"Using safe approach point X:{safe_target[0]:.2f} Y:{safe_target[1]:.2f}."
            )
        return safe_target, None

    def _limit_goal_distance(self, goal_pos, drone_pos):
        """Keep each high-level target within the trained curriculum distance range."""
        goal_pos = np.array(goal_pos, dtype=float)
        delta_xy = goal_pos[:2] - drone_pos[:2]
        dist_xy = np.linalg.norm(delta_xy)
        if dist_xy <= self.curriculum_max_distance or dist_xy < 1e-6:
            return goal_pos

        limited = goal_pos.copy()
        limited[:2] = drone_pos[:2] + delta_xy / dist_xy * self.curriculum_max_distance
        safe_limited = self._nearest_safe_point(float(limited[0]), float(limited[1]))
        if safe_limited is not None:
            limited = np.array(safe_limited, dtype=float)
        return limited

    def _generate_search_waypoints(self):
        """Generate lawnmower waypoints (skipped when sequential SLAM mission is enabled)."""
        if self._uses_sequential_mission():
            return []
        env = self.env.unwrapped
        cells = None
        if not getattr(env, "_walkable_grid_unreliable", False):
            cells = getattr(env, "_walkable_spawn_cells", None)
        if cells is not None and cells.shape[0] >= 3:
            wps = self._generate_waypoints_from_walkable_cells(cells)
            if wps:
                print(f"[Brain] Generated {len(wps)} waypoints from walkable floor grid.")
                return wps

        # Add safety margins to boundaries
        start_x = self.min_x + self.safety_margin
        end_x = self.max_x - self.safety_margin
        start_y = self.min_y + self.safety_margin
        end_y = self.max_y - self.safety_margin
        
        # Step through room
        x_steps = np.arange(start_x, end_x, self.step_size)
        y_steps = np.arange(start_y, end_y, self.step_size)
        
        # Build list
        wps = []
        skipped = 0
        for i, x in enumerate(x_steps):
            # Alternate Y direction for smooth continuous snake flight path
            curr_y_steps = y_steps if i % 2 == 0 else y_steps[::-1]
            for y in curr_y_steps:
                # Keep waypoints at drone default flight height (z = 1.0m)
                safe_wp = self._nearest_safe_point(float(x), float(y))
                if safe_wp is None:
                    skipped += 1
                    continue
                if not wps or math.hypot(safe_wp[0] - wps[-1][0], safe_wp[1] - wps[-1][1]) > 0.5:
                    wps.append(safe_wp)
                    
        # If no waypoints were generated, add a basic centered grid fallback
        if not wps:
            center_wp = self._nearest_safe_point(0.0, 0.0)
            wps = [center_wp if center_wp is not None else (0.0, 0.0, 1.0)]

        if skipped > 0:
            print(f"[Brain] Skipped {skipped} unsafe waypoint candidates with no nearby safe replacement.")
            
        return wps

    def capture_mission_snapshot(self):
        """Save SLAM progress so a crash reset can continue the search."""
        snap = {
            "state": self.state,
            "current_wp_idx": self.current_wp_idx,
            "found_person": self.found_person,
            "target_person_pos": (
                self.target_person_pos.copy()
                if self.target_person_pos is not None
                else None
            ),
            "scan_yaw_accumulated": self.scan_yaw_accumulated,
            "scan_lock_pos": (
                self.scan_lock_pos.copy() if self.scan_lock_pos is not None else None
            ),
            "segment_idx": self.segment_idx,
            "nav_target": self.nav_target.copy() if self.nav_target is not None else None,
            "mission_finished": self.mission_finished,
        }
        if hasattr(self, "_visited"):
            snap["visited"] = self._visited.copy()
        return snap

    def force_skip_to_next_checkpoint(self) -> bool:
        """Escape corridor crash loop — advance nav target without a 360 scan."""
        if not self._uses_sequential or self.state != "GOTO_WAYPOINT":
            return False
        seq = self._get_spawn_sequence()
        if not seq or self.nav_target is None:
            return False
        tgt_idx = self._checkpoint_index_for_target(self.nav_target, seq)
        at_idx = tgt_idx if tgt_idx is not None else self.segment_idx
        next_idx = min(at_idx + 1, len(seq) - 1)
        if next_idx <= at_idx:
            return False
        self.segment_idx = at_idx
        self.nav_target = np.array(seq[next_idx], dtype=float)
        self.state = "GOTO_WAYPOINT"
        self.scan_yaw_accumulated = 0.0
        self.scan_lock_pos = None
        self._scan_announced_segment = -1
        print(
            f"[Brain] Crash-loop escape — skipping to {self.get_segment_label(next_idx)} "
            f"({self.nav_target[0]:.1f}, {self.nav_target[1]:.1f})."
        )
        return True

    def resync_nav_target_from_sequence(self) -> None:
        """Fix stale/wrong nav targets after crash loops (e.g. USD corridor coords)."""
        if not self._uses_sequential or self.state != "GOTO_WAYPOINT":
            return
        seq = self._get_spawn_sequence()
        if not seq:
            return
        next_idx = min(self.segment_idx + 1, len(seq) - 1)
        self.nav_target = np.array(seq[next_idx], dtype=float)

    def prepare_crash_respawn(self) -> None:
        """After a crash reset: spawn at room entrance and resume GOTO — never inherit mid-360° SCAN."""
        if not self._uses_sequential:
            return
        self.found_person = False
        self.target_person_pos = None
        self._pending_rescue_pos = None
        self.scan_yaw_accumulated = 0.0
        self.scan_lock_pos = None
        self._pending_scan_snap_pos = None
        self._arrival_dwell_steps = 0
        self._scan_announced_segment = -1
        self.last_drone_yaw = None
        self.mission_finished = False

        seq = self._get_spawn_sequence()
        if not seq:
            return

        # Keep the active nav target after a crash (do not rewind to segment+1).
        if self.nav_target is None:
            next_idx = min(self.segment_idx + 1, len(seq) - 1)
            self.nav_target = np.array(seq[next_idx], dtype=float)
        self.state = "GOTO_WAYPOINT"
        tgt = self.nav_target
        print(
            f"[Brain] Crash recovery — continue GOTO toward ({tgt[0]:.1f}, {tgt[1]:.1f}) "
            f"(segment {self.segment_idx + 1}, no 360 until target reached)."
        )

    def restore_mission_snapshot(self, snapshot) -> None:
        """Restore SLAM progress after respawning inside the map."""
        if snapshot is None:
            self.reset_mission_from_start()
            return

        if self._uses_sequential_mission():
            self.segment_idx = int(snapshot.get("segment_idx", 0))
            self.nav_target = snapshot.get("nav_target")
            if self.nav_target is not None:
                self.nav_target = np.array(self.nav_target, dtype=float)
            self.mission_finished = bool(snapshot.get("mission_finished", False))
            self.current_wp_idx = int(snapshot.get("current_wp_idx", 0))
            self.found_person = False
            self.target_person_pos = None
            self.scan_yaw_accumulated = 0.0
            self.scan_lock_pos = None
            self.last_drone_yaw = None
            self._scan_announced_segment = -1
            self._arrival_dwell_steps = 0
            if hasattr(self, "_visited") and "visited" in snapshot:
                self._visited = snapshot["visited"].copy()
            # State/GOTO finalized in prepare_crash_respawn() after spawn placement.
            return

        if snapshot.get("found_person", False):
            self.state = "SCAN"
            self.current_wp_idx = 0
            self.found_person = False
            self.target_person_pos = None
            self.scan_yaw_accumulated = 0.0
            self.scan_lock_pos = None
            self.last_drone_yaw = None
            return

        self.state = snapshot.get("state", "SCAN")
        self.current_wp_idx = int(snapshot.get("current_wp_idx", 0))
        self.found_person = False
        self.target_person_pos = None
        self.scan_yaw_accumulated = float(snapshot.get("scan_yaw_accumulated", 0.0))
        self.scan_lock_pos = None
        self.last_drone_yaw = None
        if self.state == "SCAN":
            self.scan_yaw_accumulated = 0.0
        if self.current_wp_idx >= len(self.waypoints):
            self.current_wp_idx = 0
            self.state = "SCAN"

    def try_complete_goto_arrival(self, drone_local_xy) -> bool:
        """Optional debug-only stuck skip; disabled by default (brain_allow_stuck_arrival_skip=False)."""
        if not getattr(self.env.unwrapped.cfg, "brain_allow_stuck_arrival_skip", False):
            return False
        if self.state != "GOTO_WAYPOINT" or self.nav_target is None or not self._uses_sequential:
            return False
        cfg = self.env.unwrapped.cfg
        arrive_r = float(getattr(cfg, "brain_spawn_arrival_radius", 2.5))
        d_pos = np.array(drone_local_xy, dtype=float)
        dist = np.linalg.norm(self.nav_target[:2] - d_pos[:2])
        if dist > arrive_r * 1.5:
            return False

        seq = self._get_spawn_sequence()
        next_idx = self.segment_idx + 1
        self._mark_visited_at(d_pos[0], d_pos[1])
        if next_idx == len(seq) - 1:
            self.state = "COMPLETE"
            self.mission_finished = True
            print("[Brain] Near-target arrival — reached finish point. Mission complete.")
        else:
            self.segment_idx = next_idx
            self.state = "SCAN"
            self.scan_yaw_accumulated = 0.0
            self.scan_lock_pos = None
            self.nav_target = None
            print(
                f"[Brain] Near-target arrival at ({d_pos[0]:.1f}, {d_pos[1]:.1f}) "
                f"→ starting SCAN at spawn{self.segment_idx + 1}..."
            )
        return True

    def get_brain_goal_local(self, drone_local_xy=None):
        """Return the current high-level goal in env-local coordinates (x, y, z)."""
        if self.found_person and self.target_person_pos is not None:
            return np.array(self.target_person_pos, dtype=float)
        if self.state == "SCAN":
            if drone_local_xy is not None:
                return np.array([drone_local_xy[0], drone_local_xy[1], 1.0], dtype=float)
            if self.scan_lock_pos is not None:
                return np.array(self.scan_lock_pos, dtype=float)
            if self._uses_sequential:
                seq = self._get_spawn_sequence()
                pt = seq[min(self.segment_idx, len(seq) - 1)]
                return np.array(pt, dtype=float)
        if self.state == "GOTO_WAYPOINT" and self._uses_sequential and self.nav_target is not None:
            return np.array(self.nav_target, dtype=float)
        if self.waypoints:
            idx = min(self.current_wp_idx, len(self.waypoints) - 1)
            return np.array(self.waypoints[idx], dtype=float)
        if self._uses_sequential:
            seq = self._get_spawn_sequence()
            return np.array(seq[0], dtype=float)
        return np.array([0.0, 0.0, 1.0], dtype=float)

    def reached_finish_point(self) -> bool:
        return bool(self.mission_finished and not self.found_person)

    def _generate_waypoints_from_walkable_cells(self, cells_tensor):
        """Build a lawnmower path from parsed walkable floor cells (R-shaped maps)."""
        cells = cells_tensor.detach().cpu().numpy()
        if cells.shape[0] == 0:
            return []

        res = float(getattr(self.env.unwrapped, "_walkable_grid_res", 0.4))
        x_bins = np.round(cells[:, 0] / max(res, 0.2)) * max(res, 0.2)
        unique_x = np.unique(x_bins)
        wps = []
        skipped = 0

        for i, x_col in enumerate(sorted(unique_x)):
            col = cells[np.abs(cells[:, 0] - x_col) <= res * 0.75]
            if col.shape[0] == 0:
                continue
            col = col[np.argsort(col[:, 1])]
            if i % 2 == 1:
                col = col[::-1]
            for pt in col:
                safe_wp = self._nearest_safe_point(float(pt[0]), float(pt[1]))
                if safe_wp is None:
                    skipped += 1
                    continue
                if not wps or math.hypot(safe_wp[0] - wps[-1][0], safe_wp[1] - wps[-1][1]) > 0.5:
                    wps.append(safe_wp)

        if skipped > 0:
            print(f"[Brain] Skipped {skipped} walkable-grid waypoint candidates with no safe replacement.")
        return wps

    def update(self, person_found, person_world_xyz, drone_pos, drone_quat):
        """
        Updates the Brain State Machine and computes the high-level goal position and target yaw.
        :param person_found: Boolean tensor of shape (num_envs,) indicating if YOLO sees a human.
        :param person_world_xyz: Float tensor of shape (num_envs, 3) indicating human's position.
        :param drone_pos: Float tensor of shape (num_envs, 3) representing drone position.
        :param drone_quat: Float tensor of shape (num_envs, 4) representing drone orientation.
        :return: (desired_pos_w, target_yaw)
        """
        # Convert tensors to python scalars/arrays for single env0 high-level planning
        env_origin = self.env.unwrapped._terrain.env_origins[0].cpu().numpy()
        d_pos_w = drone_pos[0].cpu().numpy()
        d_pos = d_pos_w - env_origin  # env-local frame (matches waypoints / map_obstacles)
        d_quat = drone_quat[0].cpu().numpy()
        
        # Compute yaw angle of drone
        qw, qx, qy, qz = d_quat
        drone_yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        if self.last_drone_yaw is None:
            self.last_drone_yaw = drone_yaw

        # -------------------------------------------------------------
        # State Machine Transitions & Actions
        # -------------------------------------------------------------
        
        # If person found by YOLO, transition to APPROACH (only when rescue is armed — final room+)
        if person_found[0].item() and not self.found_person:
            safe_target, reject_reason = self._resolve_rescue_target(
                person_world_xyz[0].cpu().numpy() - env_origin
            )
            if reject_reason is not None:
                print(f"[Brain] Ignoring detection — {reject_reason}.")
            elif safe_target is not None:
                p_pos = person_world_xyz[0].cpu().numpy() - env_origin
                if self.state == "SCAN":
                    self._pending_rescue_pos = safe_target
                    self.person_noted_anywhere = True
                    print(
                        f"\n[Brain] Person detected during 360 SCAN at "
                        f"X:{p_pos[0]:.2f} Y:{p_pos[1]:.2f} — finishing rotation, then approach."
                    )
                else:
                    self.found_person = True
                    self.target_person_pos = safe_target
                    self.state = "APPROACH_TARGET"
                    print(
                        f"\n[Brain] !!! YOLOV11 DETECTED HUMAN AT: "
                        f"X:{p_pos[0]:.2f} Y:{p_pos[1]:.2f} Z:{p_pos[2]:.2f} !!!"
                    )
                    print("[Brain] Changing state to APPROACH_TARGET.")

        if self.state == "SCAN":
            if self.scan_lock_pos is None:
                if self._uses_sequential and self._scan_announced_segment != self.segment_idx:
                    self._scan_announced_segment = self.segment_idx
                    label = self.get_segment_label()
                    print(
                        f"\n[Brain] *** 360 SCAN at {label.upper()} "
                        f"(segment {self.segment_idx + 1}) "
                        f"position X:{d_pos[0]:.2f} Y:{d_pos[1]:.2f} Z:{d_pos[2]:.2f} — rotating for YOLO ***\n"
                    )
                # Spin in place where the drone actually arrived (not the checkpoint coord).
                self.scan_lock_pos = np.array(
                    [float(d_pos[0]), float(d_pos[1]), float(d_pos[2])], dtype=float
                )
                self._pending_scan_snap_pos = None
            
            yaw_diff = wrap_to_pi_scalar(drone_yaw - self.last_drone_yaw)
            self.scan_yaw_accumulated += abs(yaw_diff)
            target_yaw = drone_yaw + 0.15
            desired_pos_w = self.scan_lock_pos
            
            if self.scan_yaw_accumulated >= 2 * math.pi:
                self.scan_yaw_accumulated = 0.0
                self.scan_lock_pos = None
                self._mark_visited_at(d_pos[0], d_pos[1])

                if self._uses_sequential:
                    seq = self._get_spawn_sequence()
                    next_idx = self.segment_idx + 1
                    if next_idx >= len(seq):
                        if self._pending_rescue_pos is not None:
                            self.found_person = True
                            self.target_person_pos = self._pending_rescue_pos
                            self._pending_rescue_pos = None
                            self.state = "APPROACH_TARGET"
                            tp = self.target_person_pos
                            print(
                                "[Brain] Final scan complete — approaching detected person at "
                                f"X:{tp[0]:.2f} Y:{tp[1]:.2f} Z:{tp[2]:.2f}."
                            )
                        else:
                            self.state = "COMPLETE"
                            self.mission_finished = True
                            print("[Brain] Scan complete — no further segments. Mission complete.")
                    else:
                        self.nav_target = np.array(seq[next_idx], dtype=float)
                        self.state = "GOTO_WAYPOINT"
                        dest_label = self.get_segment_label(next_idx)
                        visited, total = self.coverage_stats()
                        print(
                            f"[Brain] Scan at {self.get_segment_label()} complete "
                            f"(coverage {visited}/{total} cells). Navigating to {dest_label}..."
                        )
                else:
                    self.state = "GOTO_WAYPOINT"
                    print(f"[Brain] Scan completed. Navigating to waypoint {self.current_wp_idx}/{len(self.waypoints)}...")
                
        elif self.state == "GOTO_WAYPOINT":
            cfg = self.env.unwrapped.cfg
            skip_scan = set(getattr(cfg, "brain_skip_scan_segment_indices", ()))

            if self._uses_sequential:
                if self.nav_target is None:
                    seq = self._get_spawn_sequence()
                    next_idx = min(self.segment_idx + 1, len(seq) - 1)
                    self.nav_target = np.array(seq[next_idx], dtype=float)
                    self.state = "GOTO_WAYPOINT"
                    desired_pos_w = np.array(self.nav_target, dtype=float)
                    dx = desired_pos_w[0] - d_pos[0]
                    dy = desired_pos_w[1] - d_pos[1]
                    target_yaw = math.atan2(dy, dx)
                else:
                    desired_pos_w = np.array(self.nav_target, dtype=float)
                    dx = desired_pos_w[0] - d_pos[0]
                    dy = desired_pos_w[1] - d_pos[1]
                    target_yaw = math.atan2(dy, dx)
                    seq = self._get_spawn_sequence()
                    tgt_idx = self._checkpoint_index_for_target(self.nav_target, seq)
                    arrive_r = self._arrival_radius_for_target(tgt_idx, seq)
                    dist_to_target = np.linalg.norm(desired_pos_w[:2] - d_pos[:2])
                    dwell_need = int(getattr(cfg, "brain_scan_arrival_dwell_steps", 8))
                    can_arrive = dist_to_target < arrive_r and (
                        tgt_idx is None
                        or self._can_count_arrival(d_pos, tgt_idx, seq, arrive_r)
                    )
                    if can_arrive:
                        self._arrival_dwell_steps += 1
                    else:
                        self._arrival_dwell_steps = 0

                    if self._arrival_dwell_steps >= dwell_need:
                        self._arrival_dwell_steps = 0
                        self._mark_visited_at(d_pos[0], d_pos[1])
                        if tgt_idx is not None and tgt_idx > self.segment_idx:
                            self.segment_idx = tgt_idx
                        elif tgt_idx is None:
                            self.segment_idx = min(self.segment_idx + 1, len(seq) - 1)

                        at_idx = self.segment_idx
                        if at_idx in skip_scan:
                            next_idx = at_idx + 1
                            if next_idx >= len(seq):
                                self.state = "COMPLETE"
                                self.mission_finished = True
                                self.nav_target = None
                                print(
                                    f"[Brain] Passed {self.get_segment_label(at_idx)} "
                                    f"(narrow — no 360). Mission complete."
                                )
                            else:
                                self.nav_target = np.array(seq[next_idx], dtype=float)
                                self.state = "GOTO_WAYPOINT"
                                print(
                                    f"[Brain] Passed {self.get_segment_label(at_idx)} "
                                    f"(narrow corridor — no 360). "
                                    f"Navigating to {self.get_segment_label(next_idx)}..."
                                )
                        else:
                            self.state = "SCAN"
                            self.scan_yaw_accumulated = 0.0
                            self.scan_lock_pos = None
                            self.nav_target = None
                            self._scan_announced_segment = -1
                            print(
                                f"[Brain] Reached {self.get_segment_label()} "
                                f"(dist {dist_to_target:.2f}m). Starting 360 SCAN..."
                            )
            elif self.current_wp_idx >= len(self.waypoints):
                self.state = "COMPLETE"
                print(f"[Brain] Checked all waypoints. Search complete.")
                desired_pos_w = d_pos
                target_yaw = drone_yaw
            else:
                wp = self.waypoints[self.current_wp_idx]
                desired_pos_w = np.array(wp)
                
                dx = wp[0] - d_pos[0]
                dy = wp[1] - d_pos[1]
                target_yaw = math.atan2(dy, dx)
                
                dist_to_wp = np.linalg.norm(desired_pos_w[:2] - d_pos[:2])
                wp_arrive_r = float(getattr(cfg, "brain_scan_arrival_radius", 1.0))
                if dist_to_wp < wp_arrive_r:
                    self.current_wp_idx += 1
                    if self.current_wp_idx >= len(self.waypoints):
                        self.state = "COMPLETE"
                        print("[Brain] Reached final waypoint. Search complete.")
                    else:
                        print(f"[Brain] Reached waypoint. Continuing to waypoint {self.current_wp_idx}/{len(self.waypoints)}...")
                    
        elif self.state == "APPROACH_TARGET":
            # Action: Head straight to detected human
            desired_pos_w = self.target_person_pos
            
            # Look at target
            dx = self.target_person_pos[0] - d_pos[0]
            dy = self.target_person_pos[1] - d_pos[1]
            target_yaw = math.atan2(dy, dx)
            
            # Check distance to target
            dist_to_target = np.linalg.norm(self.target_person_pos - d_pos)
            if dist_to_target < 0.8:
                self.state = "COMPLETE"
                print(f"\n[Brain] SUCCESS: Reached search and rescue target person location!")
                print(f"  ↳ FINAL RESCUE COORDINATES relative to entrance: X:{self.target_person_pos[0]:.2f}m, Y:{self.target_person_pos[1]:.2f}m, Z:{self.target_person_pos[2]:.2f}m")
                
        elif self.state == "COMPLETE":
            if self.target_person_pos is not None:
                desired_pos_w = self.target_person_pos
            elif self._uses_sequential:
                seq = self._get_spawn_sequence()
                desired_pos_w = np.array(seq[-1], dtype=float)
            else:
                desired_pos_w = d_pos
            target_yaw = drone_yaw
            
        else:
            # Fallback
            desired_pos_w = d_pos
            target_yaw = drone_yaw

        self.last_drone_yaw = drone_yaw
        # Rescue / sequential nav: do not cap goal distance to curriculum range.
        skip_limit = self.state in ("APPROACH_TARGET", "COMPLETE") or (
            self._uses_sequential and self.state == "GOTO_WAYPOINT"
        )
        if not skip_limit:
            desired_pos_w = self._limit_goal_distance(desired_pos_w, d_pos)
        
        # Convert back to torch Tensors matching environment shape [num_envs, ...]
        desired_pos_w_tensor = torch.zeros((self.env.unwrapped.num_envs, 3), device=self.device)
        desired_pos_w_tensor[:, 0] = float(desired_pos_w[0]) + self.env.unwrapped._terrain.env_origins[:, 0]
        desired_pos_w_tensor[:, 1] = float(desired_pos_w[1]) + self.env.unwrapped._terrain.env_origins[:, 1]
        desired_pos_w_tensor[:, 2] = float(desired_pos_w[2])
        
        target_yaw_tensor = torch.ones(self.env.unwrapped.num_envs, device=self.device) * float(target_yaw)
        
        return desired_pos_w_tensor, target_yaw_tensor

def wrap_to_pi_scalar(x):
    """Wrap angle in radians to [-pi, pi]."""
    while x > math.pi:
        x -= 2 * math.pi
    while x < -math.pi:
        x += 2 * math.pi
    return x
