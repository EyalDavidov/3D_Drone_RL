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
        
        # Extract boundaries and obstacles dynamically from USD stage
        self.min_x, self.max_x, self.min_y, self.max_y, self.obstacles = self._extract_map_data()
        
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

    def _extract_map_data(self):
        """
        Dynamically calculate room boundaries and extract internal obstacles.
        Prioritizes the environment's configured map_obstacles (which defines the active room/arena),
        falling back to USD stage parsing if config is empty.
        """
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
            print(f"[Brain] Warning: Could not parse map_obstacles config ({e}). Trying USD stage.")

        # 2. Fallback to USD stage parsing
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
        """
        Generate continuous boustrophedon (lawnmower) search waypoints.
        """
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
        
        # If person found by YOLO, transition to APPROACH state
        if person_found[0].item() and not self.found_person:
            p_pos = person_world_xyz[0].cpu().numpy() - env_origin
            # Safety check: ensure coordinates are valid/finite
            if np.all(np.isfinite(p_pos)):
                safe_target = self._nearest_safe_point(float(p_pos[0]), float(p_pos[1]))
                if safe_target is None:
                    print(
                        f"[Brain] Ignoring detected target at unsafe location "
                        f"X:{p_pos[0]:.2f} Y:{p_pos[1]:.2f}; no safe approach point found nearby."
                    )
                    safe_target = None
                else:
                    safe_target = np.array(safe_target)
                    if np.linalg.norm(safe_target[:2] - p_pos[:2]) > 0.05:
                        print(
                            f"[Brain] Target is too close to an obstacle. "
                            f"Using safe approach point X:{safe_target[0]:.2f} Y:{safe_target[1]:.2f}."
                        )

                if safe_target is not None:
                    self.found_person = True
                    self.target_person_pos = safe_target
                    self.state = "APPROACH_TARGET"
                    print(f"\n[Brain] !!! YOLOV11 DETECTED HUMAN AT: X:{p_pos[0]:.2f} Y:{p_pos[1]:.2f} Z:{p_pos[2]:.2f} !!!")
                    print(f"[Brain] Changing state to APPROACH_TARGET.")

        if self.state == "SCAN":
            if self.scan_lock_pos is None:
                self.scan_lock_pos = np.array([d_pos[0], d_pos[1], 1.0])
            
            # Action: Rotate in place to cover 360 degrees
            # Calculate change in yaw since last update
            yaw_diff = wrap_to_pi_scalar(drone_yaw - self.last_drone_yaw)
            self.scan_yaw_accumulated += abs(yaw_diff)
            
            # Spin command: update target yaw slightly ahead
            target_yaw = drone_yaw + 0.15  # smooth spinning
            
            # Desired position is locked location (stay in place while spinning)
            desired_pos_w = self.scan_lock_pos
            
            # Check if 360 degree scan is completed
            if self.scan_yaw_accumulated >= 2 * math.pi:
                self.scan_yaw_accumulated = 0.0
                self.scan_lock_pos = None
                self.state = "GOTO_WAYPOINT"
                print(f"[Brain] Scan completed. Navigating to waypoint {self.current_wp_idx}/{len(self.waypoints)}...")
                
        elif self.state == "GOTO_WAYPOINT":
            # Action: Navigate to target waypoint
            if self.current_wp_idx >= len(self.waypoints):
                self.state = "COMPLETE"
                print(f"[Brain] Checked all waypoints. Search complete.")
                desired_pos_w = d_pos
                target_yaw = drone_yaw
            else:
                wp = self.waypoints[self.current_wp_idx]
                desired_pos_w = np.array(wp)
                
                # Face target waypoint
                dx = wp[0] - d_pos[0]
                dy = wp[1] - d_pos[1]
                target_yaw = math.atan2(dy, dx)
                
                # Check distance to waypoint
                dist_to_wp = np.linalg.norm(desired_pos_w[:2] - d_pos[:2])
                if dist_to_wp < 1.0:  # within threshold (larger than training goal_radius)
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
            # Action: Hover at target
            if self.target_person_pos is not None:
                desired_pos_w = self.target_person_pos
            else:
                desired_pos_w = d_pos
            target_yaw = drone_yaw
            
        else:
            # Fallback
            desired_pos_w = d_pos
            target_yaw = drone_yaw

        self.last_drone_yaw = drone_yaw
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
