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
            safety_margin=0.20,
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

    def reset_mission_from_start(self) -> None:
        """Keep room-1 spawn from sequential config but start in SLAM EXPLORE (not SCAN)."""
        super().reset_mission_from_start()
        self.state = "EXPLORE"
        self.active_frontier = None
        self.astar_path_world = []
        self.explore_step_count = 0
        self.last_scan_pos = None
        self.rescued_people = []
        self.blacklisted_frontiers = []
        self.active_frontier_ticks = 0

    def capture_mission_snapshot(self):
        # Decouple from parent sequential checks, always create a valid snapshot dictionary
        snap = {}
        snap["segment_idx"] = int(self.segment_idx)
        snap["rescued_people"] = [p.copy() for p in self.rescued_people]
        if self.last_scan_pos is not None:
            snap["last_scan_pos"] = self.last_scan_pos.copy()
            
        return snap

    def restore_mission_snapshot(self, snap):
        if not snap:
            return
        self.segment_idx = snap.get("segment_idx", 0)
        self.rescued_people = [np.array(p) for p in snap.get("rescued_people", [])]
        if "last_scan_pos" in snap and snap["last_scan_pos"] is not None:
            self.last_scan_pos = snap["last_scan_pos"].copy()
        
        # Clear state/target so that on recovery we force immediate safe path planning
        self.state = "EXPLORE"
        self.active_frontier = None
        self.astar_path_world = []
        self.explore_step_count = 50  # force immediate path generation
        self.blacklisted_frontiers = []
        self.active_frontier_ticks = 0

    def coverage_stats(self) -> tuple[int, int]:
        # Count cells that have been mapped (either free or occupied, i.e., not unknown 0.5)
        prob = self.mapper.get_occupancy_grid()
        mapped = (prob < 0.35) | (prob > 0.65)
        if hasattr(self.mapper, "walkable_mask") and self.mapper.walkable_mask is not None:
            visited = int(np.sum(mapped & self.mapper.walkable_mask))
            total = int(np.sum(self.mapper.walkable_mask))
        else:
            visited = int(np.sum(mapped))
            total = int(prob.size)
        return visited, total

    def get_segment_label(self, idx=0):
        return f"SLAM {self.state}"

    def update(self, person_found, person_world_xyz, drone_pos, drone_quat):
        """SLAM-driven high-level brain update logic."""
        env_origin = self.env._terrain.env_origins[0].cpu().numpy()
        d_pos_w = drone_pos[0].cpu().numpy()
        d_quat = drone_quat[0].cpu().numpy()

        qw, qx, qy, qz = d_quat
        drone_yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))

        depth_tensor = self.env._tiled_camera.data.output.get("depth")
        if depth_tensor is not None:
            depth_np = torch.squeeze(depth_tensor[0]).detach().cpu().numpy()
            self.mapper.update_from_depth(
                depth_np, d_pos_w, d_quat,
                focal_length=float(self.env.cfg.tiled_camera.spawn.focal_length),
                horizontal_aperture=float(self.env.cfg.tiled_camera.spawn.horizontal_aperture),
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

        desired_pos_w = np.zeros(3, dtype=np.float32)
        target_yaw = drone_yaw
        cruise_z = 1.0

        if self.state == "EXPLORE":
            self.explore_step_count += 1

            if self.active_frontier is not None:
                self.active_frontier_ticks += 1
                # If we've been trying to reach the active frontier for more than 15 seconds (150 steps), blacklist it!
                if self.active_frontier_ticks > 150:
                    centroid = self.active_frontier["centroid_world"]
                    print(
                        f"\n[SLAM Brain] Active frontier at X:{centroid[0]:.2f} Y:{centroid[1]:.2f} "
                        f"is UNREACHABLE after {self.active_frontier_ticks} steps. Blacklisting it to prevent loop."
                    )
                    self.blacklisted_frontiers.append(centroid)
                    self.active_frontier = None
                    self.astar_path_world = []
                    self.active_frontier_ticks = 0
                    self.explore_step_count = 50  # force immediate replanning in this frame
            else:
                self.active_frontier_ticks = 0

            dist_to_f = (
                float(np.linalg.norm(d_pos_w[:2] - np.array(self.active_frontier["centroid_world"])))
                if self.active_frontier is not None
                else float("inf")
            )

            if self.active_frontier is not None and dist_to_f < 0.6:
                # Check distance traveled since the last scan to avoid redundant scanning
                dist_since_last_scan = 999.0
                if getattr(self, "last_scan_pos", None) is not None:
                    dist_since_last_scan = float(np.linalg.norm(d_pos_w[:2] - self.last_scan_pos[:2]))
                    
                if dist_since_last_scan > 2.5:
                    print(f"[SLAM Brain] Reached frontier. Travelled {dist_since_last_scan:.2f}m since last scan. Initiating 360 SCAN.")
                    self.state = "SCAN"
                    self.env._scan_step_count = 0
                    self.last_scan_pos = d_pos_w.copy()
                    self.explore_step_count = 0
                    desired_pos_w[:] = d_pos_w
                    desired_pos_w_tensor = torch.tensor(desired_pos_w, device=drone_pos.device).repeat(self.env.num_envs, 1)
                    target_yaw_tensor = torch.tensor(target_yaw, device=drone_pos.device).repeat(self.env.num_envs)
                    return desired_pos_w_tensor, target_yaw_tensor
                else:
                    print(f"[SLAM Brain] Reached frontier, but skipping redundant scan (only {dist_since_last_scan:.2f}m from last scan). Clearing target.")
                    self.active_frontier = None
                    self.active_frontier_ticks = 0
                    # Fall through to planning block to plan new path in this very frame!

            need_target = self.active_frontier is None
            periodic_replan = (
                self.active_frontier is not None and self.explore_step_count >= 50
            )

            if need_target:
                self.explore_step_count = 0
                self.active_frontier_ticks = 0
                frontiers = self.mapper.detect_frontiers(min_size=8)
                
                # Filter out frontiers near blacklisted coordinates
                frontiers = [
                    f for f in frontiers
                    if np.linalg.norm(d_pos_w[:2] - np.array(f["centroid_world"])) > 1.8
                    and not any(np.linalg.norm(np.array(f["centroid_world"]) - np.array(b)) < 1.5 for b in self.blacklisted_frontiers)
                ]

                if not frontiers:
                    prob = self.mapper.get_occupancy_grid()
                    explored_cells = int(np.sum(prob < 0.35))
                    drone_y = float(d_pos_w[1])

                    # Hard requirement: drone must have physically entered room 4
                    # (Y <= -16.0 is inside the room-4 entrance corridor).
                    # Also require enough total cells explored to cover all 4 rooms.
                    # 12000 cells @ 0.10m² = ~120 m² ≈ full 4-room footprint.
                    reached_room4 = drone_y <= -16.0
                    enough_explored = explored_cells > 12000

                    if reached_room4 and enough_explored:
                        print(
                            f"[SLAM Brain] All frontiers cleared AND room 4 reached. "
                            f"Exploration COMPLETE ({explored_cells} cells, Y={drone_y:.1f}m)."
                        )
                        self.state = "COMPLETE"
                        self.mission_finished = True
                    elif not reached_room4:
                        # Force the drone toward room 4 by synthesising a temporary
                        # waypoint at the room-4 entrance so it doesn't idle here.
                        room4_entry = np.array([-2.0, -17.0], dtype=np.float32)
                        print(
                            f"[SLAM Brain] No visible frontiers but room 4 not reached "
                            f"(Y={drone_y:.1f}m, cells={explored_cells}). "
                            f"Driving toward room-4 entrance {room4_entry}."
                        )
                        # Inject a synthetic frontier so EXPLORE logic picks it up
                        self.active_frontier = {
                            "centroid_world": room4_entry.tolist(),
                            "cells": [],
                        }
                        self.astar_path_world = [room4_entry.tolist()]
                        self.active_frontier_ticks = 0
                    else:
                        print(
                            f"[SLAM Brain] No frontiers visible but not enough explored yet "
                            f"({explored_cells} cells, Y={drone_y:.1f}m). Waiting for more map data."
                        )
                        self.active_frontier = None
                        self.astar_path_world = []
                else:
                    # Room boundaries (world Y): Room 1 is shallowest (Y≈+2),
                    # Room 4 is deepest (Y≈-21).
                    def _frontier_room(f_y: float) -> int:
                        if f_y > -3.0:
                            return 1
                        elif f_y > -9.0:
                            return 2
                        elif f_y > -17.0:
                            return 3
                        return 4

                    # Always clear the shallowest room that still has frontiers
                    # before advancing to deeper rooms.  Use a 200 m "virtual
                    # distance" penalty per room number gap so distance within a
                    # room never overrides room priority.
                    min_room = min(
                        _frontier_room(float(f["centroid_world"][1])) for f in frontiers
                    )

                    def _frontier_score(f):
                        cw   = np.array(f["centroid_world"])
                        dist = float(np.linalg.norm(d_pos_w[:2] - cw))
                        room = _frontier_room(float(cw[1]))
                        # 200 m gap per room ensures room priority is always dominant
                        return (room - min_room) * 200.0 + dist

                    sorted_frontiers = sorted(frontiers, key=_frontier_score)
                    grid_path = None
                    selected_f = None
                    inflated = self.mapper.get_inflated_grid()
                    start_r, start_c = self.mapper.world_to_grid(d_pos_w[0], d_pos_w[1])

                    for f in sorted_frontiers:
                        goal_r, goal_c = self.mapper.world_to_grid(
                            f["centroid_world"][0], f["centroid_world"][1]
                        )
                        inflated_temp = inflated.copy()
                        # Clear a 5x5 region around start and goal to guarantee connectivity near obstacles
                        for r_off in range(-2, 3):
                            for c_off in range(-2, 3):
                                if self.mapper.is_in_bounds(start_r + r_off, start_c + c_off):
                                    inflated_temp[start_r + r_off, start_c + c_off] = 0
                                if self.mapper.is_in_bounds(goal_r + r_off, goal_c + c_off):
                                    inflated_temp[goal_r + r_off, goal_c + c_off] = 0

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
                    else:
                        self.active_frontier = sorted_frontiers[0]
                        self.astar_path_world = [self.active_frontier["centroid_world"]]
                        print(
                            "[SLAM Brain] All paths blocked. Fallback straight line to: "
                            f"{self.active_frontier['centroid_world']}"
                        )

            elif periodic_replan:
                self.explore_step_count = 0
                inflated = self.mapper.get_inflated_grid()
                start_r, start_c = self.mapper.world_to_grid(d_pos_w[0], d_pos_w[1])
                goal_r, goal_c = self.mapper.world_to_grid(
                    self.active_frontier["centroid_world"][0],
                    self.active_frontier["centroid_world"][1],
                )
                inflated_temp = inflated.copy()
                # Clear a 5x5 region around start and goal to guarantee connectivity near obstacles
                for r_off in range(-2, 3):
                    for c_off in range(-2, 3):
                        if self.mapper.is_in_bounds(start_r + r_off, start_c + c_off):
                            inflated_temp[start_r + r_off, start_c + c_off] = 0
                        if self.mapper.is_in_bounds(goal_r + r_off, goal_c + c_off):
                            inflated_temp[goal_r + r_off, goal_c + c_off] = 0
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
                
                # Look ahead from the closest node index to find a node at least 0.4 meters ahead
                next_target = self.astar_path_world[-1]
                for node in self.astar_path_world[closest_idx:]:
                    if np.linalg.norm(d_pos_2d - np.array(node)) > 0.4:
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
        cfg.brain_real_slam_mode = True
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
            steps_in_scan = int(getattr(self, "_scan_step_count", 0))
            if steps_in_scan >= 130:
                print("[SLAM Environment] Scan spin finished. Returning to EXPLORE.")
                self._brain.active_frontier = None
                self._brain.astar_path_world = []
                self._brain.state = "EXPLORE"
                self._scan_step_count = 0

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
        # Respawn the drone at the center checkpoint of the room it crashed in
        seq = getattr(self.cfg, "brain_spawn_sequence", None)
        if seq and len(seq) > 0 and crash_local is not None:
            crash_xy = np.array(crash_local[:2])
            distances = [np.linalg.norm(crash_xy - np.array(pt[:2])) for pt in seq]
            closest_idx = int(np.argmin(distances))
            sx, sy, sz = seq[closest_idx]
            
            if hasattr(self, "_brain"):
                self._brain.segment_idx = closest_idx
                
            print(f"[SLAM Environment] Crash recovery — Respawning at Room {closest_idx + 1} checkpoint: ({sx:.2f}, {sy:.2f}, {sz:.2f})")
            
            spawn_x = torch.full((env_count,), sx, device=device)
            spawn_y = torch.full((env_count,), sy, device=device)
            spawn_z = torch.full((env_count,), sz, device=device)
            return spawn_x, spawn_y, spawn_z
            
        return super()._sample_brain_spawn_xyz(env_count, crash_local=crash_local, force_checkpoint=force_checkpoint)
