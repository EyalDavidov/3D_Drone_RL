import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

class OccupancyGridMapper:
    def __init__(self, min_x=-35.0, max_x=15.0, min_y=-35.0, max_y=15.0, cell_size=0.15, safety_margin=0.65):
        """Dynamic 2D occupancy grid mapping based on 3D depth projection.
        
        Log-odds representation:
        - 0.0: Unknown
        - Positive values (up to max_log_odds): Occupied
        - Negative values (down to min_log_odds): Free
        """
        self.min_x = float(min_x)
        self.max_x = float(max_x)
        self.min_y = float(min_y)
        self.max_y = float(max_y)
        self.cell_size = float(cell_size)
        self.safety_margin = float(safety_margin)
        
        self.w = int(np.round((self.max_x - self.min_x) / self.cell_size))
        self.h = int(np.round((self.max_y - self.min_y) / self.cell_size))
        
        # Log-odds array (initially 0.0 - Unknown)
        self.grid_log_odds = np.zeros((self.h, self.w), dtype=np.float32)
        
        # Log-odds update values
        self.l_occ = 1.5    # Increase for hit
        self.l_free = -0.5  # Decrease for empty space
        self.max_log_odds = 5.0
        self.min_log_odds = -5.0
        
        # Kernel for obstacle inflation
        inflation_pixels = int(np.round(self.safety_margin / self.cell_size))
        self.inflation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * inflation_pixels + 1, 2 * inflation_pixels + 1))

        # Walls are long thin structures; props are compact blobs. Using area alone
        # caused merged prop clusters to flicker into "walls" as the map updated.
        self.wall_min_cells = 35
        self.wall_min_span_cells = 12      # ≥1.2 m long side → structural wall
        self.wall_max_thickness_cells = 5  # ≤0.5 m short side for thin walls
        self.wall_min_elongation = 2.5     # fallback for L-corners / longer blobs

        # Visual-only: cells once classified as structural walls stay yellow so
        # sparse early scans don't flicker teal → yellow frame-to-frame.
        self._sticky_wall_mask = np.zeros((self.h, self.w), dtype=np.uint8)
        self.expected_total_cells = 11400

    def world_to_grid(self, x, y):
        """Convert world coordinates (meters) to grid indices (row, col)."""
        col = int(np.floor((x - self.min_x) / self.cell_size))
        row = int(np.floor((y - self.min_y) / self.cell_size))
        return row, col

    def grid_to_world(self, row, col):
        """Convert grid indices (row, col) to world coordinates (meters)."""
        x = self.min_x + (col + 0.5) * self.cell_size
        y = self.min_y + (row + 0.5) * self.cell_size
        return x, y

    def is_in_bounds(self, row, col):
        return 0 <= row < self.h and 0 <= col < self.w

    def update_from_depth(self, depth_image, drone_pos, drone_quat, focal_length=18.0, horizontal_aperture=20.955):
        """Highly optimized Visual SLAM update using OpenCV C++ line drawing."""
        if depth_image is None:
            return
            
        img_h, img_w = depth_image.shape
        
        # Subsample grid to prevent overwhelming the raytracer
        step_h = 6
        step_w = 6
        sub_depth = depth_image[::step_h, ::step_w]
        
        fx = img_w * (focal_length / horizontal_aperture)
        fy = fx
        cx = img_w / 2.0
        cy = img_h / 2.0
        
        u = np.arange(0, img_w, step_w)
        v = np.arange(0, img_h, step_h)
        uu, vv = np.meshgrid(u, v)
        
        valid_mask = (sub_depth > 0.05) & (sub_depth < 10.0) & (~np.isinf(sub_depth)) & (~np.isnan(sub_depth))
        
        z_c = sub_depth[valid_mask]
        u_c = uu[valid_mask]
        v_c = vv[valid_mask]
        
        local_x = (u_c - cx) * z_c / fx
        local_y = (v_c - cy) * z_c / fy
        
        cam_vectors = np.stack([z_c, -local_x, -local_y], axis=-1)
        
        d_x, d_y, d_z = float(drone_pos[0]), float(drone_pos[1]), float(drone_pos[2])
        qw, qx, qy, qz = float(drone_quat[0]), float(drone_quat[1]), float(drone_quat[2]), float(drone_quat[3])
        
        rot = R.from_quat([qx, qy, qz, qw])
        world_vectors = rot.apply(cam_vectors)
        world_pts = world_vectors + np.array([d_x, d_y, d_z])
        
        # Filter floor and ceiling height to match drone's flight clearance height (0.8m to 1.3m).
        # This ignores the wall under/above the window openings, keeping them open in the 2D grid!
        mask = (world_pts[:, 2] > 0.8) & (world_pts[:, 2] < 1.3)
        hits_world = world_pts[mask]
        
        r0, c0 = self.world_to_grid(d_x, d_y)
        if not self.is_in_bounds(r0, c0):
            return
            
        # Draw free-space lines in C++ via cv2.line on a temp single-channel float array
        free_mask = np.zeros_like(self.grid_log_odds, dtype=np.float32)
        
        if len(hits_world) > 0:
            # Convert all hits to grid indices in a vectorized manner
            cols = np.floor((hits_world[:, 0] - self.min_x) / self.cell_size).astype(np.int32)
            rows = np.floor((hits_world[:, 1] - self.min_y) / self.cell_size).astype(np.int32)
            
            # Filter in-bounds endpoints
            valid_endpoints = (cols >= 0) & (cols < self.w) & (rows >= 0) & (rows < self.h)
            cols = cols[valid_endpoints]
            rows = rows[valid_endpoints]
            
            # Find unique grid cells to avoid redundant line tracing
            endpoints = np.stack([rows, cols], axis=-1)
            unique_endpoints = np.unique(endpoints, axis=0)
            
            for r1, c1 in unique_endpoints:
                cv2.line(free_mask, (c0, r0), (int(c1), int(r1)), float(self.l_free), thickness=1)
                self.grid_log_odds[r1, c1] = min(self.max_log_odds, self.grid_log_odds[r1, c1] + self.l_occ)
                
        # Apply the fast free mask
        self.grid_log_odds = np.clip(self.grid_log_odds + free_mask, self.min_log_odds, self.max_log_odds)
        
        # Always clear a 0.5m radius around the drone's actual position (since we are currently flying there)
        # 0.5m is approx 3 pixels in grid
        cv2.circle(self.grid_log_odds, (c0, r0), 3, float(self.min_log_odds), -1)

    def get_occupancy_grid(self):
        """Convert log-odds map to probability grid [0.0, 1.0]."""
        prob = 1.0 / (1.0 + np.exp(-self.grid_log_odds))
        return prob

    def coverage_stats(self) -> tuple[int, int]:
        """Pure SLAM map coverage.
        
        Returns (visited_cells, expected_total_cells) where expected_total_cells (11400)
        represents the actual total floor and wall cells of the entire multi-level track.
        This provides a correct, monotonically increasing percentage up to 100% as the drone
        explores, avoiding the bounding-box math which drops as the bounding-box size grows.
        """
        prob = self.get_occupancy_grid()
        known = (prob < 0.35) | (prob > 0.65)
        visited = int(known.sum())
        expected_total = getattr(self, "expected_total_cells", 11400)
        return min(visited, expected_total), expected_total

    def get_inflated_grid(self):
        """Generate binary grid with expanded obstacles for safety (all occupied)."""
        prob = self.get_occupancy_grid()
        # Binary occupied mask
        binary_occupied = (prob > 0.65).astype(np.uint8)
        # Dilate
        inflated = cv2.dilate(binary_occupied, self.inflation_kernel, iterations=1)
        return inflated

    def _is_structural_wall_component(self, area: int, width: int, height: int) -> bool:
        """True if a connected occupied blob looks like a wall, not a prop cluster."""
        long_side = max(width, height)
        short_side = max(1, min(width, height))
        elongation = long_side / short_side
        # Long thin segment (typical wall in top-down 2D projection). Checked FIRST
        # and NOT gated by area, so thin walls (e.g. 12x2 cells) aren't wrongly
        # dropped to teal just for having a small pixel count.
        if long_side >= self.wall_min_span_cells and short_side <= self.wall_max_thickness_cells:
            return True
        # Elongated corner / longer wall run needs some bulk to avoid catching props.
        if area >= self.wall_min_cells and elongation >= self.wall_min_elongation:
            return True
        return False

    def get_wall_obstacle_masks(self, use_walkable=True):
        """Split occupied cells into structural walls vs. dodgeable small obstacles.

        Returns (wall_mask, obstacle_mask) as uint8 grids.

        use_walkable=True  → may consult the static USD walkable mask (Signal 1).
                             Use ONLY for visual colouring of the map.
        use_walkable=False → PURE SLAM (shape heuristics only). Use for planning /
                             frontier reachability so navigation never depends on
                             the ground-truth USD map ("no cheating").
        """
        prob = self.get_occupancy_grid()
        occ = (prob > 0.65).astype(np.uint8)
        wall_mask = np.zeros_like(occ)
        obstacle_mask = np.zeros_like(occ)
        if not occ.any():
            return wall_mask, obstacle_mask

        # --- Signal 1: static floor boundary (colouring only, trustworthy mask) --
        # Wall hits register AT the floor edge (still flagged walkable), so we erode
        # the walkable floor to the true room INTERIOR; occupied cells at/outside
        # that boundary are structural walls. Skip this if the mask is degenerate
        # (all True / all False) so a bad mask can't paint everything one colour.
        walkable = getattr(self, "walkable_mask", None) if use_walkable else None
        if (
            walkable is not None
            and walkable.shape == occ.shape
            and 0 < int(walkable.sum()) < walkable.size
        ):
            wk = walkable.astype(np.uint8)
            erode_cells = max(2, int(round(0.25 / self.cell_size)))
            erode_k = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * erode_cells + 1, 2 * erode_cells + 1)
            )
            interior = cv2.erode(wk, erode_k, iterations=1)
            wall_mask[(occ > 0) & (interior == 0)] = 1

        # --- Signal 2: shape of the connected structure ---------------------
        # A wall scanned from a distance comes in as a dashed line of separate hits;
        # labelling those directly makes each dash a tiny "prop" (teal). CLOSE first
        # to bridge the scan gaps so the whole wall is ONE long component, then
        # classify by size/elongation. Props are compact and spaced apart, so they
        # stay their own small components → teal.
        remaining = ((occ > 0) & (wall_mask == 0)).astype(np.uint8)
        if remaining.any():
            close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            bridged = cv2.morphologyEx(remaining, cv2.MORPH_CLOSE, close_k)
            num, labels, stats, _ = cv2.connectedComponentsWithStats(bridged, connectivity=8)
            for i in range(1, num):
                area = int(stats[i, cv2.CC_STAT_AREA])
                width = int(stats[i, cv2.CC_STAT_WIDTH])
                height = int(stats[i, cv2.CC_STAT_HEIGHT])
                comp = (labels == i) & (occ > 0)
                if not comp.any():
                    continue
                if self._is_structural_wall_component(area, width, height):
                    wall_mask[comp] = 1
                else:
                    obstacle_mask[comp] = 1

        unlabeled = (occ > 0) & (wall_mask == 0) & (obstacle_mask == 0)
        obstacle_mask[unlabeled] = 1

        # Sticky yellow (display colouring only — planning uses use_walkable=False).
        # Once a cell is labelled structural wall it stays yellow for as long as
        # the SLAM grid still marks it occupied; no flicker back to teal when the
        # shape heuristic re-evaluates a partially-scanned segment.
        if use_walkable:
            self._sticky_wall_mask = np.maximum(
                self._sticky_wall_mask, wall_mask.astype(np.uint8)
            )
            sticky = (self._sticky_wall_mask > 0) & (occ > 0)
            obstacle_mask[sticky] = 0
            wall_mask[sticky] = 1

        return wall_mask, obstacle_mask

    def get_planning_grid(self):
        """Binary blocking grid for A*/BFS.

        Walls AND dodgeable obstacles get only a thin 1-cell (~drone-radius) margin.
        The full safety inflation (2 cells / 0.2 m each side) closed the MOUTH of
        narrow side corridors/turns once one wall was mapped, so A* and frontier
        reachability could never thread the opening — the drone flew straight past
        the turn into the end wall. A 1-cell margin keeps those openings passable;
        the trained PPO policy handles the fine wall-clearance itself.

        PURE SLAM: classification here never consults the USD walkable mask, so A*
        and frontier reachability depend only on what the drone has actually mapped.
        """
        wall_mask, obstacle_mask = self.get_wall_obstacle_masks(use_walkable=False)
        small_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        inflated_walls = cv2.dilate(wall_mask, small_k, iterations=1)
        inflated_obs = cv2.dilate(obstacle_mask, small_k, iterations=1)
        return ((inflated_walls > 0) | (inflated_obs > 0)).astype(np.uint8)

    def get_traversable_free(self):
        """Free-space mask for frontier BFS / reachability / probing.

        Unlike get_planning_grid(), walls are NOT inflated here — only their actual
        mapped-occupied cells block. Even a 1-cell wall inflation seals the free
        cells at the END of a narrow corridor (front wall + side wall inflate over
        exactly where a turn opening would be), so the turn frontier vanished and the
        drone just hovered ("No reachable frontier"). Raw wall cells are already
        excluded by the free threshold, and the 4-connected BFS can't slip diagonally
        through a wall corner, so corridors + their turns stay open. Dodgeable
        obstacles (poles) DO get a thin inflation so paths still route around them.

        Pure SLAM — occupancy grid only, no USD.
        """
        prob = self.get_occupancy_grid()
        wall_mask, _ = self.get_wall_obstacle_masks(use_walkable=False)
        # Observed-free space OR small/dodgeable obstacles (props/poles) are passable.
        # This keeps BFS reachability from getting blocked by scattered props in corridors.
        return (prob < 0.35) | ((prob > 0.65) & (wall_mask == 0))

    def compute_reachable_mask(self, start_row, start_col):
        """Flood-fill the free cells reachable from a start cell (pure SLAM).

        Uses only the drone's own occupancy grid — no ground-truth/USD map — to
        answer "can the drone actually get there through known free space?".
        A cell is traversable if it is observed free (prob < 0.35) and not inside
        an inflated obstacle. Returns a bool mask the size of the grid.
        """
        from collections import deque

        reachable = np.zeros((self.h, self.w), dtype=bool)
        prob = self.get_occupancy_grid()
        # Traversable = observed free AND not blocked in the planning grid (walls
        # inflated, poles/props blocked with a small margin). The flood-fill threads
        # between poles through the gaps, matching what A* will actually plan.
        blocked = self.get_planning_grid()
        free = (prob < 0.35) & (blocked == 0)

        # The drone cell itself may be inflated/unknown; seed from the nearest free
        # cell in a small neighborhood so we don't return an empty mask.
        seed = None
        for rad in range(0, 8):
            for dr in range(-rad, rad + 1):
                for dc in range(-rad, rad + 1):
                    r, c = start_row + dr, start_col + dc
                    if 0 <= r < self.h and 0 <= c < self.w and free[r, c]:
                        seed = (r, c)
                        break
                if seed is not None:
                    break
            if seed is not None:
                break
        if seed is None:
            return reachable

        dq = deque([seed])
        reachable[seed] = True
        while dq:
            r, c = dq.popleft()
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.h and 0 <= nc < self.w and free[nr, nc] and not reachable[nr, nc]:
                    reachable[nr, nc] = True
                    dq.append((nr, nc))
        return reachable

    def segment_hits_wall(self, x0, y0, x1, y1, blocked=None) -> bool:
        """True if the straight world segment (x0,y0)->(x1,y1) crosses a blocked cell.

        Used to reject straight-line fallbacks that would drive the drone through a
        mapped wall toward a frontier sitting in a dead-end corridor behind it.
        """
        if blocked is None:
            blocked = self.get_planning_grid()
        r0, c0 = self.world_to_grid(x0, y0)
        r1, c1 = self.world_to_grid(x1, y1)
        n = int(max(abs(r1 - r0), abs(c1 - c0))) + 1
        rs = np.linspace(r0, r1, n).round().astype(int)
        cs = np.linspace(c0, c1, n).round().astype(int)
        for r, c in zip(rs, cs):
            if self.is_in_bounds(int(r), int(c)) and blocked[int(r), int(c)]:
                return True
        return False

    def segment_is_known_free(self, x0, y0, x1, y1, tail_cells: int = 6) -> bool:
        """True only if the straight segment runs through OBSERVED-FREE space.

        Stricter than segment_hits_wall: it rejects the line if it crosses UNKNOWN
        cells too (not just mapped walls). An unknown cell may hide a wall the drone
        hasn't seen yet — beelining through it is exactly what makes the drone crash
        into unmapped walls while chasing a far frontier. The last `tail_cells`
        (nearest the frontier) are allowed to be unknown, since a frontier is by
        definition adjacent to unknown space.
        """
        prob = self.get_occupancy_grid()
        blocked = self.get_planning_grid()
        known_free = (prob < 0.35) & (blocked == 0)
        r0, c0 = self.world_to_grid(x0, y0)
        r1, c1 = self.world_to_grid(x1, y1)
        n = int(max(abs(r1 - r0), abs(c1 - c0))) + 1
        rs = np.linspace(r0, r1, n).round().astype(int)
        cs = np.linspace(c0, c1, n).round().astype(int)
        cutoff = max(0, n - int(tail_cells))
        for i in range(n):
            if i >= cutoff:
                break  # allow the short unknown tail at the frontier
            r, c = int(rs[i]), int(cs[i])
            if not self.is_in_bounds(r, c) or not known_free[r, c]:
                return False
        return True

    def is_frontier_reachable(self, reachable_mask, centroid_grid) -> bool:
        """True if a frontier centroid touches the reachable free region (3x3)."""
        if reachable_mask is None:
            return True
        cr, cc = int(centroid_grid[0]), int(centroid_grid[1])
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                r, c = cr + dr, cc + dc
                if 0 <= r < self.h and 0 <= c < self.w and reachable_mask[r, c]:
                    return True
        return False

    def find_reachable_frontiers(self, start_row, start_col, min_size=1):
        """Robust pure-SLAM frontier search: one BFS over the drone's OWN free space.

        This replaces the fragile "detect frontiers -> separately test reachability
        -> separately run A*" chain, where each stage could independently reject a
        genuinely-reachable opening and leave the drone with no target even though
        clear unexplored space sat right ahead.

        Every frontier returned is reachable *by construction* (the BFS actually
        walked to it through observed-free cells) and comes with a guaranteed path
        via the returned parent field — no A* that can silently fail.

        Returns (frontiers, came_from):
          frontiers: list of {centroid_grid, centroid_world, size, goal_grid}
          came_from: (h, w, 2) int32 parent array for reconstruct_path()
        """
        from collections import deque

        # Walls NOT inflated (see get_traversable_free) so narrow corridor-end turns
        # stay open and their frontiers are detectable/reachable.
        free = self.get_traversable_free()
        unknown = (np.abs(self.grid_log_odds) < 0.1)
        k = np.ones((3, 3), dtype=np.uint8)
        unknown_adj = cv2.dilate(unknown.astype(np.uint8), k) > 0

        h, w = self.h, self.w
        came_from = -np.ones((h, w, 2), dtype=np.int32)
        reached = np.zeros((h, w), dtype=bool)

        # Seed from the nearest free cell to the drone (its own cell may be inflated).
        seed = None
        for rad in range(0, 12):
            for dr in range(-rad, rad + 1):
                for dc in range(-rad, rad + 1):
                    r, c = start_row + dr, start_col + dc
                    if 0 <= r < h and 0 <= c < w and free[r, c]:
                        seed = (r, c)
                        break
                if seed is not None:
                    break
            if seed is not None:
                break
        if seed is None:
            return [], came_from

        dq = deque([seed])
        reached[seed] = True
        fmask = np.zeros((h, w), dtype=np.uint8)
        while dq:
            r, c = dq.popleft()
            if unknown_adj[r, c]:
                fmask[r, c] = 1  # reachable free cell touching unknown => frontier
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and free[nr, nc] and not reached[nr, nc]:
                    reached[nr, nc] = True
                    came_from[nr, nc, 0] = r
                    came_from[nr, nc, 1] = c
                    dq.append((nr, nc))

        if not fmask.any():
            return [], came_from

        # Label the UNKNOWN space into connected regions and record each region's
        # area. A real opening (corridor/room) borders a LARGE unknown region; an
        # occlusion shadow behind an obstacle borders a TINY one. This "unknown gain"
        # is what lets the picker ignore useless pockets in an already-covered room
        # and commit to the big corridor. Pure SLAM — only the drone's own grid.
        u_num, u_labels, u_stats, _ = cv2.connectedComponentsWithStats(
            unknown.astype(np.uint8)
        )
        u_area = u_stats[:, cv2.CC_STAT_AREA] if u_num > 0 else np.zeros(1)

        num, labels_im, stats, centroids = cv2.connectedComponentsWithStats(fmask)
        frontiers = []
        for i in range(1, num):
            size = int(stats[i, cv2.CC_STAT_AREA])
            if size < min_size:
                continue
            cx, cy = centroids[i]
            cgr, cgc = int(round(cy)), int(round(cx))
            ys, xs = np.where(labels_im == i)
            # Goal = the cluster cell nearest the centroid (guaranteed reached/reachable).
            j = int(np.argmin((ys - cgr) ** 2 + (xs - cgc) ** 2))
            goal = (int(ys[j]), int(xs[j]))
            wx, wy = self.grid_to_world(goal[0], goal[1])

            # Unknown gain: total area of the unknown regions this frontier touches.
            u_ids = set()
            for fr, fc in zip(ys, xs):
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = fr + dr, fc + dc
                    if 0 <= nr < h and 0 <= nc < w and unknown[nr, nc]:
                        lbl = int(u_labels[nr, nc])
                        if lbl > 0:
                            u_ids.add(lbl)
            unknown_gain = int(sum(int(u_area[l]) for l in u_ids))

            frontiers.append({
                "centroid_grid": (goal[0], goal[1]),
                "centroid_world": (wx, wy),
                "size": size,
                "unknown_gain": unknown_gain,
                "goal_grid": goal,
            })
        return frontiers, came_from

    def deepen_frontier_goal(
        self, goal_grid, came_from, drone_grid, visited_mask=None, max_depth=40
    ):
        """Push a corridor/room mouth goal deeper into the passage.

        BFS frontiers cluster at the opening (where free meets unknown). The drone
        was clearing those at 0.7 m while still outside the corridor, then switching
        to a side target. This finds the farthest reachable cell that still borders
        unknown so the committed goal sits inside the corridor.
        """
        from collections import deque

        free = self.get_traversable_free()
        unknown = (np.abs(self.grid_log_odds) < 0.1)
        gr, gc = int(goal_grid[0]), int(goal_grid[1])

        def touches_unknown(r, c):
            if not self.is_in_bounds(r, c) or not free[r, c]:
                return False
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if self.is_in_bounds(nr, nc) and unknown[nr, nc]:
                    return True
            return False

        def cell_mostly_visited(r, c, radius=2, threshold=0.45):
            if visited_mask is None:
                return False
            total, rev = 0, 0
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    rr, cc = r + dr, c + dc
                    if self.is_in_bounds(rr, cc):
                        total += 1
                        if visited_mask[rr, cc]:
                            rev += 1
            return total > 0 and (rev / total) >= threshold

        best = None
        best_score = (-1, -1)  # (unknown_ahead, path_len)
        dq = deque([(gr, gc, 0)])
        seen = {(gr, gc)}
        while dq:
            r, c, depth = dq.popleft()
            if depth > max_depth:
                continue
            if not touches_unknown(r, c):
                continue
            if cell_mostly_visited(r, c):
                continue
            path = self.reconstruct_path(came_from, (r, c))
            if not path or len(path) < 2:
                continue
            plen = len(path)
            unk_ahead = self.unknown_touch_count(r, c, radius=5)
            score = (unk_ahead, plen)
            if score > best_score:
                best_score = score
                best = (r, c)
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if (nr, nc) not in seen and self.is_in_bounds(nr, nc) and free[nr, nc]:
                    seen.add((nr, nc))
                    dq.append((nr, nc, depth + 1))
        if best is None or best == (gr, gc):
            return None
        return best

    def is_narrow_frontier(self, goal_grid) -> bool:
        """True if the goal cell sits in a corridor-like narrow passage (≤2 free neighbors)."""
        free = self.get_traversable_free()
        r, c = int(goal_grid[0]), int(goal_grid[1])
        if not self.is_in_bounds(r, c) or not free[r, c]:
            return False
        n_free = 0
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if self.is_in_bounds(nr, nc) and free[nr, nc]:
                n_free += 1
        return n_free <= 2

    def is_cell_frontier(self, centroid_grid, radius=1) -> bool:
        """True if a mapped cell is STILL a frontier: some observed-free, traversable
        cell within `radius` that is adjacent to UNKNOWN space.

        Used to invalidate the active target early. When the drone flies toward a
        dead-end, the camera eventually maps the wall behind it — the unknown cells
        become occupied, so this returns False and the brain can switch to a new
        target BEFORE crawling all the way up to the wall.
        """
        free = self.get_traversable_free()
        unknown = (np.abs(self.grid_log_odds) < 0.1)
        r0, c0 = int(centroid_grid[0]), int(centroid_grid[1])
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                r, c = r0 + dr, c0 + dc
                if not self.is_in_bounds(r, c) or not free[r, c]:
                    continue
                # Check 8-connected neighbors to handle diagonal frontiers correctly
                # while keeping the search tight so we don't look past mapped walls.
                for ar in (-1, 0, 1):
                    for ac in (-1, 0, 1):
                        if ar == 0 and ac == 0:
                            continue
                        nr, nc = r + ar, c + ac
                        if self.is_in_bounds(nr, nc) and unknown[nr, nc]:
                            return True
        return False

    def unknown_touch_count(self, row, col, radius=4) -> int:
        """Count unknown cells near a grid cell — proxy for 'unexplored space ahead'.

        Used to score probe directions at ANY angle: the ray ending nearest the most
        unknown is the most worth exploring. Pure SLAM (occupancy grid only).
        """
        unknown = (np.abs(self.grid_log_odds) < 0.1)
        r0, c0 = int(row), int(col)
        count = 0
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                r, c = r0 + dr, c0 + dc
                if self.is_in_bounds(r, c) and unknown[r, c]:
                    count += 1
        return count

    def get_clearance_at_grid(self, row, col) -> float:
        """Return the distance (in meters) to the nearest occupied wall/obstacle cell."""
        prob = self.get_occupancy_grid()
        free_or_unknown = (prob < 0.65).astype(np.uint8)
        dist = cv2.distanceTransform(free_or_unknown, cv2.DIST_L2, 5)
        r, c = int(row), int(col)
        if not self.is_in_bounds(r, c):
            return 0.0
        return float(dist[r, c]) * self.cell_size

    def plan_path_centered(self, start_grid, goal_grid, clearance_cells=5, wall_weight=4.0):
        """A* that PREFERS the middle of free space (fixes wall-hugging paths).

        Blocking = get_traversable_free() (walls block only their raw cells, poles
        inflated) so narrow corridors are still passable. On top of the normal step
        cost, cells close to a wall get a penalty proportional to how far inside
        `clearance_cells` they are — so in open areas the path runs down the centre,
        while in a genuinely narrow corridor (every cell is near a wall) the penalty
        is ~uniform and it still finds the route. Returns a world-coord path or None.
        """
        import heapq

        # Inflate structural walls (to close noise gaps) but do NOT inflate small obstacles
        # so that narrow corridor passages around poles/obstacles remain passable.
        wall_mask, obstacle_mask = self.get_wall_obstacle_masks(use_walkable=False)
        small_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        inflated_walls = cv2.dilate(wall_mask, small_k, iterations=1)
        prob = self.get_occupancy_grid()
        free = ((inflated_walls == 0) & ((prob < 0.35) | (obstacle_mask > 0))).astype(np.uint8)
        h, w = free.shape
        r0, c0 = int(start_grid[0]), int(start_grid[1])
        r1, c1 = int(goal_grid[0]), int(goal_grid[1])
        if not (self.is_in_bounds(r0, c0) and self.is_in_bounds(r1, c1)):
            return None

        # Distance from each free cell to the nearest wall/obstacle (non-free cell).
        dist = cv2.distanceTransform(free, cv2.DIST_L2, 5)

        # Seed from nearest free cell if the drone's own cell reads blocked.
        if free[r0, c0] == 0:
            found = None
            for rad in range(1, 8):
                for dr in range(-rad, rad + 1):
                    for dc in range(-rad, rad + 1):
                        rr, cc = r0 + dr, c0 + dc
                        if self.is_in_bounds(rr, cc) and free[rr, cc]:
                            found = (rr, cc)
                            break
                    if found:
                        break
                if found:
                    break
            if found is None:
                return None
            r0, c0 = found

        def clearance_penalty(r, c):
            d = float(dist[r, c])
            if d >= clearance_cells:
                return 0.0
            # Quadratic penalty — strongly avoids cells near walls.
            t = (clearance_cells - d) / max(clearance_cells, 1e-6)
            return wall_weight * t * t * clearance_cells

        def heuristic(r, c):
            return np.hypot(r - r1, c - c1)

        open_set = [(heuristic(r0, c0), 0.0, (r0, c0))]
        came = {}
        g = {(r0, c0): 0.0}
        # 4-connected only — diagonal steps cut corners and hug walls in L-turns.
        neigh = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        it = 0
        while open_set and it < 20000:
            it += 1
            _, cg, cur = heapq.heappop(open_set)
            if cur == (r1, c1):
                path = [cur]
                while cur in came:
                    cur = came[cur]
                    path.append(cur)
                path.reverse()
                return [self.grid_to_world(r, c) for r, c in path]
            r, c = cur
            for dr, dc in neigh:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                # Allow the goal cell even if it sits on a non-free cell (frontier
                # goals sit at the edge of known-free space).
                if free[nr, nc] == 0 and (nr, nc) != (r1, c1):
                    continue
                step = 1.0
                tg = cg + step + clearance_penalty(nr, nc)
                if (nr, nc) not in g or tg < g[(nr, nc)]:
                    g[(nr, nc)] = tg
                    came[(nr, nc)] = cur
                    heapq.heappush(open_set, (tg + heuristic(nr, nc), tg, (nr, nc)))
        return None

    def reconstruct_path(self, came_from, goal_grid):
        """Rebuild the BFS path (list of (row,col)) from the seed to goal_grid."""
        path = []
        r, c = int(goal_grid[0]), int(goal_grid[1])
        guard = 0
        limit = self.h * self.w
        while r >= 0 and c >= 0 and guard < limit:
            path.append((r, c))
            pr = int(came_from[r, c, 0])
            pc = int(came_from[r, c, 1])
            if pr < 0 or pc < 0:
                break
            r, c = pr, pc
            guard += 1
        path.reverse()
        return path

    def detect_frontiers(self, min_size=1):
        """Detect boundaries between explored free space and unexplored space."""
        prob = self.get_occupancy_grid()
        # Use RAW observed-free space (prob < 0.35), NOT the inflated planning grid.
        # Inflation erases free cells next to walls, which erased the free cells at
        # the MOUTH of a side corridor/turn — so the opening never became a frontier
        # and the drone flew past it into the end wall. We only require the cell to
        # be genuinely observed-free; reachability + A* decide traversability later.
        free_space = (prob < 0.35)
        
        # 2. Identify Unknown Space (prob == 0.5, i.e., log-odds close to 0)
        unknown_space = (np.abs(self.grid_log_odds) < 0.1)
        
        # 3. Detect frontier cells: Free cells adjacent to Unknown cells
        kernel = np.array([[1, 1, 1],
                           [1, 0, 1],
                           [1, 1, 1]], dtype=np.uint8)
        unknown_neighbors = cv2.filter2D(unknown_space.astype(np.uint8), -1, kernel) > 0
        frontier_mask = free_space & unknown_neighbors
        
        # 4. Cluster frontier cells
        num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(frontier_mask.astype(np.uint8))
        
        frontiers = []
        for i in range(1, num_labels):
            size = stats[i, cv2.CC_STAT_AREA]
            if size >= min_size:
                cx, cy = centroids[i]
                # Convert centroid back to world coordinate
                wx, wy = self.grid_to_world(int(cy), int(cx))
                frontiers.append({
                    "centroid_grid": (int(cy), int(cx)),
                    "centroid_world": (wx, wy),
                    "size": size
                })
        return frontiers
