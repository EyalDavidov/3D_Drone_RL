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

    def get_wall_obstacle_masks(self):
        """Split occupied cells into structural walls vs. dodgeable small obstacles.

        Returns (wall_mask, obstacle_mask) as uint8 grids. Perimeter / house
        boundary cells are always walls (yellow in the 3D view). Compact blobs
        inside walkable rooms are props (teal). Shape heuristics handle the rest.
        """
        prob = self.get_occupancy_grid()
        occ = (prob > 0.65).astype(np.uint8)
        wall_mask = np.zeros_like(occ)
        obstacle_mask = np.zeros_like(occ)
        if not occ.any():
            return wall_mask, obstacle_mask

        # --- Signal 1: static floor boundary (only if the mask is trustworthy) ---
        # Wall hits register AT the floor edge (still flagged walkable), so we erode
        # the walkable floor to the true room INTERIOR; occupied cells at/outside
        # that boundary are structural walls. Skip this if the mask is degenerate
        # (all True / all False) so a bad mask can't paint everything one colour.
        walkable = getattr(self, "walkable_mask", None)
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
        return wall_mask, obstacle_mask

    def get_planning_grid(self):
        """Binary blocking grid for A*/BFS.

        Walls get the full safety inflation. Dodgeable obstacles (poles/props) block
        their own footprint plus a 1-cell drone-radius margin, so A* routes AROUND
        each pole *through the gaps between them* instead of aiming straight at one
        and relying on the policy to weave (the forced-yaw navigator can't). The
        small margin still leaves narrow passages open.
        """
        wall_mask, obstacle_mask = self.get_wall_obstacle_masks()
        inflated_walls = cv2.dilate(wall_mask, self.inflation_kernel, iterations=1)
        small_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        inflated_obs = cv2.dilate(obstacle_mask, small_k, iterations=1)
        return ((inflated_walls > 0) | (inflated_obs > 0)).astype(np.uint8)

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

    def detect_frontiers(self, min_size=5):
        """Detect boundaries between explored free space and unexplored space."""
        prob = self.get_occupancy_grid()
        # Use the WALLS-only planning grid so cells next to dodgeable props still
        # count as free — otherwise inflated prop fields hide the frontiers behind
        # them and the drone never plans past a corridor full of boxes.
        inflated = self.get_planning_grid()
        
        # 1. Identify Free Space (prob < 0.35 and not inflated wall)
        free_space = (prob < 0.35) & (inflated == 0)
        
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
