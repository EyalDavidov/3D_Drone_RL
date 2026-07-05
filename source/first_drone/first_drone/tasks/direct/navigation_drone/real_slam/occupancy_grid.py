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
        """Generate binary grid with expanded obstacles for safety."""
        prob = self.get_occupancy_grid()
        # Binary occupied mask
        binary_occupied = (prob > 0.65).astype(np.uint8)
        # Dilate
        inflated = cv2.dilate(binary_occupied, self.inflation_kernel, iterations=1)
        return inflated

    def detect_frontiers(self, min_size=5):
        """Detect boundaries between explored free space and unexplored space."""
        prob = self.get_occupancy_grid()
        inflated = self.get_inflated_grid()
        
        # 1. Identify Free Space (prob < 0.35 and not inflated obstacle)
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
