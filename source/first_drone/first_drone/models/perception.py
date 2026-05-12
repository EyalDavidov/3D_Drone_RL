import numpy as np
import torch
import os
from ultralytics import YOLO
from pathlib import Path

class PerceptionModule:
    def __init__(self, use_mock=False):
        """
        Initialize the perception module.
        :param use_mock: If True, uses dummy data. If False, runs real YOLO.
        """
        self.use_mock = use_mock
        self.detection_count = 0  # Counter for saved detections
        
        # Create output directory for successful detections
        self.output_dir = Path("debug_yolo_detections")
        self.output_dir.mkdir(exist_ok=True)
        
        if not self.use_mock:
            # נתיב אבסולוטי כדי להבטיח שהוא מוצא את המודל ללא קשר מאיפה מריצים
            yolo_path = r'D:\isaac\3D_Drone_RL\YOLO\yolo11n.pt'
            self.yolo_model = YOLO(yolo_path)

    def process_camera_data(self, rgb_image, depth_image, drone_pos=None, drone_quat=None):
        """
        Processes images to return target detection and latent representation.
        Expects rgb_image and depth_image as PyTorch Tensors from Isaac Sim.
        Optionally takes drone_pos and drone_quat to calculate true global GPS coordinates.
        """
        
        # We extract the batch size (number of drones) from the tensor shape
        # Typically rgb_image shape is (num_envs, height, width, channels)
        batch_size = rgb_image.shape[0] if isinstance(rgb_image, torch.Tensor) else 1
        
        if self.use_mock:
            # RL Team expects batch outputs, so we return tensors matching num_envs
            target_found = torch.zeros(batch_size, dtype=torch.bool, device=rgb_image.device)
            target_coords = torch.zeros((batch_size, 2), dtype=torch.float32, device=rgb_image.device)
            latent_depth_vector = torch.zeros((batch_size, 32), dtype=torch.float32, device=rgb_image.device)
            return target_found, target_coords, latent_depth_vector

        # ---------------------------------------------------------
        # 1. Pre-process the Tensor for YOLO
        # ---------------------------------------------------------
        # Convert PyTorch Tensor to NumPy array on CPU
        if isinstance(rgb_image, torch.Tensor):
            rgb_array = rgb_image.detach().cpu().numpy()
        else:
            rgb_array = rgb_image

        # Isaac Sim outputs RGBA (4 channels). YOLO needs RGB (3 channels).
        # We slice the array to keep only the first 3 channels.
        if rgb_array.shape[-1] == 4:
            rgb_array = rgb_array[..., :3]

        # Isaac Sim often outputs float32 values between 0.0 and 1.0.
        # YOLO expects standard image formats with values 0 to 255.
        if rgb_array.dtype == np.float32 or rgb_array.dtype == np.float64:
            if rgb_array.max() <= 1.0:
                rgb_array = (rgb_array * 255.0).astype(np.uint8)

        # IMPORTANT: Running YOLO on 4096 images simultaneously will crash the GPU memory.
        # For this integration phase, we will only run YOLO on the camera of drone 0.
        single_env_image_rgb = rgb_array[0]
        # YOLO (via Ultralytics) expects numpy arrays in BGR format (like OpenCV).
        # We must convert Isaac Sim's RGB array to BGR before passing it to the model.
        single_env_image_bgr = single_env_image_rgb[:, :, ::-1]

        # ---------------------------------------------------------
        # 2. Run Real YOLO Detection
        # ---------------------------------------------------------
        # Run YOLO with a base threshold of 0.50 to filter out weak detections natively
        results = self.yolo_model(single_env_image_bgr, verbose=False, conf=0.50)
        
        # We keep all objects because the model internally filtered EVERYTHING > 0.50
        filtered_results = results[0]
        
        # We need to collect the custom text overlays to draw them on the frame later
        custom_texts = []
        
        # Check if we found any persons for RL and saving logic
        has_persons = False
        for box in filtered_results.boxes:
            if int(box.cls[0]) == 0:
                has_persons = True
                
                # We do a quick pre-calculation here to get coordinates for the OpenCV frame
                x_center, y_center, _, _ = box.xywh[0].tolist()
                px = max(0, min(int(x_center), single_env_image_bgr.shape[1] - 1))
                py = max(0, min(int(y_center), single_env_image_bgr.shape[0] - 1))
                
                # Fetch depth
                if isinstance(depth_image, torch.Tensor):
                    depth_array = depth_image[0].detach().cpu().numpy()
                else:
                    depth_array = depth_image[0]
                z_depth = float(np.squeeze(depth_array[py, px]))
                if np.isinf(z_depth): z_depth = 10.0
                
                # Optional: Calculate SLAM coordinates to display
                if drone_pos is not None and drone_quat is not None:
                    fx = single_env_image_bgr.shape[1] * (24.0 / 20.955)
                    cx = single_env_image_bgr.shape[1] / 2.0
                    cy = single_env_image_bgr.shape[0] / 2.0
                    local_x = (x_center - cx) * z_depth / fx
                    local_y = (y_center - cy) * z_depth / fx
                    
                    d_x = float(drone_pos[0, 0].item())
                    d_y = float(drone_pos[0, 1].item())
                    d_z = float(drone_pos[0, 2].item())
                    qw = float(drone_quat[0, 0].item())
                    qx = float(drone_quat[0, 1].item())
                    qy = float(drone_quat[0, 2].item())
                    qz = float(drone_quat[0, 3].item())
                    
                    # --- Full 3D Rotation Matrix (Fix for Pitch/Roll during flight) ---
                    # Instead of just Yaw, we use the full quaternion to rotate the camera vector.
                    from scipy.spatial.transform import Rotation as R
                    rot = R.from_quat([qx, qy, qz, qw])  # scipy expects [x, y, z, w]
                    
                    # Local camera vector: 
                    # X_drone(Forward) = z_depth
                    # Y_drone(Left)    = -local_x
                    # Z_drone(Up)      = -local_y
                    cam_vector = np.array([z_depth, -local_x, -local_y])
                    
                    # Rotate the vector to world coordinates
                    world_vector = rot.apply(cam_vector)
                    
                    t_x = d_x + world_vector[0]
                    t_y = d_y + world_vector[1]
                    t_z = d_z + world_vector[2]
                    # ------------------------------------------------------------------
                    
                    # Convert Local Cartesian (meters) to Simulated GPS (Decimal Degrees)
                    # Assuming origin (0,0) is at Lat=32.1234, Lon=34.1234
                    anchor_lat, anchor_lon = 32.1234, 34.1234
                    lat_offset_per_m = 1.0 / 111320.0
                    lon_offset_per_m = 1.0 / (111320.0 * math.cos(math.radians(anchor_lat)))
                    
                    # Assuming X is North(+) and Y is East(+)
                    target_lat = anchor_lat + (t_x * lat_offset_per_m)
                    target_lon = anchor_lon + (t_y * lon_offset_per_m)
                    
                    # Save text and position (placing it near the bounding box)
                    # YOLO's label is about 25 pixels tall sitting exactly above base_y.
                    # We push ours to 35px and 55px above to strictly avoid overlap.
                    bbox_x1, bbox_y1, _, _ = box.xyxy[0].tolist()
                    base_x = int(bbox_x1)
                    base_y = int(bbox_y1)
                    
                    gps_str = f"GPS: {target_lat:.6f}, {target_lon:.6f}"
                    local_str = f"XYZ: X:{t_x:.1f} Y:{t_y:.1f} Z:{t_z:.1f}"
                    
                    custom_texts.append((gps_str, base_x, max(20, base_y - 55)))
                    custom_texts.append((local_str, base_x, max(40, base_y - 35)))
                    
        self.detection_count += 1
        
        # --- Live OpenCV Window Display ---
        import cv2
        annotated_frame = filtered_results.plot()  # Returns BGR format array with ONLY filtered boxes
        
        # Draw our custom Rescue Coordinates on the frame!
        for txt, p_x, p_y in custom_texts:
            # Color is BGR, so (0, 0, 255) is Red. OpenCV text thickness 2 for better visibility.
            cv2.putText(annotated_frame, txt, (p_x, p_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
        cv2.imshow("Isaac Sim POV - Live YOLO", annotated_frame)
        cv2.waitKey(1) # 1 millisecond delay to update the live frame
        # ----------------------------------
        
        # User requested to only save up to 3 successful pictures to avoid spam
        if not hasattr(self, 'saved_good_pictures'):
            self.saved_good_pictures = 0
            
        if has_persons and self.saved_good_pictures < 3:
            self.saved_good_pictures += 1
            output_path = self.output_dir / f"detection_{self.saved_good_pictures:02d}.jpg"
            # Convert BGR to RGB for proper color display
            annotated_frame_rgb = annotated_frame[:, :, ::-1]
            # Save using PIL or opencv
            from PIL import Image
            Image.fromarray(annotated_frame_rgb).save(str(output_path))
            print(f"[YOLO] ✓ Detected persons! Saved: {output_path} ({self.saved_good_pictures}/3)")
        
        # Prepare output tensors for the RL team
        target_found_batch = torch.zeros(batch_size, dtype=torch.bool, device=rgb_image.device)
        target_coords_batch = torch.zeros((batch_size, 2), dtype=torch.float32, device=rgb_image.device)
        
        # Extract detection results for drone 0
        for box in filtered_results.boxes:
            if int(box.cls[0]) == 0:  # Class 0 is 'person'
                target_found_batch[0] = True
                
                # --- Step 1: Pixel center ---
                x_center, y_center, _, _ = box.xywh[0].tolist()
                px = int(x_center)
                py = int(y_center)
                
                # Bounds check
                img_h, img_w = single_env_image_bgr.shape[:2]
                px = max(0, min(px, img_w - 1))
                py = max(0, min(py, img_h - 1))
                
                # --- Step 2: Depth Sampling ---
                if isinstance(depth_image, torch.Tensor):
                    depth_array = depth_image[0].detach().cpu().numpy()
                else:
                    depth_array = depth_image[0]
                
                # Extract Z depth (meters) from the matrix at [py, px]
                z_depth = float(np.squeeze(depth_array[py, px]))
                if np.isinf(z_depth): 
                    z_depth = 10.0  # Handle sky/infinity
                    
                # --- Step 3: De-projection to Meters ---
                # Calculate focal length in pixels using Isaac Sim pinhole physics
                # fx = width * (focal_length_mm / horizontal_aperture_mm)
                # Configuration values: focus=24.0, aperture=20.955
                fx = img_w * (24.0 / 20.955)
                fy = fx  # Square pixels assumption
                cx = img_w / 2.0
                cy = img_h / 2.0
                
                # Mathematics for projection
                local_x = (x_center - cx) * z_depth / fx
                local_y = (y_center - cy) * z_depth / fy
                
                # --- Step 4: Calculate Coordinates Relative to Entrance (SLAM Simulation) ---
                global_msg = ""
                if drone_pos is not None and drone_quat is not None:
                    # In a GPS-denied disaster zone, the drone uses SLAM/VIO (Visual Inertial Odometry)
                    # to track its own movement relative to its starting point (The Entrance).
                    # Here, the physics engine perfectly simulates the perfect SLAM tracking relative to origin.
                    
                    # Grab drone's SLAM-tracked coordinates (meters from entrance)
                    d_x = float(drone_pos[0, 0].item())
                    d_y = float(drone_pos[0, 1].item())
                    d_z = float(drone_pos[0, 2].item())
                    
                    # Grab drone's quaternion (w, x, y, z)
                    qw = float(drone_quat[0, 0].item())
                    qx = float(drone_quat[0, 1].item())
                    qy = float(drone_quat[0, 2].item())
                    qz = float(drone_quat[0, 3].item())
                    
                    # Convert quaternion to yaw angle (heading)
                    import math
                    yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
                    
                    # Calculate target's position RELATIVE TO THE ENTRANCE (0,0,0)
                    target_x_entrance = d_x + (z_depth * math.cos(yaw)) - (local_x * math.sin(yaw))
                    target_y_entrance = d_y + (z_depth * math.sin(yaw)) + (local_x * math.cos(yaw))
                    target_z_entrance = d_z - local_y  # Camera Y is down, so we subtract from drone Z height
                    
                    global_msg = f"\n   ↳ [RESCUE COORDS] Target is {target_x_entrance:.1f}m Forward, {target_y_entrance:.1f}m Right, and {target_z_entrance:.1f}m High relative to the Building Entrance!"
                
                # --- Step 5: Logging ---
                print(f"[ALARM] Person found! Dist: {z_depth:.2f}m, Local X: {local_x:.2f}m{global_msg}")
                
                # ---------------------------------------------
                
                target_coords_batch[0, 0] = x_center
                target_coords_batch[0, 1] = y_center
                break 

        # ---------------------------------------------------------
        # 3. VAE Section (Placeholder)
        # ---------------------------------------------------------
        latent_depth_vector = torch.zeros((batch_size, 32), dtype=torch.float32, device=rgb_image.device)
        
        return target_found_batch, target_coords_batch, latent_depth_vector