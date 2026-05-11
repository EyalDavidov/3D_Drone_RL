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

    def process_camera_data(self, rgb_image, depth_image):
        """
        Processes images to return target detection and latent representation.
        Expects rgb_image and depth_image as PyTorch Tensors from Isaac Sim.
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
        results = self.yolo_model(single_env_image_bgr, verbose=False)
        
        # Extract detection info and save successful detections
        has_persons = False
        for box in results[0].boxes:
            # Class 0 is 'person'.
            if int(box.cls[0]) == 0 and float(box.conf[0]) > 0.80:
                has_persons = True
                break
        
        self.detection_count += 1
        
        # --- Live OpenCV Window Display ---
        import cv2
        annotated_frame = results[0].plot()  # Returns BGR format array
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
        for box in results[0].boxes:
            if int(box.cls[0]) == 0:  # Class 0 is 'person'
                target_found_batch[0] = True
                x_center, y_center, _, _ = box.xywh[0].tolist()
                target_coords_batch[0, 0] = x_center
                target_coords_batch[0, 1] = y_center
                break 

        # ---------------------------------------------------------
        # 3. VAE Section (Placeholder)
        # ---------------------------------------------------------
        latent_depth_vector = torch.zeros((batch_size, 32), dtype=torch.float32, device=rgb_image.device)
        
        return target_found_batch, target_coords_batch, latent_depth_vector