import numpy as np
import torch
from ultralytics import YOLO

class PerceptionModule:
    def __init__(self, use_mock=False):
        """
        Initialize the perception module.
        :param use_mock: If True, uses dummy data. If False, runs real YOLO.
        """
        self.use_mock = use_mock
        
        if not self.use_mock:
            self.yolo_model = YOLO('yolo11n.pt')

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
        single_env_image = rgb_array[0]

        # ---------------------------------------------------------
        # 2. Run Real YOLO Detection
        # ---------------------------------------------------------
        results = self.yolo_model(single_env_image, verbose=False)
        
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