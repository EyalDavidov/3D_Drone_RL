import numpy as np
from ultralytics import YOLO

class PerceptionModule:
    def __init__(self, use_mock=False):
        """
        Initialize the perception module.
        :param use_mock: If True, uses dummy data. If False, runs real YOLO.
        """
        self.use_mock = use_mock
        
        if not self.use_mock:
            # Load the actual YOLO model (the .pt file you just tested)
            self.yolo_model = YOLO('yolo11n.pt')

    def process_camera_data(self, rgb_image, depth_image):
        """
        Processes images to return target detection and latent representation.
        """
        # --- Logic for RL Team (Niv and Eyal) to use while CV is in progress ---
        if self.use_mock:
            return False, None, np.zeros(32, dtype=np.float32)
        
        # --- Real Computer Vision Logic ---
        # 1. Run YOLO detection on the RGB frame
        # verbose=False prevents the console from being flooded with logs
        results = self.yolo_model(rgb_image, verbose=False)
        
        target_found = False
        target_coords = None
        
        # 2. Extract detection results
        for box in results[0].boxes:
            # Class 0 is 'person' in COCO dataset
            if int(box.cls[0]) == 0:
                target_found = True
                # Get center coordinates (x, y)
                x_center, y_center, _, _ = box.xywh[0].tolist()
                target_coords = (x_center, y_center)
                break # Stop after finding the first person
                
        # --- VAE Section (Placeholder for now) ---
        # Soon, we will add the VAE compression here to handle the depth_image
        latent_depth_vector = np.zeros(32, dtype=np.float32)
        
        return target_found, target_coords, latent_depth_vector