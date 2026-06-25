import math
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

_REPO_ROOT = Path(__file__).resolve().parents[5]


class PerceptionModule:
    def __init__(
        self,
        use_mock=False,
        person_conf_threshold: float = 0.95,
        min_bbox_area_frac: float = 0.008,
        min_depth_m: float = 0.8,
        max_depth_m: float = 25.0,
    ):
        """Initialize YOLO-based person detection for the Brain module."""
        self.use_mock = use_mock
        self.person_conf_threshold = float(person_conf_threshold)
        self.min_bbox_area_frac = float(min_bbox_area_frac)
        self.min_depth_m = float(min_depth_m)
        self.max_depth_m = float(max_depth_m)
        self.detection_count = 0
        self.last_best_person_conf = 0.0
        self.output_dir = Path("debug_yolo_detections")
        self.output_dir.mkdir(exist_ok=True)

        if not self.use_mock:
            yolo_path = _REPO_ROOT / "YOLO" / "yolo11n.pt"
            if not yolo_path.exists():
                yolo_path = Path(r"D:\isaac\3D_Drone_RL\YOLO\yolo11n.pt")
            self.yolo_model = YOLO(str(yolo_path))
            print(
                f"[Perception] YOLO person-only mode: conf>={self.person_conf_threshold:.0%}, "
                f"min_bbox_area={self.min_bbox_area_frac:.1%}\n"
            )

    def process_camera_data(self, rgb_image, depth_image, drone_pos=None, drone_quat=None):
        """Run YOLO detection and depth-based 3D localization.

        Only class-0 (person) detections above ``person_conf_threshold`` are accepted.

        Returns:
            person_found: bool tensor of shape (num_envs,)
            person_world_xyz: float tensor of shape (num_envs, 3) in world frame
        """
        batch_size = rgb_image.shape[0] if isinstance(rgb_image, torch.Tensor) else 1
        device = rgb_image.device if isinstance(rgb_image, torch.Tensor) else "cpu"

        person_found = torch.zeros(batch_size, dtype=torch.bool, device=device)
        person_world_xyz = torch.zeros((batch_size, 3), dtype=torch.float32, device=device)

        if self.use_mock:
            return person_found, person_world_xyz

        rgb_array = rgb_image.detach().cpu().numpy() if isinstance(rgb_image, torch.Tensor) else rgb_image
        if rgb_array.shape[-1] == 4:
            rgb_array = rgb_array[..., :3]
        if rgb_array.dtype in (np.float32, np.float64) and rgb_array.max() <= 1.0:
            rgb_array = (rgb_array * 255.0).astype(np.uint8)

        single_env_image_bgr = rgb_array[0][:, :, ::-1]
        img_h, img_w = single_env_image_bgr.shape[:2]
        img_area = float(img_h * img_w)

        # Person class only — run at low conf to report best candidate, accept only above threshold.
        results = self.yolo_model(
            single_env_image_bgr,
            verbose=False,
            conf=0.20,
            classes=[0],
        )
        filtered_results = results[0]

        custom_texts = []
        has_confirmed_person = False
        best_person = None  # (conf, box, world_xyz, z_depth, local_x, local_y)
        best_person_conf = 0.0

        for box in filtered_results.boxes:
            if int(box.cls[0]) != 0:
                continue

            conf = float(box.conf[0].item())
            if int(box.cls[0]) == 0:
                best_person_conf = max(best_person_conf, conf)
            if conf < self.person_conf_threshold:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            bbox_area_frac = max(0.0, (x2 - x1) * (y2 - y1)) / img_area
            if bbox_area_frac < self.min_bbox_area_frac:
                continue

            x_center, y_center, _, _ = box.xywh[0].tolist()
            px = max(0, min(int(x_center), img_w - 1))
            py = max(0, min(int(y_center), img_h - 1))

            if isinstance(depth_image, torch.Tensor):
                depth_array = depth_image[0].detach().cpu().numpy()
            else:
                depth_array = depth_image[0]
            z_depth = float(np.squeeze(depth_array[py, px]))
            if np.isinf(z_depth) or z_depth <= 0.0:
                z_depth = 10.0
            if z_depth < self.min_depth_m or z_depth > self.max_depth_m:
                continue

            if drone_pos is None or drone_quat is None:
                continue

            fx = img_w * (24.0 / 20.955)
            fy = fx
            cx = img_w / 2.0
            cy = img_h / 2.0
            local_x = (x_center - cx) * z_depth / fx
            local_y = (y_center - cy) * z_depth / fy

            d_x = float(drone_pos[0, 0].item())
            d_y = float(drone_pos[0, 1].item())
            d_z = float(drone_pos[0, 2].item())
            qw = float(drone_quat[0, 0].item())
            qx = float(drone_quat[0, 1].item())
            qy = float(drone_quat[0, 2].item())
            qz = float(drone_quat[0, 3].item())

            from scipy.spatial.transform import Rotation as R

            rot = R.from_quat([qx, qy, qz, qw])
            cam_vector = np.array([z_depth, -local_x, -local_y])
            world_vector = rot.apply(cam_vector)

            t_x = d_x + world_vector[0]
            t_y = d_y + world_vector[1]
            t_z = d_z + world_vector[2]

            candidate = (conf, box, (t_x, t_y, t_z), z_depth, local_x, local_y)
            if best_person is None or conf > best_person[0]:
                best_person = candidate

        if best_person is not None:
            conf, box, (t_x, t_y, t_z), z_depth, local_x, local_y = best_person
            has_confirmed_person = True
            person_found[0] = True
            person_world_xyz[0, 0] = t_x
            person_world_xyz[0, 1] = t_y
            person_world_xyz[0, 2] = t_z

            anchor_lat, anchor_lon = 32.1234, 34.1234
            lat_offset_per_m = 1.0 / 111320.0
            lon_offset_per_m = 1.0 / (111320.0 * math.cos(math.radians(anchor_lat)))
            target_lat = anchor_lat + (t_x * lat_offset_per_m)
            target_lon = anchor_lon + (t_y * lon_offset_per_m)

            bbox_x1, bbox_y1, _, _ = box.xyxy[0].tolist()
            base_x = int(bbox_x1)
            base_y = int(bbox_y1)
            custom_texts.append((f"PERSON {conf:.0%}", base_x, max(5, base_y - 70)))
            custom_texts.append((f"GPS: {target_lat:.6f}, {target_lon:.6f}", base_x, max(20, base_y - 55)))
            custom_texts.append((f"XYZ: X:{t_x:.1f} Y:{t_y:.1f} Z:{t_z:.1f}", base_x, max(40, base_y - 35)))

            print(
                f"[ALARM] Person confirmed ({conf:.0%})! Dist: {z_depth:.2f}m, Local X: {local_x:.2f}m\n"
                f"   ↳ [RESCUE COORDS] Target is {t_x:.1f}m Forward, {t_y:.1f}m Right, "
                f"and {t_z:.1f}m High relative to the Building Entrance!"
            )

        self.detection_count += 1
        self.last_best_person_conf = best_person_conf

        if self.detection_count % 100 == 0:
            if has_confirmed_person:
                print(f"[YOLO] Person CONFIRMED at {best_person_conf:.0%} (threshold {self.person_conf_threshold:.0%})")
            elif best_person_conf > 0.0:
                print(
                    f"[YOLO] Person seen but REJECTED: best conf={best_person_conf:.0%} "
                    f"(need {self.person_conf_threshold:.0%})"
                )
            else:
                print(f"[YOLO] No person in view (threshold {self.person_conf_threshold:.0%})")

        try:
            import cv2

            annotated_frame = filtered_results.plot()
            status_line = (
                f"Person {best_person_conf:.0%} (need {self.person_conf_threshold:.0%})"
                if best_person_conf > 0.0
                else f"No person (need {self.person_conf_threshold:.0%})"
            )
            cv2.putText(
                annotated_frame, status_line, (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2,
            )
            for txt, p_x, p_y in custom_texts:
                cv2.putText(annotated_frame, txt, (p_x, p_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.imshow("Isaac Sim POV - Live YOLO", annotated_frame)
            cv2.waitKey(1)
        except Exception:
            annotated_frame = filtered_results.plot()
            for txt, p_x, p_y in custom_texts:
                try:
                    import cv2

                    cv2.putText(annotated_frame, txt, (p_x, p_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                except Exception:
                    pass

        if self.detection_count % 100 == 0:
            diag_path = self.output_dir / f"diagnostic_{self.detection_count:04d}.jpg"
            try:
                from PIL import Image

                Image.fromarray(annotated_frame[:, :, ::-1]).save(str(diag_path))
                print(f"[YOLO DEBUG] Saved diagnostic frame: {diag_path}")
            except Exception as e:
                print(f"[YOLO DEBUG] Failed to save diagnostic frame: {e}")

        if not hasattr(self, "saved_good_pictures"):
            self.saved_good_pictures = 0
        if has_confirmed_person and self.saved_good_pictures < 3:
            self.saved_good_pictures += 1
            output_path = self.output_dir / f"detection_{self.saved_good_pictures:02d}.jpg"
            try:
                from PIL import Image

                Image.fromarray(annotated_frame[:, :, ::-1]).save(str(output_path))
                print(f"[YOLO] High-confidence person saved: {output_path} ({self.saved_good_pictures}/3)")
            except Exception as e:
                print(f"[YOLO] Failed to save detection image: {e}")

        return person_found, person_world_xyz
