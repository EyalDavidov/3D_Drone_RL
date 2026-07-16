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
        noted_conf_threshold: float = 0.60,
        min_bbox_area_frac: float = 0.0001,
        min_bbox_height_frac: float = 0.02,
        min_person_aspect: float = 1.65,
        noted_confirm_frames: int = 3,
        min_depth_m: float = 0.8,
        max_depth_m: float = 25.0,
        camera_focal_length: float = 7.5,
        camera_horizontal_aperture: float = 20.955,
        yolo_imgsz: int = 640,
        yolo_camera_upscale: int = 2,
        yolo_sharpen: bool = True,
        rescue_person_slots: list[dict] | None = None,
        person_match_radius: float = 5.0,
        yolo_clahe: bool = False,
        show_opencv: bool = True,
    ):
        """Initialize YOLO-based person detection for the Brain module."""
        self.use_mock = use_mock
        self.show_opencv = bool(show_opencv)
        self.yolo_clahe = bool(yolo_clahe)
        self.person_conf_threshold = float(person_conf_threshold)
        self.noted_conf_threshold = float(noted_conf_threshold)
        self.min_bbox_area_frac = float(min_bbox_area_frac)
        self.min_bbox_height_frac = float(min_bbox_height_frac)
        self.min_person_aspect = float(min_person_aspect)
        self.noted_confirm_frames = max(1, int(noted_confirm_frames))
        self.min_depth_m = float(min_depth_m)
        self.max_depth_m = float(max_depth_m)
        self.camera_focal_length = float(camera_focal_length)
        self.camera_horizontal_aperture = float(camera_horizontal_aperture)
        self.yolo_imgsz = int(yolo_imgsz)
        self.yolo_camera_upscale = max(1, int(yolo_camera_upscale))
        self.yolo_sharpen = bool(yolo_sharpen)
        self.detection_count = 0
        self.last_best_person_conf = 0.0
        self.person_ever_detected = False
        self.output_dir = Path("debug_yolo_detections")
        self.output_dir.mkdir(exist_ok=True)
        # Clear yolo_saves and debug_yolo_detections directories on initialization to clean up old runs
        try:
            saves_dir = Path(r"D:\isaac\3D_Drone_RL\scripts\dashboard\static\yolo_saves")
            if saves_dir.exists():
                for f in saves_dir.glob("*.jpg"):
                    try:
                        f.unlink()
                    except Exception:
                        pass
            for f in self.output_dir.glob("*.jpg"):
                try:
                    f.unlink()
                except Exception:
                    pass
            print("[Perception] Cleared old YOLO detection image saves.")
        except Exception as e_clear:
            print(f"[Perception] Warning: failed to clear old saves: {e_clear}")
        self._display_initialized = False
        self._alert_window_initialized = False
        # ── Layout constants ──────────────────────────────────────────────
        self._video_width = 680
        self._sidebar_width = 320
        self._display_width = self._video_width + self._sidebar_width
        self._display_height = 580
        self._header_h = 52
        self._footer_h = 60
        self._video_area_h = self._display_height - self._header_h - self._footer_h
        self._sidebar_card_h = 100
        # ─────────────────────────────────────────────────────────────────
        self._detection_log: list[dict] = []
        self._person_best_conf: dict[str, float] = {}
        self.frame_confirmed_persons: list[dict] = []
        self._sidebar_scroll = 0
        self._sidebar_trackbar_ready = False
        self._alert_width = 1280
        self._alert_height = 720
        self._simple_view_width = 480
        self._simple_view_height = 270
        self._window_name = "Brain Nav - YOLO"
        self._alert_window_name = "RESCUE ALERT - Person Detected"
        self._last_display_frame = None
        self._display_error_logged = False
        # ── Cyber-Industrial HUD palette (BGR for OpenCV) ─────────────────
        # Mirrors the web dashboard's glassmorphism theme: deep navy-black
        # surfaces, dim cyan strokes, and neon cyan/magenta/lime/amber accents.
        self._hc = {
            "bg":        (26, 15, 10),    # deep navy-black base
            "panel":     (38, 22, 15),    # header / footer / sidebar bars
            "card":      (52, 32, 22),    # rescue-log cards
            "bar_bg":    (50, 34, 24),    # meter track
            "edge":      (86, 64, 40),    # dim cyan-blue stroke
            "edge_lit":  (200, 168, 60),  # lit cyan stroke
            "cyan":      (255, 243, 47),  # neon cyan  (#2ff3ff)
            "magenta":   (149, 45, 255),  # neon magenta (#ff2d95)
            "lime":      (60, 255, 182),  # neon lime  (#b6ff3c)
            "amber":     (32, 176, 255),  # neon amber (#ffb020)
            "text":      (255, 252, 234), # near-white
            "text_dim":  (184, 163, 148), # secondary
            "text_muted":(139, 116, 100), # muted labels
        }
        self._noted_streak = 0
        self._active_scan_label: str | None = None
        self._operator_alert_until = 0
        self._last_alert_frame = None
        self._last_intel: dict | None = None
        self._rescue_person_slots = list(rescue_person_slots or [])
        self._person_match_radius = float(person_match_radius)
        # ── Native web-HUD payload (rendered by the browser, not OpenCV) ──
        # Published every YOLO pass so the dashboard's native component can draw
        # a fully synced HUD: same frame the boxes were computed on.
        self._web_frame_bgr: np.ndarray | None = None
        self._web_frame_left_bgr: np.ndarray | None = None
        self._web_frame_right_bgr: np.ndarray | None = None
        self._web_boxes: list[dict] = []
        self._web_boxes_left: list[dict] = []
        self._web_boxes_right: list[dict] = []
        self._web_state: dict | None = None

        if not self.use_mock:
            yolo_path = _REPO_ROOT / "YOLO" / "yolo11n.pt"
            if not yolo_path.exists():
                yolo_path = Path(r"D:\isaac\3D_Drone_RL\YOLO\yolo11n.pt")
            self._yolo_device = "cuda:0" if torch.cuda.is_available() else "cpu"
            self.yolo_model = YOLO(str(yolo_path))
            if self._yolo_device.startswith("cuda"):
                self.yolo_model.to(self._yolo_device)
            print(
                f"[Perception] YOLO on {self._yolo_device} | rescue>={self.person_conf_threshold:.0%}, "
                f"noted>={self.noted_conf_threshold:.0%}, imgsz={self.yolo_imgsz}, "
                f"upscale={self.yolo_camera_upscale}x, confirm_frames={self.noted_confirm_frames}\n"
            )
        else:
            self._yolo_device = "cpu"

    def _bbox_passes_person_shape(self, box, img_w: int, img_h: int) -> bool:
        """Verify that the detected bbox matches standing human proportions."""
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        return self._bbox_passes_person_shape_xy(x1, y1, x2, y2, img_w, img_h)

    @staticmethod
    def _box_xyxy_scaled(box, scale: float) -> tuple[float, float, float, float]:
        s = float(scale)
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        return x1 * s, y1 * s, x2 * s, y2 * s

    def _bbox_passes_person_shape_xy(
        self, x1: float, y1: float, x2: float, y2: float, img_w: int, img_h: int
    ) -> bool:
        bw = max(0.0, x2 - x1)
        bh = max(0.0, y2 - y1)
        if bw < 1.0 or bh < 2.0:
            return False

        # 1. Height fraction check (must not be a tiny sliver)
        h_frac = bh / img_h
        if h_frac < self.min_bbox_height_frac:
            return False

        # 2. Area fraction check
        area_frac = (bw * bh) / (img_w * img_h)
        if area_frac < self.min_bbox_area_frac:
            return False

        aspect = bw / bh

        # 3. Min aspect ratio — reject thin poles/pillars (aspect << 0.12)
        # A real human standing: aspect ~ 0.25–0.80 (width/height)
        # A cylindrical pole from drone view: aspect ~ 0.03–0.10
        if aspect < 0.12:
            return False

        # 4. Max aspect ratio — reject very wide/horizontal shapes
        # Relax this limit to 3.0 if the detection is close-up (occupies >= 25% of screen height)
        max_aspect = 3.0 if h_frac >= 0.25 else self.min_person_aspect
        if aspect > max_aspect:
            return False

        return True

    def _init_alert_window(self) -> None:
        if self._alert_window_initialized:
            return
        try:
            import cv2

            cv2.namedWindow(self._alert_window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self._alert_window_name, self._alert_width, self._alert_height)
            cv2.moveWindow(self._alert_window_name, 680, 40)
            self._alert_window_initialized = True
        except Exception:
            pass

    def _trigger_operator_alert(self, frame_bgr: np.ndarray, conf: float, scan_label: str | None) -> None:
        """Console banner for operator; visual alert stays in the main YOLO HUD window."""
        self._operator_alert_until = self.detection_count + 120
        self._last_alert_frame = frame_bgr.copy()
        where = (scan_label or "person detected").upper()
        banner = (
            f"\n{'=' * 62}\n"
            f"  *** PERSON DETECTED  |  {conf:.0%} confidence  |  {where} ***\n"
            f"  See Brain Nav - YOLO window. Saved to debug_yolo_detections/\n"
            f"{'=' * 62}\n"
        )
        print(banner)

    def _sidebar_visible_cards(self) -> int:
        usable = self._display_height - self._header_h - 40
        return max(1, usable // self._sidebar_card_h)

    def _on_sidebar_scroll(self, val: int) -> None:
        self._sidebar_scroll = int(val)

    @staticmethod
    def _local_xyz_to_gps(x: float, y: float) -> tuple[float, float]:
        anchor_lat, anchor_lon = 32.1234, 34.1234
        lat = anchor_lat + (x / 111320.0)
        lon = anchor_lon + (y / (111320.0 * math.cos(math.radians(anchor_lat))))
        return lat, lon

    def _deproject_bbox_center(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        img_w: int,
        img_h: int,
        depth_image,
        drone_pos,
        drone_quat,
        *,
        relax_depth: bool = False,
    ) -> tuple[tuple[float, float, float], float, float, float] | None:
        """Return (world_xyz, z_depth, local_x, local_y) from a person bbox."""
        if drone_pos is None or drone_quat is None:
            return None

        x_center = 0.5 * (x1 + x2)
        y_center = 0.5 * (y1 + y2)
        px = max(0, min(int(x_center), img_w - 1))
        py = max(0, min(int(y_center), img_h - 1))

        if depth_image is not None:
            if isinstance(depth_image, torch.Tensor):
                depth_array = depth_image[0].detach().cpu().numpy()
            else:
                depth_array = depth_image[0]
            
            depth_array = np.squeeze(depth_array)
            
            # Robust neighborhood depth sampling (5x5 window) to prevent de-projection
            # from hitting background walls/void when querying thin/side-profile mannequins
            r_start = max(0, py - 2)
            r_end = min(img_h, py + 3)
            c_start = max(0, px - 2)
            c_end = min(img_w, px + 3)
            
            neighborhood = depth_array[r_start:r_end, c_start:c_end]
            valid_mask = (neighborhood > 0.05) & (~np.isinf(neighborhood)) & (~np.isnan(neighborhood))
            if np.any(valid_mask):
                # 15th percentile filters single-pixel noise but picks the foreground object
                z_depth = float(np.percentile(neighborhood[valid_mask], 15))
            else:
                z_depth = float(depth_array[py, px])
        else:
            z_depth = 5.0  # fallback when side camera has no depth
        if np.isinf(z_depth) or z_depth <= 0.0:
            z_depth = 10.0
        if relax_depth:
            z_depth = float(np.clip(z_depth, self.min_depth_m, self.max_depth_m))
        elif z_depth < self.min_depth_m or z_depth > self.max_depth_m:
            return None

        fx = img_w * (self.camera_focal_length / self.camera_horizontal_aperture)
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
        return (t_x, t_y, t_z), z_depth, local_x, local_y

    def _match_rescue_person_slot(
        self, xyz: tuple[float, float, float] | None
    ) -> dict | None:
        """Map a detection to one of the rescue-person slots (dynamically created if none matches)."""
        if xyz is None:
            return None
        
        # Convert detection coordinates to GPS lat/lon
        lat, lon = self._local_xyz_to_gps(float(xyz[0]), float(xyz[1]))
        
        best_slot = None
        # Match if the GPS coordinates are within 0.0001 degrees (4 decimal places)
        limit_gps = 0.0001
        best_diff = limit_gps + 0.00001
        
        if self._rescue_person_slots:
            for slot in self._rescue_person_slots:
                slat, slon = self._local_xyz_to_gps(float(slot["xyz"][0]), float(slot["xyz"][1]))
                lat_diff = abs(lat - slat)
                lon_diff = abs(lon - slon)
                if lat_diff < limit_gps and lon_diff < limit_gps:
                    total_diff = lat_diff + lon_diff
                    if total_diff < best_diff:
                        best_diff = total_diff
                        best_slot = slot
                        
        if best_slot is not None:
            return best_slot
            
        # Create a new dynamic slot
        new_idx = len(self._rescue_person_slots) + 1
        new_slot = {
            "id": f"person_{new_idx}",
            "xyz": (float(xyz[0]), float(xyz[1]), float(xyz[2])),
            "label": f"Person {new_idx}"
        }
        self._rescue_person_slots.append(new_slot)
        return new_slot



    def _person_log_key_from_world_xy(self, x: float, y: float) -> str:
        """Stable log id for a distinct world position (~2 m cells)."""
        bx = int(round(float(x) / 2.0))
        by = int(round(float(y) / 2.0))
        return f"pos_{bx}_{by}"

    def _dedupe_confirmed_detections(
        self, items: list[dict], merge_dist: float = 1.5
    ) -> list[dict]:
        """Merge same-frame duplicates from multiple cameras / boxes."""
        cam_rank = {"front": 0, "left": 1, "right": 1}

        def _rank(cam: str | None) -> int:
            return cam_rank.get(str(cam or "front"), 2)

        merged: list[dict] = []
        for item in sorted(items, key=lambda d: -float(d.get("conf", 0.0))):
            xyz = item.get("xyz")
            if xyz is None:
                continue
            ix, iy = float(xyz[0]), float(xyz[1])
            dup_idx = None
            for i, kept in enumerate(merged):
                kxyz = kept.get("xyz")
                if kxyz is None:
                    continue
                if math.hypot(ix - float(kxyz[0]), iy - float(kxyz[1])) < merge_dist:
                    dup_idx = i
                    break
            if dup_idx is not None:
                kept = merged[dup_idx]
                best_conf = max(float(kept.get("conf", 0.0)), float(item.get("conf", 0.0)))
                # Keep the highest confidence, but prefer front-cam XYZ when the same
                # person is seen from multiple cameras (side cams often lack depth).
                if _rank(item.get("cam")) < _rank(kept.get("cam")):
                    merged[dup_idx] = {**item, "conf": best_conf}
                else:
                    merged[dup_idx] = {**kept, "conf": best_conf}
            else:
                merged.append(item)
        return merged

    def _person_log_key(
        self, scan_label: str | None, xyz: tuple[float, float, float] | None
    ) -> str:
        """One log row per physical rescue person — keyed by configured slot, not scan label."""
        slot = self._match_rescue_person_slot(xyz)
        if slot is not None:
            return str(slot["id"])
        if xyz is not None:
            return self._person_log_key_from_world_xy(float(xyz[0]), float(xyz[1]))
        label = (scan_label or "person_detected").strip().lower().replace(" ", "_")
        return label

    def _append_detection_log(
        self,
        conf: float,
        xyz: tuple[float, float, float] | None,
        scan_label: str | None,
        *,
        frame_idx: int,
    ) -> bool:
        """Upsert one log row per person — keep only the best confidence for that person."""
        person_key = self._person_log_key(scan_label, xyz)
        prev_best = self._person_best_conf.get(person_key, 0.0)
        if float(conf) <= prev_best + 1e-6:
            return False

        gps_lat, gps_lon = (None, None)
        if xyz is not None:
            gps_lat, gps_lon = self._local_xyz_to_gps(float(xyz[0]), float(xyz[1]))

        slot = self._match_rescue_person_slot(xyz)
        display_label = (
            str(slot["label"]).upper()
            if slot is not None
            else (scan_label or "person detected").upper()
        )

        entry = {
            "person_key": person_key,
            "conf": float(conf),
            "xyz": xyz,
            "gps_lat": gps_lat,
            "gps_lon": gps_lon,
            "label": display_label,
            "frame": int(frame_idx),
        }
        self._person_best_conf[person_key] = float(conf)

        replaced = False
        for i, existing in enumerate(self._detection_log):
            if existing.get("person_key") == person_key:
                self._detection_log[i] = entry
                replaced = True
                break
        if not replaced:
            self._detection_log.insert(0, entry)
        self._detection_log.sort(key=lambda e: (-float(e["conf"]), -int(e["frame"])))
        self._detection_log = self._detection_log[:30]
        self._sidebar_scroll = 0
        return True

    def _sync_sidebar_trackbar(self) -> None:
        # No trackbar — max 3 rescue persons always fit in the sidebar.
        pass

    # ── colour helpers ────────────────────────────────────────────────────
    def _conf_color(self, conf: float) -> tuple[int, int, int]:
        """Neon BGR that steps lime → cyan → amber as confidence rises."""
        hc = self._hc
        if conf >= 0.80:
            return hc["lime"]
        if conf >= 0.60:
            return hc["cyan"]
        return hc["amber"]

    def _slot_accent(self, person_key: str) -> tuple[int, int, int]:
        hc = self._hc
        palette = [hc["cyan"], hc["lime"], hc["amber"], hc["magenta"]]
        idx = hash(person_key) % len(palette)
        return palette[idx]

    # ── drawing primitives ────────────────────────────────────────────────
    @staticmethod
    def _filled_pill(canvas, x0: int, y0: int, x1: int, y1: int,
                     color: tuple, *, alpha: float = 1.0) -> None:
        import cv2

        r = (y1 - y0) // 2
        if alpha < 1.0:
            roi = canvas[y0:y1, x0:x1].copy()
            overlay = roi.copy()
            cv2.rectangle(overlay, (r, 0), (x1 - x0 - r, y1 - y0), color, -1)
            cv2.circle(overlay, (r, r), r, color, -1)
            cv2.circle(overlay, (x1 - x0 - r, r), r, color, -1)
            canvas[y0:y1, x0:x1] = cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0)
        else:
            cv2.rectangle(canvas, (x0 + r, y0), (x1 - r, y1), color, -1)
            cv2.circle(canvas, (x0 + r, y0 + r), r, color, -1)
            cv2.circle(canvas, (x1 - r, y0 + r), r, color, -1)

    @staticmethod
    def _draw_corner_brackets(canvas, x0: int, y0: int, x1: int, y1: int,
                              color: tuple, *, arm: int = 20, thickness: int = 2) -> None:
        """Neon corner brackets (HUD framing) around a rectangle."""
        import cv2

        arm = max(4, min(arm, (x1 - x0) // 3, (y1 - y0) // 3))
        for (cx, cy, dx, dy) in [
            (x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)
        ]:
            cv2.line(canvas, (cx, cy), (cx + dx * arm, cy), color, thickness, cv2.LINE_AA)
            cv2.line(canvas, (cx, cy), (cx, cy + dy * arm), color, thickness, cv2.LINE_AA)

    @staticmethod
    def _center_text(canvas, text: str, x0: int, w: int, y: int,
                     scale: float, color: tuple) -> None:
        """Draw horizontally centered text within [x0, x0+w]."""
        import cv2

        tw = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0]
        cv2.putText(canvas, text, (x0 + (w - tw) // 2, y),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)

    @staticmethod
    def _pill_outline(canvas, x0: int, y0: int, x1: int, y1: int, color: tuple) -> None:
        """Rounded outline matching _filled_pill geometry."""
        import cv2

        r = (y1 - y0) // 2
        cv2.line(canvas, (x0 + r, y0), (x1 - r, y0), color, 1, cv2.LINE_AA)
        cv2.line(canvas, (x0 + r, y1), (x1 - r, y1), color, 1, cv2.LINE_AA)
        cv2.ellipse(canvas, (x0 + r, y0 + r), (r, r), 90, 90, 270, color, 1, cv2.LINE_AA)
        cv2.ellipse(canvas, (x1 - r, y0 + r), (r, r), 270, 90, 270, color, 1, cv2.LINE_AA)

    def _conf_bar(self, canvas, x0: int, y0: int, w: int, h: int,
                  conf: float, thresh: float,
                  filled_color: tuple, bg_color: tuple | None = None) -> None:
        import cv2

        hc = self._hc
        bg_color = bg_color or hc["bar_bg"]
        cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), bg_color, -1)
        fill = int(w * min(max(conf, 0.0), 1.0))
        if fill > 0:
            cv2.rectangle(canvas, (x0, y0), (x0 + fill, y0 + h), filled_color, -1)
        # Threshold marker (neon cyan tick)
        tx = x0 + int(w * thresh)
        cv2.line(canvas, (tx, y0 - 3), (tx, y0 + h + 3), hc["cyan"], 2)

    def _draw_detection_sidebar(self, canvas: np.ndarray) -> None:
        import cv2

        hc = self._hc
        x0 = self._video_width
        w = self._sidebar_width
        h = self._display_height
        hh = self._header_h

        # Sidebar background (slightly lifted navy) + neon divider from video
        cv2.rectangle(canvas, (x0, 0), (x0 + w - 1, h - 1), hc["panel"], -1)
        cv2.line(canvas, (x0, 0), (x0, h - 1), hc["edge"], 1)
        cv2.line(canvas, (x0 + 1, 0), (x0 + 1, h - 1), hc["cyan"], 1)

        # Sidebar header
        cv2.rectangle(canvas, (x0, 0), (x0 + w - 1, hh - 1), hc["card"], -1)
        cv2.line(canvas, (x0, hh - 1), (x0 + w - 1, hh - 1), hc["edge"], 1)

        # "RESCUE LOG" title with dot indicator
        num = len(self._detection_log)
        cv2.circle(canvas, (x0 + 18, hh // 2), 5,
                   hc["cyan"] if num > 0 else hc["text_muted"], -1, cv2.LINE_AA)
        cv2.putText(
            canvas, "RESCUE LOG", (x0 + 30, hh // 2 + 6),
            cv2.FONT_HERSHEY_DUPLEX, 0.52, hc["text"], 1, cv2.LINE_AA,
        )
        if num > 0:
            badge_txt = str(num)
            tw = cv2.getTextSize(badge_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)[0][0]
            bx = x0 + w - 14 - tw - 10
            self._filled_pill(canvas, bx - 4, hh // 2 - 10, bx + tw + 10, hh // 2 + 10,
                               hc["cyan"])
            cv2.putText(canvas, badge_txt, (bx, hh // 2 + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (12, 18, 26), 1, cv2.LINE_AA)

        # Cards area
        card_top = hh + 6
        visible = self._sidebar_visible_cards()
        start = min(self._sidebar_scroll, max(0, num - visible))
        shown = self._detection_log[start: start + visible]

        if not shown:
            mid_y = card_top + (h - card_top) // 2
            cv2.circle(canvas, (x0 + w // 2, mid_y - 40), 16, hc["edge"], 1, cv2.LINE_AA)
            cv2.circle(canvas, (x0 + w // 2, mid_y - 40), 3, hc["edge"], -1, cv2.LINE_AA)
            self._center_text(canvas, "NO DETECTIONS YET", x0, w, mid_y - 6,
                              0.48, hc["text_dim"])
            self._center_text(canvas, "Awaiting scan...", x0, w, mid_y + 16,
                              0.40, hc["text_muted"])
            return

        card_h = self._sidebar_card_h
        for i, entry in enumerate(shown):
            cy0 = card_top + i * card_h + 4
            cy1 = cy0 + card_h - 8
            cx0 = x0 + 10
            cx1 = x0 + w - 10

            # Card background (frosted navy) + neon top hairline + edge
            cv2.rectangle(canvas, (cx0, cy0), (cx1, cy1), hc["card"], -1)
            cv2.rectangle(canvas, (cx0, cy0), (cx1, cy1), hc["edge"], 1)

            # Left accent stripe keyed to the person slot
            accent = self._slot_accent(entry.get("person_key", ""))
            cv2.rectangle(canvas, (cx0, cy0), (cx0 + 4, cy1), accent, -1)

            conf = float(entry["conf"])
            conf_pct = int(round(conf * 100))
            c_col = self._conf_color(conf)

            # ── Row 1: label (left) + conf badge (right)
            label_txt = entry["label"][:20]
            cv2.putText(canvas, label_txt, (cx0 + 14, cy0 + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, hc["text"], 1, cv2.LINE_AA)

            badge_txt = f"{conf_pct}%"
            btw = cv2.getTextSize(badge_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)[0][0]
            bpx = cx1 - btw - 16
            py0b, py1b = cy0 + 8, cy0 + 28
            self._filled_pill(canvas, bpx - 6, py0b, bpx + btw + 8, py1b, c_col)
            cv2.putText(canvas, badge_txt, (bpx, py0b + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (12, 18, 12), 1, cv2.LINE_AA)

            # ── Row 2: thin confidence bar
            bar_y = cy0 + 34
            self._conf_bar(canvas, cx0 + 14, bar_y, cx1 - cx0 - 22, 6,
                           conf, self.person_conf_threshold, c_col)

            # ── Row 3 & 4: GPS coords
            gps_lat = entry.get("gps_lat")
            gps_lon = entry.get("gps_lon")
            xyz = entry.get("xyz")
            y3 = cy0 + 54
            y4 = cy0 + 72
            if gps_lat is not None and gps_lon is not None:
                cv2.putText(canvas, "LAT", (cx0 + 14, y3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.36, hc["text_muted"], 1, cv2.LINE_AA)
                cv2.putText(canvas, f"{gps_lat:.6f}", (cx0 + 46, y3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, hc["cyan"], 1, cv2.LINE_AA)
                cv2.putText(canvas, "LON", (cx0 + 14, y4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.36, hc["text_muted"], 1, cv2.LINE_AA)
                cv2.putText(canvas, f"{gps_lon:.6f}", (cx0 + 46, y4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, hc["cyan"], 1, cv2.LINE_AA)
            elif xyz is not None:
                xyzstr = f"X{xyz[0]:.1f}  Y{xyz[1]:.1f}  Z{xyz[2]:.1f}"
                cv2.putText(canvas, xyzstr, (cx0 + 14, y3 + 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, hc["cyan"], 1, cv2.LINE_AA)

    def _init_display_window(self) -> None:
        """Create a resizable OpenCV window once."""
        if self._display_initialized:
            return
        try:
            import cv2

            cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self._window_name, self._display_width, self._display_height)
            cv2.moveWindow(self._window_name, 40, 40)
            try:
                cv2.startWindowThread()
            except Exception:
                pass
            self._display_initialized = True
        except Exception as exc:
            if not self._display_error_logged:
                print(f"[Perception] OpenCV window init failed: {exc}")
                self._display_error_logged = True

    def _make_simple_view(self, bgr: np.ndarray) -> np.ndarray:
        import cv2

        return cv2.resize(bgr, (480, 270), interpolation=cv2.INTER_LINEAR)

    def _annotate_detections(
        self,
        bgr: np.ndarray,
        results,
        draw_threshold: float | None = None,
        coord_scale: float = 1.0,
    ) -> np.ndarray:
        """Draw person detections with corner-bracket targeting style."""
        import cv2

        vis = bgr.copy()
        thresh = float(draw_threshold if draw_threshold is not None else self.noted_conf_threshold)
        
        # Determine format: list of keep_candidates vs YOLO Results object
        if isinstance(results, list):
            # Format: (conf, box, x1, y1, x2, y2, scale_type)
            boxes_to_draw = []
            for item in results:
                conf = item[0]
                if conf < 0.15:
                    continue
                x1, y1, x2, y2 = item[2], item[3], item[4], item[5]
                boxes_to_draw.append((conf, x1, y1, x2, y2))
        else:
            if results is None or getattr(results, "boxes", None) is None:
                return vis
            cs = float(coord_scale)
            boxes_to_draw = []
            for box in results.boxes:
                if int(box.cls[0]) != 0:
                    continue
                conf = float(box.conf[0].item())
                if conf < 0.15:
                    continue
                x1, y1, x2, y2 = self._box_xyxy_scaled(box, cs)
                boxes_to_draw.append((conf, x1, y1, x2, y2))

        for conf, x1, y1, x2, y2 in boxes_to_draw:
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            if conf >= self.person_conf_threshold:
                color = self._hc["lime"]
            elif conf >= thresh:
                color = self._hc["cyan"]
            else:
                color = self._hc["amber"]
            # Corner-bracket style instead of full rectangle
            arm = max(6, min((x2 - x1) // 4, (y2 - y1) // 4, 20))
            t = 2
            for (cx, cy, dx, dy) in [
                (x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)
            ]:
                cv2.line(vis, (cx, cy), (cx + dx * arm, cy), color, t)
                cv2.line(vis, (cx, cy), (cx, cy + dy * arm), color, t)
            # Confidence label (bottom-right of bbox, no overlap with top label)
            label = f"{conf:.0%}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            lx = max(x1, x2 - tw - 4)
            ly = min(y2 - 2, vis.shape[0] - 4)
            cv2.rectangle(vis, (lx - 2, ly - th - 2), (lx + tw + 2, ly + 2), (10, 10, 10), -1)
            cv2.putText(vis, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
        return vis

    def _collect_web_boxes(
        self, results, coord_scale: float, img_w: int, img_h: int,
    ) -> list[dict]:
        """Normalized person boxes (0..1) for the native web HUD overlay.

        Mirrors _annotate_detections' selection (person class, conf>=0.15) so the
        browser draws exactly what the OpenCV HUD would, but as crisp vector
        overlays instead of a baked-in bitmap.
        """
        boxes: list[dict] = []
        iw = max(1.0, float(img_w))
        ih = max(1.0, float(img_h))
        
        if isinstance(results, list):
            # Format: (conf, box, x1, y1, x2, y2, scale_type)
            boxes_to_collect = []
            for item in results:
                conf = item[0]
                if conf < 0.15:
                    continue
                x1, y1, x2, y2 = item[2], item[3], item[4], item[5]
                boxes_to_collect.append((conf, x1, y1, x2, y2))
        else:
            if results is None or getattr(results, "boxes", None) is None:
                return boxes
            cs = float(coord_scale)
            boxes_to_collect = []
            for box in results.boxes:
                if int(box.cls[0]) != 0:
                    continue
                conf = float(box.conf[0].item())
                if conf < 0.15:
                    continue
                x1, y1, x2, y2 = self._box_xyxy_scaled(box, cs)
                boxes_to_collect.append((conf, x1, y1, x2, y2))

        for conf, x1, y1, x2, y2 in boxes_to_collect:
            try:
                if conf >= self.person_conf_threshold:
                    tier = "confirmed"
                elif conf >= self.noted_conf_threshold:
                    tier = "noted"
                else:
                    tier = "low"
                boxes.append({
                    "x": round(max(0.0, min(1.0, x1 / iw)), 4),
                    "y": round(max(0.0, min(1.0, y1 / ih)), 4),
                    "w": round(max(0.0, min(1.0, (x2 - x1) / iw)), 4),
                    "h": round(max(0.0, min(1.0, (y2 - y1) / ih)), 4),
                    "conf": round(conf, 4),
                    "tier": tier,
                })
            except Exception:
                continue
        boxes.sort(key=lambda b: -b["conf"])
        return boxes

    def _draw_intel_panel(
        self,
        canvas: np.ndarray,
        intel: dict,
        *,
        video_x0: int,
        video_y0: int,
        video_w: int,
        video_h: int,
    ) -> None:
        """Fixed-position detection info box at the bottom of the video area."""
        import cv2

        if not intel:
            return
        hc = self._hc
        pad_x, pad_y = 12, 8
        line_h = 22
        lines = []
        if intel.get("label"):
            lines.append(("LOC", intel["label"].upper(), hc["text"]))
        if intel.get("conf") is not None:
            lines.append(("CONF", f"{int(intel['conf'] * 100)}%", self._conf_color(intel["conf"])))
        if intel.get("gps_lat") is not None:
            lines.append(("LAT", f"{intel['gps_lat']:.6f}", hc["cyan"]))
            lines.append(("LON", f"{intel['gps_lon']:.6f}", hc["cyan"]))
        if intel.get("dist") is not None:
            lines.append(("DIST", f"{intel['dist']:.1f} m", hc["text_dim"]))

        if not lines:
            return

        panel_w = 214
        panel_h = pad_y * 2 + len(lines) * line_h + 4
        px = video_x0 + 8
        py = video_y0 + video_h - panel_h - 8

        # Frosted-glass background (blend toward deep navy)
        roi = canvas[py: py + panel_h, px: px + panel_w]
        if roi.shape[0] < 1 or roi.shape[1] < 1:
            return
        glass = np.full_like(roi, hc["bg"])
        canvas[py: py + panel_h, px: px + panel_w] = cv2.addWeighted(glass, 0.6, roi, 0.4, 0)

        # Neon left accent + edge
        cv2.rectangle(canvas, (px, py), (px + 3, py + panel_h), hc["cyan"], -1)
        cv2.rectangle(canvas, (px, py), (px + panel_w - 1, py + panel_h - 1), hc["edge"], 1)

        for i, (key, val, col) in enumerate(lines):
            ry = py + pad_y + i * line_h + 14
            cv2.putText(canvas, key, (px + 10, ry),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, hc["text_muted"], 1, cv2.LINE_AA)
            cv2.putText(canvas, val, (px + 52, ry),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)

    def _draw_hud_panel(
        self,
        frame: np.ndarray,
        *,
        best_conf: float,
        has_confirmed: bool,
        detection_count: int,
        custom_texts: list,
        noted_deferred: bool = False,
        has_noted: bool = False,
        rescue_armed: bool = True,
        scan_label: str | None = None,
        operator_alert: bool = False,
        alert_conf: float = 0.0,
    ) -> np.ndarray:
        """Render the full HUD: header · video · intel-panel · footer · sidebar."""
        import cv2

        hh = self._header_h
        fh = self._footer_h
        vw = self._video_width
        dh = self._display_height
        dw = self._display_width

        hc = self._hc
        canvas = np.zeros((dh, dw, 3), dtype=np.uint8)
        canvas[:] = hc["bg"]  # deep navy-black base

        # Subtle vertical gradient wash over the whole canvas for depth
        grad = np.linspace(1.0, 0.72, dh, dtype=np.float32)[:, None, None]
        canvas[:] = np.clip(canvas.astype(np.float32) * grad, 0, 255).astype(np.uint8)

        # ── 1. Scale + place camera feed ──────────────────────────────────
        src_h, src_w = frame.shape[:2]
        avail_h = dh - hh - fh
        scale = min(vw / max(src_w, 1), avail_h / max(src_h, 1))
        view_w = int(src_w * scale)
        view_h = int(src_h * scale)
        mx = (vw - view_w) // 2
        vy0 = hh + (avail_h - view_h) // 2

        upscaled = cv2.resize(frame, (view_w, view_h), interpolation=cv2.INTER_LANCZOS4)
        canvas[vy0: vy0 + view_h, mx: mx + view_w] = upscaled

        # Neon corner brackets around the video feed (HUD framing)
        self._draw_corner_brackets(
            canvas, mx - 2, vy0 - 2, mx + view_w + 1, vy0 + view_h + 1,
            hc["edge_lit"], arm=22, thickness=2,
        )
        cv2.rectangle(canvas, (mx - 2, vy0 - 2), (mx + view_w + 1, vy0 + view_h + 1),
                      hc["edge"], 1)

        # ── 2. Header bar ─────────────────────────────────────────────────
        cv2.rectangle(canvas, (0, 0), (vw - 1, hh - 1), hc["panel"], -1)
        # Neon underline (double stroke → subtle glow)
        cv2.line(canvas, (0, hh - 1), (vw - 1, hh - 1), hc["edge"], 2)
        cv2.line(canvas, (0, hh - 1), (vw - 1, hh - 1), hc["cyan"], 1)

        # Left: app title
        cv2.putText(canvas, "BRAIN NAV", (14, hh - 17),
                    cv2.FONT_HERSHEY_DUPLEX, 0.58, hc["cyan"], 1, cv2.LINE_AA)
        cv2.line(canvas, (128, 12), (128, hh - 12), hc["edge"], 1)
        cv2.putText(canvas, "YOLO11 PERSON DETECTION", (142, hh - 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, hc["text_dim"], 1, cv2.LINE_AA)

        # Right: status pill — cohesive neon states
        if has_confirmed:
            pill_txt  = "TARGET CONFIRMED"
            pill_bg, pill_fg = hc["lime"], (12, 22, 8)
        elif operator_alert or has_noted:
            pct = int(round((alert_conf or best_conf) * 100))
            if (alert_conf or best_conf) >= self.person_conf_threshold:
                pill_txt  = f"HUMAN DETECTED  {pct}%"
                pill_bg, pill_fg = hc["magenta"], (255, 255, 255)
            elif rescue_armed:
                pill_txt  = f"CONTACT  {pct}%"
                pill_bg, pill_fg = hc["amber"], (14, 20, 30)
            else:
                pill_txt  = f"NOTED  {pct}%"
                pill_bg, pill_fg = hc["amber"], (14, 20, 30)
        elif noted_deferred:
            pill_txt  = "NOTED - CONTINUE"
            pill_bg, pill_fg = hc["amber"], (14, 20, 30)
        elif scan_label:
            pill_txt  = f"SCAN  {scan_label.upper()}"
            pill_bg, pill_fg = hc["card"], hc["cyan"]
        else:
            pill_txt  = "SCANNING"
            pill_bg, pill_fg = hc["card"], hc["text_dim"]

        ptw = cv2.getTextSize(pill_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)[0][0]
        pill_w = ptw + 42            # room for a leading status dot
        pill_x1 = vw - 16
        pill_x0 = pill_x1 - pill_w
        pill_y0, pill_y1 = 10, hh - 10
        pill_cy = (pill_y0 + pill_y1) // 2
        # Status dot + pill; dark pills get a neon outline so they read on navy
        dot_col = hc["cyan"] if pill_bg == hc["card"] else pill_bg
        self._filled_pill(canvas, pill_x0, pill_y0, pill_x1, pill_y1, pill_bg)
        if pill_bg == hc["card"]:
            self._pill_outline(canvas, pill_x0, pill_y0, pill_x1, pill_y1, hc["edge"])
        cv2.circle(canvas, (pill_x0 + 16, pill_cy), 4, dot_col, -1, cv2.LINE_AA)
        cv2.putText(canvas, pill_txt, (pill_x0 + 28, pill_y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, pill_fg, 1, cv2.LINE_AA)

        # ── 3. Intel panel (detected person info, fixed bottom-left of video) ──
        if self._last_intel:
            self._draw_intel_panel(
                canvas, self._last_intel,
                video_x0=mx, video_y0=vy0, video_w=view_w, video_h=view_h,
            )

        # ── 4. Operator alert overlay (sleek magenta HUD, no ugly red) ────
        if operator_alert:
            pulse_on = (detection_count // 8) % 2 == 0
            alert_col = hc["lime"] if has_confirmed else hc["magenta"]
            # Neon corner brackets + thin frame around the video (pulses)
            self._draw_corner_brackets(
                canvas, mx - 2, vy0 - 2, mx + view_w + 1, vy0 + view_h + 1,
                alert_col, arm=26, thickness=2 if pulse_on else 1,
            )
            if pulse_on:
                cv2.rectangle(canvas, (mx - 2, vy0 - 2),
                              (mx + view_w + 1, vy0 + view_h + 1), alert_col, 1)

            # Frosted-glass banner strip at the top of the video frame
            banner_h = 30
            bx0, by0 = mx, vy0
            bx1, by1 = mx + view_w, vy0 + banner_h
            roi = canvas[by0:by1, bx0:bx1]
            if roi.shape[0] > 0 and roi.shape[1] > 0:
                overlay = np.full_like(roi, (18, 12, 26))  # deep navy glass
                canvas[by0:by1, bx0:bx1] = cv2.addWeighted(overlay, 0.62, roi, 0.38, 0)
                cv2.line(canvas, (bx0, by1), (bx1, by1), alert_col, 1, cv2.LINE_AA)

            label = "TARGET CONFIRMED" if has_confirmed else "HUMAN DETECTED"
            ac_txt = f"{label}  {int(round(alert_conf * 100))}%"
            if scan_label:
                ac_txt += f"   |   {scan_label.upper()}"
            cv2.circle(canvas, (bx0 + 16, vy0 + banner_h // 2), 5, alert_col, -1, cv2.LINE_AA)
            cv2.putText(canvas, ac_txt, (bx0 + 30, vy0 + 20),
                        cv2.FONT_HERSHEY_DUPLEX, 0.46, hc["text"], 1, cv2.LINE_AA)

        # ── 5. Footer bar ─────────────────────────────────────────────────
        fy0 = dh - fh
        cv2.rectangle(canvas, (0, fy0), (vw - 1, dh - 1), hc["panel"], -1)
        cv2.line(canvas, (0, fy0), (vw - 1, fy0), hc["edge"], 2)
        cv2.line(canvas, (0, fy0), (vw - 1, fy0), hc["cyan"], 1)

        conf_pct = int(round(best_conf * 100))
        thresh_pct = int(round(self.person_conf_threshold * 100))

        # Confidence readout
        bar_color = hc["lime"] if has_confirmed else self._conf_color(best_conf)
        cv2.putText(canvas, f"{conf_pct}%", (14, fy0 + 24),
                    cv2.FONT_HERSHEY_DUPLEX, 0.74, bar_color, 1, cv2.LINE_AA)
        cv2.putText(canvas, "CONF", (16, fy0 + 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, hc["text_muted"], 1, cv2.LINE_AA)

        # Confidence bar (rounded track feel via inner padding)
        bx, by, bw, bh2 = 74, fy0 + 14, vw - 190, 14
        cv2.rectangle(canvas, (bx - 1, by - 1), (bx + bw + 1, by + bh2 + 1), hc["edge"], 1)
        self._conf_bar(canvas, bx, by, bw, bh2, best_conf, self.person_conf_threshold, bar_color)

        # Threshold label below bar
        tx = bx + int(bw * self.person_conf_threshold)
        cv2.putText(canvas, f"{thresh_pct}%", (tx - 10, fy0 + 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, hc["cyan"], 1, cv2.LINE_AA)

        # Frame counter (right side)
        cv2.putText(canvas, f"#{detection_count}", (vw - 100, fy0 + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, hc["text_dim"], 1, cv2.LINE_AA)
        cv2.putText(canvas, "cls:person", (vw - 100, fy0 + 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, hc["text_muted"], 1, cv2.LINE_AA)

        # ── 6. Sidebar ────────────────────────────────────────────────────
        self._draw_detection_sidebar(canvas)

        return canvas

    def _show_detection_window(
        self,
        annotated_frame: np.ndarray,
        *,
        best_conf: float,
        has_confirmed: bool,
        detection_count: int,
        custom_texts: list,
        noted_deferred: bool = False,
        has_noted: bool = False,
        rescue_armed: bool = True,
        scan_label: str | None = None,
        operator_alert: bool = False,
        alert_conf: float = 0.0,
    ) -> None:
        """Display the YOLO feed in a compact HUD window."""
        import cv2

        if not self.show_opencv:
            # Web dashboard renders the HUD — skip expensive OpenCV compositing.
            self._last_display_frame = self._make_simple_view(annotated_frame)
            return

        self._init_display_window()
        self._sync_sidebar_trackbar()
        display = None
        try:
            display = self._draw_hud_panel(
                annotated_frame,
                best_conf=best_conf,
                has_confirmed=has_confirmed,
                detection_count=detection_count,
                custom_texts=custom_texts,
                noted_deferred=noted_deferred,
                has_noted=has_noted,
                rescue_armed=rescue_armed,
                scan_label=scan_label,
                operator_alert=operator_alert,
                alert_conf=alert_conf,
            )
        except Exception as exc:
            if not self._display_error_logged:
                print(f"[Perception] HUD render failed ({exc}), using simple view.")
                self._display_error_logged = True
            display = self._make_simple_view(annotated_frame)

        if display is not None:
            self._last_display_frame = display.copy()

        try:
            cv2.imshow(self._window_name, display)
            cv2.waitKey(1)
        except Exception as exc:
            if not self._display_error_logged:
                print(f"[Perception] OpenCV imshow failed: {exc}")
                self._display_error_logged = True

    def _save_detection_frame(
        self, frame_bgr: np.ndarray, prefix: str, cam: str = ""
    ) -> None:
        """Save an annotated or HUD frame to debug_yolo_detections/ and yolo_saves/."""
        cam_tag = f"_{cam}" if cam else ""
        tag = f"{prefix}_{self.detection_count:06d}{cam_tag}.jpg"
        output_path = self.output_dir / tag
        try:
            from PIL import Image

            img = Image.fromarray(frame_bgr[:, :, ::-1])
            img.save(str(output_path))
            print(f"[YOLO] Saved {prefix} detection: {output_path}")

            # Also save to dashboard saves folder for real-time telemetry
            try:
                dash_dir = Path(r"D:\isaac\3D_Drone_RL\scripts\dashboard\static\yolo_saves")
                subfolder = getattr(self, "yolo_saves_subfolder", "")
                if subfolder:
                    dash_dir = dash_dir / subfolder
                dash_dir.mkdir(parents=True, exist_ok=True)  # ensure it always exists
                dash_path = dash_dir / tag
                img.save(str(dash_path))
            except Exception as e_dash:
                print(f"[YOLO] Warning: failed to copy detection to dashboard: {e_dash}")
        except Exception as exc:
            print(f"[YOLO] Failed to save {prefix} image: {exc}")

    def process_camera_data(
        self,
        rgb_image,
        depth_image,
        drone_pos=None,
        drone_quat=None,
        rescue_armed: bool = True,
        scan_label: str | None = None,
        rgb_left=None,
        depth_left=None,
        rgb_right=None,
        depth_right=None,
    ):
        """Run YOLO detection and depth-based 3D localization across front and side cameras.

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

        self.frame_confirmed_persons = []

        if scan_label and scan_label != self._active_scan_label:
            self._active_scan_label = scan_label
            self._noted_streak = 0
        elif scan_label is None and self._active_scan_label is not None:
            self._active_scan_label = None

        import cv2

        up = self.yolo_camera_upscale
        accept_threshold = (
            self.person_conf_threshold if rescue_armed else self.noted_conf_threshold
        )

        # Helper to process a single camera stream
        def process_single_cam(rgb_img, depth_img, yaw_offset_deg):
            if rgb_img is None:
                return None, None, 0.0, 0.0, [], None, None, None, []
            # depth_img can be None for side cameras — 3D localisation will use a
            # conservative fallback depth (5 m) via relax_depth=True in _deproject_bbox_center

            rgb_arr = rgb_img.detach().cpu().numpy() if isinstance(rgb_img, torch.Tensor) else rgb_img
            if rgb_arr.shape[-1] == 4:
                rgb_arr = rgb_arr[..., :3]

            if self.detection_count % 100 == 0:
                print(f"[YOLO Debug] Input image min={rgb_arr.min():.2f} max={rgb_arr.max():.2f} dtype={rgb_arr.dtype}")

            if rgb_arr.dtype in (np.float32, np.float64):
                if rgb_arr.max() <= 1.0:
                    rgb_arr = np.clip(np.round(rgb_arr * 255.0), 0, 255).astype(np.uint8)
                else:
                    rgb_arr = np.clip(np.round(rgb_arr), 0, 255).astype(np.uint8)
            elif rgb_arr.dtype != np.uint8:
                rgb_arr = rgb_arr.astype(np.uint8)

            img_bgr = rgb_arr[0][:, :, ::-1]
            img_h, img_w = img_bgr.shape[:2]
            img_area = float(img_h * img_w)

            yolo_bgr = img_bgr
            if up > 1:
                yolo_bgr = cv2.resize(
                    img_bgr,
                    (img_w * up, img_h * up),
                    interpolation=cv2.INTER_LINEAR,
                )
            if self.yolo_sharpen:
                blur = cv2.GaussianBlur(yolo_bgr, (0, 0), sigmaX=1.0)
                yolo_bgr = cv2.addWeighted(yolo_bgr, 1.6, blur, -0.6, 0)

            if self.yolo_clahe:
                lab = cv2.cvtColor(yolo_bgr, cv2.COLOR_BGR2LAB)
                l_ch, a_ch, b_ch = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
                l_ch = clahe.apply(l_ch)
                yolo_bgr = cv2.cvtColor(cv2.merge([l_ch, a_ch, b_ch]), cv2.COLOR_LAB2BGR)

            _yolo_dev = getattr(self, "_yolo_device", "cpu")
            _yolo_half = str(_yolo_dev).startswith("cuda")

            # --- Dual-scale YOLO: run at BOTH high-res upscaled and low-res native ---
            results_hi = self.yolo_model(
                yolo_bgr, verbose=False, conf=0.15, classes=[0],
                imgsz=self.yolo_imgsz, device=_yolo_dev, half=_yolo_half,
            )
            results_lo = self.yolo_model(
                img_bgr, verbose=False, conf=0.15, classes=[0],
                imgsz=576, device=_yolo_dev, half=_yolo_half,
            )

            # Define IoU helper locally
            def compute_iou(box1, box2):
                bx1 = max(box1[0], box2[0])
                by1 = max(box1[1], box2[1])
                bx2 = min(box1[2], box2[2])
                by2 = min(box1[3], box2[3])
                inter_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
                box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
                box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
                union_area = box1_area + box2_area - inter_area
                if union_area <= 0:
                    return 0.0
                return inter_area / union_area

            # Collect candidates from both scales
            candidates = []
            
            # 1. High-res upscaled results
            if results_hi and len(results_hi) > 0 and results_hi[0].boxes is not None:
                for box in results_hi[0].boxes:
                    if int(box.cls[0]) != 0:
                        continue
                    conf = float(box.conf[0].item())
                    x1, y1, x2, y2 = self._box_xyxy_scaled(box, 1.0 / float(up) if up > 1 else 1.0)
                    candidates.append((conf, box, x1, y1, x2, y2, f"Upscale {self.yolo_imgsz}"))

            # 2. Native resolution results
            if results_lo and len(results_lo) > 0 and results_lo[0].boxes is not None:
                for box in results_lo[0].boxes:
                    if int(box.cls[0]) != 0:
                        continue
                    conf = float(box.conf[0].item())
                    x1, y1, x2, y2 = self._box_xyxy_scaled(box, 1.0)
                    candidates.append((conf, box, x1, y1, x2, y2, "Native 576"))

            # Sort by confidence descending
            candidates = sorted(candidates, key=lambda x: x[0], reverse=True)

            # Non-Maximum Suppression (NMS)
            keep_candidates = []
            for item in candidates:
                conf, box, x1, y1, x2, y2, scale_label = item
                overlap = False
                for k_conf, k_box, k_x1, k_y1, k_x2, k_y2, k_label in keep_candidates:
                    iou = compute_iou((x1, y1, x2, y2), (k_x1, k_y1, k_x2, k_y2))
                    
                    # Containment check: if one box is mostly inside another
                    area_item = (x2 - x1) * (y2 - y1)
                    area_k = (k_x2 - k_x1) * (k_y2 - k_y1)
                    inter_x1 = max(x1, k_x1)
                    inter_y1 = max(y1, k_y1)
                    inter_x2 = min(x2, k_x2)
                    inter_y2 = min(y2, k_y2)
                    inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
                    min_area = min(area_item, area_k)
                    containment = inter_area / min_area if min_area > 0 else 0.0
                    
                    if iou > 0.45 or containment > 0.70:
                        overlap = True
                        break
                if not overlap:
                    keep_candidates.append(item)

            if keep_candidates:
                print(
                    f"[YOLO Debug] Cam Yaw {yaw_offset_deg}°: Merged YOLO detected {len(keep_candidates)} person(s) "
                    f"with confidences: {['%.1f%% (%s)' % (c[0]*100, c[6]) for c in keep_candidates]}"
                )

            best_person = None
            best_bbox = None
            best_person_conf = 0.0
            raw_yolo_best = 0.0
            confirmed_list: list = []

            # Compute effective quaternion for this camera's yaw offset
            eff_quat = drone_quat
            if yaw_offset_deg != 0.0 and drone_quat is not None:
                from scipy.spatial.transform import Rotation as R
                qw_d = float(drone_quat[0, 0].item())
                qx_d = float(drone_quat[0, 1].item())
                qy_d = float(drone_quat[0, 2].item())
                qz_d = float(drone_quat[0, 3].item())
                rot_drone = R.from_quat([qx_d, qy_d, qz_d, qw_d])
                rot_offset = R.from_euler('z', yaw_offset_deg, degrees=True)
                rot_eff = rot_drone * rot_offset
                q_eff = rot_eff.as_quat()  # [x, y, z, w]
                eff_quat = torch.tensor([[q_eff[3], q_eff[0], q_eff[1], q_eff[2]]], device=drone_quat.device, dtype=drone_quat.dtype)

            for conf, box, x1, y1, x2, y2, scale_label in keep_candidates:
                raw_yolo_best = max(raw_yolo_best, conf)
                if not self._bbox_passes_person_shape_xy(x1, y1, x2, y2, img_w, img_h):
                    bw = max(0.0, x2 - x1)
                    bh = max(0.0, y2 - y1)
                    aspect = bw / bh if bh > 0 else 0
                    print(
                        f"[YOLO Debug] Cam Yaw {yaw_offset_deg}°: Rejected detection (conf {conf:.1%}) "
                        f"due to shape filtering (aspect={bw:.1f}/{bh:.1f}={aspect:.2f}, "
                        f"height_frac={bh/img_h:.3f}, area_frac={(bw*bh)/(img_w*img_h):.5f})"
                    )
                    continue

                best_person_conf = max(best_person_conf, conf)

                if conf >= self.noted_conf_threshold:
                    if best_bbox is None or conf > best_bbox[0]:
                        best_bbox = (conf, box, x1, y1, x2, y2)

                if conf < accept_threshold:
                    continue

                deproj = self._deproject_bbox_center(
                    x1, y1, x2, y2, img_w, img_h, depth_img, drone_pos, eff_quat,
                    relax_depth=True,
                )
                if deproj is None:
                    continue
                (t_x, t_y, t_z), z_depth, local_x, local_y = deproj

                candidate = (conf, (x1, y1, x2, y2), (t_x, t_y, t_z), z_depth, local_x, local_y)
                confirmed_list.append(candidate)
                if best_person is None or conf > best_person[0]:
                    best_person = candidate

            # Build annotated frame
            display_bgr = img_bgr.copy()
            if self.yolo_sharpen:
                blur = cv2.GaussianBlur(display_bgr, (0, 0), sigmaX=1.0)
                display_bgr = cv2.addWeighted(display_bgr, 1.6, blur, -0.6, 0)

            annotated_frame = self._annotate_detections(
                display_bgr,
                keep_candidates,
                draw_threshold=0.15,
            )

            web_boxes = self._collect_web_boxes(keep_candidates, 1.0, img_w, img_h)

            # Compute log_xyz for noted/detected candidates if needed
            log_xyz_local = None
            if best_person is not None:
                log_xyz_local = (float(best_person[2][0]), float(best_person[2][1]), float(best_person[2][2]))
            elif best_bbox is not None:
                _, _, bx1, by1, bx2, by2 = best_bbox
                noted_deproj = self._deproject_bbox_center(
                    bx1, by1, bx2, by2, img_w, img_h, depth_img, drone_pos, eff_quat,
                    relax_depth=True,
                )
                if noted_deproj is not None:
                    log_xyz_local = noted_deproj[0]

            return (
                best_person, best_bbox, best_person_conf, raw_yolo_best,
                web_boxes, annotated_frame, img_bgr, log_xyz_local, confirmed_list,
            )

        # 1. Process Front
        res_f = process_single_cam(rgb_image, depth_image, 0.0)
        (
            best_person, best_bbox, best_person_conf, raw_yolo_best, web_boxes,
            annotated_frame, front_bgr, log_xyz_f, confirmed_f,
        ) = res_f
        self._web_frame_bgr = front_bgr
        self._web_boxes = web_boxes

        # 2. Process Left
        if self.detection_count % 200 == 0:
            print(f"[YOLO Diag] Left cam: rgb={'OK '+str(rgb_left.shape) if rgb_left is not None else 'None'}, depth={'OK' if depth_left is not None else 'None'}")
        res_l = process_single_cam(rgb_left, depth_left, 90.0)
        (
            best_person_l, best_bbox_l, best_person_conf_l, raw_yolo_best_l, web_boxes_l,
            annotated_frame_l, left_bgr, log_xyz_l, confirmed_l,
        ) = res_l
        self._web_frame_left_bgr = left_bgr
        self._web_boxes_left = web_boxes_l

        # 3. Process Right
        if self.detection_count % 200 == 0:
            print(f"[YOLO Diag] Right cam: rgb={'OK '+str(rgb_right.shape) if rgb_right is not None else 'None'}, depth={'OK' if depth_right is not None else 'None'}")
        res_r = process_single_cam(rgb_right, depth_right, -90.0)
        (
            best_person_r, best_bbox_r, best_person_conf_r, raw_yolo_best_r, web_boxes_r,
            annotated_frame_r, right_bgr, log_xyz_r, confirmed_r,
        ) = res_r
        self._web_frame_right_bgr = right_bgr
        self._web_boxes_right = web_boxes_r

        all_confirmed_raw: list[dict] = []
        for cam_name, clist in (
            ("front", confirmed_f or []),
            ("left", confirmed_l or []),
            ("right", confirmed_r or []),
        ):
            for conf, _bbox, xyz, z_depth, local_x, local_y in clist:
                all_confirmed_raw.append({
                    "conf": float(conf),
                    "cam": cam_name,
                    "xyz": (float(xyz[0]), float(xyz[1]), float(xyz[2])),
                    "z_depth": float(z_depth),
                    "local_x": float(local_x),
                    "local_y": float(local_y),
                })
        merged_confirmed = self._dedupe_confirmed_detections(all_confirmed_raw)
        if rescue_armed:
            self.frame_confirmed_persons = [
                {
                    "conf": float(m["conf"]),
                    "world_xyz": m["xyz"],
                    "cam": m.get("cam", "front"),
                    "z_depth": float(m.get("z_depth", 0.0)),
                }
                for m in merged_confirmed
                if float(m["conf"]) >= self.person_conf_threshold
            ]

        # Find best noted / confirmed person across all cameras
        cam_frames = {
            "front": annotated_frame,
            "left": annotated_frame_l,
            "right": annotated_frame_r,
        }
        detected_persons = []
        if best_person is not None:
            detected_persons.append(("front", best_person))
        if best_person_l is not None:
            detected_persons.append(("left", best_person_l))
        if best_person_r is not None:
            detected_persons.append(("right", best_person_r))

        overall_best_person = None
        confirmed_cam = None
        if detected_persons:
            detected_persons.sort(key=lambda item: -item[1][0])
            confirmed_cam, overall_best_person = detected_persons[0]

        noted_candidates = []
        if best_bbox is not None:
            noted_candidates.append(("front", best_bbox))
        if best_bbox_l is not None:
            noted_candidates.append(("left", best_bbox_l))
        if best_bbox_r is not None:
            noted_candidates.append(("right", best_bbox_r))

        overall_best_bbox = None
        noted_cam = None
        if noted_candidates:
            noted_candidates.sort(key=lambda item: -item[1][0])
            noted_cam, overall_best_bbox = noted_candidates[0]

        overall_best_person_conf = max(best_person_conf, best_person_conf_l, best_person_conf_r)
        overall_raw_yolo_best = max(raw_yolo_best, raw_yolo_best_l, raw_yolo_best_r)

        display_conf = overall_best_person_conf
        if overall_best_bbox is not None:
            display_conf = max(display_conf, overall_best_bbox[0])
        if display_conf <= 0.0 and overall_raw_yolo_best > 0.0:
            display_conf = overall_raw_yolo_best

        has_noted = False
        candidate_conf = overall_best_bbox[0] if overall_best_bbox is not None else 0.0
        if overall_best_bbox is not None and candidate_conf >= self.noted_conf_threshold:
            self._noted_streak += 1
        else:
            self._noted_streak = max(0, self._noted_streak - 1)

        if self._noted_streak >= self.noted_confirm_frames and overall_best_bbox is not None:
            has_noted = True
            display_conf = max(display_conf, candidate_conf)

        has_confirmed_person = False
        if overall_best_person is None and not self.frame_confirmed_persons:
            self._last_intel = None

        if self.frame_confirmed_persons:
            best_det = max(self.frame_confirmed_persons, key=lambda d: float(d["conf"]))
            conf = float(best_det["conf"])
            t_x, t_y, t_z = best_det["world_xyz"]
            z_depth = float(best_det.get("z_depth", 0.0))
            confirmed_cam = best_det.get("cam", confirmed_cam)
            has_confirmed_person = True
            person_found[0] = True
            person_world_xyz[0, 0] = t_x
            person_world_xyz[0, 1] = t_y
            person_world_xyz[0, 2] = t_z

            target_lat, target_lon = self._local_xyz_to_gps(t_x, t_y)
            slot = self._match_rescue_person_slot((t_x, t_y, t_z))
            self._last_intel = {
                "conf": conf,
                "label": slot["label"] if slot else (scan_label or "person detected"),
                "gps_lat": target_lat,
                "gps_lon": target_lon,
                "dist": z_depth,
            }

            print(
                f"[ALARM] {len(self.frame_confirmed_persons)} person(s) confirmed "
                f"(best {conf:.0%}) via {confirmed_cam or 'front'} cam! "
                f"Dist: {z_depth:.2f}m\n"
                f"   ↳ [RESCUE COORDS] Target is {t_x:.1f}m Forward, {t_y:.1f}m Right, "
                f"and {t_z:.1f}m High relative to the Building Entrance!"
            )
        elif overall_best_person is not None:
            conf, (bx1, by1, bx2, by2), (t_x, t_y, t_z), z_depth, local_x, local_y = overall_best_person
            has_confirmed_person = True
            person_found[0] = True
            person_world_xyz[0, 0] = t_x
            person_world_xyz[0, 1] = t_y
            person_world_xyz[0, 2] = t_z

            target_lat, target_lon = self._local_xyz_to_gps(t_x, t_y)

            slot = self._match_rescue_person_slot((t_x, t_y, t_z))
            self._last_intel = {
                "conf": conf,
                "label": slot["label"] if slot else (scan_label or "person detected"),
                "gps_lat": target_lat,
                "gps_lon": target_lon,
                "dist": z_depth,
            }

            print(
                f"[ALARM] Person confirmed ({conf:.0%}) via {confirmed_cam or 'front'} cam! "
                f"Dist: {z_depth:.2f}m, Local X: {local_x:.2f}m\n"
                f"   ↳ [RESCUE COORDS] Target is {t_x:.1f}m Forward, {t_y:.1f}m Right, "
                f"and {t_z:.1f}m High relative to the Building Entrance!"
            )

        self.detection_count += 1
        if display_conf > 0.0:
            self.last_best_person_conf = max(self.last_best_person_conf * 0.995, display_conf)
        else:
            self.last_best_person_conf *= 0.992
        alert_conf = candidate_conf if has_noted else display_conf

        log_xyz = None
        if overall_best_person is not None:
            log_xyz = (float(overall_best_person[2][0]), float(overall_best_person[2][1]), float(overall_best_person[2][2]))
        elif overall_best_bbox is not None:
            if best_bbox is not None and overall_best_bbox[0] == best_bbox[0]:
                log_xyz = log_xyz_f
            elif best_bbox_l is not None and overall_best_bbox[0] == best_bbox_l[0]:
                log_xyz = log_xyz_l
            elif best_bbox_r is not None and overall_best_bbox[0] == best_bbox_r[0]:
                log_xyz = log_xyz_r

            if log_xyz is not None and self._last_intel is None and (has_noted or candidate_conf >= self.noted_conf_threshold):
                t_x, t_y, t_z = log_xyz
                target_lat, target_lon = self._local_xyz_to_gps(t_x, t_y)
                slot = self._match_rescue_person_slot(log_xyz)
                # Find corresponding depth of detection
                z_depth = 5.0
                if best_bbox is not None and overall_best_bbox[0] == best_bbox[0] and res_f[0] is not None:
                    z_depth = res_f[0][3]
                elif best_bbox_l is not None and overall_best_bbox[0] == best_bbox_l[0] and res_l[0] is not None:
                    z_depth = res_l[0][3]
                elif best_bbox_r is not None and overall_best_bbox[0] == best_bbox_r[0] and res_r[0] is not None:
                    z_depth = res_r[0][3]

                self._last_intel = {
                    "conf": candidate_conf,
                    "label": slot["label"] if slot else (scan_label or "person detected"),
                    "gps_lat": target_lat,
                    "gps_lon": target_lon,
                    "dist": z_depth,
                }

        person_key = self._person_log_key(scan_label, log_xyz)
        person_seen = has_noted or has_confirmed_person
        should_log = person_seen and float(alert_conf) > self._person_best_conf.get(person_key, 0.0) + 1e-6
        logged_any = False

        if rescue_armed and self.frame_confirmed_persons:
            for det in self.frame_confirmed_persons:
                xyz = det["world_xyz"]
                conf = float(det["conf"])
                pk = self._person_log_key(scan_label, xyz)
                if conf <= self._person_best_conf.get(pk, 0.0) + 1e-6:
                    continue
                saved_new = self._append_detection_log(
                    conf, xyz, scan_label, frame_idx=self.detection_count
                )
                if saved_new:
                    logged_any = True
                    win_cam = det.get("cam", "front")
                    alert_frame = cam_frames.get(win_cam)
                    if alert_frame is None:
                        alert_frame = annotated_frame
                    self._trigger_operator_alert(alert_frame, conf, scan_label)
                    
                    # Save frame for this newly logged person if confidence is high
                    if conf >= 0.70:
                        save_frame = self._make_simple_view(alert_frame)
                        self._save_detection_frame(save_frame, "detection", cam=win_cam)
        elif should_log:
            saved_new = self._append_detection_log(
                alert_conf, log_xyz, scan_label, frame_idx=self.detection_count
            )
            if saved_new:
                logged_any = True
                win_cam = confirmed_cam if has_confirmed_person else noted_cam
                alert_frame = cam_frames.get(win_cam or "front")
                if alert_frame is None:
                    alert_frame = annotated_frame
                self._trigger_operator_alert(alert_frame, alert_conf, scan_label)
        if has_noted or has_confirmed_person:
            self._operator_alert_until = max(
                self._operator_alert_until, self.detection_count + 45
            )
        operator_alert = self.detection_count <= self._operator_alert_until

        if has_noted or has_confirmed_person:
            self.person_ever_detected = True

        noted_deferred = (has_noted or has_confirmed_person) and not rescue_armed
        if noted_deferred:
            if has_confirmed_person:
                has_confirmed_person = False
                person_found[0] = False
            print(
                f"[YOLO] Person NOTED in {scan_label or 'rooms 1–3'} "
                f"(conf {display_conf:.0%}) — continuing mission, no GPS approach."
            )
        elif has_noted and rescue_armed and logged_any:
            print(
                f"[YOLO] Person SEEN at {display_conf:.0%} in {scan_label or 'scan'} "
                f"(need {self.person_conf_threshold:.0%} for rescue) — saved to debug_yolo_detections/"
            )
        elif has_noted and logged_any:
            print(
                f"[YOLO] Person NOTED at {display_conf:.0%} in {scan_label or 'rooms 1–3'} "
                f"— continuing mission."
            )
        elif has_confirmed_person and logged_any:
            print(
                f"[YOLO] Person CONFIRMED at {display_conf:.0%} "
                f"(threshold {self.person_conf_threshold:.0%})"
            )
        elif self.detection_count % 100 == 0:
            if display_conf > 0.0:
                print(
                    f"[YOLO] Person seen but REJECTED: best conf={display_conf:.0%} "
                    f"(need {self.person_conf_threshold:.0%})"
                )
            else:
                print(f"[YOLO] No person in view (threshold {self.person_conf_threshold:.0%})")

        self._web_state = {
            "has_confirmed":  bool(has_confirmed_person),
            "has_noted":      bool(has_noted and not has_confirmed_person),
            "noted_deferred": bool(noted_deferred),
            "operator_alert": bool(operator_alert),
            "rescue_armed":   bool(rescue_armed),
            "alert_conf":     float(alert_conf),
            "display_conf":   float(display_conf),
            "scan_label":     scan_label,
        }

        try:
            self._show_detection_window(
                annotated_frame,
                best_conf=display_conf,
                has_confirmed=has_confirmed_person,
                detection_count=self.detection_count,
                custom_texts=[],
                noted_deferred=noted_deferred,
                has_noted=has_noted and not has_confirmed_person,
                rescue_armed=rescue_armed,
                scan_label=scan_label,
                operator_alert=operator_alert,
                alert_conf=alert_conf,
            )
        except Exception as exc:
            if not self._display_error_logged:
                print(f"[Perception] Detection window error: {exc}")
                self._display_error_logged = True

        should_save = should_log and (float(alert_conf) >= 0.70) and not (rescue_armed and has_confirmed_person)
        if should_save:
            prefix = "detection" if has_confirmed_person else "noted"
            win_cam = confirmed_cam if has_confirmed_person else noted_cam
            det_frame = cam_frames.get(win_cam or "front")
            if det_frame is not None:
                save_frame = self._make_simple_view(det_frame)
            else:
                save_frame = getattr(self, "_last_display_frame", None)
                if save_frame is None:
                    save_frame = self._make_simple_view(annotated_frame)
            self._save_detection_frame(save_frame, prefix, cam=win_cam or "front")

        return person_found, person_world_xyz
