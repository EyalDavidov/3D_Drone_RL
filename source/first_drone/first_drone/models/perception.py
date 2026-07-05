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
        min_person_aspect: float = 0.30,
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
    ):
        """Initialize YOLO-based person detection for the Brain module."""
        self.use_mock = use_mock
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
        self._web_boxes: list[dict] = []
        self._web_state: dict | None = None

        if not self.use_mock:
            yolo_path = _REPO_ROOT / "YOLO" / "yolo11n.pt"
            if not yolo_path.exists():
                yolo_path = Path(r"D:\isaac\3D_Drone_RL\YOLO\yolo11n.pt")
            self.yolo_model = YOLO(str(yolo_path))
            print(
                f"[Perception] YOLO person-only mode: rescue>={self.person_conf_threshold:.0%}, "
                f"noted>={self.noted_conf_threshold:.0%}, "
                f"min_bbox_area={self.min_bbox_area_frac:.1%}, "
                f"confirm_frames={self.noted_confirm_frames}\n"
            )

    def _bbox_passes_person_shape(self, box, img_w: int, img_h: int) -> bool:
        """Always return True to bypass geometric checks, matching feature/perception-module."""
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        bw = max(0.0, x2 - x1)
        bh = max(0.0, y2 - y1)
        if bw < 2.0 or bh < 2.0:
            return False
        return True

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
        if bw < 2.0 or bh < 2.0:
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
        where = (scan_label or "camera view").upper()
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

    def _match_rescue_person_slot(
        self, xyz: tuple[float, float, float] | None
    ) -> dict | None:
        """Map a detection to one of the fixed rescue-person slots (max 3 entries)."""
        if xyz is None or not self._rescue_person_slots:
            return None
        best_slot = None
        best_dist = float(self._person_match_radius) + 1.0
        x, y = float(xyz[0]), float(xyz[1])
        for slot in self._rescue_person_slots:
            sx, sy = float(slot["xyz"][0]), float(slot["xyz"][1])
            dist = math.hypot(x - sx, y - sy)
            if dist <= self._person_match_radius and dist < best_dist:
                best_dist = dist
                best_slot = slot
        return best_slot

    def _person_log_key(
        self, scan_label: str | None, xyz: tuple[float, float, float] | None
    ) -> str:
        """One log row per physical rescue person — keyed by configured slot, not scan label."""
        slot = self._match_rescue_person_slot(xyz)
        if slot is not None:
            return str(slot["id"])
        label = (scan_label or "camera_view").strip().lower().replace(" ", "_")
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
            else (scan_label or "camera view").upper()
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
        self._detection_log = self._detection_log[:20]
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
        if results.boxes is None:
            return vis
        thresh = float(draw_threshold if draw_threshold is not None else self.noted_conf_threshold)
        cs = float(coord_scale)
        for box in results.boxes:
            if int(box.cls[0]) != 0:
                continue
            conf = float(box.conf[0].item())
            if conf < 0.15:
                continue
            x1, y1, x2, y2 = self._box_xyxy_scaled(box, cs)
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
        if results is None or results.boxes is None:
            return boxes
        cs = float(coord_scale)
        iw = max(1.0, float(img_w))
        ih = max(1.0, float(img_h))
        for box in results.boxes:
            try:
                if int(box.cls[0]) != 0:
                    continue
                conf = float(box.conf[0].item())
                if conf < 0.15:
                    continue
                x1, y1, x2, y2 = self._box_xyxy_scaled(box, cs)
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

    def _save_detection_frame(self, frame_bgr: np.ndarray, prefix: str) -> None:
        """Save an annotated or HUD frame to debug_yolo_detections/."""
        tag = f"{prefix}_{self.detection_count:06d}.jpg"
        output_path = self.output_dir / tag
        try:
            from PIL import Image

            Image.fromarray(frame_bgr[:, :, ::-1]).save(str(output_path))
            print(f"[YOLO] Saved {prefix} detection: {output_path}")
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
    ):
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

        if scan_label and scan_label != self._active_scan_label:
            self._active_scan_label = scan_label
            self._noted_streak = 0
        elif scan_label is None and self._active_scan_label is not None:
            self._active_scan_label = None

        rgb_array = rgb_image.detach().cpu().numpy() if isinstance(rgb_image, torch.Tensor) else rgb_image
        if rgb_array.shape[-1] == 4:
            rgb_array = rgb_array[..., :3]
        if rgb_array.dtype in (np.float32, np.float64) and rgb_array.max() <= 1.0:
            rgb_array = (rgb_array * 255.0).astype(np.uint8)

        single_env_image_bgr = rgb_array[0][:, :, ::-1]
        img_h, img_w = single_env_image_bgr.shape[:2]
        img_area = float(img_h * img_w)

        import cv2

        up = self.yolo_camera_upscale
        yolo_bgr = single_env_image_bgr
        if up > 1:
            yolo_bgr = cv2.resize(
                single_env_image_bgr,
                (img_w * up, img_h * up),
                interpolation=cv2.INTER_LANCZOS4,
            )
        if self.yolo_sharpen:
            blur = cv2.GaussianBlur(yolo_bgr, (0, 0), sigmaX=1.0)
            yolo_bgr = cv2.addWeighted(yolo_bgr, 1.4, blur, -0.4, 0)

        # Boost local contrast if enabled (defaults to False since we now colorize models)
        if self.yolo_clahe:
            lab = cv2.cvtColor(yolo_bgr, cv2.COLOR_BGR2LAB)
            l_ch, a_ch, b_ch = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            l_ch = clahe.apply(l_ch)
            yolo_bgr = cv2.cvtColor(cv2.merge([l_ch, a_ch, b_ch]), cv2.COLOR_LAB2BGR)

        results = self.yolo_model(
            yolo_bgr,
            verbose=False,
            conf=0.15,
            classes=[0],
            imgsz=self.yolo_imgsz,
        )
        filtered_results = results[0]

        custom_texts = []
        has_confirmed_person = False
        best_person = None
        best_bbox = None
        best_person_conf = 0.0
        raw_yolo_best = 0.0
        accept_threshold = (
            self.person_conf_threshold if rescue_armed else self.noted_conf_threshold
        )

        coord_back = 1.0 / float(up) if up > 1 else 1.0

        for box in filtered_results.boxes:
            if int(box.cls[0]) != 0:
                continue
            conf = float(box.conf[0].item())
            raw_yolo_best = max(raw_yolo_best, conf)
            x1, y1, x2, y2 = self._box_xyxy_scaled(box, coord_back)
            if not self._bbox_passes_person_shape_xy(x1, y1, x2, y2, img_w, img_h):
                continue

            best_person_conf = max(best_person_conf, conf)

            bbox_area_frac = max(0.0, (x2 - x1) * (y2 - y1)) / img_area
            if conf >= self.noted_conf_threshold:
                if best_bbox is None or conf > best_bbox[0]:
                    best_bbox = (conf, box, x1, y1, x2, y2)

            if conf < accept_threshold:
                continue

            x_center = 0.5 * (x1 + x2)
            y_center = 0.5 * (y1 + y2)
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

            candidate = (conf, (x1, y1, x2, y2), (t_x, t_y, t_z), z_depth, local_x, local_y)
            if best_person is None or conf > best_person[0]:
                best_person = candidate

        display_conf = best_person_conf
        if best_bbox is not None:
            display_conf = max(display_conf, best_bbox[0])
        if display_conf <= 0.0 and raw_yolo_best > 0.0:
            display_conf = raw_yolo_best

        has_noted = False
        candidate_conf = best_bbox[0] if best_bbox is not None else 0.0
        if best_bbox is not None and candidate_conf >= self.noted_conf_threshold:
            self._noted_streak += 1
        else:
            self._noted_streak = 0

        if self._noted_streak >= self.noted_confirm_frames and best_bbox is not None:
            has_noted = True
            display_conf = max(display_conf, candidate_conf)

        prev_noted = False  # legacy flag — per-person tracking uses _person_best_conf

        if best_person is None:
            self._last_intel = None

        if best_person is not None:
            conf, (bx1, by1, bx2, by2), (t_x, t_y, t_z), z_depth, local_x, local_y = best_person
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

            slot = self._match_rescue_person_slot((t_x, t_y, t_z))
            self._last_intel = {
                "conf": conf,
                "label": slot["label"] if slot else (scan_label or "camera view"),
                "gps_lat": target_lat,
                "gps_lon": target_lon,
                "dist": z_depth,
            }

            print(
                f"[ALARM] Person confirmed ({conf:.0%})! Dist: {z_depth:.2f}m, Local X: {local_x:.2f}m\n"
                f"   ↳ [RESCUE COORDS] Target is {t_x:.1f}m Forward, {t_y:.1f}m Right, "
                f"and {t_z:.1f}m High relative to the Building Entrance!"
            )

        self.detection_count += 1
        self.last_best_person_conf = display_conf
        alert_conf = candidate_conf if has_noted else display_conf

        log_xyz = None
        if best_person is not None:
            log_xyz = (float(best_person[2][0]), float(best_person[2][1]), float(best_person[2][2]))
        person_key = self._person_log_key(scan_label, log_xyz)
        person_seen = has_noted or has_confirmed_person
        should_log = person_seen and float(alert_conf) > self._person_best_conf.get(person_key, 0.0) + 1e-6
        new_detection_event = should_log

        display_bgr = single_env_image_bgr
        if self.yolo_sharpen:
            blur = cv2.GaussianBlur(display_bgr, (0, 0), sigmaX=0.8)
            display_bgr = cv2.addWeighted(display_bgr, 1.35, blur, -0.35, 0)

        annotated_frame = self._annotate_detections(
            display_bgr,
            filtered_results,
            draw_threshold=0.15,
            coord_scale=coord_back,
        )
        if should_log:
            saved_new = self._append_detection_log(
                alert_conf, log_xyz, scan_label, frame_idx=self.detection_count
            )
            if saved_new:
                self._trigger_operator_alert(annotated_frame, alert_conf, scan_label)
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
        elif has_noted and rescue_armed and should_log:
            print(
                f"[YOLO] Person SEEN at {display_conf:.0%} in {scan_label or 'scan'} "
                f"(need {self.person_conf_threshold:.0%} for rescue) — saved to debug_yolo_detections/"
            )
        elif has_noted and should_log:
            print(
                f"[YOLO] Person NOTED at {display_conf:.0%} in {scan_label or 'rooms 1–3'} "
                f"— continuing mission."
            )
        elif has_confirmed_person and should_log:
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

        # ── Publish native web-HUD payload (synced: same frame as the boxes) ──
        self._web_frame_bgr = display_bgr
        self._web_boxes = self._collect_web_boxes(
            filtered_results, coord_back, img_w, img_h,
        )
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

        save_frame = getattr(self, "_last_display_frame", None)
        if save_frame is None:
            save_frame = self._make_simple_view(annotated_frame)
        should_save = should_log
        if should_save:
            prefix = "detection" if has_confirmed_person else "noted"
            self._save_detection_frame(save_frame, prefix)

        return person_found, person_world_xyz
