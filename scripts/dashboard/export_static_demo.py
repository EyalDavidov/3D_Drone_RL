"""Export top N latest flight recordings for static GitHub Pages dashboard.

Optimizes file size for static web deployment by:
1. Downsampling total frame count to ~1,000 max frames.
2. Trimming bulky debug camera base64 images while preserving the main YOLO camera feed.

Usage:
    python scripts/dashboard/export_static_demo.py              # Export latest 5 flights
    python scripts/dashboard/export_static_demo.py --count 5
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent
_RECORDINGS_DIR = Path(
    os.getenv("DASHBOARD_RECORDINGS_DIR", str(_REPO_ROOT / "recordings"))
).expanduser().resolve()
_STATIC_REC_DIR = _SCRIPT_DIR / "static" / "recordings"

_TARGET_SAMPLE_FRAMES = 1000  # Target frame count per exported demo flight


def _format_size(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.1f} KB"
    if num_bytes < 1024 * 1024 * 1024:
        return f"{num_bytes / (1024 * 1024):.1f} MB"
    return f"{num_bytes / (1024 * 1024 * 1024):.2f} GB"


def _read_recording_meta(filename: str) -> dict:
    meta_path = _RECORDINGS_DIR / f"{filename}.meta.json"
    if not meta_path.exists():
        return {}
    try:
        with open(meta_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _optimize_frame(obj: dict, frame_index: int) -> dict:
    """Optimize single frame object for fast web transmission."""
    if obj.get("_record_type") == "session_header":
        return obj

    # Copy frame
    optimized = dict(obj)

    # Trim heavy non-essential images, keeping main yolo_frame on every 3rd frame or on detections
    images = optimized.get("images")
    if isinstance(images, dict) and images:
        trimmed_images = {}
        # Keep yolo_frame for HUD presentation
        if "yolo_frame" in images and (frame_index % 3 == 0 or bool(images.get("captured_frames"))):
            trimmed_images["yolo_frame"] = images["yolo_frame"]
        if "captured_frames" in images:
            trimmed_images["captured_frames"] = images["captured_frames"]
        optimized["images"] = trimmed_images

    return optimized


def process_and_export_recording(src_path: Path, max_target_frames: int = _TARGET_SAMPLE_FRAMES) -> dict | None:
    filename = src_path.name
    src_size = src_path.stat().st_size
    print(f"[Export] Processing {filename} ({_format_size(src_size)})...")

    # Read lines
    raw_lines = []
    header_obj = None

    try:
        with open(src_path, "r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                line_str = line.strip()
                if not line_str:
                    continue
                try:
                    obj = json.loads(line_str)
                    if obj.get("_record_type") == "session_header":
                        header_obj = obj
                    else:
                        raw_lines.append(obj)
                except Exception:
                    pass
    except Exception as e:
        print(f"[Export] Error reading {filename}: {e}")
        return None

    total_frames = len(raw_lines)
    if total_frames == 0:
        print(f"[Export] Skipping {filename} (no frame data).")
        return None

    # Determine stride
    stride = max(1, int(total_frames / float(max_target_frames)))
    sampled_frames = raw_lines[::stride]
    if raw_lines[-1] not in sampled_frames:
        sampled_frames.append(raw_lines[-1])

    print(f"  -> Original: {total_frames} frames | Sampled: {len(sampled_frames)} frames (stride x{stride})")

    # Destination path
    dest_path = _STATIC_REC_DIR / filename
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    max_coverage = 0.0
    crash_reason = ""
    status = "-"

    with open(dest_path, "w", encoding="utf-8") as out_fh:
        if header_obj:
            out_fh.write(json.dumps(header_obj, ensure_ascii=False) + "\n")

        for idx, frame in enumerate(sampled_frames):
            opt_frame = _optimize_frame(frame, idx)
            out_fh.write(json.dumps(opt_frame, ensure_ascii=False) + "\n")

            # Extract metrics
            max_coverage = max(max_coverage, float(frame.get("map_explored_pct", 0) or 0))
            ms = frame.get("mission_status") or {}
            if ms.get("crash_reason") and not crash_reason:
                crash_reason = str(ms["crash_reason"])
                status = "CRASH"

    dest_size = dest_path.stat().st_size
    print(f"  -> Exported to static/recordings/{filename} ({_format_size(dest_size)})")

    first_obj = sampled_frames[0] if sampled_frames else {}
    last_obj = sampled_frames[-1] if sampled_frames else {}

    # Extract date from filename (flight_YYYYMMDD_HHMMSS.jsonl)
    date_stem = filename.replace("flight_", "").replace(".jsonl", "")
    date_str = date_stem
    try:
        parts = date_stem.split("_")
        if len(parts) == 2:
            d_part, t_part = parts[0], parts[1]
            y, m, d = d_part[:4], d_part[4:6], d_part[6:]
            hh, mm = t_part[:2], t_part[2:4]
            date_str = f"{d}/{m}/{y} {hh}:{mm}"
    except Exception:
        pass

    spawn_info = last_obj.get("spawn_info", {})
    total_targets = spawn_info.get("total", 2)
    detected_targets = spawn_info.get("detected", last_obj.get("people_found", 0))
    coverage = max(float(last_obj.get("map_explored_pct", 0) or 0), max_coverage)
    duration = float(last_obj.get("level_time", 0.0) or 0.0)
    ms = last_obj.get("mission_status") or {}
    if not crash_reason:
        crash_reason = str(ms.get("crash_reason") or "")
    if status == "-":
        status = str(ms.get("status") or last_obj.get("slam_state") or "-")
    level = last_obj.get("level", first_obj.get("level", 1))

    rec_meta = _read_recording_meta(filename)
    title = str(rec_meta.get("title") or "").strip()

    return {
        "filename": filename,
        "title": title,
        "display_title": title or filename,
        "date": date_str,
        "targets_total": total_targets,
        "targets_found": detected_targets,
        "coverage": coverage,
        "duration": duration,
        "frames": len(sampled_frames),
        "status": status,
        "crash_reason": crash_reason,
        "level": level,
        "file_size": _format_size(dest_size),
        "file_bytes": dest_size,
        "large_file": False,
        "recommended_stride": 1,
    }


def main():
    parser = argparse.ArgumentParser(description="Export top N flight recordings for static web hosting")
    parser.add_argument("--count", type=int, default=5, help="Number of latest recordings to export (default: 5)")
    args = parser.parse_args()

    print("=" * 60)
    print(f"  RL Drone Dashboard - Static Demo Exporter")
    print(f"  Exporting top {args.count} latest flight recordings")
    print("=" * 60)

    if not _RECORDINGS_DIR.exists():
        print(f"[ERROR] Recordings directory not found: {_RECORDINGS_DIR}")
        return

    # Find all flight_*.jsonl files sorted by timestamp (newest first)
    files = [f for f in _RECORDINGS_DIR.glob("flight_*.jsonl") if not f.name.endswith(".meta.json")]
    files.sort(key=lambda x: x.name, reverse=True)

    if not files:
        print("[ERROR] No flight_*.jsonl recording files found.")
        return

    target_files = files[: args.count]
    print(f"[Export] Found {len(files)} total recordings. Exporting latest {len(target_files)}:\n")

    _STATIC_REC_DIR.mkdir(parents=True, exist_ok=True)
    exported_metadata = []

    for src_file in target_files:
        meta = process_and_export_recording(src_file)
        if meta:
            exported_metadata.append(meta)

    # Write manifest.json
    manifest_path = _STATIC_REC_DIR / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(exported_metadata, fh, ensure_ascii=False, indent=2)

    total_bytes = sum(m["file_bytes"] for m in exported_metadata)
    print("\n" + "=" * 60)
    print(f"  SUCCESS! Exported {len(exported_metadata)} flight recordings.")
    print(f"  Manifest: {manifest_path}")
    print(f"  Total Export Size: {_format_size(total_bytes)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
