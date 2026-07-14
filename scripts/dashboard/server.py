"""Dashboard server — HTTP + WebSocket for the RL Drone Dashboard.

Usage:
    python scripts/dashboard/server.py          # mock-data mode (default)
    python scripts/dashboard/server.py --port 8000

Bidirectional WebSocket: streams telemetry to clients,
receives commands (e.g., set_level) from clients.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import threading
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
try:
    from http.server import ThreadingHTTPServer
except ImportError:
    from socketserver import ThreadingMixIn
    class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
        daemon_threads = True
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_STATIC_DIR = _SCRIPT_DIR / "static"
_REPO_ROOT = _SCRIPT_DIR.parent.parent
_RECORDINGS_DIR = Path(
    os.getenv("DASHBOARD_RECORDINGS_DIR", str(_REPO_ROOT / "recordings"))
).expanduser().resolve()
_LAPTOP_HOST_IP = os.getenv("DASHBOARD_LAPTOP_IP", "100.97.155.78")
_LARGE_RECORDING_BYTES = 250 * 1024 * 1024
_REPLAY_TARGET_FRAMES = 2500


def _safe_recording_filename(filename: str | None) -> str | None:
    if not filename or ".." in filename or "/" in filename or "\\" in filename:
        return None
    return filename


def _recording_file_path(filename: str) -> Path:
    if filename.startswith("run_"):
        return _RECORDINGS_DIR / filename / "telemetry.jsonl"
    return _RECORDINGS_DIR / filename


def _recording_meta_path(filename: str) -> Path:
    if filename.startswith("run_"):
        return _RECORDINGS_DIR / filename / "recording_meta.json"
    return _RECORDINGS_DIR / f"{filename}.meta.json"


def _read_recording_meta(filename: str) -> dict:
    meta_path = _recording_meta_path(filename)
    if not meta_path.exists():
        return {}
    try:
        with open(meta_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _format_size(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.1f} KB"
    if num_bytes < 1024 * 1024 * 1024:
        return f"{num_bytes / (1024 * 1024):.1f} MB"
    return f"{num_bytes / (1024 * 1024 * 1024):.2f} GB"


def _read_first_nonempty_line(path: Path) -> str | None:
    with open(path, "rb") as fh:
        for raw in fh:
            line = raw.strip()
            if line:
                return line.decode("utf-8", errors="ignore")
    return None


def _read_last_nonempty_line(path: Path, block_size: int = 1024 * 1024) -> str | None:
    size = path.stat().st_size
    if size <= 0:
        return None
    data = b""
    with open(path, "rb") as fh:
        pos = size
        while pos > 0:
            read_size = min(block_size, pos)
            pos -= read_size
            fh.seek(pos)
            data = fh.read(read_size) + data
            lines = [ln.strip() for ln in data.splitlines() if ln.strip()]
            if lines:
                return lines[-1].decode("utf-8", errors="ignore")
    return None


def _sample_jsonl_metadata(path: Path, file_size: int) -> tuple[int, float, str, str]:
    """Estimate frames/coverage/status quickly without scanning multi-GB logs."""
    max_coverage = 0.0
    crash_reason = ""
    status = "-"
    sampled_lines = 0
    sampled_bytes = 0

    try:
        with open(path, "rb") as fh:
            for raw in fh:
                sampled_bytes += len(raw)
                line = raw.strip()
                if line:
                    sampled_lines += 1
                    try:
                        d = json.loads(line)
                        max_coverage = max(max_coverage, float(d.get("map_explored_pct", 0) or 0))
                        ms = d.get("mission_status") or {}
                        if ms.get("crash_reason") and not crash_reason:
                            crash_reason = str(ms["crash_reason"])
                            status = "CRASH"
                    except Exception:
                        pass
                if sampled_bytes >= 4 * 1024 * 1024 or sampled_lines >= 250:
                    break
    except Exception:
        pass

    frame_estimate = sampled_lines
    if sampled_bytes > 0 and file_size > sampled_bytes:
        avg_line_bytes = sampled_bytes / max(sampled_lines, 1)
        frame_estimate = max(sampled_lines, int(file_size / max(avg_line_bytes, 1.0)))
    return frame_estimate, max_coverage, crash_reason, status


def _iter_sampled_jsonl_lines(path: Path, max_frames: int):
    """Return a responsive, whole-flight sample from huge JSONL recordings."""
    size = path.stat().st_size
    if size <= 0 or max_frames <= 0:
        return

    emitted = 0
    seen_offsets = set()

    with open(path, "rb") as fh:
        first_offset = fh.tell()
        first_raw = fh.readline()
        if first_raw.strip():
            seen_offsets.add(first_offset)
            try:
                first_obj = json.loads(first_raw)
                is_header = first_obj.get("_record_type") == "session_header"
            except Exception:
                is_header = False
            yield first_raw
            if not is_header:
                emitted += 1
                if emitted >= max_frames:
                    return

        data_start = fh.tell()
        if data_start >= size:
            return

        slots = max_frames - emitted
        if slots <= 0:
            return

        span = max(1, size - data_start)
        sample_slots = max(0, slots - 1)
        for idx in range(sample_slots):
            offset = data_start + int((idx * span) / max(1, sample_slots))
            offset = min(max(data_start, offset), max(data_start, size - 1))
            fh.seek(offset)
            if offset > data_start:
                fh.readline()
            line_offset = fh.tell()
            raw = fh.readline()
            if not raw or not raw.strip() or line_offset in seen_offsets:
                continue
            seen_offsets.add(line_offset)
            yield raw
            emitted += 1
            if emitted >= max_frames:
                return

        if emitted < max_frames:
            last_line = _read_last_nonempty_line(path)
            if last_line:
                yield (last_line + "\n").encode("utf-8")


class _StaticHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, directory=None, **kwargs):
        super().__init__(*args, directory=str(_STATIC_DIR), **kwargs)

    def log_message(self, format, *args):
        pass

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.end_headers()

    def do_POST(self):
        try:
            self._handle_post_route()
        except (ConnectionAbortedError, ConnectionResetError):
            pass
        except Exception as e:
            print(f"[Dashboard Server] Error in do_POST: {e}")
            self.send_error(500, "Server error")

    def _handle_post_route(self):
        if self.path != "/api/recording_meta":
            self.send_error(404, "Not found")
            return

        try:
            length = int(self.headers.get("Content-Length", "0") or "0")
        except Exception:
            length = 0
        raw = self.rfile.read(min(length, 64 * 1024))
        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception:
            self.send_error(400, "Invalid JSON")
            return

        filename = _safe_recording_filename(payload.get("filename"))
        if not filename:
            self.send_error(400, "Invalid filename")
            return

        file_path = _recording_file_path(filename)
        if not file_path.exists():
            self.send_error(404, "Recording not found")
            return

        title = str(payload.get("title") or "").strip()
        title = title[:80]
        meta = _read_recording_meta(filename)
        if title:
            meta["title"] = title
        else:
            meta.pop("title", None)
        meta_path = _recording_meta_path(filename)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=2)

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"ok": True, "title": meta.get("title", "")}).encode("utf-8"))

    def do_GET(self):
        try:
            self._handle_get_route()
        except (ConnectionAbortedError, ConnectionResetError):
            # Quietly ignore aborted/reset connections from browser page refreshes
            pass
        except Exception as e:
            print(f"[Dashboard Server] Error in do_GET: {e}")

    def _handle_get_route(self):
        if self.path == "/api/recordings":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            
            recordings = []
            try:
                rec_dir = _RECORDINGS_DIR
                if rec_dir.exists():
                    # Scan both file styles for backward compatibility
                    files_to_scan = []
                    
                    # 1. Scan flight_*.jsonl files directly
                    for f in rec_dir.glob("flight_*.jsonl"):
                        if f.name.endswith(".meta.json"):
                            continue
                        files_to_scan.append((f, f.name, f.stem.replace("flight_", "")))
                    
                    # 2. Scan run_ subdirectories containing telemetry.jsonl
                    for run_folder in [d for d in rec_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]:
                        telemetry_file = run_folder / "telemetry.jsonl"
                        if telemetry_file.exists():
                            files_to_scan.append((telemetry_file, run_folder.name, run_folder.name.replace("run_", "")))
                    
                    # Sort descending by date/timestamp stem
                    files_to_scan.sort(key=lambda x: x[2], reverse=True)
                    
                    for f, display_name, date_stem in files_to_scan:
                        summary = {}
                        first_summary = {}
                        max_coverage = 0.0
                        crash_reason = ""
                        status = "-"
                        file_size = f.stat().st_size
                        frame_count, max_coverage, sampled_crash, sampled_status = _sample_jsonl_metadata(f, file_size)
                        if sampled_crash:
                            crash_reason = sampled_crash
                            status = sampled_status
                        try:
                            first_line = _read_first_nonempty_line(f)
                            last_line = _read_last_nonempty_line(f)
                            if first_line:
                                try:
                                    first_summary = json.loads(first_line)
                                except Exception:
                                    first_summary = {}
                            if last_line:
                                try:
                                    summary = json.loads(last_line)
                                except Exception:
                                    summary = {}
                                if summary:
                                    max_coverage = max(
                                        max_coverage, float(summary.get("map_explored_pct", 0) or 0)
                                    )
                                    ms = summary.get("mission_status") or {}
                                    if ms.get("crash_reason") and not crash_reason:
                                        crash_reason = str(ms["crash_reason"])
                                        status = "CRASH"
                        except Exception:
                            pass
                        
                        # Extract date
                        date_str = date_stem
                        try:
                            # format 20260712_141600 as 12/07/2026 14:16
                            parts = date_str.split("_")
                            if len(parts) == 2:
                                d_part, t_part = parts[0], parts[1]
                                y, m, d = d_part[:4], d_part[4:6], d_part[6:]
                                hh, mm = t_part[:2], t_part[2:4]
                                date_str = f"{d}/{m}/{y} {hh}:{mm}"
                        except Exception:
                            pass
                        
                        spawn_info = summary.get("spawn_info", {})
                        total_targets = spawn_info.get("total", 2)
                        detected_targets = spawn_info.get("detected", summary.get("people_found", 0))
                        coverage = max(
                            float(summary.get("map_explored_pct", 0) or 0),
                            max_coverage,
                        )
                        duration = summary.get("level_time", 0.0)
                        ms = summary.get("mission_status") or {}
                        if not crash_reason:
                            crash_reason = str(ms.get("crash_reason") or "")
                        if status == "-":
                            status = str(ms.get("status") or summary.get("slam_state") or "-")
                        level = summary.get("level", first_summary.get("level", 1))
                        size_str = _format_size(file_size)
                        large_file = file_size >= _LARGE_RECORDING_BYTES
                        recommended_stride = max(
                            1,
                            int((frame_count + _REPLAY_TARGET_FRAMES - 1) / _REPLAY_TARGET_FRAMES),
                        )
                        rec_meta = _read_recording_meta(display_name)
                        title = str(rec_meta.get("title") or "").strip()

                        recordings.append({
                            "filename": display_name,
                            "title": title,
                            "display_title": title or display_name,
                            "date": date_str,
                            "targets_total": total_targets,
                            "targets_found": detected_targets,
                            "coverage": coverage,
                            "duration": duration,
                            "frames": frame_count,
                            "status": status,
                            "crash_reason": crash_reason,
                            "level": level,
                            "file_size": size_str,
                            "file_bytes": file_size,
                            "large_file": large_file,
                            "frames_estimated": large_file,
                            "recommended_stride": recommended_stride,
                        })
            except Exception as e:
                print(f"[Dashboard Server] Error listing recordings: {e}")
                
            self.wfile.write(json.dumps(recordings).encode("utf-8"))
            return
            
        elif self.path.startswith("/api/recording?"):
            from urllib.parse import parse_qs, urlparse
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            filename = params.get("file", [None])[0]
            try:
                stride = max(1, int(params.get("stride", ["1"])[0] or "1"))
            except Exception:
                stride = 1
            try:
                max_frames = int(params.get("max_frames", ["0"])[0] or "0")
            except Exception:
                max_frames = 0
            
            if not filename or ".." in filename or "/" in filename or "\\" in filename:
                self.send_error(400, "Invalid filename")
                return
                
            rec_dir = _RECORDINGS_DIR
            if filename.startswith("run_"):
                file_path = rec_dir / filename / "telemetry.jsonl"
            else:
                file_path = rec_dir / filename
                
            if not file_path.exists():
                self.send_error(404, "Recording not found")
                return
                
            accept_encoding = self.headers.get("Accept-Encoding", "")
            use_gzip = "gzip" in accept_encoding
            
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("X-Recording-Stride", str(stride))
            self.send_header("X-Recording-Max-Frames", str(max_frames))
            if use_gzip:
                self.send_header("Content-Encoding", "gzip")
            self.end_headers()
            
            try:
                file_size = file_path.stat().st_size

                def iter_recording_lines():
                    emitted = 0
                    frame_idx = 0
                    with open(file_path, "rb") as f:
                        for raw in f:
                            if not raw.strip():
                                continue
                            try:
                                obj = json.loads(raw)
                                is_header = obj.get("_record_type") == "session_header"
                            except Exception:
                                is_header = False
                            should_emit = is_header or stride <= 1 or (frame_idx % stride == 0)
                            if should_emit:
                                yield raw
                                if not is_header:
                                    emitted += 1
                                    if max_frames > 0 and emitted >= max_frames:
                                        break
                            if not is_header:
                                frame_idx += 1

                line_source = (
                    _iter_sampled_jsonl_lines(file_path, max_frames)
                    if max_frames > 0 and file_size >= _LARGE_RECORDING_BYTES
                    else iter_recording_lines()
                )

                if use_gzip:
                    import gzip
                    with gzip.GzipFile(fileobj=self.wfile, mode="wb", compresslevel=6) as gzip_file:
                        for raw_line in line_source:
                            gzip_file.write(raw_line)
                else:
                    for raw_line in line_source:
                        self.wfile.write(raw_line)
            except Exception as stream_err:
                print(f"[Dashboard Server] Error streaming recording file {filename}: {stream_err}")
            return
            
        super().do_GET()


def _run_http_server(port: int):
    server = ThreadingHTTPServer(("0.0.0.0", port), _StaticHandler)
    print(f"[Dashboard] HTTP server running at http://localhost:{port}")
    print(f"[Dashboard] HTTP laptop link: http://{_LAPTOP_HOST_IP}:{port}")
    server.serve_forever()


async def _ws_handler(websocket, telemetry_source):
    """Bidirectional WebSocket handler: sends telemetry, receives commands."""
    print(f"[Dashboard] WebSocket client connected: {websocket.remote_address}")

    async def sender():
        try:
            while True:
                data = telemetry_source.tick()
                await websocket.send(json.dumps(data))
                await asyncio.sleep(1.0 / telemetry_source.tick_rate)
        except Exception:
            pass

    async def receiver():
        try:
            async for message in websocket:
                try:
                    cmd = json.loads(message)
                    if cmd.get("command") == "set_level":
                        level = cmd["level"]
                        if level == "auto":
                            telemetry_source.force_level = None
                            print("[Dashboard] Level mode: AUTO")
                        else:
                            lv = int(level) - 1  # 0-indexed
                            telemetry_source.force_level = lv
                            telemetry_source._reset_to_level(lv)
                    elif cmd.get("command") == "spawn_random_targets":
                        count = int(cmd.get("count", 2))
                        telemetry_source.pending_spawn_count = count
                        print(f"[Dashboard] Spawn command received: {count} targets")
                except (json.JSONDecodeError, KeyError, ValueError):
                    pass
        except Exception:
            pass

    send_task = asyncio.create_task(sender())
    recv_task = asyncio.create_task(receiver())

    done, pending = await asyncio.wait(
        {send_task, recv_task},
        return_when=asyncio.FIRST_COMPLETED,
    )
    for task in pending:
        task.cancel()

    print(f"[Dashboard] WebSocket client disconnected: {websocket.remote_address}")


async def _run_ws_server(port: int, telemetry_source):
    try:
        import websockets
    except ImportError:
        print("\n[ERROR] The 'websockets' package is required.")
        print("Install it with:  pip install websockets\n")
        sys.exit(1)

    try:
        server = await websockets.serve(
            lambda ws: _ws_handler(ws, telemetry_source),
            "0.0.0.0",
            port,
        )
    except OSError as exc:
        if getattr(exc, "errno", None) == 10048:
            print(f"\n[Dashboard] WebSocket port {port} is already in use.")
            print("[Dashboard] Stop the existing process or choose a different --ws-port.")
        raise
    print(f"[Dashboard] WebSocket server running at ws://localhost:{port}")
    print(f"[Dashboard] WebSocket laptop link: ws://{_LAPTOP_HOST_IP}:{port}")
    await server.wait_closed()


def start_dashboard_server(
    http_port: int = 8000,
    ws_port: int = 8001,
    telemetry_source=None,
    open_browser: bool = False,
    blocking: bool = True,
):
    if telemetry_source is None:
        from mock_data import MockDroneTelemetry
        telemetry_source = MockDroneTelemetry(tick_rate=20.0)
        print("[Dashboard] Using MOCK telemetry data")

    http_thread = threading.Thread(target=_run_http_server, args=(http_port,), daemon=True)
    http_thread.start()

    if open_browser:
        url = f"http://localhost:{http_port}?ws_port={ws_port}"
        print(f"[Dashboard] Opening browser at {url}")
        webbrowser.open(url)

    if blocking:
        asyncio.run(_run_ws_server(ws_port, telemetry_source))
    else:
        def _run_ws_in_thread():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(_run_ws_server(ws_port, telemetry_source))

        ws_thread = threading.Thread(target=_run_ws_in_thread, daemon=True)
        ws_thread.start()
        return http_thread, ws_thread


def main():
    parser = argparse.ArgumentParser(description="RL Drone Dashboard Server")
    parser.add_argument("--http-port", type=int, default=8000)
    parser.add_argument("--ws-port", type=int, default=8001)
    parser.add_argument("--open-browser", action="store_true", help="Automatically open browser on launch")
    args = parser.parse_args()

    print("=" * 60)
    print("  RL Drone Play-Mode Dashboard")
    print("  Mock Data Mode")
    print("=" * 60)

    try:
        start_dashboard_server(
            http_port=args.http_port,
            ws_port=args.ws_port,
            open_browser=args.open_browser,
            blocking=True,
        )
    except OSError as exc:
        if getattr(exc, "errno", None) == 10048:
            print("[Dashboard] Startup failed because the WebSocket port is busy.")
            print(f"[Dashboard] Try: python scripts\\dashboard\\server.py --ws-port {args.ws_port + 1}")
            return
        raise
    except KeyboardInterrupt:
        print("\n[Dashboard] Server stopped.")


if __name__ == "__main__":
    main()
