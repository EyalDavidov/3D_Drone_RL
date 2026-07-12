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


class _StaticHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, directory=None, **kwargs):
        super().__init__(*args, directory=str(_STATIC_DIR), **kwargs)

    def log_message(self, format, *args):
        pass

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

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
                        status = "—"
                        frame_count = 0
                        try:
                            first_line = None
                            last_line = None
                            with open(f, "r", encoding="utf-8") as file_fh:
                                for raw_line in file_fh:
                                    line = raw_line.strip()
                                    if not line:
                                        continue
                                    if first_line is None:
                                        first_line = line
                                    last_line = line
                                    frame_count += 1
                                    if frame_count % 50 == 1:
                                        try:
                                            d = json.loads(line)
                                            max_coverage = max(
                                                max_coverage, float(d.get("map_explored_pct", 0) or 0)
                                            )
                                            ms = d.get("mission_status") or {}
                                            if ms.get("crash_reason") and not crash_reason:
                                                crash_reason = str(ms["crash_reason"])
                                                status = "CRASH"
                                        except Exception:
                                            pass
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
                        if status == "—":
                            status = str(ms.get("status") or summary.get("slam_state") or "—")
                        level = summary.get("level", first_summary.get("level", 1))
                        file_size = f.stat().st_size
                        if file_size < 1024:
                            size_str = f"{file_size} B"
                        elif file_size < 1024 * 1024:
                            size_str = f"{file_size / 1024:.1f} KB"
                        else:
                            size_str = f"{file_size / (1024 * 1024):.1f} MB"

                        recordings.append({
                            "filename": display_name,
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
                        })
            except Exception as e:
                print(f"[Dashboard Server] Error listing recordings: {e}")
                
            self.wfile.write(json.dumps(recordings).encode("utf-8"))
            return
            
        elif self.path.startswith("/api/recording?file="):
            from urllib.parse import parse_qs, urlparse
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            filename = params.get("file", [None])[0]
            
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
            self.send_header("Content-Type", "text/plain")
            if use_gzip:
                self.send_header("Content-Encoding", "gzip")
            self.end_headers()
            
            try:
                if use_gzip:
                    import gzip
                    with open(file_path, "rb") as f:
                        with gzip.GzipFile(fileobj=self.wfile, mode="wb", compresslevel=6) as gzip_file:
                            while True:
                                chunk = f.read(65536)  # 64 KB chunk size
                                if not chunk:
                                    break
                                gzip_file.write(chunk)
                else:
                    with open(file_path, "rb") as f:
                        while True:
                            chunk = f.read(65536)  # 64 KB chunk size
                            if not chunk:
                                break
                            self.wfile.write(chunk)
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
