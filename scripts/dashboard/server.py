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
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_STATIC_DIR = _SCRIPT_DIR / "static"


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
                import json
                rec_dir = Path(r"D:\isaac\3D_Drone_RL\recordings")
                if rec_dir.exists():
                    # Scan flight_*.jsonl files directly
                    for f in sorted(rec_dir.glob("flight_*.jsonl"), reverse=True):
                        summary = {}
                        first_summary = {}
                        max_coverage = 0.0
                        crash_reason = ""
                        status = "—"
                        frame_count = 0
                        try:
                            with open(f, "r", encoding="utf-8") as file_fh:
                                lines = [ln.strip() for ln in file_fh if ln.strip()]
                            frame_count = len(lines)
                            if lines:
                                try:
                                    first_summary = json.loads(lines[0])
                                except Exception:
                                    first_summary = {}
                                try:
                                    summary = json.loads(lines[-1])
                                except Exception:
                                    summary = {}
                                step = max(1, frame_count // 80)
                                for i in range(0, frame_count, step):
                                    try:
                                        d = json.loads(lines[i])
                                        max_coverage = max(
                                            max_coverage, float(d.get("map_explored_pct", 0) or 0)
                                        )
                                        ms = d.get("mission_status") or {}
                                        if ms.get("crash_reason") and not crash_reason:
                                            crash_reason = str(ms["crash_reason"])
                                            status = "CRASH"
                                    except Exception:
                                        pass
                        except Exception:
                            pass
                        
                        # Extract date from filename: flight_20260712_141600.jsonl
                        name_stem = f.stem
                        date_str = name_stem.replace("flight_", "")
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
                            "filename": f.name,
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
                
            file_path = Path(r"D:\isaac\3D_Drone_RL\recordings") / filename
            if not file_path.exists():
                self.send_error(404, "Recording not found")
                return
                
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            with open(file_path, "rb") as f:
                self.wfile.write(f.read())
            return
            
        super().do_GET()


def _run_http_server(port: int):
    server = HTTPServer(("0.0.0.0", port), _StaticHandler)
    print(f"[Dashboard] HTTP server running at http://localhost:{port}")
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

    server = await websockets.serve(
        lambda ws: _ws_handler(ws, telemetry_source),
        "0.0.0.0",
        port,
    )
    print(f"[Dashboard] WebSocket server running at ws://localhost:{port}")
    await server.wait_closed()


def start_dashboard_server(
    http_port: int = 8000,
    ws_port: int = 8001,
    telemetry_source=None,
    open_browser: bool = True,
    blocking: bool = True,
):
    if telemetry_source is None:
        from mock_data import MockDroneTelemetry
        telemetry_source = MockDroneTelemetry(tick_rate=20.0)
        print("[Dashboard] Using MOCK telemetry data")

    http_thread = threading.Thread(target=_run_http_server, args=(http_port,), daemon=True)
    http_thread.start()

    if open_browser:
        url = f"http://localhost:{http_port}"
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
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("  RL Drone Play-Mode Dashboard")
    print("  Mock Data Mode")
    print("=" * 60)

    try:
        start_dashboard_server(
            http_port=args.http_port,
            ws_port=args.ws_port,
            open_browser=not args.no_browser,
            blocking=True,
        )
    except KeyboardInterrupt:
        print("\n[Dashboard] Server stopped.")


if __name__ == "__main__":
    main()
