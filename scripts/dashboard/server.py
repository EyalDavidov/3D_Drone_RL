"""Dashboard server — HTTP + WebSocket for the RL Drone Dashboard.

Usage:
    python scripts/dashboard/server.py          # mock-data mode (default)
    python scripts/dashboard/server.py --port 8000

When integrated with play.py, the server is started in a background thread
and receives real telemetry instead of mock data.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import threading
import webbrowser
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

# ---------------------------------------------------------------------------
# Resolve paths
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_STATIC_DIR = _SCRIPT_DIR / "static"


# ---------------------------------------------------------------------------
# HTTP server (serves static files)
# ---------------------------------------------------------------------------
class _StaticHandler(SimpleHTTPRequestHandler):
    """Serve files from the static/ directory."""

    def __init__(self, *args, directory=None, **kwargs):
        super().__init__(*args, directory=str(_STATIC_DIR), **kwargs)

    def log_message(self, format, *args):
        # Suppress noisy HTTP logs
        pass

    def end_headers(self):
        # Add CORS headers for local development
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()


def _run_http_server(port: int):
    """Run the HTTP server in a separate thread."""
    server = HTTPServer(("0.0.0.0", port), _StaticHandler)
    print(f"[Dashboard] HTTP server running at http://localhost:{port}")
    server.serve_forever()


# ---------------------------------------------------------------------------
# WebSocket server (streams telemetry)
# ---------------------------------------------------------------------------
async def _ws_handler(websocket, telemetry_source):
    """Handle a single WebSocket connection."""
    print(f"[Dashboard] WebSocket client connected: {websocket.remote_address}")
    try:
        while True:
            data = telemetry_source.tick()
            await websocket.send(json.dumps(data))
            await asyncio.sleep(1.0 / telemetry_source.tick_rate)
    except Exception:
        print(f"[Dashboard] WebSocket client disconnected: {websocket.remote_address}")


async def _run_ws_server(port: int, telemetry_source):
    """Run the WebSocket server."""
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


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def start_dashboard_server(
    http_port: int = 8000,
    ws_port: int = 8001,
    telemetry_source=None,
    open_browser: bool = True,
    blocking: bool = True,
):
    """Start the dashboard (HTTP + WebSocket servers).
    
    Args:
        http_port: Port for the HTTP static file server.
        ws_port: Port for the WebSocket telemetry stream.
        telemetry_source: Object with a .tick() method returning telemetry dict.
                          If None, uses MockDroneTelemetry.
        open_browser: Whether to open the browser automatically.
        blocking: If True, blocks the calling thread. If False, runs in background threads.
    """
    # Default to mock data
    if telemetry_source is None:
        from mock_data import MockDroneTelemetry
        telemetry_source = MockDroneTelemetry(tick_rate=20.0)
        print("[Dashboard] Using MOCK telemetry data")

    # Start HTTP server in a background thread
    http_thread = threading.Thread(
        target=_run_http_server,
        args=(http_port,),
        daemon=True,
    )
    http_thread.start()

    # Open browser
    if open_browser:
        url = f"http://localhost:{http_port}"
        print(f"[Dashboard] Opening browser at {url}")
        webbrowser.open(url)

    # Run WebSocket server
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
    parser.add_argument("--http-port", type=int, default=8000, help="HTTP server port (default: 8000)")
    parser.add_argument("--ws-port", type=int, default=8001, help="WebSocket server port (default: 8001)")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser automatically")
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
