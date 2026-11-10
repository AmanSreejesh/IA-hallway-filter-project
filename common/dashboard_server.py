"""
The touchscreen dashboard: a tiny web server that main.py runs alongside the camera loop.

  - /          the dashboard page (files in common/dashboard/)
  - /stream    the camera feed, with the filters already drawn on, as an MJPEG stream
  - /api/state what's on right now (polled by the page about once a second)
  - /api/do    the page's buttons: start (tap the resting screen), pick a filter, take turns, calibrate,
               set the 30-inch distance, debug view

It only uses Python's standard library, so there's nothing extra to install. It listens on
127.0.0.1 by default, which means only the computer it runs on can see the camera feed.
Nothing is saved: frames are turned into JPEGs in memory, sent, and thrown away.
"""
import json
import mimetypes
import os
import queue
import shutil
import subprocess
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

PAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard")
JPEG_QUALITY = 80          # lower = less work for the Pi, blurrier picture
ACTIONS = {"start", "select", "auto", "next", "calibrate", "zone", "debug"}
mimetypes.add_type("font/woff2", ".woff2")


class Dashboard:
    def __init__(self, host="127.0.0.1", port=8000):
        self.commands = queue.Queue()
        self.state = {}
        self._cond = threading.Condition()
        self._frame = None
        self._frame_id = 0
        self._jpeg = None
        self._jpeg_id = -1
        self.viewers = 0
        self.running = True
        handler = type("Handler", (_Handler,), {"dashboard": self})
        self.server = None
        for p in range(port, port + 10):   # if the port is busy, try the next few
            try:
                self.server = ThreadingHTTPServer((host, p), handler)
                break
            except OSError:
                continue
        if self.server is None:
            raise OSError(f"no free port between {port} and {port + 9}")
        self.server.daemon_threads = True
        self.url = f"http://{'localhost' if host in ('127.0.0.1', '0.0.0.0') else host}:{self.server.server_port}/"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    # --- called by main.py ---
    def publish(self, frame) -> None:
        """Hand over the newest finished frame. Skipped entirely when nobody is watching."""
        if not self.viewers:
            return
        with self._cond:
            self._frame = frame
            self._frame_id += 1
            self._cond.notify_all()

    def pending(self):
        """Commands from the page since last time, e.g. ("select", 1) or ("debug", None)."""
        while True:
            try:
                yield self.commands.get_nowait()
            except queue.Empty:
                return

    def close(self) -> None:
        self.running = False
        with self._cond:
            self._cond.notify_all()
        self.server.shutdown()
        self.server.server_close()

    # --- used by the request handlers ---
    def next_jpeg(self, last_id):
        """Wait for a frame newer than `last_id`. Each frame is encoded once, however many
        screens are watching."""
        with self._cond:
            self._cond.wait_for(lambda: self._frame_id != last_id or not self.running, timeout=1.0)
            if self._frame_id == last_id or self._frame is None:
                return None, last_id
            if self._jpeg_id != self._frame_id:
                ok, buf = cv2.imencode(".jpg", self._frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                if ok:
                    self._jpeg, self._jpeg_id = buf.tobytes(), self._frame_id
            return self._jpeg, self._frame_id


class _Handler(BaseHTTPRequestHandler):
    dashboard = None   # filled in by Dashboard
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):   # keep the terminal quiet
        pass

    def _send(self, code, body=b"", kind="text/plain; charset=utf-8", cache="no-store"):
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/stream":
            return self._stream()
        if path == "/api/state":
            return self._send(200, json.dumps(self.dashboard.state).encode(), "application/json")
        if path == "/":
            path = "/index.html"
        # Only serve files that are really inside common/dashboard/
        full = os.path.realpath(os.path.join(PAGE_DIR, path.lstrip("/")))
        if not full.startswith(PAGE_DIR + os.sep) or not os.path.isfile(full):
            return self._send(404, b"Not found")
        with open(full, "rb") as f:
            body = f.read()
        kind = mimetypes.guess_type(full)[0] or "application/octet-stream"
        if kind.startswith("text/") or kind.endswith("javascript"):
            kind += "; charset=utf-8"
        self._send(200, body, kind, "max-age=86400" if "/fonts/" in path else "no-cache")

    def do_POST(self):
        if self.path != "/api/do":
            return self._send(404, b"Not found")
        try:
            size = min(int(self.headers.get("Content-Length", 0)), 1024)
            data = json.loads(self.rfile.read(size) or b"{}")
            action = data.get("action")
            if action not in ACTIONS:
                raise ValueError(action)
            value = data.get("value")
            if action == "select" and not isinstance(value, int):
                raise ValueError(value)
            if action == "start" and value is not None and not isinstance(value, int):
                raise ValueError(value)   # start, optionally with the filter they tapped
        except (ValueError, TypeError, AttributeError):
            return self._send(400, b"Bad request")
        self.dashboard.commands.put((action, value))
        self._send(204)

    def _stream(self):
        d = self.dashboard
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        with d._cond:
            d.viewers += 1
        last = 0
        try:
            while d.running:
                jpeg, last = d.next_jpeg(last)
                if jpeg is None:
                    continue
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                 + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass
        finally:
            with d._cond:
                d.viewers -= 1
            self.close_connection = True


def open_browser(url: str, kiosk: bool) -> None:
    """On the Pi, open Chromium full screen with no browser bars (kiosk mode).
    Anywhere else, open a normal tab; tapping the page once makes it full screen."""
    chromium = shutil.which("chromium-browser") or shutil.which("chromium")
    if kiosk and chromium:
        subprocess.Popen([chromium, "--kiosk", "--noerrdialogs", "--disable-infobars", "--no-first-run",
                          "--disable-session-crashed-bubble", "--check-for-update-interval=31536000",
                          "--password-store=basic", url],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        webbrowser.open(url)
