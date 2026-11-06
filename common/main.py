"""
Main hallway filter system.
Combines each team member's filter and applies the shared features to all of them.

Run from the repo root:
    python common/main.py              # every filter, taking turns
    python common/main.py emoji        # only the filters you name (for testing your own)
    python common/main.py sharingan    # Aman's procedural iris effect
    python common/main.py astronaut clown_nose

Face tracking: each filter says which face tracker it wants, and this file uses whichever one
the filter on screen asks for (switching automatically when the filters take turns):
  - "mediapipe": follows 478 points per face and measures the head's 3D angle. Best for filters
    that turn and tilt with your head (the 3D emoji uses it). Needs: pip install mediapipe
  - "yunet": finds 5 points per face. Lighter, and doesn't need anything extra.
If MediaPipe isn't available, YuNet is used for compatible filters; iris filters are skipped.
TRACKER below can force one tracker for everything.

Opt-in zone: distance comes from how far apart the eyes look on camera (see estimate_distance).
Eyes are set closer together in kids and some adults, so each person's own eye gap is measured
from the size of their iris, which is almost the same in everyone (see iris_sample).
Filters show at 30 inches or closer: fully visible at 25 inches, fading out toward 30.
Set it up once with a tape measure: stand with your eyes exactly 30 inches from the camera,
look at the screen, and press z (or "Set 30-inch distance" in the staff controls).

Dashboard: the camera fills a full-screen web page (common/dashboard/). A small filter wheel sits
at the bottom: tap it to open it, spin it or tap a slice to pick a filter. A picked filter stays
until nobody has been in view for FILTER_SWITCH_SECONDS, then the filters take turns again.
On the Pi it opens Chromium in kiosk mode by itself; anywhere else it opens a browser tab (tap it
once to go full screen). The page is only reachable from this computer. Staff controls
(calibrate, debug view) open by holding the top-left corner of the screen for 2 seconds.
    python common/main.py --window      # the old OpenCV window instead of the dashboard
    python common/main.py --no-browser  # start the dashboard without opening a browser

Keys (in the dashboard page or the --window window):
       n = next filter   d = debug view (distance, head angles, FPS -- for calibrating)
       c = calibrate "looking forward": look straight at the screen and press c (saved for next time)
       z = set the 30-inch distance: eyes exactly 30 inches from the camera, then press z (saved)
       f = full screen, left/right arrows = turn the wheel (dashboard only)
Window only:  q = quit   l = start/stop recording head angles to common/pose_log.csv
In dashboard mode, stop the program with Ctrl+C in the terminal.

How a filter plugs in (see viraj/emoji_filter.py for an example):
  - Make a class with a `name`, an `apply(frame, face, opacity)` method that draws onto
    `frame`, and optionally a `forget(face_id)` method that's called when a person leaves.
  - Optionally set `tracker = "mediapipe"` on the class (the default is "yunet").
  - Add it to FILTERS below.
  - Filters only draw. This file decides who gets a filter, so every filter gets the same
    opt-in zone, fade, timer, and multi-user behaviour.

Packages: anything missing (OpenCV, NumPy, MediaPipe, moderngl) is installed automatically the
first time this runs (see installer.py).

Camera frames are never saved. Calibration and optional numeric pose logs are stored locally.
"""
import argparse
import csv
import importlib
import json
import math
import os
import sys
import tempfile
import threading
import time
from collections import deque
import urllib.request
from datetime import datetime, time as dtime

COMMON_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(COMMON_DIR))   # repo root, so "viraj.emoji_filter" imports
sys.path.insert(1, COMMON_DIR)                     # so "dashboard_server" imports

from installer import ensure_packages  # noqa: E402
ensure_packages()   # installs OpenCV, NumPy, MediaPipe... the first time, if they're missing

import cv2  # noqa: E402
import numpy as np  # noqa: E402

MODEL_PATH = os.path.join(COMMON_DIR, "face_detection_yunet_2023mar.onnx")
MODEL_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
             "face_detection_yunet/face_detection_yunet_2023mar.onnx")
MEDIAPIPE_MODEL_PATH = os.path.join(COMMON_DIR, "face_landmarker.task")
MEDIAPIPE_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                       "face_landmarker/float16/1/face_landmarker.task")

# name -> (module, class). Add your filter here.
FILTERS = {
    "emoji": ("viraj.emoji_filter", "EmojiFilter"),
    "sharingan": ("aman.sharingan_improved", "SharinganFilter"),
    "helmet": ("ishan.helmet_filter", "HelmetFilter"),
    "clown_nose": ("aryan.clown_nose_filter", "ClownNoseFilter"),
}
# Preserve the earlier astronaut command without duplicating the helmet in the wheel.
FILTER_ALIASES = {"astronaut": "helmet"}

# --- Feature toggles (True = on, False = off) ---
USE_OPT_IN_ZONE = True    # only people at or closer than ZONE_ENTER_INCHES get a filter
USE_WALK_BY_CHECK = True  # 3-second privacy intro: stay in the zone, facing the screen, for DWELL_SECONDS
                          # before your filter starts. People hurrying past never get one (issue #7)
USE_ANGLE_LIMIT = True    # stop drawing past MAX_ANGLE, independently for each person
USE_FADE_IN = True        # fade by distance between FADE_FULL_INCHES and ZONE_ENTER_INCHES
                          # (off = fully visible anywhere in the zone)
USE_TIMER = True          # 1 minute note and 3 minute reminder
USE_MULTI_USER = True     # everyone in view gets a filter (off = only the closest person)
USE_ENGAGEMENT = True     # rotate between filters every FILTER_SWITCH_SECONDS ('n' still works when off)
USE_SCHEDULE = True       # filters are off from FILTERS_OFF_AT to FILTERS_ON_AT (local time). Set to False
                          # for after-school events like showcases
USE_HINT_TEXT = True      # "Step up close..." message when nobody is using a filter

# --- Settings (keep in sync with constraints.md) ---
TRACKER = None            # None = whatever each filter asks for, or "mediapipe" / "yunet" to force one
CAMERA_SIZE = (1280, 720)   # widescreen, so the picture fills a 16:9 screen without cutting much off
PROCESS_HEIGHT = 480        # frames are shrunk to this height (854x480) before anything else, which
                            # keeps the Pi as fast as the old 640x480 and the screen still looks sharp
DETECT_SCALE = 0.5        # look for faces on a half-size copy of the frame (~4x less work on the Pi).
                          # People in the opt-in zone are big on screen, so nothing is lost.
NEW_FACE_EVERY = 3        # while someone's being tracked, scan the whole picture for new people every N frames
SEARCH_ANGLES = (40, -40, 80, -80, 130, -130, 180)   # tilts tried when nobody is in view (YuNet)
MEDIAPIPE_MAX_FACES = 8   # each extra face costs time on the Pi

# Opt-in zone. Distance comes from how far apart the eyes look on screen (see estimate_distance).
FOCAL_PX = 600            # camera focal length in pixels for a 640-wide picture (scaled to the real
                          # width automatically). Used for head angles and people off to the side
EYE_GAP_INCHES = 2.5      # average distance between a person's eyes, used until theirs is measured
ZONE_ENTER_INCHES = 30    # filters show at or closer than this (2.5 ft)...
FADE_FULL_INCHES = 25     # ...fully visible at or closer than this, fading out from here to 30 in
FADE_START = 0.1          # opacity right at the edge of the zone (30 in)
ZONE_CALIBRATE_SECONDS = 1.5   # pressing z averages the eye gap over this long
# Each person's own eye gap, measured against their iris (MediaPipe only; YuNet uses the average)
IRIS_MM = 11.7            # width of the coloured part of the eye: about the same in everyone over ~3
IRIS_MIN_SAMPLES = 4      # good frames needed before a person's own eye gap replaces the average
                          # (it keeps getting more exact with every good frame after that)
IRIS_MAX_TURN = 25        # only measure when the head is turned left/right less than this...
IRIS_MAX_SPEED = 2.0      # ...isn't moving faster than this (face widths per second, blur)...
IRIS_MIN_PX = 5           # ...and the iris is at least this many pixels wide

MAX_ANGLE = 60            # filters show up to and including 60 degrees on every axis (turn, nod,
                          # tilt) and stop past that, e.g. a passerby in profile (90 degrees)

# Smoothing for motion diagnostics; drawing uses current-frame geometry to avoid lag.
STILL_SMOOTHING = 1.0     # lower = steadier when still, but more lag (try 0.5 to 2)
MOVING_RESPONSE = 5.0     # higher = catches up faster when you move (try 2 to 10)


# Timer
TIMER_GENTLE_SECONDS = 60     # small note that a minute has passed
TIMER_FIRM_SECONDS = 180      # bigger reminder to get back to work
GENTLE_SHOW_SECONDS = 10      # how long the small note stays up
TIMER_RESET_SECONDS = 5       # step away (or look away) this long and your timer starts over.
                              # The camera losing your face for a moment doesn't count

# Multi-user tracking
LOST_GRACE_SECONDS = 0.5      # keep a person around this long if the detector misses them

# Engagement
FILTER_SWITCH_SECONDS = 30    # rotate to the next filter this often

# Schedule: filters turn off after dismissal so nobody heading to the buses gets stopped,
# and come back on in the morning
FILTERS_OFF_AT = dtime(16, 15)   # 4:15 pm
FILTERS_ON_AT = dtime(7, 30)     # 7:30 am


def filters_are_off(local_now=None) -> bool:
    """Use the display computer's local clock. Off from exactly 4:15 pm, overnight, until exactly
    7:30 am (7:30 itself is on)."""
    local_now = datetime.now() if local_now is None else local_now
    now = local_now.time()
    return USE_SCHEDULE and (now >= FILTERS_OFF_AT or now < FILTERS_ON_AT)


# Looking forward. The camera sits above or below the screen, so someone looking at the screen
# looks slightly down or up to the camera (and each tracker has its own small bias). These are the
# head angles that count as "looking straight ahead". Press c while looking at the screen to set
# them; they're saved to calibration.json (separately for YuNet and MediaPipe) and loaded every time.
CALIBRATION_PATH = os.path.join(COMMON_DIR, "calibration.json")
FORWARD = {"yaw": 0.0, "pitch": 0.0}
DISTANCE_FOCAL = None     # the camera's focal length in pixels, measured by pressing z (None = FOCAL_PX)
POSE_LOG_PATH = os.path.join(COMMON_DIR, "pose_log.csv")


# --- Imports: each team member's filter ---
def load_filters(names: list) -> list:
    """Import the filters by name. A broken filter is skipped instead of crashing everything."""
    filters = []
    loaded = set()
    for name in names or FILTERS:
        name = FILTER_ALIASES.get(name, name)
        if name in loaded:
            continue
        if name not in FILTERS:
            print(f"Unknown filter '{name}'. Choices: {', '.join(FILTERS)}")
            continue
        module_name, class_name = FILTERS[name]
        try:
            f = getattr(importlib.import_module(module_name), class_name)()
        except Exception as e:
            print(f"Skipping filter '{name}': {e}")
            continue
        filters.append(f)
        loaded.add(name)
        print(f"Loaded filter: {name}")
    return filters


def download(url: str, path: str) -> None:
    """Atomically cache a model so interrupted downloads cannot become live models."""
    if os.path.exists(path) and os.path.getsize(path) > 100_000:
        return
    print(f"Downloading {os.path.basename(path)} (one time only)...")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=os.path.dirname(path) or ".",
                                         prefix=os.path.basename(path) + ".",
                                         suffix=".part", delete=False) as output:
            temporary = output.name
            with urllib.request.urlopen(url, timeout=60) as source:
                while chunk := source.read(1024 * 1024):
                    output.write(chunk)
        if os.path.getsize(temporary) <= 100_000:
            raise RuntimeError(f"Incomplete model download: {os.path.basename(path)}")
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)
    print("Done.")


# --- Face detector ---
# detect() returns (row, tilt, pose, landmarks) per face:
#   row  = [x, y, w, h, eye1 x,y, eye2 x,y, nose x,y, mouth1 x,y, mouth2 x,y, score]
#   tilt = head tilt in degrees if the detector knows it, else None
#   pose = (yaw, pitch) if the detector measures them itself (MediaPipe does), else None
#   landmarks = 478x2 pixel coordinates (MediaPipe), else None
# YuNet always lists the points in the same order (eye, eye, nose, mouth corner, mouth corner),
# even when the head is upside down, so the order tells us which way the face is pointing.
class YuNetDetector:
    def __init__(self):
        download(MODEL_URL, MODEL_PATH)
        self.net = cv2.FaceDetectorYN.create(MODEL_PATH, "", CAMERA_SIZE, score_threshold=0.8)
        self.input_size = None
        self.search_step = 0
        self.frame_count = 0
        self.prev_gray = None

    def _detect(self, small: np.ndarray, center=None, angle: float = 0.0) -> list:
        """Find faces in `small`, first turning the picture by `angle` degrees around `center`.

        YuNet was trained on upright faces: on a tilted head its points stay level (at a 30
        degree tilt it still reported the eyes as flat) and past ~40 degrees it loses the face.
        Turning the picture so the face is upright, then turning the points back, fixes both.
        """
        size = (small.shape[1], small.shape[0])
        if size != self.input_size:
            self.net.setInputSize(size)
            self.input_size = size
        if angle == 0:
            _, faces = self.net.detect(small)
            return [] if faces is None else list(faces)
        turn = cv2.getRotationMatrix2D(center, angle, 1.0)
        _, faces = self.net.detect(cv2.warpAffine(small, turn, size))
        if faces is None:
            return []
        back = cv2.invertAffineTransform(turn)
        out = []
        for f in faces:
            pts = f[4:14].reshape(5, 2) @ back[:, :2].T + back[:, 2]
            x, y, w, h = f[:4]
            corners = np.array([[x, y], [x + w, y], [x, y + h], [x + w, y + h]]) @ back[:, :2].T + back[:, 2]
            (x0, y0), (x1, y1) = corners.min(axis=0), corners.max(axis=0)
            out.append(np.array([x0, y0, x1 - x0, y1 - y0, *pts.ravel(), f[14]], dtype=np.float32))
        return out

    def _tilt_change(self, gray: np.ndarray, center: tuple, radius: float) -> float:
        """How many degrees the face around `center` rotated since last frame.

        Follows dozens of small details on the face (optical flow) and measures how much
        they turned together. Much more accurate for tilting than YuNet's points.
        """
        if self.prev_gray is None or self.prev_gray.shape != gray.shape:
            return 0.0
        mask = np.zeros_like(gray)
        cv2.circle(mask, (int(center[0]), int(center[1])), int(radius), 255, -1)
        old = cv2.goodFeaturesToTrack(self.prev_gray, 60, 0.01, 3, mask=mask)
        if old is None or len(old) < 6:
            return 0.0
        new, ok, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, old, None, winSize=(15, 15), maxLevel=2)
        ok = ok.ravel() == 1
        if ok.sum() < 6:
            return 0.0
        motion, _ = cv2.estimateAffinePartial2D(old[ok], new[ok], method=cv2.RANSAC, ransacReprojThreshold=2.0)
        return 0.0 if motion is None else math.degrees(math.atan2(motion[1, 0], motion[0, 0]))

    def detect(self, frame: np.ndarray, tracked: list = ()) -> list:
        """Returns (row, tilt, None) per face. `tracked` = (nose, tilt, size) for each person from
        last frame. Their new tilt is last tilt + how far the face rotated (optical flow), and
        they're looked for in a copy of the picture turned so their face is upright.
        For brand-new faces, tilt is None (main.py starts it from the eye line)."""
        # Detect on a shrunken copy (the biggest speed-up on the Pi), then scale the points back up.
        small = cv2.resize(frame, None, fx=DETECT_SCALE, fy=DETECT_SCALE, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        self.frame_count += 1
        # Look over the whole picture for new people (every frame if nobody's here yet)
        found = [(f, None) for f in self._detect(small)] if not tracked or self.frame_count % NEW_FACE_EVERY == 0 else []

        for nose, tilt, size in tracked:
            center = (float(nose[0] * DETECT_SCALE), float(nose[1] * DETECT_SCALE))
            near = size * DETECT_SCALE * 2.5
            tilt = wrap(tilt + self._tilt_change(gray, center, size * DETECT_SCALE * 1.6))
            redo = [f for f in self._detect(small, center, tilt)
                    if np.hypot(f[8] - center[0], f[9] - center[1]) < near]
            # This person's upright-picture result (if any) is replaced by the straightened one
            same = [f for f, _ in found if np.hypot(f[8] - center[0], f[9] - center[1]) < near]
            found = [(f, t) for f, t in found if np.hypot(f[8] - center[0], f[9] - center[1]) >= near]
            if redo or same:
                found.append(((redo or same)[0], tilt))

        # Nobody in view: also try one tilted angle per frame, to catch someone already tilting their head
        if not found and not tracked:
            angle = SEARCH_ANGLES[self.search_step % len(SEARCH_ANGLES)]
            self.search_step += 1
            found = [(f, None) for f in self._detect(small, (small.shape[1] / 2, small.shape[0] / 2), angle)]

        self.prev_gray = gray
        for f, _ in found:
            f[:14] /= DETECT_SCALE
        return [(f, tilt, None, None) for f, tilt in found]


class MediaPipeDetector:
    """MediaPipe's Face Landmarker: follows 478 points on each face from frame to frame, handles
    tilted heads, and measures the head's 3D angle itself."""
    provides_landmarks = True
    # Which of the 478 points become our 5 (eye, eye, nose, mouth corner, mouth corner). The first
    # of each pair is the one on the left of the screen when the head is upright.
    EYE_A, EYE_B, NOSE_TIP, MOUTH = (33, 133), (362, 263), 1, (61, 291)
    TO_OPENCV = np.diag([1.0, -1.0, -1.0])   # MediaPipe's axes (y up, z toward you) -> OpenCV's

    def __init__(self):
        import mediapipe as mp   # only needed when a filter asks for MediaPipe
        from mediapipe.tasks.python import BaseOptions, vision
        self.mp = mp
        download(MEDIAPIPE_MODEL_URL, MEDIAPIPE_MODEL_PATH)
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MEDIAPIPE_MODEL_PATH),
            # VIDEO mode follows faces from frame to frame instead of searching the whole
            # picture every time: steadier, and much faster on the Pi
            running_mode=vision.RunningMode.VIDEO,
            num_faces=MEDIAPIPE_MAX_FACES,
            output_facial_transformation_matrixes=True)   # MediaPipe's own 3D head angle
        self.landmarker = vision.FaceLandmarker.create_from_options(options)
        self.last_ms = 0

    def detect(self, frame: np.ndarray, tracked: list = ()) -> list:
        """Detect each face independently in this exact frame, including iris points.

        Turning the whole image for the closest face made other people's opposite
        head tilts harder to detect. Each face supplies its own pose instead.
        """
        h, w = frame.shape[:2]
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ms = max(int(time.monotonic() * 1000), self.last_ms + 1)
        self.last_ms = ms
        result = self.landmarker.detect_for_video(image, ms)
        return self.decode(result, w, h)

    @classmethod
    def decode(cls, result, w, h):
        out = []
        for index, landmarks in enumerate(result.face_landmarks):
            if len(landmarks) != 478:
                continue
            pts = np.array([(p.x * w, p.y * h) for p in landmarks], dtype=np.float32)
            if not np.isfinite(pts).all():
                continue
            (x0, y0), (x1, y1) = pts[:468].min(axis=0), pts[:468].max(axis=0)
            row = np.array([x0, y0, x1 - x0, y1 - y0,
                            *pts[list(cls.EYE_A)].mean(axis=0), *pts[list(cls.EYE_B)].mean(axis=0),
                            *pts[cls.NOSE_TIP], *pts[cls.MOUTH[0]], *pts[cls.MOUTH[1]], 1.0], dtype=np.float32)
            try:
                matrix = np.asarray(result.facial_transformation_matrixes[index])
                if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
                    raise ValueError("Invalid head pose")
                # Remove Procrustes scale before extracting angles.
                u, singular, vt = np.linalg.svd(cls.TO_OPENCV @ matrix[:3, :3] @ cls.TO_OPENCV)
                if singular.min() < 1e-6 or np.linalg.det(u @ vt) < 0:
                    raise ValueError("Degenerate head pose")
                yaw, pitch, roll = angles_from_rotation(u @ vt)
            except (IndexError, ValueError, np.linalg.LinAlgError):
                yaw = pitch = roll = float("inf")
            out.append((row, roll, (yaw, pitch), pts))
        return out

    def close(self):
        self.landmarker.close()


def make_detector(name: str):
    """Start the face tracker called `name` ("mediapipe" or "yunet"). Falls back to YuNet if
    MediaPipe isn't installed or won't start."""
    if name == "mediapipe":
        try:
            detector = MediaPipeDetector()
            print("Face tracking: MediaPipe")
            return detector
        except Exception as e:
            print(f"MediaPipe isn't available ({e}); using YuNet instead. To get it: pip install mediapipe")
    print("Face tracking: YuNet")
    return YuNetDetector()


def load_calibration():
    """Ignore corrupt installation settings instead of preventing startup."""
    if not os.path.exists(CALIBRATION_PATH):
        return {}
    try:
        with open(CALIBRATION_PATH) as source:
            saved = json.load(source)
        if not isinstance(saved, dict):
            raise ValueError("expected a calibration object")
        clean = {}
        for tracker, values in saved.items():
            if not isinstance(values, dict):
                continue
            angles = {axis: float(values.get(axis, 0)) for axis in ("yaw", "pitch")}
            if all(math.isfinite(v) and abs(v) <= 90 for v in angles.values()):
                clean[tracker] = angles
                focal = float(values.get("distance_focal", 0) or 0)
                if not focal and values.get("distance_k"):   # saved by an older version
                    focal = float(values["distance_k"]) / EYE_GAP_INCHES
                if math.isfinite(focal) and focal > 0:
                    clean[tracker]["distance_focal"] = focal
        return clean
    except (OSError, ValueError, TypeError) as error:
        print(f"Ignoring invalid calibration ({error}); press c to recalibrate.")
        return {}


# --- Camera ---
class Camera:
    """Reads the camera on its own thread and keeps only the newest frame.

    Without this, a slow Pi falls behind and the camera's queue hands it frames from a
    moment ago, so the filter looks like it's lagging behind your face.
    """

    def __init__(self, index=0):
        if sys.platform.startswith("win"):
            backend = cv2.CAP_DSHOW
        elif sys.platform.startswith("linux"):
            backend = cv2.CAP_V4L2   # read the webcam directly; OpenCV's GStreamer route fails on the Pi
        else:
            backend = cv2.CAP_ANY
        self.cap = cv2.VideoCapture(index, backend)
        if self.cap.isOpened():
            # MJPG lets USB webcams send 1280x720 at full speed (raw frames are too big for USB 2)
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_SIZE[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_SIZE[1])
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._start(self.cap.isOpened())

    def _start(self, running):
        self.frame = None
        self.new_frame = threading.Event()
        self.running = running
        self.thread = threading.Thread(target=self._loop, daemon=True)
        if self.running:
            self.thread.start()

    def _grab(self):
        ok, frame = self.cap.read()
        return frame if ok else None

    def _close(self):
        self.cap.release()

    def _loop(self):
        try:
            while self.running:
                self.frame = self._grab()
                self.running = self.frame is not None
                self.new_frame.set()
        except Exception:   # cv2.error, or the Pi camera library giving up
            self.frame = None
        finally:
            self.running = False
            self.new_frame.set()

    def read(self):
        """Wait for a frame we haven't seen yet. Returns None if the camera stopped."""
        if not self.new_frame.wait(2.0):
            return None  # A stalled stream must not reuse an old face position.
        self.new_frame.clear()
        return self.frame

    def release(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(1.0)
        self._close()

    def shows_picture(self, seconds=3.0):
        """True once a frame isn't pure black. Some cameras (like an iPhone the Mac picked up
        as a Continuity Camera) open fine but only ever send black frames."""
        end = time.time() + seconds
        while self.running and time.time() < end:
            frame = self.read()
            if frame is not None and frame.max() > 30:
                return True
        return False


class PiCamera(Camera):
    """The Raspberry Pi camera module (ribbon cable). OpenCV can't read it, so this uses picamera2."""

    def __init__(self):
        from picamera2 import Picamera2   # comes with Raspberry Pi OS
        self.cam = Picamera2()
        # "RGB888" is stored blue-green-red, which is the order OpenCV wants
        self.cam.configure(self.cam.create_video_configuration(main={"size": CAMERA_SIZE, "format": "RGB888"}))
        self.cam.start()
        self._start(True)

    def _grab(self):
        return self.cam.capture_array()

    def _close(self):
        self.cam.stop()
        self.cam.close()


def open_camera(index=None):
    """Open the camera at index, or if none was asked for, the first one that shows a picture.
    On a Raspberry Pi with no working USB webcam, try the ribbon-cable camera module."""
    if index is not None:
        return Camera(index)
    fallback = None
    for i in range(4):
        cam = Camera(i)
        if not cam.running:
            cam.release()
            continue
        if cam.shows_picture():
            if i:
                print(f"Using camera {i} (the ones before it only sent black frames).")
            if fallback:
                fallback.release()
            return cam
        print(f"Camera {i} is only sending black frames, trying the next one...")
        if fallback:
            cam.release()
        else:
            fallback = cam
    if is_raspberry_pi():
        if fallback:   # free the camera so picamera2 can have it
            fallback.release()
            fallback = None
        try:
            cam = PiCamera()
            print("Using the Raspberry Pi camera module.")
            return cam
        except Exception as error:
            print(f"No Raspberry Pi camera module either ({error}).")
    return fallback or Camera(0)


# --- Raspberry Pi ---
def is_raspberry_pi() -> bool:
    """True when running on a Raspberry Pi (it writes its model name to this file)."""
    try:
        with open("/proc/device-tree/model") as f:
            return "raspberry pi" in f.read().lower()
    except OSError:
        return False


# --- Face data handed to every filter ---
class Face:
    """One tracked person. Drawing positions, size, pose and mesh follow the current frame.

    id                       stays the same while the person is in view
    box                      (x, y, w, h)
    left_eye, right_eye, nose, mouth_left, mouth_right   (x, y) points (left = left on screen when upright)
    size                     face size in pixels (about the eye gap, but steady when you turn)
    roll                     head tilt in degrees, -180 to 180 (ear toward shoulder; 180 = upside down)
    yaw                      head turn left/right in degrees (0 = facing the camera, positive = turned toward screen right)
    pitch                    head nod up/down in degrees (positive = looking down)
    distance                 estimated inches from the camera
    """

    def __init__(self, face_id: int):
        self.id = face_id
        self.points = None      # 7x2 array: box corner, box size, eyes, nose, mouth corners
        self.size = 0.0
        self.roll = self.yaw = self.pitch = 0.0
        self.distance = 999.0
        self.speed = 0.0        # face widths per second
        # system state (filters don't need these)
        self.point_filter = Smoother()
        self.iris_samples = deque(maxlen=90)   # (eye gap in inches, iris width in pixels) from good frames
        self.eye_gap = None     # this person's own eye gap in inches, once IRIS_MIN_SAMPLES are in
        self.pose_guess = None  # last 3D head pose, used as the starting point for the next frame
        self.tilt = None        # current tilt: MediaPipe pose or YuNet optical flow
        self.landmarks = None   # current-frame iris/eyelid points; None for YuNet
        self.measured_angles = (float("inf"),) * 3  # calibrated yaw, pitch, roll
        self.raw_yaw = self.raw_pitch = 0.0   # this frame's measurement, before calibration and smoothing
        self.last_seen = 0.0
        self.opted_in = False   # done the privacy intro
        self.eligible_since = None     # when they last became eligible (the intro's start)
        self.ineligible_since = None   # when they last stopped being eligible
        self.opacity = 0.0
        self.session_start = None
        self.last_eligible = 0.0

    box = property(lambda self: (*self.points[0], *self.points[1]))
    left_eye = property(lambda self: tuple(self.points[2]))
    right_eye = property(lambda self: tuple(self.points[3]))
    nose = property(lambda self: tuple(self.points[4]))
    mouth_left = property(lambda self: tuple(self.points[5]))
    mouth_right = property(lambda self: tuple(self.points[6]))


# --- Smoothing ---
def smooth(old, new, amount: float):
    """Blend toward the new value. amount = how much of the new value to take (0..1)."""
    return new if old is None else old + (new - old) * amount


class Smoother:
    """Smooths a value a lot while it's still and hardly at all while it's moving.

    A fixed amount of smoothing always lags behind (the old problem). This is the
    "One Euro filter": it measures how fast the value is changing and loosens up to match.
    """

    def __init__(self):
        self.value = None
        self.velocity = None

    @staticmethod
    def _amount(dt: float, cutoff):
        return 1 / (1 + 1 / (2 * math.pi * cutoff * dt))

    def __call__(self, new, dt: float, scale: float = 1.0):
        new = np.asarray(new, dtype=np.float64)
        if self.value is None or dt <= 0:
            self.value, self.velocity = new, np.zeros_like(new)
            return self.value
        self.velocity = smooth(self.velocity, (new - self.value) / dt, self._amount(dt, 1.0))
        # scale makes "fast" mean the same thing whether you're near the camera or far away
        cutoff = STILL_SMOOTHING + MOVING_RESPONSE * np.abs(self.velocity) / scale
        self.value = smooth(self.value, new, self._amount(dt, cutoff))
        return self.value


# --- Head pose ---
# Rough 3D positions (mm) of the 5 face points on an average head, with the nose tip at 0.
# Axes match OpenCV's camera: x = right, y = down, z = away from the camera.
FACE_MODEL = np.array([
    [-31.5, -42.0, 27.0],   # eye (left on screen)
    [31.5, -42.0, 27.0],    # eye (right on screen)
    [0.0, 0.0, 0.0],        # nose tip
    [-25.0, 38.0, 18.0],    # mouth corner (left on screen)
    [25.0, 38.0, 18.0],     # mouth corner (right on screen)
])


def raw_points(det: np.ndarray) -> np.ndarray:
    """YuNet row -> 7x2 points: box corner, box size, eye, eye, nose, mouth corner, mouth corner.
    Kept in YuNet's order (not sorted left/right), so it still works when the head is tilted
    sideways or upside down."""
    return det[:14].reshape(7, 2).astype(np.float64)


def angles_from_rotation(R: np.ndarray) -> tuple:
    """3x3 rotation (OpenCV camera axes) -> (yaw, pitch, roll) in degrees, in this file's directions.
    Uses the same order the emoji filter builds its rotation in (roll, then turn, then nod), so
    big combined tilts and turns come back exactly instead of getting mixed up."""
    yaw = math.degrees(math.asin(max(-1.0, min(1.0, R[2, 0]))))
    pitch = math.degrees(math.atan2(R[2, 1], R[2, 2]))
    roll = math.degrees(math.atan2(R[1, 0], R[0, 0]))
    return yaw, pitch, roll


def solve_head_pose(face: Face, pts: np.ndarray, frame_size: tuple) -> tuple:
    """Fit the 3D face model to the 5 points (eye, eye, nose, mouth corner, mouth corner)
    to get (yaw, pitch). This is a real 3D fit, so turning and nodding are measured together."""
    w, h = frame_size
    camera = np.array([[FOCAL_PX, 0, w / 2], [0, FOCAL_PX, h / 2], [0, 0, 1]], dtype=np.float64)
    if face.pose_guess is None:
        eye_dist = max(float(np.hypot(*(pts[1] - pts[0]))), 1.0)
        face.pose_guess = (np.zeros((3, 1)), np.array([[0.0], [0.0], [FOCAL_PX * 63 / eye_dist]]))
    rvec, tvec = (v.copy() for v in face.pose_guess)
    ok, rvec, tvec = cv2.solvePnP(FACE_MODEL, pts, camera, None, rvec, tvec,
                                  useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
    yaw, pitch, _ = angles_from_rotation(cv2.Rodrigues(rvec)[0]) if ok else (90.0, 90.0, 0.0)
    if not ok or abs(yaw) > 80 or abs(pitch) > 80 or tvec[2, 0] <= 0:
        face.pose_guess = None   # fit went wrong -- start fresh next frame
        return float("nan"), float("nan")
    face.pose_guess = (rvec, tvec)
    return yaw, pitch


def wrap(angle: float) -> float:
    """Keep an angle between -180 and 180."""
    return (angle + 180) % 360 - 180


def face_size(p: np.ndarray, yaw: float, pitch: float) -> float:
    """Deproject the stable eye-corner gap; gaze does not affect zone distance."""
    eye_dist = max(float(np.linalg.norm(p[3] - p[2])), 1.0)
    cos_yaw = max(math.cos(math.radians(yaw)), 0.5) if math.isfinite(yaw) else 1.0
    return eye_dist / cos_yaw


def frontal_eye_gap(p: np.ndarray, yaw: float, frame_size: tuple) -> float:
    """How wide, in pixels, the eye gap would look if the head faced the camera from where it is.

    Pinhole camera, focal length f, picture centre c. For eyes at depth Z along the ray
    r = (point - c) / f, an eye-line vector d (|d| = G inches) shows up on screen as
        g = f / Z * (d_xy - d_z * r)
    A head turned by yaw has |d_xy| = G cos(yaw) and d_z = G sin(yaw), taking d from the eye on the
    left of the screen to the eye on the right (positive yaw = turned toward screen right, so the
    right-hand eye is the farther one). Writing k = f * G / Z (the answer), |g / k + sin(yaw) r| = cos(yaw):
        |g|^2 t^2 + 2 sin(yaw) (g . r) t + sin(yaw)^2 |r|^2 - cos(yaw)^2 = 0,   k = 1 / t
    With yaw = 0 this is just |g|; straight ahead of the camera it's |g| / cos(yaw)."""
    w, h = frame_size
    g = p[3] - p[2]
    if g[0] < 0:   # point it left to right on screen (tilts are limited to 60 degrees)
        g = -g
    gg = max(float(g @ g), 1.0)
    r = ((p[2] + p[3]) / 2 - (w / 2, h / 2)) / FOCAL_PX
    turn = math.radians(max(-60.0, min(60.0, yaw))) if math.isfinite(yaw) else 0.0   # past 60 never shows
    sin_t, cos_t = math.sin(turn), math.cos(turn)
    gr = float(g @ r)
    t = (-sin_t * gr + math.sqrt(max((sin_t * gr) ** 2 - gg * (sin_t ** 2 * float(r @ r) - cos_t ** 2), 0.0))) / gg
    return 1.0 / t if t > 0 else math.sqrt(gg)


def off_axis(p: np.ndarray, frame_size: tuple) -> float:
    """How much farther someone is than their depth, because they're off to the side or up/down:
    a point (dx, dy) pixels from the middle of the picture is depth * sqrt(1 + (dx/f)^2 + (dy/f)^2) away."""
    w, h = frame_size
    mid = (p[2] + p[3]) / 2   # halfway between the eyes
    return math.hypot((mid[0] - w / 2) / FOCAL_PX, (mid[1] - h / 2) / FOCAL_PX, 1.0)


def estimate_distance(p: np.ndarray, raw_yaw: float, frame_size: tuple, gap_inches: float = None) -> float:
    """Inches from the camera to the point between the eyes.

    An eye gap G inches wide at depth Z looks k = f * G / Z pixels wide when facing the camera
    (frontal_eye_gap undoes a head turn), so Z = f * G / k and the distance is Z * off_axis.
    G is this person's own eye gap (gap_inches, measured from their iris) or the average.
    f is the camera's focal length, measured by pressing z with someone exactly ZONE_ENTER_INCHES
    away, which makes the zone edge exact for this camera and its settings."""
    focal = DISTANCE_FOCAL or FOCAL_PX
    gap = gap_inches or EYE_GAP_INCHES
    return focal * gap / frontal_eye_gap(p, raw_yaw, frame_size) * off_axis(p, frame_size)


def calibration_sample(face, frame_size: tuple, inches: float = None) -> float:
    """f * G for someone standing exactly `inches` away (estimate_distance run backwards).
    Divide by their eye gap to get the camera's focal length."""
    inches = ZONE_ENTER_INCHES if inches is None else inches
    return frontal_eye_gap(face.points, face.raw_yaw, frame_size) * inches / off_axis(face.points, frame_size)


# MediaPipe points for each eye: iris centre, iris edge left/right, iris edge top/bottom, eye corners
IRIS_EYES = ((468, (469, 471), (470, 472), (33, 133)),
             (473, (474, 476), (475, 477), (362, 263)))


def iris_sample(face, frame_size: tuple):
    """This frame's measurement of the person's own eye gap, as (inches, iris width in pixels),
    or None if the frame isn't good enough to measure from.

    The iris is about IRIS_MM wide in everyone, so eye gap = IRIS_MM * (eye gap px / iris px).
    Both are measured in the same frame, so walking toward or away from the camera cancels out.
    Only the iris's side-to-side width is used: looking up or down (at a screen below the camera,
    say) squashes it top to bottom but not side to side. Frames are skipped when the head is
    turned left/right, the eyes look sideways (the iris then looks narrower), the face is moving
    fast (blur), or the iris is too small to measure."""
    lm = face.landmarks
    if lm is None or lm.shape != (478, 2) or not np.isfinite(lm).all():
        return None
    if not math.isfinite(face.raw_yaw) or abs(face.raw_yaw) > IRIS_MAX_TURN or face.speed > IRIS_MAX_SPEED:
        return None
    widths = []
    for centre, across, _, (corner_a, corner_b) in IRIS_EYES:
        axis = lm[corner_b] - lm[corner_a]
        length = float(axis @ axis)
        if length < 1.0:
            return None
        if not 0.38 <= float((lm[centre] - lm[corner_a]) @ axis) / length <= 0.62:
            return None   # looking sideways (the iris would look narrower)
        widths.append(float(np.linalg.norm(lm[across[0]] - lm[across[1]])))
    iris_px = sum(widths) / len(widths)
    if iris_px < IRIS_MIN_PX:
        return None
    inches = IRIS_MM / 25.4 * frontal_eye_gap(face.points, face.raw_yaw, frame_size) / iris_px
    return (inches, iris_px) if 1.6 <= inches <= 3.2 else None   # 40-80 mm, else a bad detection


def weighted_median(samples) -> float:
    """Median of (value, weight) pairs. Closer frames (bigger iris) count for more."""
    ordered = sorted(samples)
    half = sum(w for _, w in ordered) / 2
    total = 0.0
    for value, weight in ordered:
        total += weight
        if total >= half:
            return value
    return ordered[-1][0]


def update_face(face: Face, det: np.ndarray, tilt, pose, now: float, frame_size: tuple, landmarks=None) -> None:
    dt = now - face.last_seen if face.points is not None else 0.0
    scale = max(face.size, 1.0)
    if tilt is not None:
        face.tilt = tilt                # carried over from last frame by optical flow
    elif face.tilt is None:            # new face: start from the eye line
        face.tilt = math.degrees(math.atan2(det[7] - det[5], det[6] - det[4]))
    # A detection gap: the filter hides (update_opacity decides whether the intro replays)
    if dt > LOST_GRACE_SECONDS:
        face.opacity = 0.0
        face.iris_samples.clear()   # could be someone else now
        face.eye_gap = None
    face.landmarks = landmarks
    face.points = raw_points(det)
    face.point_filter(face.points, dt, scale)
    # Diagnostic motion estimate only: head sways must not be treated as walking past.
    face.speed = float(np.hypot(*face.point_filter.velocity[4])) / (scale * 2.6)

    # Turn the points so the face is upright, then fit the 3D head there. That way the turn and
    # nod come out the same however far the head is tilted, and the tilt itself comes from
    # optical flow (YuNet's own points barely follow a tilt).
    if pose is not None:
        yaw, pitch = pose   # the detector measured it (MediaPipe)
    else:
        c, s = math.cos(math.radians(-face.tilt)), math.sin(math.radians(-face.tilt))
        nose = face.points[4]
        upright = (face.points[2:7] - nose) @ np.array([[c, s], [-s, c]]) + nose
        yaw, pitch = solve_head_pose(face, upright, frame_size)
    face.raw_yaw, face.raw_pitch = yaw, pitch
    yaw, pitch = yaw - FORWARD["yaw"], pitch - FORWARD["pitch"]   # so looking at the screen = 0, 0
    face.measured_angles = (yaw, pitch, wrap(face.tilt))
    # A filter must turn by the measured amount on this frame, not lag behind
    # the person's movement while the old angle is being averaged away.
    if np.isfinite(face.measured_angles).all():
        face.yaw, face.pitch, face.roll = face.measured_angles
    # Use current geometry for all renderers: smoothing scale/anchors leaves an uncovered
    # nose on an approach and trails behind a head sway, especially at close distances.
    face.size = face_size(face.points, face.raw_yaw, face.raw_pitch)
    sample = iris_sample(face, frame_size)
    if sample:
        face.iris_samples.append(sample)
    if len(face.iris_samples) >= IRIS_MIN_SAMPLES:
        face.eye_gap = weighted_median(face.iris_samples)
    face.distance = estimate_distance(face.points, face.raw_yaw, frame_size, face.eye_gap)
    face.last_seen = now


# --- Multi-user ---
# Match each detection to the closest person from last frame, so everyone keeps their id.
def track(faces: dict, detections: list, now: float, next_id: list, frame_size: tuple) -> None:
    unmatched = dict(faces)
    for det, tilt, pose, landmarks in sorted(detections, key=lambda d: -d[0][2]):   # biggest (closest) faces first
        nose = det[8:10]
        best, best_dist = None, None
        for face in unmatched.values():
            d = float(np.hypot(*(face.points[4] - nose)))
            if d <= max(face.size * 2.6, float(det[2]) * 0.75, 60) and (best_dist is None or d < best_dist):
                best, best_dist = face, d
        if best is None:
            best = Face(next_id[0])
            next_id[0] += 1
            faces[best.id] = best
        else:
            del unmatched[best.id]
        update_face(best, det, tilt, pose, now, frame_size, landmarks)


def within_angle_limit(face: Face) -> bool:
    """A current-frame boundary, shared by every filter and detector. Exactly MAX_ANGLE still counts."""
    return bool(np.isfinite(face.measured_angles).all() and
                max(abs(a) for a in face.measured_angles) <= MAX_ANGLE + 1e-5)


# --- Opt-in zone ---
# No floor-height or full-face-box requirement: seated/short visitors and close-ups use
# exactly the same rules. Clip artwork at the image edges, not at a maximum face size.
def is_eligible(face: Face, frame_w: int, frame_h: int) -> bool:
    if not math.isfinite(face.distance) or face.distance <= 0:
        return False
    # Landmark rows are float32; do not reject an exact boundary because of
    # sub-pixel arithmetic (the tolerance is only a thousandth of an inch).
    if USE_OPT_IN_ZONE and face.distance > ZONE_ENTER_INCHES + 1e-3:
        return False
    if USE_ANGLE_LIMIT and not within_angle_limit(face):
        return False
    nx, ny = face.nose
    return 0 <= nx < frame_w and 0 <= ny < frame_h


# --- Privacy intro (3-second wait) ---
DWELL_SECONDS = 3.0           # in the zone, facing the screen, this long before a filter starts;
                              # the dashboard shows the privacy note in the middle meanwhile
DWELL_GRACE_SECONDS = 0.5     # a blink or a missed frame this short doesn't restart the 3 seconds


# --- Fade ---
# Opacity comes from distance: fully visible at FADE_FULL_INCHES or closer, fading to FADE_START at
# the edge of the zone and gone past it. Leaving the zone or turning away hides it immediately.
def zone_opacity(distance: float) -> float:
    if not USE_OPT_IN_ZONE:
        return 1.0
    # (a thousandth of an inch of slack, so float rounding can't reject someone standing on the line)
    if not math.isfinite(distance) or distance <= 0 or distance > ZONE_ENTER_INCHES + 1e-3:
        return 0.0
    if not USE_FADE_IN or distance <= FADE_FULL_INCHES:
        return 1.0
    edge = min(1.0, (ZONE_ENTER_INCHES - distance) / (ZONE_ENTER_INCHES - FADE_FULL_INCHES))
    return FADE_START + (1 - FADE_START) * max(0.0, edge)


def reset_activation(face: Face) -> None:
    face.opted_in = False
    face.eligible_since = face.ineligible_since = None
    face.opacity = 0.0


def update_opacity(face: Face, eligible: bool, now: float, dt: float) -> None:
    """Someone new has to stay eligible for DWELL_SECONDS (the privacy intro) before their filter
    shows. After that it follows their distance, and hides the moment they're not eligible. Coming
    back within TIMER_RESET_SECONDS doesn't replay the intro."""
    if not eligible:
        face.opacity = 0.0
        if face.ineligible_since is None:
            face.ineligible_since = now
        away = now - face.ineligible_since
        if away > DWELL_GRACE_SECONDS:
            face.eligible_since = None     # still waiting: the 3 seconds start over
        if away > TIMER_RESET_SECONDS:
            face.opted_in = False          # gone for a while: the next visit gets the intro again
        return
    face.ineligible_since = None
    if face.eligible_since is None:
        face.eligible_since = now
    if not face.opted_in and (not USE_WALK_BY_CHECK or now - face.eligible_since >= DWELL_SECONDS - 1e-9):
        face.opted_in = True
    if not face.opted_in:
        face.opacity = 0.0
        return
    face.last_eligible = now
    face.opacity = zone_opacity(face.distance)


# --- Timer ---
# Gently remind people how long they have been using the filter.
def draw_text(frame, text, x, y, scale, color, thickness=1):
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x = int(min(max(x - tw / 2, 4), frame.shape[1] - tw - 4))
    y = int(max(y, th + 4))
    cv2.rectangle(frame, (x - 4, y - th - 4), (x + tw + 4, y + base + 2), (0, 0, 0), -1)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_timer(frame, face: Face, now: float, draw: bool = True) -> tuple:
    """Draw the gentle note above this person (unless the dashboard shows it instead).
    Returns (gentle note showing, needs the firm reminder)."""
    if face.opacity > 0 and face.session_start is None:
        face.session_start = now
    if face.session_start is not None and now - face.last_eligible > TIMER_RESET_SECONDS:
        face.session_start = None
    if face.session_start is None or face.opacity <= 0:
        return False, False
    used = now - face.session_start
    gentle = TIMER_GENTLE_SECONDS <= used < TIMER_GENTLE_SECONDS + GENTLE_SHOW_SECONDS
    if gentle and draw:
        x, y, w, _ = face.box
        draw_text(frame, "1 minute has passed", x + w / 2, y - 10, 0.5, (255, 255, 255))
    return gentle, used >= TIMER_FIRM_SECONDS


def draw_debug(frame, face: Face) -> None:
    x, y, w, h = (int(v) for v in face.box)
    color = (0, 255, 0) if face.opted_in else (0, 0, 255)
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1)
    draw_text(frame, f"#{face.id} {face.distance:.1f}in  filter {face.opacity * 100:.0f}%  "
                     f"yaw{face.yaw:.0f} pitch{face.pitch:.0f} roll{face.roll:.0f}",
              x + w / 2, y + h + 20, 0.4, color)
    eyes = f"eye gap {face.eye_gap * 25.4:.0f}mm (measured)" if face.eye_gap else \
        f"eye gap {EYE_GAP_INCHES * 25.4:.0f}mm (average, measuring {len(face.iris_samples)}/{IRIS_MIN_SAMPLES})"
    draw_text(frame, eyes, x + w / 2, y + h + 36, 0.4, color)


def process_faces(frame, faces, filters, active, now, dt, filters_off=None, draw_notes=True):
    """Shared opt-in, multi-user, angle gate, fade and timer.

    Filters draw only. Every registered filter passes through these same rules.
    Returns (faces seen this frame, firm reminder, anyone has a filter, gentle note showing).
    """
    h, w = frame.shape[:2]
    if filters_off is None:
        filters_off = filters_are_off()
    # Multi-user off: only the closest person (biggest face) can get a filter
    seen_faces = [f for f in faces.values() if f.last_seen == now]
    closest = max(seen_faces, key=lambda f: f.size).id if seen_faces else None

    firm = gentle = anyone = False
    for face in list(faces.values()):
        seen = face.last_seen == now
        allowed = USE_MULTI_USER or face.id == closest
        update_opacity(face, seen and allowed and not filters_off and is_eligible(face, w, h), now, dt)
        # Forget people who are gone. Someone with a timer running is kept for TIMER_RESET_SECONDS,
        # so the camera losing their face for a moment doesn't restart their timer.
        keep = TIMER_RESET_SECONDS if USE_TIMER and face.session_start is not None else LOST_GRACE_SECONDS
        if now - face.last_seen > keep and face.opacity <= 0:
            for f in filters:
                if hasattr(f, "forget"):
                    f.forget(face.id)
            del faces[face.id]
            continue
        if seen and allowed and face.opacity > 0:
            anyone = True
            active.apply(frame, face, face.opacity)
        if USE_TIMER:
            g, f = draw_timer(frame, face, now, draw_notes)
            gentle, firm = gentle or g, firm or f

    return seen_faces, firm, anyone, gentle


# --- Main camera loop ---
KEY_ACTIONS = {ord('n'): "next", ord('d'): "debug", ord('c'): "calibrate", ord('z'): "zone", ord('l'): "log"}


def save_calibration(saved: dict, what: str = None) -> None:
    what = what or f"Calibrated looking forward: yaw {FORWARD['yaw']}, pitch {FORWARD['pitch']}"
    try:
        with open(CALIBRATION_PATH + ".part", "w") as out:
            json.dump(saved, out, indent=2)
        os.replace(CALIBRATION_PATH + ".part", CALIBRATION_PATH)
        print(f"{what} (saved)")
    except OSError as error:
        print(f"Calibration active for this session; could not save it: {error}")


def main(argv=None):
    global FOCAL_PX, MEDIAPIPE_MAX_FACES, DISTANCE_FOCAL
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("filters", nargs="*", help="Available: " + ", ".join(FILTERS) +
                        " (astronaut is an alias for helmet)")
    parser.add_argument("--camera", type=int, default=None,
                        help="camera number (default: the first one that shows a picture)")
    parser.add_argument("--max-faces", type=int, default=MEDIAPIPE_MAX_FACES)
    parser.add_argument("--focal-px", type=float, default=FOCAL_PX)
    parser.add_argument("--window", action="store_true", help="show an OpenCV window instead of the dashboard")
    parser.add_argument("--port", type=int, default=8000, help="dashboard port (default 8000)")
    parser.add_argument("--no-browser", action="store_true", help="don't open a browser for the dashboard")
    args = parser.parse_args(argv)
    if args.max_faces < 1 or not math.isfinite(args.focal_px) or args.focal_px <= 0:
        parser.error("--max-faces and --focal-px must be positive")
    FOCAL_PX, MEDIAPIPE_MAX_FACES = args.focal_px, args.max_faces
    filters = load_filters(args.filters)
    if not filters:
        print("Error: no filters loaded.")
        return 1
    detectors = {}            # trackers are started the first time a filter needs them
    tracker = None
    saved = load_calibration()  # "looking forward" calibration, one per tracker

    cam = open_camera(args.camera)
    if not cam.running:
        print("Error: Could not open a camera. Check it's plugged in (unplug and replug a USB webcam),\n"
              "and that no other program is using it. On a Pi, `rpicam-hello` tests a camera module.")
        cam.release()
        return 1

    dashboard = None
    if not args.window:
        from dashboard_server import Dashboard, open_browser
        try:
            dashboard = Dashboard(port=args.port)
        except OSError as error:
            print(f"Error: could not start the dashboard ({error}). Try --window.")
            cam.release()
            return 1
        print(f"Dashboard: {dashboard.url}  (Ctrl+C to stop)")
        if not args.no_browser:
            open_browser(dashboard.url, kiosk=is_raspberry_pi())
    else:
        print("Running. q = quit, n = next filter, d = debug view, c = calibrate forward, l = record angles.")
    log_file = log = None

    faces = {}
    next_id = [0]
    current = 0
    auto = USE_ENGAGEMENT     # taking turns between filters (the dashboard can pick one instead)
    next_switch = time.monotonic() + FILTER_SWITCH_SECONDS
    debug = False
    last = time.monotonic()
    last_state = 0.0
    fps = 0.0
    nobody_since = time.monotonic()   # taking turns starts again once everyone has left
    focal_set = False
    zone_cal = None           # pressing z: eye-gap samples from someone exactly 30 inches away
    zone_note = ""

    try:
        while True:
            frame = cam.read()
            if frame is None:
                print("Error: Failed to grab frame.")
                return 1
            if frame.shape[0] != PROCESS_HEIGHT:
                scale = PROCESS_HEIGHT / frame.shape[0]
                frame = cv2.resize(frame, (round(frame.shape[1] * scale), PROCESS_HEIGHT), interpolation=cv2.INTER_AREA)
            frame = cv2.flip(frame, 1)   # mirror view, like a selfie camera
            h, w = frame.shape[:2]
            if not focal_set:   # FOCAL_PX is for a 640-wide picture; most webcams' widescreen mode
                FOCAL_PX = args.focal_px * w / 640   # keeps the same side-to-side view
                focal_set = True
            now = time.monotonic()
            dt, last = now - last, now
            fps = smooth(fps, 1 / max(dt, 1e-3), 0.1)

            # Engagement: switch between the team's filters
            if auto and now >= next_switch and len(filters) > 1:
                current = (current + 1) % len(filters)
                next_switch = now + FILTER_SWITCH_SECONDS
            active = filters[current]

            # Use the face tracker this filter asks for
            wanted = TRACKER or getattr(active, "tracker", "yunet")
            if wanted not in detectors:
                detectors[wanted] = make_detector(wanted)
            detector = detectors[wanted]
            if getattr(active, "requires_landmarks", False) and not getattr(detector, "provides_landmarks", False):
                print(f"Skipping {active.name}: iris/eyelid landmarks require MediaPipe. "
                      "Install requirements.txt and use the MediaPipe tracker.")
                filters.pop(current)
                if not filters:
                    return 1
                current %= len(filters)
                next_switch = now + FILTER_SWITCH_SECONDS
                continue
            name = type(detector).__name__
            if name != tracker:
                tracker = name
                for f in filters:   # each tracker keeps its own track of people, so start fresh
                    for face_id in faces:
                        if hasattr(f, "forget"):
                            f.forget(face_id)
                faces.clear()
                mine = saved.get(tracker, {})
                FORWARD.update(yaw=mine.get("yaw", 0.0), pitch=mine.get("pitch", 0.0))
                DISTANCE_FOCAL = mine.get("distance_focal")

            tracked = [(f.nose, f.tilt, f.size) for f in faces.values()]
            track(faces, detector.detect(frame, tracked), now, next_id, (w, h))
            # Check after inference: a slow detector may have crossed dismissal time.
            filters_off = filters_are_off()
            # In dashboard mode the page shows the notes and reminders in its own type, so the
            # picture stays clean. The --window view draws them onto the frame like before.
            draw_notes = dashboard is None
            seen_faces, firm, anyone, gentle = process_faces(
                frame, faces, filters, active, now, dt, filters_off, draw_notes)
            # Privacy intro: someone is in the zone but hasn't had their 3 seconds yet, and nobody
            # is using a filter. The page shows the privacy note in the middle with a countdown.
            waiting = [f for f in seen_faces if f.eligible_since is not None and not f.opted_in]
            intro = (max(min(1.0, (now - f.eligible_since) / DWELL_SECONDS) for f in waiting)
                     if USE_WALK_BY_CHECK and waiting and not anyone else None)
            # Timer on screen: the longest anyone in front of the screen right now has had a filter
            timer = max((now - f.session_start for f in seen_faces
                         if f.session_start is not None and f.opacity > 0), default=None) if USE_TIMER else None
            # Setting the 30-inch distance: average the closest person's eye gap for a moment
            if zone_cal is not None:
                if seen_faces:
                    closest = max(seen_faces, key=lambda f: f.size)
                    zone_cal["samples"].append(calibration_sample(closest, (w, h)))
                    zone_cal["gap"] = closest.eye_gap   # their own eye gap, refined while they stand there
                if now >= zone_cal["until"]:
                    if len(zone_cal["samples"]) >= 5:
                        gap = zone_cal.get("gap") or EYE_GAP_INCHES
                        DISTANCE_FOCAL = float(np.median(zone_cal["samples"])) / gap
                        saved.setdefault(tracker, {})["distance_focal"] = round(DISTANCE_FOCAL, 3)
                        saved[tracker].pop("distance_k", None)
                        whose = "your measured" if zone_cal.get("gap") else "the average"
                        save_calibration(saved, f"Set the {ZONE_ENTER_INCHES}-inch distance for {tracker} "
                                                f"(using {whose} eye gap, {gap * 25.4:.0f} mm)")
                        zone_note = f"Done: {ZONE_ENTER_INCHES} inches is set"
                    else:
                        zone_note = "Couldn't see a face. Stand 30 in away, look at the screen, try again"
                        print(zone_note)
                    zone_cal = None
            # Someone picked a filter on the screen: it stays until everyone has left for a while,
            # then the filters go back to taking turns for the next people.
            if seen_faces:
                nobody_since = now
            elif not auto and USE_ENGAGEMENT and now - nobody_since > FILTER_SWITCH_SECONDS:
                auto = True
                next_switch = now + FILTER_SWITCH_SECONDS
            if debug:
                for face in faces.values():
                    draw_debug(frame, face)
                draw_text(frame, f"Filter: {active.name}  {fps:.0f} FPS", 140, 20, 0.5, (255, 255, 0))

            # --- Show the frame, then collect what people asked for ---
            actions = []
            if dashboard:
                dashboard.publish(frame)
                if now - last_state > 0.2:   # the page polls about once a second; no need to rebuild every frame
                    last_state = now
                    dashboard.state = {
                        "filters": [f.name for f in filters], "active": current, "auto": auto,
                        "switch_in": max(0.0, next_switch - now) if auto and len(filters) > 1 else None,
                        "switch_every": FILTER_SWITCH_SECONDS,
                        "hint": USE_HINT_TEXT and not anyone and not filters_off and intro is None,
                        "active": anyone,
                        "intro": None if intro is None else round(intro, 3),   # 0 -> 1 over 3 seconds
                        "dwell": DWELL_SECONDS,
                        "gentle": gentle, "firm": firm, "off": filters_off,
                        "people": len(seen_faces), "debug": debug, "fps": round(fps),
                        "empty_for": 0.0 if seen_faces else round(now - nobody_since, 1),   # seconds nobody's been in view
                        "tracker": "MediaPipe" if "MediaPipe" in tracker else "YuNet",
                        "zone_note": zone_note,
                        "timer": timer,
                    }
                actions = list(dashboard.pending())
            else:
                if intro is not None:
                    left = math.ceil(DWELL_SECONDS * (1 - intro))
                    draw_text(frame, "This camera doesn't record. Pictures never leave this computer.", w / 2, h / 2 - 12,
                              0.6, (255, 255, 255), 2)
                    draw_text(frame, f"Your filter starts in {left}", w / 2, h / 2 + 18, 0.7, (255, 255, 255), 2)
                if timer is not None:
                    draw_text(frame, f"Timer {int(timer) // 60}:{int(timer) % 60:02d}", 60, 20, 0.6, (255, 255, 255), 2)
                if firm:
                    draw_text(frame, "3 minutes! Time to get back to your work.", w / 2, 40, 0.8, (0, 200, 255), 2)
                if filters_off:
                    draw_text(frame, "Filters are off right now. They're back at 7:30 a.m.", w / 2, h - 16, 0.5, (200, 200, 200))
                elif USE_HINT_TEXT and not anyone:
                    draw_text(frame, "Step up close and look here to try a filter", w / 2, h - 16, 0.5, (255, 255, 255))
                draw_text(frame, "This camera doesn't record: pictures never leave this computer and are never saved.",
                          w / 2, h - 52, 0.42, (255, 255, 255))
                draw_text(frame, "To stay off the screen, step out of the camera's view.", w / 2, h - 36, 0.42, (255, 255, 255))
                cv2.imshow("Hallway Filter", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27) or cv2.getWindowProperty("Hallway Filter", cv2.WND_PROP_VISIBLE) < 1:
                    break
                if key in KEY_ACTIONS:
                    actions.append((KEY_ACTIONS[key], None))

            for action, value in actions:
                if action == "next":
                    current = (current + 1) % len(filters)
                    next_switch = now + FILTER_SWITCH_SECONDS
                elif action == "select" and 0 <= value < len(filters):
                    current, auto = value, False   # stay on the filter they picked
                    nobody_since = now   # someone's at the screen, even if the camera hasn't found them yet
                elif action == "auto":
                    auto = True
                    next_switch = now + FILTER_SWITCH_SECONDS
                elif action == "calibrate" and seen_faces:
                    # whoever is closest is looking straight at the screen right now
                    f = max(seen_faces, key=lambda f: f.size)
                    FORWARD.update(yaw=round(f.raw_yaw, 1), pitch=round(f.raw_pitch, 1))
                    saved.setdefault(tracker, {}).update(FORWARD)
                    save_calibration(saved)
                elif action == "zone":
                    zone_cal = {"until": now + ZONE_CALIBRATE_SECONDS, "samples": []}
                    zone_note = "Measuring... hold still"
                elif action == "debug":
                    debug = not debug
                    for f in filters:
                        f.debug = debug   # lets a filter draw its own debug info too
                elif action == "log":
                    if log_file:
                        log_file.close()
                        log_file = log = None
                        print(f"Stopped recording. Saved to {POSE_LOG_PATH}")
                    else:
                        log_file = open(POSE_LOG_PATH, "w", newline="")
                        log = csv.writer(log_file)
                        log.writerow(["time", "face", "raw_yaw", "raw_pitch", "tilt", "yaw", "pitch", "roll",
                                      "nose_x", "nose_y", "size", "distance", "opacity"])
                        print(f"Recording head angles to {POSE_LOG_PATH} (press l again to stop)")
                if dashboard:
                    last_state = 0.0   # send the change to the page right away
            if log:
                for f in seen_faces:
                    log.writerow([f"{now:.3f}", f.id, f"{f.raw_yaw:.2f}", f"{f.raw_pitch:.2f}", f"{f.tilt:.2f}",
                                  f"{f.yaw:.2f}", f"{f.pitch:.2f}", f"{f.roll:.2f}", f"{f.nose[0]:.1f}", f"{f.nose[1]:.1f}",
                                  f"{f.size:.1f}", f"{f.distance:.1f}", f"{f.opacity:.2f}"])

    except KeyboardInterrupt:
        pass
    except (OSError, RuntimeError, ValueError, cv2.error) as error:
        print(f"Hallway filter stopped: {error}", file=sys.stderr)
        return 1
    finally:
        cam.release()
        if dashboard:
            dashboard.close()
        if log_file:
            log_file.close()
        for detector in detectors.values():
            if hasattr(detector, "close"):
                detector.close()
        if args.window:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())
