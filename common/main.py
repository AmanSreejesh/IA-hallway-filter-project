"""
Main hallway filter system.
Combines each team member's filter and applies the shared features to all of them.

Run from the repo root:
    python common/main.py              # every filter, taking turns
    python common/main.py emoji        # only the filters you name (for testing your own)

Face tracking: each filter says which face tracker it wants, and this file uses whichever one
the filter on screen asks for (switching automatically when the filters take turns):
  - "mediapipe": follows 478 points per face and measures the head's 3D angle. Best for filters
    that turn and tilt with your head (the 3D emoji uses it). Needs: pip install mediapipe
  - "yunet": finds 5 points per face. Lighter, and doesn't need anything extra.
If MediaPipe isn't installed, YuNet is used instead. TRACKER below can force one for everything.

Opt-in zone: on a Raspberry Pi with the ultrasonic sensor wired up (see the wiring notes below),
distance comes from the sensor. Anywhere else it's estimated from the size of the face.

Keys:  q = quit   n = next filter   d = debug view (distance, head angles, FPS -- for calibrating)
       c = calibrate "looking forward": look straight at the screen and press c (saved for next time)
       l = start/stop recording head angles to common/pose_log.csv (numbers only, for finding problems)

How a filter plugs in (see viraj/emoji_filter.py for an example):
  - Make a class with a `name`, an `apply(frame, face, opacity)` method that draws onto
    `frame`, and optionally a `forget(face_id)` method that's called when a person leaves.
  - Optionally set `tracker = "mediapipe"` on the class (the default is "yunet").
  - Add it to FILTERS below.
  - Filters only draw. This file decides who gets a filter, so every filter gets the same
    opt-in zone, fade-in, timer, and multi-user behaviour.

Nothing is ever saved. Each frame is grabbed, drawn on, shown, and then thrown away.
"""
import csv
import importlib
import json
import math
import os
import sys
import threading
import time
import urllib.request
from datetime import datetime, time as dtime

import cv2
import numpy as np

COMMON_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(COMMON_DIR))   # repo root, so "viraj.emoji_filter" imports

MODEL_PATH = os.path.join(COMMON_DIR, "face_detection_yunet_2023mar.onnx")
MODEL_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
             "face_detection_yunet/face_detection_yunet_2023mar.onnx")
MEDIAPIPE_MODEL_PATH = os.path.join(COMMON_DIR, "face_landmarker.task")
MEDIAPIPE_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                       "face_landmarker/float16/1/face_landmarker.task")

# name -> (module, class). Add your filter here.
FILTERS = {
    "emoji": ("viraj.emoji_filter", "EmojiFilter"),
    # "sharingan": ("aman.sharingan_filter", "SharinganFilter"),
    # "astronaut": ("ishan.astronaut_filter", "AstronautFilter"),
    # "clown_nose": ("aryan.clown_nose_filter", "ClownNoseFilter"),
}

# --- Feature toggles (True = on, False = off) ---
USE_OPT_IN_ZONE = True    # only people closer than ZONE_ENTER_INCHES and fully on screen get a filter
USE_WALK_BY_CHECK = True  # people walking past (faster than WALKING_SPEED) don't get a filter
USE_ANGLE_LIMIT = False   # filter goes away when the head turns/tilts past MAX_ANGLE
USE_FADE_IN = True        # wait DWELL_SECONDS, then fade in (off = filter appears/disappears instantly)
USE_TIMER = True          # 1 minute note and 3 minute reminder
USE_MULTI_USER = True     # everyone in view gets a filter (off = only the closest person)
USE_ENGAGEMENT = True     # rotate between filters every FILTER_SWITCH_SECONDS ('n' still works when off)
USE_SCHEDULE = False       # filters turn off at FILTERS_OFF_AT
USE_HINT_TEXT = True      # "Step up close..." message when nobody is using a filter
USE_ULTRASONIC = True     # on a Raspberry Pi, measure the opt-in zone with an ultrasonic sensor
                          # (ignored on other computers, which use the camera estimate instead)

# --- Settings (keep in sync with constraints.md) ---
TRACKER = None            # None = whatever each filter asks for, or "mediapipe" / "yunet" to force one
CAMERA_SIZE = (640, 480)
DETECT_SCALE = 0.5        # look for faces on a half-size copy of the frame (~4x less work on the Pi).
                          # People in the opt-in zone are big on screen, so nothing is lost.
NEW_FACE_EVERY = 3        # while someone's being tracked, scan the whole picture for new people every N frames
SEARCH_ANGLES = (40, -40, 80, -80, 130, -130, 180)   # tilts tried when nobody is in view (YuNet)
MEDIAPIPE_MAX_FACES = 3   # each extra face costs time on the Pi
MEDIAPIPE_MAX_FLIP = 60   # a tilt this many degrees away from last frame is a mistake, not a real head
MEDIAPIPE_STRAIGHTEN_ABOVE = 20   # if the closest person tilts more than this, turn the picture so their face is upright

# Opt-in zone. On a Raspberry Pi with the ultrasonic sensor, distance comes from the sensor.
# Otherwise it's estimated from how far apart the eyes look on screen.
FOCAL_PX = 600            # camera focal length in pixels -- calibrate with the 'd' debug view
EYE_GAP_INCHES = 2.5      # average distance between a person's eyes
ZONE_ENTER_INCHES = 30    # step closer than this to opt in
ZONE_EXIT_INCHES = 33     # ...and you have to step back past this to opt out.
                          # The gap between the two stops flicker at the border.
# Ultrasonic sensor (Raspberry Pi only). An HC-SR04 sends out a click of sound and times the echo,
# which measures distance far more accurately than guessing from the size of a face on camera.
#
# WIRING (HC-SR04 -> Raspberry Pi 40-pin header; pin numbers are the physical pin numbers):
#
#     HC-SR04 VCC  ->  pin 2  (5V)
#     HC-SR04 GND  ->  pin 6  (GND)
#     HC-SR04 TRIG ->  pin 16 (GPIO 23)        direct wire is fine (the Pi sends 3.3V, the sensor accepts it)
#     HC-SR04 ECHO ->  1k ohm resistor ->  pin 18 (GPIO 24)
#                                     |
#                                     +->  2k ohm resistor ->  pin 14 (GND)
#
#   ECHO sends back 5V, which can damage the Pi's 3.3V pins. The 1k + 2k resistors form a
#   voltage divider that drops it to ~3.3V. (No 2k? Two 1k in series works. A 3.3V-only
#   "HC-SR04P" sensor powered from pin 1 doesn't need the resistors.)
#
#   Mount the sensor right next to the camera, pointing the same way, at about chest-to-face
#   height. It measures the closest thing in a ~15 degree cone in front of it, so keep walls and
#   furniture out of that cone (or farther away than ZONE_EXIT_INCHES).
#
#   Software: gpiozero comes with Raspberry Pi OS. If it's missing: sudo apt install python3-gpiozero
#   (if you use a virtual environment, create it with --system-site-packages so it can see gpiozero).
SENSOR_TRIGGER_GPIO = 23
SENSOR_ECHO_GPIO = 24
SENSOR_MAX_INCHES = 120   # readings past this count as "nobody there"

WALKING_SPEED = 2.0       # moving faster than this (face widths per second) = walking past, no filter
MAX_ANGLE = 45            # turn/tilt your head past this and the filter goes away
RETURN_ANGLE = 40         # ...and comes back once you're under this (same anti-flicker idea)

# Smoothing. Still faces get smoothed a lot (no jitter); moving faces barely at all (no lag).
STILL_SMOOTHING = 1.0     # lower = steadier when still, but more lag (try 0.5 to 2)
MOVING_RESPONSE = 5.0     # higher = catches up faster when you move (try 2 to 10)

# Fade-in: people walking past shouldn't get a filter slapped on them.
DWELL_SECONDS = 0.5       # must stay in the zone this long before anything shows
LEAVE_SECONDS = 0.3       # must be out of the zone this long before it fades away
                          # (so one bad frame -- a blink, a missed detection -- can't flicker it)
FADE_START = 0.1          # first visible opacity
FADE_IN_SECONDS = 0.4     # time to go from FADE_START to fully visible
FADE_OUT_SECONDS = 0.3

# Timer
TIMER_GENTLE_SECONDS = 60     # small note that a minute has passed
TIMER_FIRM_SECONDS = 180      # bigger reminder to get back to work
GENTLE_SHOW_SECONDS = 5       # how long the small note stays up
TIMER_RESET_SECONDS = 5       # step away this long and your timer starts over

# Multi-user tracking
LOST_GRACE_SECONDS = 0.5      # keep a person around this long if the detector misses them

# Engagement
FILTER_SWITCH_SECONDS = 30    # rotate to the next filter this often

# Schedule: filters turn off after dismissal so nobody heading to the buses gets stopped
FILTERS_OFF_AT = dtime(16, 15)

# Looking forward. The camera sits above or below the screen, so someone looking at the screen
# looks slightly down or up to the camera (and each tracker has its own small bias). These are the
# head angles that count as "looking straight ahead". Press c while looking at the screen to set
# them; they're saved to calibration.json (separately for YuNet and MediaPipe) and loaded every time.
CALIBRATION_PATH = os.path.join(COMMON_DIR, "calibration.json")
FORWARD = {"yaw": 0.0, "pitch": 0.0}
POSE_LOG_PATH = os.path.join(COMMON_DIR, "pose_log.csv")


# --- Imports: each team member's filter ---
def load_filters(names: list) -> list:
    """Import the filters by name. A broken filter is skipped instead of crashing everything."""
    filters = []
    for name in names or FILTERS:
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
        print(f"Loaded filter: {name}")
    return filters


def download(url: str, path: str) -> None:
    """Download a model file the first time the program runs."""
    if os.path.exists(path) and os.path.getsize(path) > 100_000:
        return
    print(f"Downloading {os.path.basename(path)} (one time only)...")
    urllib.request.urlretrieve(url, path)
    print("Done.")


# --- Face detector ---
# detect() returns (row, tilt, pose) per face:
#   row  = [x, y, w, h, eye1 x,y, eye2 x,y, nose x,y, mouth1 x,y, mouth2 x,y, score]
#   tilt = head tilt in degrees if the detector knows it, else None
#   pose = (yaw, pitch) if the detector measures them itself (MediaPipe does), else None
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
        return [(f, tilt, None) for f, tilt in found]


class MediaPipeDetector:
    """MediaPipe's Face Landmarker: follows 478 points on each face from frame to frame, handles
    tilted heads, and measures the head's 3D angle itself."""
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
        """Returns (row, tilt, (yaw, pitch)) per face.
        `tracked` = (nose, tilt, size) for each person from last frame."""
        h, w = frame.shape[:2]
        # MediaPipe loses faces tilted past ~50 degrees. If the closest person is tilted, turn the
        # picture around their face so it's upright, then turn the results back afterwards.
        turn = None
        known = [t for t in tracked if t[1] is not None]
        if known:
            nose, tilt, _ = max(known, key=lambda t: t[2])
            if abs(tilt) > MEDIAPIPE_STRAIGHTEN_ABOVE:
                turn = cv2.getRotationMatrix2D((float(nose[0]), float(nose[1])), tilt, 1.0)
                frame = cv2.warpAffine(frame, turn, (w, h))
                back = cv2.invertAffineTransform(turn)
                c, s = math.cos(math.radians(tilt)), math.sin(math.radians(tilt))
                spin = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])   # the same turn, in 3D
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ms = max(int(time.monotonic() * 1000), self.last_ms + 1)   # MediaPipe needs increasing timestamps
        self.last_ms = ms
        result = self.landmarker.detect_for_video(image, ms)

        out = []
        for landmarks, matrix in zip(result.face_landmarks, result.facial_transformation_matrixes):
            pts = np.array([(p.x * w, p.y * h) for p in landmarks], dtype=np.float32)   # 0..1 -> pixels
            if turn is not None:
                pts = pts @ back[:, :2].T.astype(np.float32) + back[:, 2].astype(np.float32)
            (x0, y0), (x1, y1) = pts.min(axis=0), pts.max(axis=0)
            row = np.array([x0, y0, x1 - x0, y1 - y0,
                            *pts[list(self.EYE_A)].mean(axis=0), *pts[list(self.EYE_B)].mean(axis=0),
                            *pts[self.NOSE_TIP], *pts[self.MOUTH[0]], *pts[self.MOUTH[1]], 1.0], dtype=np.float32)
            rotation = self.TO_OPENCV @ np.asarray(matrix)[:3, :3] @ self.TO_OPENCV
            if turn is not None:
                rotation = spin @ rotation
            yaw, pitch, roll = angles_from_rotation(rotation)
            # On a very tilted head MediaPipe occasionally finds the face upside down. A real head
            # can't flip that far between two frames, so skip it (the tracker holds the last position).
            nose = pts[self.NOSE_TIP]
            if any(np.hypot(*(np.asarray(n) - nose)) < size * 2.5 and abs(wrap(roll - tilt)) > MEDIAPIPE_MAX_FLIP
                   for n, tilt, size in tracked if tilt is not None):
                continue
            out.append((row, roll, (yaw, pitch)))
        return out


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


# --- Camera ---
class Camera:
    """Reads the camera on its own thread and keeps only the newest frame.

    Without this, a slow Pi falls behind and the camera's queue hands it frames from a
    moment ago, so the filter looks like it's lagging behind your face.
    """

    def __init__(self):
        backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(0, backend)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_SIZE[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_SIZE[1])
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.frame = None
        self.new_frame = threading.Event()
        self.running = self.cap.isOpened()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        if self.running:
            self.thread.start()

    def _loop(self):
        while self.running:
            ok, frame = self.cap.read()
            self.frame = frame if ok else None
            self.running = ok
            self.new_frame.set()

    def read(self):
        """Wait for a frame we haven't seen yet. Returns None if the camera stopped."""
        self.new_frame.wait(2.0)
        self.new_frame.clear()
        return self.frame

    def release(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(1.0)
        self.cap.release()


# --- Ultrasonic sensor (opt-in zone on the Raspberry Pi) ---
def is_raspberry_pi() -> bool:
    """True when running on a Raspberry Pi (it writes its model name to this file)."""
    try:
        with open("/proc/device-tree/model") as f:
            return "raspberry pi" in f.read().lower()
    except OSError:
        return False


class UltrasonicSensor:
    """HC-SR04 distance sensor. gpiozero pings it on a background thread and averages the
    last few readings, so reading it costs nothing in the main loop. See the wiring above."""

    def __init__(self):
        from gpiozero import DistanceSensor   # only exists on the Pi, so imported here
        self.sensor = DistanceSensor(echo=SENSOR_ECHO_GPIO, trigger=SENSOR_TRIGGER_GPIO,
                                     max_distance=SENSOR_MAX_INCHES * 0.0254)

    def inches(self) -> float:
        return self.sensor.distance / 0.0254   # gpiozero reports meters

    def close(self):
        self.sensor.close()


def start_sensor():
    """The sensor on a Raspberry Pi (if turned on and wired up), otherwise None."""
    if not USE_ULTRASONIC or not is_raspberry_pi():
        return None
    try:
        sensor = UltrasonicSensor()
        print(f"Ultrasonic sensor on: {sensor.inches():.0f} in to the closest thing in front of it.")
        return sensor
    except Exception as e:
        print(f"Ultrasonic sensor not working ({e}); using the camera to estimate distance instead.")
        return None


def apply_sensor_distance(faces: list, sensor_inches: float) -> None:
    """The sensor measures one thing: how far away the closest person is. That's the closest
    (biggest) face, so it gets the sensor's distance. Everyone else gets their camera estimate
    corrected by the same amount, which also fixes any error in FOCAL_PX."""
    if not faces:
        return
    closest = max(faces, key=lambda f: f.size)
    correction = sensor_inches / max(closest.camera_distance, 1.0)
    for f in faces:
        f.distance = f.camera_distance * correction


# --- Face data handed to every filter ---
class Face:
    """One tracked person. Everything here is already smoothed.

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
        self.camera_distance = 999.0   # distance guessed from face size (main loop may replace .distance with the sensor's)
        self.speed = 0.0        # face widths per second
        # system state (filters don't need these)
        self.point_filter = Smoother()
        self.angle_filter = Smoother()
        self.size_filter = Smoother()
        self.pose_guess = None  # last 3D head pose, used as the starting point for the next frame
        self.tilt = None        # how far the head is tilted in the picture, tracked by optical flow
        self.raw_yaw = self.raw_pitch = 0.0   # this frame's measurement, before calibration and smoothing
        self.last_seen = 0.0
        self.opted_in = False
        self.change_start = None
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
        return face.yaw, face.pitch
    face.pose_guess = (rvec, tvec)
    return yaw, pitch


def wrap(angle: float) -> float:
    """Keep an angle between -180 and 180."""
    return (angle + 180) % 360 - 180


def face_size(p: np.ndarray, yaw: float, pitch: float) -> float:
    """The eye gap shrinks when you turn and the eye-to-mouth gap shrinks when you nod,
    so undo each one and average them. Eye-to-mouth is ~1.28x the eye gap on real faces,
    so it's scaled down to match. The result is about the eye gap, but steady."""
    eye_dist = max(float(np.hypot(*(p[3] - p[2]))), 1.0)
    down_len = max(float(np.hypot(*((p[5] + p[6]) / 2 - (p[2] + p[3]) / 2))), 1.0)
    cos_yaw = max(math.cos(math.radians(yaw)), 0.5)
    cos_pitch = max(math.cos(math.radians(pitch)), 0.5)
    return (eye_dist / cos_yaw + down_len / 1.28 / cos_pitch) / 2


def update_face(face: Face, det: np.ndarray, tilt, pose, now: float, frame_size: tuple) -> None:
    dt = now - face.last_seen if face.points is not None else 0.0
    scale = max(face.size, 1.0)
    if tilt is not None:
        face.tilt = tilt                # carried over from last frame by optical flow
    elif face.tilt is None:            # new face: start from the eye line
        face.tilt = math.degrees(math.atan2(det[7] - det[5], det[6] - det[4]))
    face.points = face.point_filter(raw_points(det), dt, scale)
    # How fast the nose is moving, in face widths per second (for the walk-by check)
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
    roll = face.tilt
    # Smooth the tilt the short way round (179 -> -179 is a 2 degree change, not 358)
    if face.angle_filter.value is not None:
        roll = face.angle_filter.value[0] + wrap(roll - face.angle_filter.value[0])
    roll, face.yaw, face.pitch = face.angle_filter([roll, yaw, pitch], dt, 30.0)
    face.roll = wrap(roll)
    face.size = float(face.size_filter(face_size(face.points, face.yaw, face.pitch), dt, scale))
    face.camera_distance = FOCAL_PX * EYE_GAP_INCHES / face.size
    face.distance = face.camera_distance
    face.last_seen = now


# --- Multi-user ---
# Match each detection to the closest person from last frame, so everyone keeps their id.
def track(faces: dict, detections: list, now: float, next_id: list, frame_size: tuple) -> None:
    unmatched = dict(faces)
    for det, tilt, pose in sorted(detections, key=lambda d: -d[0][2]):   # biggest (closest) faces first
        nose = det[8:10]
        best, best_dist = None, None
        for face in unmatched.values():
            d = float(np.hypot(*(face.points[4] - nose)))
            if d < max(face.size * 1.5, 60) and (best_dist is None or d < best_dist):
                best, best_dist = face, d
        if best is None:
            best = Face(next_id[0])
            next_id[0] += 1
            faces[best.id] = best
        else:
            del unmatched[best.id]
        update_face(best, det, tilt, pose, now, frame_size)


# --- Opt-in zone ---
# Only apply a filter when a person is close, looking at the screen, and fully in view.
def is_eligible(face: Face, frame_w: int, frame_h: int) -> bool:
    zone = ZONE_EXIT_INCHES if face.opted_in else ZONE_ENTER_INCHES
    angle = MAX_ANGLE if face.opted_in else RETURN_ANGLE
    if USE_OPT_IN_ZONE and face.distance > zone:
        return False
    if USE_WALK_BY_CHECK and face.speed > WALKING_SPEED:
        return False   # walking past, not stopping to play
    if USE_ANGLE_LIMIT and (abs(face.yaw) > angle or abs(face.pitch) > angle or abs(face.roll) > angle):
        return False
    if not USE_OPT_IN_ZONE:
        return True
    x, y, w, h = face.box
    if face.opted_in:
        nx, ny = face.nose   # once opted in, only the nose has to stay on screen
        return 0 <= nx < frame_w and 0 <= ny < frame_h
    return x >= 0 and y >= 0 and x + w <= frame_w and y + h <= frame_h


# --- Fade-in ---
# Start the filter at low opacity and raise it while the person stays. Fade out when they leave.
def update_opacity(face: Face, eligible: bool, now: float, dt: float) -> None:
    # Only switch in or out once the new answer has held steady for a moment. Before, a
    # single bad frame (noisy distance, a blink, a missed detection) restarted the fade,
    # which is why the opacity jumped up and down at random.
    if eligible == face.opted_in:
        face.change_start = None
    elif face.change_start is None:
        face.change_start = now
    elif now - face.change_start >= ((DWELL_SECONDS if USE_FADE_IN else 0) if eligible else LEAVE_SECONDS):
        face.opted_in = eligible
        face.change_start = None

    if not USE_FADE_IN:
        face.opacity = 1.0 if face.opted_in else 0.0
        if face.opted_in:
            face.last_eligible = now
    elif face.opted_in:
        face.last_eligible = now
        face.opacity = min(1.0, max(face.opacity, FADE_START) + dt * (1 - FADE_START) / FADE_IN_SECONDS)
    else:
        face.opacity = max(0.0, face.opacity - dt / FADE_OUT_SECONDS)


# --- Timer ---
# Gently remind people how long they have been using the filter.
def draw_text(frame, text, x, y, scale, color, thickness=1):
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x = int(min(max(x - tw / 2, 4), frame.shape[1] - tw - 4))
    y = int(max(y, th + 4))
    cv2.rectangle(frame, (x - 4, y - th - 4), (x + tw + 4, y + base + 2), (0, 0, 0), -1)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_timer(frame, face: Face, now: float) -> bool:
    """Draw the gentle note above this person. Returns True if they need the firm reminder."""
    if face.opacity > 0 and face.session_start is None:
        face.session_start = now
    if face.session_start is not None and now - face.last_eligible > TIMER_RESET_SECONDS:
        face.session_start = None
    if face.session_start is None or face.opacity <= 0:
        return False
    used = now - face.session_start
    if TIMER_GENTLE_SECONDS <= used < TIMER_GENTLE_SECONDS + GENTLE_SHOW_SECONDS:
        x, y, w, _ = face.box
        draw_text(frame, "1 minute has passed", x + w / 2, y - 10, 0.5, (255, 255, 255))
    return used >= TIMER_FIRM_SECONDS


def draw_debug(frame, face: Face) -> None:
    x, y, w, h = (int(v) for v in face.box)
    color = (0, 255, 0) if face.opted_in else (0, 0, 255)
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1)
    draw_text(frame, f"#{face.id} {face.distance:.0f}in yaw{face.yaw:.0f} pitch{face.pitch:.0f} roll{face.roll:.0f}",
              x + w / 2, y + h + 20, 0.4, color)


# --- Main camera loop ---
def main():
    filters = load_filters(sys.argv[1:])
    if not filters:
        print("Error: no filters loaded.")
        return
    detectors = {}            # trackers are started the first time a filter needs them
    tracker = None
    saved = {}                # "looking forward" calibration, one per tracker
    if os.path.exists(CALIBRATION_PATH):
        with open(CALIBRATION_PATH) as f:
            saved = json.load(f)

    sensor = start_sensor()
    cam = Camera()
    if not cam.running:
        print("Error: Could not open webcam.")
        return
    print("Running. q = quit, n = next filter, d = debug view, c = calibrate forward, l = record angles.")
    log_file = log = None

    faces = {}
    next_id = [0]
    current = 0
    next_switch = time.monotonic() + FILTER_SWITCH_SECONDS
    debug = False
    last = time.monotonic()
    fps = 0.0

    while True:
        frame = cam.read()
        if frame is None:
            print("Error: Failed to grab frame.")
            break
        frame = cv2.flip(frame, 1)   # mirror view, like a selfie camera
        h, w = frame.shape[:2]
        now = time.monotonic()
        dt, last = now - last, now
        fps = smooth(fps, 1 / max(dt, 1e-3), 0.1)
        filters_off = USE_SCHEDULE and datetime.now().time() >= FILTERS_OFF_AT

        # Engagement: switch between the team's filters
        if USE_ENGAGEMENT and now >= next_switch and len(filters) > 1:
            current = (current + 1) % len(filters)
            next_switch = now + FILTER_SWITCH_SECONDS
        active = filters[current]

        # Use the face tracker this filter asks for
        wanted = TRACKER or getattr(active, "tracker", "yunet")
        if wanted not in detectors:
            detectors[wanted] = make_detector(wanted)
        detector = detectors[wanted]
        name = type(detector).__name__
        if name != tracker:
            tracker = name
            for f in filters:   # each tracker keeps its own track of people, so start fresh
                for face_id in faces:
                    if hasattr(f, "forget"):
                        f.forget(face_id)
            faces.clear()
            FORWARD.update(yaw=0.0, pitch=0.0)
            FORWARD.update(saved.get(tracker, {}))

        tracked = [(f.nose, f.tilt, f.size) for f in faces.values()]
        track(faces, detector.detect(frame, tracked), now, next_id, (w, h))
        # Multi-user off: only the closest person (biggest face) can get a filter
        seen_faces = [f for f in faces.values() if f.last_seen == now]
        closest = max(seen_faces, key=lambda f: f.size).id if seen_faces else None
        # Opt-in zone on the Pi: use the ultrasonic sensor's distance instead of the camera's guess
        sensor_inches = sensor.inches() if sensor else None
        if sensor_inches is not None:
            apply_sensor_distance(seen_faces, sensor_inches)

        firm = False
        anyone = False
        for face in list(faces.values()):
            seen = face.last_seen == now
            allowed = USE_MULTI_USER or face.id == closest
            update_opacity(face, seen and allowed and not filters_off and is_eligible(face, w, h), now, dt)
            # Forget people who are gone and fully faded out
            if now - face.last_seen > LOST_GRACE_SECONDS and face.opacity <= 0:
                for f in filters:
                    if hasattr(f, "forget"):
                        f.forget(face.id)
                del faces[face.id]
                continue
            if face.opacity > 0:
                anyone = True
                active.apply(frame, face, face.opacity)
            if USE_TIMER:
                firm |= draw_timer(frame, face, now)
            if debug:
                draw_debug(frame, face)

        if firm:
            draw_text(frame, "3 minutes! Time to get back to your work.", w / 2, 40, 0.8, (0, 200, 255), 2)
        if filters_off:
            draw_text(frame, "Filters are off for the day", w / 2, h - 16, 0.5, (200, 200, 200))
        elif USE_HINT_TEXT and not anyone:
            draw_text(frame, "Step up close and look here to try a filter", w / 2, h - 16, 0.5, (255, 255, 255))
        if debug:
            sensor_text = f"  sensor {sensor_inches:.0f}in" if sensor_inches is not None else ""
            draw_text(frame, f"Filter: {active.name}  {fps:.0f} FPS{sensor_text}", 140, 20, 0.5, (255, 255, 0))

        cv2.imshow("Hallway Filter", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('n'):
            current = (current + 1) % len(filters)
            next_switch = now + FILTER_SWITCH_SECONDS
        if key == ord('c') and seen_faces:
            # whoever is closest is looking straight at the screen right now
            f = max(seen_faces, key=lambda f: f.size)
            FORWARD.update(yaw=round(f.raw_yaw, 1), pitch=round(f.raw_pitch, 1))
            saved[tracker] = dict(FORWARD)
            with open(CALIBRATION_PATH, "w") as out:
                json.dump(saved, out, indent=2)
            print(f"Calibrated looking forward: yaw {FORWARD['yaw']}, pitch {FORWARD['pitch']} (saved)")
        if key == ord('l'):
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
        if log:
            for f in seen_faces:
                log.writerow([f"{now:.3f}", f.id, f"{f.raw_yaw:.2f}", f"{f.raw_pitch:.2f}", f"{f.tilt:.2f}",
                              f"{f.yaw:.2f}", f"{f.pitch:.2f}", f"{f.roll:.2f}", f"{f.nose[0]:.1f}", f"{f.nose[1]:.1f}",
                              f"{f.size:.1f}", f"{f.distance:.1f}", f"{f.opacity:.2f}"])
        if key == ord('d'):
            debug = not debug
            for f in filters:
                f.debug = debug   # lets a filter draw its own debug info too

    # The frame only ever lived in memory and gets overwritten every loop. Nothing is saved.
    cam.release()
    if log_file:
        log_file.close()
    if sensor:
        sensor.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
