"""
Main hallway filter system.
Combines each team member's filter and applies the shared features to all of them.

Run from the repo root:
    python common/main.py              # every filter, taking turns
    python common/main.py emoji        # only the filters you name (for testing your own)

Keys:  q = quit   n = next filter   d = debug view (distance + head angles, for calibrating)

How a filter plugs in (see viraj/emoji_filter.py for an example):
  - Make a class with a `name`, an `apply(frame, face, opacity)` method that draws onto
    `frame`, and optionally a `forget(face_id)` method that's called when a person leaves.
  - Add it to FILTERS below.
  - Filters only draw. This file decides who gets a filter, so every filter gets the same
    opt-in zone, fade-in, timer, and multi-user behaviour.

Nothing is ever saved. Each frame is grabbed, drawn on, shown, and then thrown away.
"""
import importlib
import math
import os
import sys
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

# name -> (module, class). Add your filter here.
FILTERS = {
    "emoji": ("viraj.emoji_filter", "EmojiFilter"),
    # "sharingan": ("aman.sharingan_filter", "SharinganFilter"),
    # "astronaut": ("ishan.astronaut_filter", "AstronautFilter"),
    # "clown_nose": ("aryan.clown_nose_filter", "ClownNoseFilter"),
}

# --- Settings (keep in sync with constraints.md) ---
CAMERA_SIZE = (640, 480)

# Opt-in zone. Distance is estimated from how far apart the eyes look on screen.
FOCAL_PX = 600            # camera focal length in pixels -- calibrate with the 'd' debug view
EYE_GAP_INCHES = 2.5      # average distance between a person's eyes
ZONE_ENTER_INCHES = 30    # step closer than this to opt in
ZONE_EXIT_INCHES = 33     # ...and you have to step back past this to opt out.
                          # The gap between the two stops flicker at the border.
MAX_ANGLE = 45            # turn/tilt your head past this and the filter goes away
RETURN_ANGLE = 40         # ...and comes back once you're under this (same anti-flicker idea)

# Fade-in: people walking past shouldn't get a filter slapped on them.
DWELL_SECONDS = 1.0       # must stay opted in this long before anything shows
FADE_START = 0.1          # first visible opacity
FADE_IN_SECONDS = 2.0     # time to go from FADE_START to fully visible
FADE_OUT_SECONDS = 0.4

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
ENFORCE_SCHEDULE = True
FILTERS_OFF_AT = dtime(16, 15)


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
            filters.append(getattr(importlib.import_module(module_name), class_name)())
            print(f"Loaded filter: {name}")
        except Exception as e:
            print(f"Skipping filter '{name}': {e}")
    return filters


def load_model() -> None:
    """Download the YuNet face model the first time the program runs."""
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 100_000:
        return
    print("Downloading YuNet face model (one time only)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Done.")


# --- Face data handed to every filter ---
class Face:
    """One tracked person. Everything here is already smoothed.

    id                       stays the same while the person is in view
    box                      (x, y, w, h)
    left_eye, right_eye, nose, mouth_left, mouth_right   (x, y) points, left = left on screen
    size                     face size in pixels (about the eye gap, but steady when you turn)
    roll                     head tilt in degrees (ear toward shoulder)
    yaw                      head turn left/right in degrees (0 = facing the camera)
    pitch                    head nod up/down in degrees (positive = looking down)
    distance                 estimated inches from the camera
    """

    def __init__(self, face_id: int):
        self.id = face_id
        self.points = None      # 7x2 array: box corner, box size, eyes, nose, mouth corners
        self.size = 0.0
        self.roll = self.yaw = self.pitch = 0.0
        self.distance = 999.0
        # system state (filters don't need these)
        self.last_seen = 0.0
        self.opted_in = False
        self.dwell_start = None
        self.opacity = 0.0
        self.session_start = None
        self.last_eligible = 0.0

    box = property(lambda self: (*self.points[0], *self.points[1]))
    left_eye = property(lambda self: tuple(self.points[2]))
    right_eye = property(lambda self: tuple(self.points[3]))
    nose = property(lambda self: tuple(self.points[4]))
    mouth_left = property(lambda self: tuple(self.points[5]))
    mouth_right = property(lambda self: tuple(self.points[6]))


# --- Head pose + smoothing ---
def raw_points(det: np.ndarray) -> np.ndarray:
    """YuNet row -> 7x2 points, with eyes and mouth corners sorted left/right on screen."""
    # Row: [x, y, w, h, eye1, eye2, nose, mouth1, mouth2 (x, y each), score]
    eyes = sorted([det[4:6], det[6:8]], key=lambda p: p[0])
    mouth = sorted([det[10:12], det[12:14]], key=lambda p: p[0])
    return np.array([det[0:2], det[2:4], *eyes, det[8:10], *mouth], dtype=np.float32)


def head_pose(p: np.ndarray) -> tuple:
    """Return (roll, yaw, pitch, size) from smoothed points."""
    left_eye, right_eye, nose = p[2], p[3], p[4]
    eye_vec = right_eye - left_eye
    eye_dist = max(float(np.hypot(*eye_vec)), 1.0)
    eye_mid = (left_eye + right_eye) / 2
    mouth_mid = (p[5] + p[6]) / 2

    roll = math.degrees(math.atan2(eye_vec[1], eye_vec[0]))

    # Yaw: when you turn, the nose slides toward one eye. Measured along the eye line
    # so tilting doesn't fool it. 0.5 is roughly how far the nose sticks out compared to the eye gap.
    turn = float(np.dot(nose - eye_mid, eye_vec)) / (eye_dist * eye_dist)
    yaw = math.degrees(math.atan(turn / 0.5))

    # Pitch: when you nod, the nose slides toward your eyes or your mouth.
    # The nose normally sits ~55% of the way from eye line to mouth (tune if needed).
    down_vec = mouth_mid - eye_mid
    down_len = max(float(np.hypot(*down_vec)), 1.0)
    along = float(np.dot(nose - eye_mid, down_vec)) / (down_len * down_len)
    pitch = math.degrees(math.atan((along - 0.55) / 0.5))

    # Size: the eye gap shrinks when you turn and the eye-to-mouth gap shrinks when you nod,
    # so undo each one and average them. This was the main cause of the emoji jitter.
    cos_yaw = max(math.cos(math.radians(yaw)), 0.5)
    cos_pitch = max(math.cos(math.radians(pitch)), 0.5)
    size = (eye_dist / cos_yaw + down_len / cos_pitch) / 2
    return roll, yaw, pitch, size


def smooth(old, new, amount: float):
    """Blend toward the new value. amount = how much of the new value to take (0..1)."""
    return new if old is None else old + (new - old) * amount


def update_face(face: Face, det: np.ndarray, now: float) -> None:
    pts = raw_points(det)
    # Move fast when the person moves a lot (no lag), smooth heavily when they're still (no jitter)
    jump = 0.0 if face.points is None else float(np.abs(pts[4] - face.points[4]).max())
    amount = 0.8 if jump > max(face.size, 1.0) * 0.3 else 0.4
    face.points = smooth(face.points, pts, amount)

    roll, yaw, pitch, size = head_pose(face.points)
    first = face.size == 0.0
    face.roll = roll if first else smooth(face.roll, roll, 0.4)
    face.yaw = yaw if first else smooth(face.yaw, yaw, 0.4)
    face.pitch = pitch if first else smooth(face.pitch, pitch, 0.4)
    face.size = size if first else smooth(face.size, size, 0.15)
    face.distance = FOCAL_PX * EYE_GAP_INCHES / face.size
    face.last_seen = now


# --- Multi-user ---
# Match each detection to the closest person from last frame, so everyone keeps their id.
def track(faces: dict, detections, now: float, next_id: list) -> None:
    unmatched = dict(faces)
    for det in sorted(detections, key=lambda d: -d[2]):   # biggest (closest) faces first
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
        update_face(best, det, now)


# --- Opt-in zone ---
# Only apply a filter when a person is close, looking at the screen, and fully in view.
def is_eligible(face: Face, frame_w: int, frame_h: int) -> bool:
    if ENFORCE_SCHEDULE and datetime.now().time() >= FILTERS_OFF_AT:
        return False
    zone = ZONE_EXIT_INCHES if face.opted_in else ZONE_ENTER_INCHES
    angle = MAX_ANGLE if face.opted_in else RETURN_ANGLE
    if face.distance > zone:
        return False
    if abs(face.yaw) > angle or abs(face.pitch) > angle or abs(face.roll) > angle:
        return False
    x, y, w, h = face.box
    if face.opted_in:
        nx, ny = face.nose   # once opted in, only the nose has to stay on screen
        return 0 <= nx < frame_w and 0 <= ny < frame_h
    return x >= 0 and y >= 0 and x + w <= frame_w and y + h <= frame_h


# --- Fade-in ---
# Start the filter at low opacity and raise it while the person stays. Fade out when they leave.
def update_opacity(face: Face, eligible: bool, now: float, dt: float) -> None:
    face.opted_in = eligible
    if eligible:
        face.last_eligible = now
        if face.dwell_start is None:
            face.dwell_start = now
        if now - face.dwell_start >= DWELL_SECONDS:
            face.opacity = min(1.0, max(face.opacity, FADE_START) + dt * (1 - FADE_START) / FADE_IN_SECONDS)
    else:
        face.dwell_start = None
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
    load_model()

    backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
    cap = cv2.VideoCapture(0, backend)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_SIZE[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_SIZE[1])
    detector = cv2.FaceDetectorYN.create(MODEL_PATH, "", CAMERA_SIZE, score_threshold=0.8)
    print("Running. q = quit, n = next filter, d = debug view.")

    faces = {}
    next_id = [0]
    current = 0
    next_switch = time.monotonic() + FILTER_SWITCH_SECONDS
    debug = False
    last = time.monotonic()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break
        frame = cv2.flip(frame, 1)   # mirror view, like a selfie camera
        h, w = frame.shape[:2]
        now = time.monotonic()
        dt, last = now - last, now

        # Engagement: switch between the team's filters
        if now >= next_switch and len(filters) > 1:
            current = (current + 1) % len(filters)
            next_switch = now + FILTER_SWITCH_SECONDS
        active = filters[current]

        detector.setInputSize((w, h))
        _, detections = detector.detect(frame)
        track(faces, detections if detections is not None else [], now, next_id)

        firm = False
        anyone = False
        for face in list(faces.values()):
            # If the detector missed someone for a moment, hold their filter steady instead of flickering
            seen = face.last_seen == now
            if seen or now - face.last_seen > LOST_GRACE_SECONDS:
                update_opacity(face, seen and is_eligible(face, w, h), now, dt)
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
            firm |= draw_timer(frame, face, now)
            if debug:
                draw_debug(frame, face)

        if firm:
            draw_text(frame, "3 minutes! Time to get back to your work.", w / 2, 40, 0.8, (0, 200, 255), 2)
        if ENFORCE_SCHEDULE and datetime.now().time() >= FILTERS_OFF_AT:
            draw_text(frame, "Filters are off for the day", w / 2, h - 16, 0.5, (200, 200, 200))
        elif not anyone:
            draw_text(frame, "Step up close and look here to try a filter", w / 2, h - 16, 0.5, (255, 255, 255))
        if debug:
            draw_text(frame, f"Filter: {active.name}", 80, 20, 0.5, (255, 255, 0))

        cv2.imshow("Hallway Filter", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('n'):
            current = (current + 1) % len(filters)
            next_switch = now + FILTER_SWITCH_SECONDS
        if key == ord('d'):
            debug = not debug

    # The frame only ever lived in memory and gets overwritten every loop. Nothing is saved.
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
