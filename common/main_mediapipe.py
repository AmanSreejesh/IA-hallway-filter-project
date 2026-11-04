"""
MediaPipe version of the hallway filter.

Exactly the same system as main.py -- same opt-in zone, fade-in, timer, multi-user, ultrasonic
sensor, and filters -- but faces are tracked with MediaPipe's Face Landmarker instead of YuNet.
MediaPipe follows 478 points on each face from frame to frame, handles tilted heads, and
measures the head's 3D angle itself, so turning, nodding and tilting are far more accurate
than YuNet's 5 points can manage.

Run from the repo root:
    python common/main_mediapipe.py            # every filter, taking turns
    python common/main_mediapipe.py emoji      # only the filters you name

Needs:  pip install mediapipe   (works on 64-bit Raspberry Pi OS)
The model file (~4 MB) is downloaded automatically the first time.
"""
import math
import os
import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

import main

MODEL_PATH = os.path.join(main.COMMON_DIR, "face_landmarker.task")
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")
MAX_FACES = 3             # each extra face costs time on the Pi
MAX_FLIP = 60             # a tilt this many degrees away from last frame is a mistake, not a real head
STRAIGHTEN_ABOVE = 20     # if the closest person tilts more than this, turn the picture so their face is upright

# Which of MediaPipe's 478 points become main.py's 5 points (eye, eye, nose, mouth corner, mouth corner).
# The first of each pair is the one on the left of the screen when the head is upright.
EYE_A = (33, 133)         # corners of one eye (averaged to get its center)
EYE_B = (362, 263)        # corners of the other eye
NOSE_TIP = 1
MOUTH = (61, 291)         # mouth corners

# MediaPipe's 3D axes (y up, z toward you) -> OpenCV's camera axes (y down, z away)
TO_OPENCV = np.diag([1.0, -1.0, -1.0])


class MediaPipeDetector:
    def __init__(self):
        main.download(MODEL_URL, MODEL_PATH)
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            # VIDEO mode follows faces from frame to frame instead of searching the whole
            # picture every time: steadier, and much faster on the Pi
            running_mode=vision.RunningMode.VIDEO,
            num_faces=MAX_FACES,
            output_facial_transformation_matrixes=True)   # MediaPipe's own 3D head angle
        self.landmarker = vision.FaceLandmarker.create_from_options(options)
        self.last_ms = 0

    def detect(self, frame: np.ndarray, tracked: list = ()) -> list:
        """Returns (row, tilt, (yaw, pitch)) per face, in main.py's format.
        `tracked` = (nose, tilt, size) for each person from last frame."""
        h, w = frame.shape[:2]
        # MediaPipe loses faces tilted past ~50 degrees. If the closest person is tilted, turn the
        # picture around their face so it's upright, then turn the results back afterwards.
        turn = None
        known = [t for t in tracked if t[1] is not None]
        if known:
            nose, tilt, _ = max(known, key=lambda t: t[2])
            if abs(tilt) > STRAIGHTEN_ABOVE:
                turn = cv2.getRotationMatrix2D((float(nose[0]), float(nose[1])), tilt, 1.0)
                frame = cv2.warpAffine(frame, turn, (w, h))
                back = cv2.invertAffineTransform(turn)
                c, s = math.cos(math.radians(tilt)), math.sin(math.radians(tilt))
                spin = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])   # the same turn, in 3D
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
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
                            *pts[list(EYE_A)].mean(axis=0), *pts[list(EYE_B)].mean(axis=0),
                            *pts[NOSE_TIP], *pts[MOUTH[0]], *pts[MOUTH[1]], 1.0], dtype=np.float32)
            rotation = TO_OPENCV @ np.asarray(matrix)[:3, :3] @ TO_OPENCV
            if turn is not None:
                rotation = spin @ rotation
            yaw, pitch, roll = main.angles_from_rotation(rotation)
            # On a very tilted head MediaPipe occasionally finds the face upside down. A real head
            # can't flip that far between two frames, so skip it (main.py holds the last position).
            nose = pts[NOSE_TIP]
            if any(np.hypot(*(np.asarray(n) - nose)) < size * 2.5 and abs(main.wrap(roll - tilt)) > MAX_FLIP
                   for n, tilt, size in tracked if tilt is not None):
                continue
            out.append((row, roll, (yaw, pitch)))
        return out


if __name__ == "__main__":
    main.main(MediaPipeDetector)
