"""
Helmet filter: puts an astronaut helmet (helmet.png) over each face.

This is helmet_Test.py turned into a filter for common/main.py. The placement and tuning are the
same; main.py now handles the distance fade, the opt-in zone and the timer for every filter.

Run it on its own with:  python ishan/helmet_filter.py
"""
import math
import os
import sys

import cv2
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HELMET_PATH = os.path.join(SCRIPT_DIR, "helmet.png")

# Tuning knobs (same as helmet_Test.py)
HELMET_SCALE = 3.8   # helmet width = face width * this
FACE_Y_RATIO = 0.5   # how far down the helmet picture the middle of the face goes
WIDTH_SCALE = 0.5    # squash the picture narrower...
HEIGHT_SCALE = 0.8   # ...and taller before using it


def load_helmet() -> np.ndarray:
    helmet = cv2.imread(HELMET_PATH, cv2.IMREAD_UNCHANGED)
    if helmet is None:
        raise FileNotFoundError(f"Could not load {HELMET_PATH}")
    if helmet.ndim != 3 or helmet.shape[2] != 4:
        raise ValueError("helmet.png must have transparency (4 channels).")
    h, w = helmet.shape[:2]
    return cv2.resize(helmet, (int(w * WIDTH_SCALE), int(h * HEIGHT_SCALE)), interpolation=cv2.INTER_AREA)


class HelmetFilter:
    name = "Helmet"
    tracker = "mediapipe"   # same tracker as the other filters, so switching to it is instant

    def __init__(self):
        self.helmet = load_helmet()

    def apply(self, frame: np.ndarray, face, opacity: float) -> None:
        if opacity <= 0:
            return
        points = np.array([face.left_eye, face.right_eye, face.nose, face.mouth_left, face.mouth_right])
        center_x, center_y = points.mean(axis=0)

        # Face width from the eyes and mouth, like the original
        eye_width = abs(face.right_eye[0] - face.left_eye[0])
        mouth_width = abs(face.mouth_right[0] - face.mouth_left[0])
        face_width = max(eye_width, mouth_width, face.size)

        angle = face.roll
        if not math.isfinite(angle):   # no head pose this frame: use the eye line
            dx, dy = np.subtract(face.right_eye, face.left_eye)
            angle = math.degrees(math.atan2(dy, dx))

        h_img, w_img = self.helmet.shape[:2]
        scale = face_width * HELMET_SCALE / w_img
        anchor = (w_img / 2, h_img * FACE_Y_RATIO)
        M = cv2.getRotationMatrix2D(anchor, -angle, scale)
        M[0, 2] += center_x - anchor[0]
        M[1, 2] += center_y - anchor[1]

        # Only warp and blend the patch the helmet covers, not the whole frame (much faster on the Pi)
        corners = np.array([[0, 0, 1], [w_img, 0, 1], [0, h_img, 1], [w_img, h_img, 1]], dtype=np.float64)
        moved = corners @ M.T
        frame_h, frame_w = frame.shape[:2]
        x0, y0 = np.floor(moved.min(axis=0)).astype(int)
        x1, y1 = np.ceil(moved.max(axis=0)).astype(int)
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, frame_w), min(y1, frame_h)
        if x1 <= x0 or y1 <= y0:
            return
        M[0, 2] -= x0
        M[1, 2] -= y0
        patch = cv2.warpAffine(self.helmet, M, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

        alpha = patch[:, :, 3:4].astype(np.float32) * (min(opacity, 1.0) / 255.0)
        roi = frame[y0:y1, x0:x1]
        roi[:] = (roi * (1 - alpha) + patch[:, :, :3] * alpha).astype(np.uint8)


if __name__ == "__main__":
    # One camera and one switcher, the same as the other filters
    sys.path.insert(0, os.path.dirname(SCRIPT_DIR))
    from common.main import main
    sys.exit(main(["helmet", *sys.argv[1:]]))
