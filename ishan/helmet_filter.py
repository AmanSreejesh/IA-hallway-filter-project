"""
Helmet filter: puts an astronaut helmet (helmet.png) over each face.

The upstream HelmetFilter name delegates drawing to AstronautFilter so both
commands share the same head tracking, placement, clipping and alpha handling.
common/main.py handles activation, distance, timing and schedule for every filter.

Run it on its own with:  python ishan/helmet_filter.py
"""
import os
import sys

import cv2
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HELMET_PATH = os.path.join(SCRIPT_DIR, "helmet.png")

# Also support the upstream standalone command before importing project modules.
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(SCRIPT_DIR))
from ishan.astronaut_filter import AstronautFilter

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


class HelmetFilter(AstronautFilter):
    """Keep the upstream command and wheel name with the acceptance-tested renderer."""
    name = "Helmet"


if __name__ == "__main__":
    # One camera and one switcher, the same as the other filters
    sys.path.insert(0, os.path.dirname(SCRIPT_DIR))
    from common.main import main
    sys.exit(main(["helmet", *sys.argv[1:]]))
