"""Aryan's clown nose using the shared camera, geometry and activation rules.

Optional clown_nose.png is loaded next to this file. The procedural red rubber
nose lets a clean checkout work without the prototype's missing image.
"""
import math
from pathlib import Path

import cv2
import numpy as np

from common.overlay import draw_artwork, premultiply

NOSE_PATH = Path(__file__).with_name("clown_nose.png")
# Preserve Aryan's 0.475 * face-width sizing using a turn-independent eye gap.
NOSE_SCALE = 0.475 * 2.6


def make_nose(size=192):
    """Draw the missing asset in code, with a soft edge and a small highlight."""
    y, x = np.mgrid[:size, :size].astype(np.float32)
    x = (x + 0.5 - size / 2) / (size * 0.47)
    y = (y + 0.5 - size / 2) / (size * 0.47)
    radius = np.hypot(x, y)
    light = np.clip(1 - 0.35 * radius + 0.12 * (-x-y), 0.4, 1)
    shine = np.exp(-((x + 0.32)**2 + (y + 0.35)**2) / 0.025)
    color = np.stack([10 * light + 155 * shine, 14 * light + 170 * shine,
                      245 * light + 10 * shine], axis=-1)
    alpha = np.clip((1-radius) * size * 0.47, 0, 1) * 255
    return np.dstack((np.clip(color, 0, 255), alpha)).astype(np.uint8)


class ClownNoseFilter:
    name = "Clown Nose"
    tracker = "mediapipe"

    def __init__(self):
        image = cv2.imread(str(NOSE_PATH), cv2.IMREAD_UNCHANGED) if NOSE_PATH.exists() else make_nose()
        self.artwork = premultiply(image)

    def apply(self, frame, face, opacity):
        width = face.size * NOSE_SCALE
        # Rotate the prototype's 0.15-width upward offset with the head so the
        # detected tip stays inside the red sphere during a sway.
        angle = math.radians(face.roll)
        center = np.asarray(face.nose) + width * 0.15 * np.array([math.sin(angle), -math.cos(angle)])
        draw_artwork(frame, self.artwork, center, width, face.roll, opacity)
