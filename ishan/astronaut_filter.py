"""Ishan's transparent astronaut helmet, rendered through the shared system."""
import math
from pathlib import Path

import cv2
import numpy as np

from common.overlay import draw_artwork, premultiply

HELMET_PATH = Path(__file__).with_name("helmet.png")
HELMET_SCALE = 3.8


class AstronautFilter:
    name = "Astronaut"
    tracker = "mediapipe"

    def __init__(self):
        self.artwork = premultiply(cv2.imread(str(HELMET_PATH), cv2.IMREAD_UNCHANGED))

    def apply(self, frame, face, opacity):
        # Preserve the prototype's (0.5, 0.8) resize proportions at full resolution.
        center = np.mean([face.left_eye, face.right_eye, face.nose,
                          face.mouth_left, face.mouth_right], axis=0)
        across = max(math.cos(math.radians(face.yaw)), 0.5)
        down = max(math.cos(math.radians(face.pitch)), 0.5)
        draw_artwork(frame, self.artwork, center, face.size * HELMET_SCALE * across,
                     face.roll, opacity, aspect=1.6 * down / across)
