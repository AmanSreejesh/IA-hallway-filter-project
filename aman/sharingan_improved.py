"""MediaPipe Sharingan eye renderer, adapted from Aman's webcam prototype.

Run through the shared camera/switcher: python common/main.py sharingan
Or: python aman/sharingan_improved.py

MediaPipe inference, identity tracking and eligibility live in common/. This
module only draws into each person's current iris, clipped to their eyelids.
Artwork is drawn with OpenCV; no PNG, YuNet or image-based iris search is used.
"""

import math
from pathlib import Path
import sys

import cv2
import numpy as np

OVERLAY_ALPHA = 0.85
ART_SIZE = 256
# Ordered contours from MediaPipe's FaceMesh topology, followed by iris center
# and ring indices. Ordering is anatomical, never based on detector list order.
EYES = (
    ((33, 246, 161, 160, 159, 158, 157, 173, 133,
      155, 154, 153, 145, 144, 163, 7), 468, (469, 470, 471, 472)),
    ((263, 466, 388, 387, 386, 385, 384, 398, 362,
      382, 381, 380, 374, 373, 390, 249), 473, (474, 475, 476, 477)),
)


def smoothstep(low, high, value):
    t = np.clip((value - low) / (high - low), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def make_artwork(size=ART_SIZE):
    """Draw a red iris, black pupil/rings and four curved tomoe like the reference."""
    # Supersample once, rather than loading or resizing an external image.
    side = size * 3
    center = (side - 1) / 2
    radius = side * 0.48
    art = np.zeros((side, side, 3), np.uint8)
    art[:] = (8, 12, 235)

    def pixels(points):
        return np.rint(np.asarray(points) * radius + center).astype(np.int32)

    cv2.circle(art, tuple(pixels((0, 0))), round(radius * 0.96), (0, 0, 0), round(radius * 0.055), cv2.LINE_AA)
    cv2.circle(art, tuple(pixels((0, 0))), round(radius * 0.62), (0, 0, 0), round(radius * 0.025), cv2.LINE_AA)
    cv2.circle(art, tuple(pixels((0, 0))), round(radius * 0.20), (0, 0, 0), -1, cv2.LINE_AA)
    t = np.linspace(0, 1, 40)[:, None]
    # A curved, tapering tail attached to a round tomoe.
    controls = np.array([[0.62, 0.12], [0.90, 0.08], [0.86, -0.15], [0.76, -0.25]])
    tail = ((1-t)**3 * controls[0] + 3*(1-t)**2*t * controls[1] +
            3*(1-t)*t**2 * controls[2] + t**3 * controls[3])
    tail = np.vstack((tail, [0.74, -0.05], [0.62, -0.12]))
    for angle in np.arange(4) * math.pi / 2 - math.pi / 8:
        c, s = math.cos(angle), math.sin(angle)
        rotation = np.array([[c, -s], [s, c]])
        cv2.fillPoly(art, [pixels(tail @ rotation.T)], (0, 0, 0), cv2.LINE_AA)
        cv2.circle(art, tuple(pixels(rotation @ [0.62, 0])), round(radius * 0.12), (0, 0, 0), -1, cv2.LINE_AA)
    y, x = np.mgrid[:side, :side]
    radial = np.hypot(x - center, y - center) / radius
    alpha = (1 - smoothstep(0.95, 1.0, radial)).astype(np.float32)
    # Premultiplied color prevents dark interpolation fringes at the edge.
    rgba = np.dstack((art.astype(np.float32) / 255 * alpha[:, :, None], alpha))
    return cv2.resize(rgba, (size, size), interpolation=cv2.INTER_AREA)


def render_eye(frame, artwork, landmarks, eye, opacity, roll):
    contour_indices, center_index, ring_indices = eye
    contour = landmarks[list(contour_indices)]
    center = landmarks[center_index]
    ring = landmarks[list(ring_indices)]
    # Use the shared pose so the artwork rotates by the exact same head angle
    # supplied to every other filter. Iris landmarks supply size and position.
    width = float(np.linalg.norm(contour[8] - contour[0]))
    if width < 2:
        return
    angle = math.radians(roll)
    across = np.array([math.cos(angle), math.sin(angle)])
    down = np.array([-across[1], across[0]])
    opening = float(np.ptp((contour - center) @ down))
    if opening / width < 0.08:
        return  # Closed eye: do not paint the eyelid.
    rx = float(np.ptp((ring - center) @ across)) / 2
    ry = float(np.ptp((ring - center) @ down)) / 2
    if min(rx, ry) < 0.6 or max(rx, ry) > width * 0.45:
        return  # Missing/corrupt iris geometry must not color the whole eye.
    extent = np.abs(across) * rx + np.abs(down) * ry + 2
    h, w = frame.shape[:2]
    x0, y0 = np.maximum(np.floor(center - extent).astype(int), 0)
    x1, y1 = np.minimum(np.ceil(center + extent).astype(int) + 1, (w, h))
    if x1 <= x0 or y1 <= y0:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    dx, dy = xx - center[0], yy - center[1]
    u = (dx * across[0] + dy * across[1]) / rx
    v = (dx * down[0] + dy * down[1]) / ry
    # Artwork radius is 48% of its canvas; map iris-edge coordinates to that.
    scale = artwork.shape[0] * 0.48
    origin = (artwork.shape[0] - 1) / 2
    mapped = cv2.remap(artwork, (origin + u * scale).astype(np.float32),
                       (origin + v * scale).astype(np.float32), cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    # Clip the iris to the actual 16-point eyelid aperture, with subpixel edges.
    supersample = 4
    aperture = np.zeros(((y1-y0)*supersample, (x1-x0)*supersample), np.uint8)
    polygon = np.rint((contour - [x0, y0] + 0.5) * supersample).astype(np.int32)
    cv2.fillPoly(aperture, [polygon], 255)
    aperture = cv2.resize(aperture, (x1-x0, y1-y0), interpolation=cv2.INTER_AREA) / 255.0
    roi = frame[y0:y1, x0:x1]
    luminance = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    lighting = 0.65 + 0.45 * cv2.GaussianBlur(luminance, (0, 0), 1.0)
    # Preserve bright catchlights from the real eye.
    strength = opacity * OVERLAY_ALPHA * aperture * (1 - 0.85 * smoothstep(0.82, 0.98, luminance))
    alpha = mapped[:, :, 3] * strength
    color = mapped[:, :, :3] * lighting[:, :, None] * strength[:, :, None]
    roi[:] = np.clip(roi.astype(np.float32) * (1-alpha[:, :, None]) + 255*color, 0, 255).astype(np.uint8)


class SharinganFilter:
    name = "Sharingan"
    tracker = "mediapipe"
    requires_landmarks = True

    def __init__(self):
        self.artwork = make_artwork()

    def apply(self, frame, face, opacity):
        if opacity <= 0:
            return
        landmarks = face.landmarks
        if landmarks is None or landmarks.shape != (478, 2) or not np.isfinite(landmarks).all():
            return
        for eye in EYES:
            render_eye(frame, self.artwork, landmarks, eye, float(np.clip(opacity, 0, 1)), face.roll)


if __name__ == "__main__":
    # One camera and one switcher, including when launched by the old filename.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from common.main import main
    sys.exit(main(["sharingan", *sys.argv[1:]]))
