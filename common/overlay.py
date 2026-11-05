"""Clipped, alpha-correct image rendering. Activation belongs to common.main."""
import math

import cv2
import numpy as np


def premultiply(image):
    """Convert BGRA bytes once, avoiding dark fringes during rotation."""
    if image is None or image.ndim != 3 or image.shape[2] != 4:
        raise ValueError("Filter artwork must be a BGRA image with transparency")
    art = image.astype(np.float32) / 255.0
    art[:, :, :3] *= art[:, :, 3:4]
    return art


def draw_artwork(frame, artwork, center, width, roll, opacity, *, aspect=1.0, anchor=(0.5, 0.5)):
    """Rotate artwork about a face anchor; clip allocations to the camera frame.

    width is in pixels; aspect multiplies the native height/width ratio.
    """
    values = (*center, width, roll, opacity, aspect)
    if not all(math.isfinite(v) for v in values) or width <= 0 or aspect <= 0 or opacity <= 0:
        return
    h, w = artwork.shape[:2]
    sx, sy = width / w, width / w * aspect
    angle = math.radians(roll)
    c, s = math.cos(angle), math.sin(angle)
    linear = np.array([[c * sx, -s * sy], [s * sx, c * sy]], np.float32)
    offset = np.asarray(center) - linear @ np.array([anchor[0] * w, anchor[1] * h])
    matrix = np.column_stack((linear, offset)).astype(np.float32)
    corners = np.array([[0, 0], [w, 0], [0, h], [w, h]]) @ linear.T + offset
    frame_h, frame_w = frame.shape[:2]
    x0, y0 = np.maximum(np.floor(corners.min(axis=0)).astype(int), 0)
    x1, y1 = np.minimum(np.ceil(corners.max(axis=0)).astype(int), [frame_w, frame_h])
    if x1 <= x0 or y1 <= y0:
        return
    matrix[:, 2] -= (x0, y0)
    warped = cv2.warpAffine(artwork, matrix, (int(x1-x0), int(y1-y0)), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    strength = float(np.clip(opacity, 0, 1))
    alpha = warped[:, :, 3:4] * strength
    roi = frame[y0:y1, x0:x1]
    roi[:] = np.clip(roi * (1-alpha) + warped[:, :, :3] * (255 * strength), 0, 255).astype(np.uint8)
