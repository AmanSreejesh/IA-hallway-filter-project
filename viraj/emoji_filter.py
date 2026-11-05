"""
Emoji filter: puts a random 3D emoji over a face.

This is an attachable module for common/main.py. It only draws. main.py decides
WHO gets a filter (opt-in zone, fade-in, timer, multi-user) and hands this filter
one face at a time, with its current-frame landmarks and head angles worked out.

How it behaves:
  - Each emoji is a 3D model (see emoji_3d.py) that turns with your head: tilt (roll),
    turn left/right (yaw), and nod up/down (pitch). It's drawn with OpenGL when the
    computer supports it, otherwise with OpenCV.
  - Every person keeps their own emoji for exactly SWAP_SECONDS, then gets a new one
    that is different from their last one and from everyone else's (if possible).
  - Nothing is ever saved.

Needs: nothing extra (pip install moderngl for the faster, smoother OpenGL drawing)
"""
import math
import random
import time

import cv2
import numpy as np

from viraj.emoji_3d import EXTENT, make_renderer

# Tuning knobs
RENDERER = "auto"      # "auto" (OpenGL if possible, else OpenCV), "gl", or "software"
SIZE_SCALE = 5.0       # emoji width = face size (about the eye gap) * SIZE_SCALE. 5 covers a whole head, hair included.
HEAD_UP = 0.45         # the middle of the head (with hair) is this far above the eye line, in face sizes
HEAD_DEPTH = 1.35      # ...and this far behind the eyes. The head turns around that point, so the emoji
                       # is centered there to stay over the whole head (not just the face) when you turn.
MAX_RENDER = 512       # bigger emojis are drawn at this size and scaled up, so close-up faces stay fast
PERSPECTIVE = 3.0      # how strong the 3D effect is when you turn or nod (lower = stronger)
SWAP_SECONDS = 5       # how long each person keeps an emoji


def rotation(roll: float, yaw: float, pitch: float) -> np.ndarray:
    """3D rotation for the emoji (OpenCV camera axes: x right, y down, z away)."""
    r, y, p = (math.radians(a) for a in (roll, -yaw, pitch))
    rz = np.array([[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1]])
    ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    return rz @ ry @ rx


def head_center(face, turn: np.ndarray) -> np.ndarray:
    """The middle of the head on screen: start between the eyes, then go up and back to the
    head's real center, turned the same way as the head (`turn`), so it's right at any angle."""
    eye_mid = (np.array(face.left_eye) + np.array(face.right_eye)) / 2
    offset = turn @ np.array([0.0, -HEAD_UP, HEAD_DEPTH]) * face.size   # up is -y, back is +z
    return eye_mid + offset[:2]


def draw_emoji(frame: np.ndarray, renderer, name: str, face, opacity: float) -> None:
    """Draw the 3D emoji turned to match the head, centered on it, and blend it in.
    Draws straight onto `frame` (no copy), which saves time on the Pi."""
    turn = rotation(face.roll, face.yaw, face.pitch)
    size = max(int(face.size * SIZE_SCALE * EXTENT), 8)
    img = renderer.render(name, turn, min(size, MAX_RENDER), PERSPECTIVE)
    if img.shape[0] != size:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    cx, cy = head_center(face, turn)
    x0, y0 = int(round(cx - size / 2)), int(round(cy - size / 2))
    f_h, f_w = frame.shape[:2]
    fx0, fy0, fx1, fy1 = max(x0, 0), max(y0, 0), min(x0 + size, f_w), min(y0 + size, f_h)
    if fx1 <= fx0 or fy1 <= fy0:
        return
    # Scaled by main.py's opacity (fade-in / fade-out). The renderers hand back colors already
    # multiplied by their own transparency.
    part = img[fy0 - y0:fy1 - y0, fx0 - x0:fx1 - x0].astype(np.float32) * (opacity / 255.0)
    roi = frame[fy0:fy1, fx0:fx1]
    roi[:] = (roi * (1 - part[:, :, 3:4]) + part[:, :, :3] * 255).astype(np.uint8)


class EmojiFilter:
    name = "Emoji"
    tracker = "mediapipe"   # the 3D emoji turns with your head, so it wants MediaPipe's accurate head angles

    def __init__(self):
        self.renderer = make_renderer(RENDERER)
        self.emojis = self.renderer.names
        # face id -> [emoji index, time of next swap]
        # Each face has its own clock, so the 5 second swap doesn't get thrown off
        # when someone else walks in or out of the frame.
        self.assigned = {}

    def _pick(self, face_id: int, now: float) -> str:
        slot = self.assigned.get(face_id)
        if slot is None or now >= slot[1]:
            taken = {s[0] for fid, s in self.assigned.items() if fid != face_id}
            previous = slot[0] if slot else None
            choices = [i for i in range(len(self.emojis)) if i not in taken and i != previous]
            if not choices:   # more people than emojis -> just avoid repeating your own
                choices = [i for i in range(len(self.emojis)) if i != previous] or [0]
            slot = [random.choice(choices), now + SWAP_SECONDS]
            self.assigned[face_id] = slot
        return self.emojis[slot[0]]

    def forget(self, face_id: int) -> None:
        """Called by main.py when a person leaves, so their emoji frees up."""
        self.assigned.pop(face_id, None)

    def apply(self, frame: np.ndarray, face, opacity: float) -> None:
        draw_emoji(frame, self.renderer, self._pick(face.id, time.monotonic()), face, opacity)
