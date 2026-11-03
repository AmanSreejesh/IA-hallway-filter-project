"""
Emoji filter: puts a random emoji over a face.

This is an attachable module for common/main.py. It only draws. main.py decides
WHO gets a filter (opt-in zone, fade-in, timer, multi-user) and hands this filter
one face at a time, already smoothed, with its head angles worked out.

How it behaves:
  - The emoji turns like a card in 3D to follow your head: tilt (roll), turn left/right
    (yaw), and nod up/down (pitch). It's one small warp per face, so it runs fine on a Pi.
  - Every person keeps their own emoji for exactly SWAP_SECONDS, then gets a new one
    that is different from their last one and from everyone else's (if possible).
  - Nothing is ever saved.

Needs:
  - an "emojis" folder next to this file, full of PNGs with transparent backgrounds
"""
import glob
import math
import os
import random
import time

import cv2
import numpy as np

EMOJI_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "emojis")

# Tuning knobs
SIZE_SCALE = 3.8       # emoji width = face size (about the eye gap) * SIZE_SCALE. 3.8 covers the whole head.
CENTER_DOWN = 0.2      # where the emoji is centered: 0 = eye line, 1 = mouth
HEAD_DEPTH = 0.5       # how far behind the face the head's center is (in face sizes).
                       # Keeps the emoji over the head instead of sliding off when you turn.
PERSPECTIVE = 3.0      # how strong the 3D effect is when you turn or nod (lower = stronger)
SWAP_SECONDS = 5       # how long each person keeps an emoji
EMOJI_PIXELS = 160     # emojis get shrunk to this size when loaded (smaller = faster on the Pi)


def load_emojis() -> dict:
    """Load every PNG in the emojis folder (name -> image), keeping only ones with a transparent background."""
    emojis = {}
    for path in sorted(glob.glob(os.path.join(EMOJI_DIR, "*.png"))):
        name = os.path.splitext(os.path.basename(path))[0]
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)   # UNCHANGED keeps the alpha channel
        if img is None or img.ndim != 3 or img.shape[2] != 4:
            print(f"[emoji] Skipping {name} (needs a transparent background)")
            continue
        # Shrink big PNGs once here, so we aren't shrinking them every single frame later.
        emojis[name] = cv2.resize(img, (EMOJI_PIXELS, EMOJI_PIXELS), interpolation=cv2.INTER_AREA)
    return emojis


def rotation(roll: float, yaw: float, pitch: float) -> np.ndarray:
    """3D rotation for the emoji card (OpenCV camera axes: x right, y down, z away)."""
    r, y, p = (math.radians(a) for a in (roll, -yaw, pitch))
    rz = np.array([[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1]])
    ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    return rz @ ry @ rx


def draw_emoji(frame: np.ndarray, emoji: np.ndarray, face, opacity: float) -> None:
    """Turn the emoji like a card in 3D to match the head, move it onto the face, and blend it in.

    `face` comes from main.py (see the Face class there). Draws straight onto
    `frame` (no copy), which saves time on the Pi.
    """
    # 1. Find the middle of the head. Eyes and mouth sit flat on the face (the nose sticks
    #    out, so it slides sideways when you turn), then push back toward the center of the
    #    head based on how far it's turned.
    eye_mid = (np.array(face.left_eye) + np.array(face.right_eye)) / 2
    mouth_mid = (np.array(face.mouth_left) + np.array(face.mouth_right)) / 2
    angle = math.radians(face.roll)
    across = np.array([math.cos(angle), math.sin(angle)])   # along the eye line
    down = np.array([-math.sin(angle), math.cos(angle)])    # from forehead toward chin
    center = eye_mid + (mouth_mid - eye_mid) * CENTER_DOWN
    center -= across * face.size * HEAD_DEPTH * math.sin(math.radians(face.yaw))
    center -= down * face.size * HEAD_DEPTH * math.sin(math.radians(face.pitch))

    # 2. Turn the emoji's 4 corners in 3D (tilt, turn, nod), then project them back onto the
    #    screen. The near edge gets bigger and the far edge smaller, so nodding and turning
    #    look like real 3D instead of the emoji just getting squished.
    size = face.size * SIZE_SCALE
    half = size / 2
    corners = np.array([[-half, -half, 0], [half, -half, 0], [half, half, 0], [-half, half, 0]])
    corners = corners @ rotation(face.roll, face.yaw, face.pitch).T
    depth = size * PERSPECTIVE
    dst = center + corners[:, :2] * (depth / (depth + corners[:, 2:3]))

    # 3. Only work on a small box around the emoji instead of the whole frame.
    f_h, f_w = frame.shape[:2]
    x0, y0 = np.maximum(np.floor(dst.min(axis=0)).astype(int), 0)
    x1, y1 = np.minimum(np.ceil(dst.max(axis=0)).astype(int), (f_w, f_h))
    if x1 <= x0 or y1 <= y0:
        return

    # 4. Warp the emoji onto a transparent canvas the size of the small box
    e_h, e_w = emoji.shape[:2]
    src = np.float32([[0, 0], [e_w, 0], [e_w, e_h], [0, e_h]])
    M = cv2.getPerspectiveTransform(src, (dst - (x0, y0)).astype(np.float32))
    warped = cv2.warpPerspective(emoji, M, (x1 - x0, y1 - y0),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(0, 0, 0, 0))

    # 5. Alpha blend, scaled by main.py's opacity (fade-in / fade-out)
    roi = frame[y0:y1, x0:x1]
    alpha = warped[:, :, 3:4].astype(np.float32) * (opacity / 255.0)
    roi[:] = (roi * (1 - alpha) + warped[:, :, :3] * alpha).astype(np.uint8)


class EmojiFilter:
    name = "Emoji"

    def __init__(self):
        self.emojis = list(load_emojis().values())
        if not self.emojis:
            raise RuntimeError(f"No usable PNGs found in {EMOJI_DIR}")
        # face id -> [emoji index, time of next swap]
        # Each face has its own clock, so the 5 second swap doesn't get thrown off
        # when someone else walks in or out of the frame.
        self.assigned = {}

    def _pick(self, face_id: int, now: float) -> np.ndarray:
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
        draw_emoji(frame, self._pick(face.id, time.monotonic()), face, opacity)
