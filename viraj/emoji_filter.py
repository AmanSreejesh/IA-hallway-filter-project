"""
Emoji filter: puts a random emoji over a face.

This is an attachable module for common/main.py. It only draws. main.py decides
WHO gets a filter (opt-in zone, fade-in, timer, multi-user) and hands this filter
one face at a time, already smoothed, with its head angles worked out.

How it behaves:
  - The emoji follows head tilt (roll), gets thinner when you turn left/right (yaw),
    and gets shorter when you look up/down (pitch) -- a cheap "fake 3D" effect that
    runs fine on a Raspberry Pi.
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
SIZE_SCALE = 2.6       # emoji width = face size * SIZE_SCALE
SWAP_SECONDS = 5       # how long each person keeps an emoji
EMOJI_PIXELS = 160     # emojis get shrunk to this size when loaded (smaller = faster on the Pi)
MIN_SQUISH = 0.35      # never squish thinner than this, even when the head is turned a lot


def load_emojis() -> list:
    """Load every PNG in the emojis folder, keeping only the ones with a transparent background."""
    emojis = []
    for path in sorted(glob.glob(os.path.join(EMOJI_DIR, "*.png"))):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)   # UNCHANGED keeps the alpha channel
        if img is None or img.ndim != 3 or img.shape[2] != 4:
            print(f"[emoji] Skipping {os.path.basename(path)} (needs a transparent background)")
            continue
        # Shrink big PNGs once here, so we aren't shrinking them every single frame later.
        img = cv2.resize(img, (EMOJI_PIXELS, EMOJI_PIXELS), interpolation=cv2.INTER_AREA)
        emojis.append(img)
    return emojis


class EmojiFilter:
    name = "Emoji"

    def __init__(self):
        self.emojis = load_emojis()
        if not self.emojis:
            raise RuntimeError(f"No usable PNGs found in {EMOJI_DIR}")
        # face id -> [emoji index, time of next swap]
        # Each face has its own clock, so the 5 second swap no longer gets thrown off
        # when someone else walks in or out of the frame (old bug: emojis were handed
        # out by left-to-right order, so any change in the crowd reshuffled everyone).
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
        """Scale, rotate, squish, and move the emoji onto the face, then blend it in.

        `face` comes from main.py (see the Face class there). Draws straight onto
        `frame` (no copy), which saves time on the Pi.
        """
        emoji = self._pick(face.id, time.monotonic())
        nose = face.nose

        # 1. Size comes from main.py's smoothed face size (not the raw eye distance,
        #    which jumped around every frame and shrank whenever you turned your head).
        e_h, e_w = emoji.shape[:2]
        size = face.size * SIZE_SCALE

        # 2. Fake 3D: turning left/right makes it thinner, looking up/down makes it shorter.
        squish_x = max(math.cos(math.radians(face.yaw)), MIN_SQUISH)
        squish_y = max(math.cos(math.radians(face.pitch)), MIN_SQUISH)
        sx = size * squish_x / e_w
        sy = size * squish_y / e_h

        # 3. One transform that squishes, scales, and rotates (roll) the emoji.
        angle = math.radians(face.roll)
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        M = np.array([[cos_a * sx, -sin_a * sy, 0.0],
                      [sin_a * sx,  cos_a * sy, 0.0]], dtype=np.float32)

        # 4. Only work on a small box around the face instead of the whole frame.
        f_h, f_w = frame.shape[:2]
        half = int(size * 0.75) + 1
        x0, y0 = max(int(nose[0]) - half, 0), max(int(nose[1]) - half, 0)
        x1, y1 = min(int(nose[0]) + half, f_w), min(int(nose[1]) + half, f_h)
        if x1 <= x0 or y1 <= y0:
            return

        # 5. Slide the emoji's center onto the nose (in the small box's coordinates)
        cx, cy = e_w / 2, e_h / 2
        M[0, 2] = (nose[0] - x0) - (M[0, 0] * cx + M[0, 1] * cy)
        M[1, 2] = (nose[1] - y0) - (M[1, 0] * cx + M[1, 1] * cy)

        # 6. Warp the emoji onto a transparent canvas the size of the small box
        warped = cv2.warpAffine(emoji, M, (x1 - x0, y1 - y0),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(0, 0, 0, 0))

        # 7. Alpha blend, scaled by main.py's opacity (fade-in / fade-out)
        roi = frame[y0:y1, x0:x1]
        alpha = warped[:, :, 3:4].astype(np.float32) / 255.0 * opacity
        roi[:] = (roi * (1 - alpha) + warped[:, :, :3] * alpha).astype(np.uint8)
