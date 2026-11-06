"""Ishan's procedural 3D helmet, attached to the shared hallway filter system.

The shared tracker supplies each person's current scale, pose and nose anchor.
No PNG, separate camera, pose smoothing or segmentation model is needed here.
The curved visor stays see-through so visitors can see their own faces.
Run: python common/main.py helmet (or python ishan/helmet_updated.py).
"""
import math
from pathlib import Path
import sys

import cv2
import numpy as np

VISOR_ALPHA = 0.08
# Mesh units use a 63 mm eye gap, matching the prototype's face model.
MODEL_EYE_GAP = 63.0
NOSE_ANCHOR = np.array([0.0, 0.0, -12.0])


def generate_grid_tris(grid, color, alpha=1.0):
    """Generate a list of triangles from a surface grid."""
    tris = []
    rows, cols, _ = grid.shape
    for r in range(rows - 1):
        for c in range(cols - 1):
            a, b = grid[r, c], grid[r, c + 1]
            d, e = grid[r + 1, c], grid[r + 1, c + 1]
            tris.append((a.copy(), b.copy(), d.copy(), color, alpha))
            tris.append((b.copy(), e.copy(), d.copy(), color, alpha))
    return tris


def make_helmet_mesh_groups():
    """Create categorized mesh groups so we can render them in strict layer order."""
    groups = {}

    # 1. Inner Core (Background blocker)
    inner_tris = []
    n_theta, n_phi = 15, 36
    inner_core = np.zeros((n_theta, n_phi, 3), np.float32)
    for i, theta in enumerate(np.linspace(0, math.pi, n_theta)):
        for j, phi in enumerate(np.linspace(0, 2 * math.pi, n_phi)):
            inner_core[i, j] = (
                55.0 * math.sin(theta) * math.sin(phi),
                10.0 + 60.0 * math.cos(theta),
                45.0 * math.sin(theta) * math.cos(phi)
            )
    inner_tris.extend(generate_grid_tris(inner_core, (40, 42, 45), 1.0))
    groups['inner'] = inner_tris

    # 2. Outer Shell (Dome & Sides)
    shell_tris = []
    dome = np.zeros((15, 48, 3), np.float32)
    for i, theta in enumerate(np.linspace(0.035, 1.95, 15)):
        for j, phi in enumerate(np.linspace(0, 2 * math.pi, 48)):
            x = 84.0 * math.sin(theta) * math.sin(phi)
            y = -18.0 - 102.0 * math.cos(theta)
            z = 70.0 * math.sin(theta) * math.cos(phi)
            dome[i, j] = (x, y, z)
    shell_tris.extend(generate_grid_tris(dome, (220, 222, 224), 1.0))

    side = np.zeros((13, 48, 3), np.float32)
    for i, y in enumerate(np.linspace(-32, 54, 13)):
        for j, phi in enumerate(np.linspace(0, 2 * math.pi, 48)):
            side[i, j] = (82.0 * math.sin(phi), y, 68.0 * math.cos(phi))

    side_tris = []
    for i in range(12):
        for j in range(47):
            phi = (j + 0.5) * 2 * math.pi / 47
            x_mid = 82.0 * math.sin(phi)
            z_mid = 68.0 * math.cos(phi)
            if z_mid < -8 and abs(x_mid) < 70:
                continue
            a, b = side[i, j], side[i, j + 1]
            d, e = side[i + 1, j], side[i + 1, j + 1]
            side_tris.append((a.copy(), b.copy(), d.copy(), (205, 208, 211), 1.0))
            side_tris.append((b.copy(), e.copy(), d.copy(), (205, 208, 211), 1.0))
    shell_tris.extend(side_tris)

    # Collar rings & hardware
    for y0, y1, rx, rz, color in [
        (51, 58, 76, 60, (204, 207, 210)),
        (58, 73, 73, 57, (226, 227, 229)),
        (73, 85, 75, 59, (157, 161, 165)),
    ]:
        ring = np.zeros((2, 49, 3), np.float32)
        for i, y in enumerate((y0, y1)):
            for j, phi in enumerate(np.linspace(0, 2 * math.pi, 49)):
                ring[i, j] = (rx * math.sin(phi), y, rz * math.cos(phi))
        shell_tris.extend(generate_grid_tris(ring, color, 1.0))

    for side_sign in (-1.0, 1.0):
        center = np.array([side_sign * 82.0, 2.0, 0.0])
        sphere = np.zeros((9, 17, 3), np.float32)
        for i, theta in enumerate(np.linspace(0, math.pi, 9)):
            for j, phi in enumerate(np.linspace(0, 2 * math.pi, 17)):
                sphere[i, j] = center + [
                    10.0 * math.sin(theta) * math.cos(phi),
                    18.0 * math.cos(theta),
                    16.0 * math.sin(theta) * math.sin(phi),
                ]
        shell_tris.extend(generate_grid_tris(sphere, (188, 192, 196), 1.0))

    groups['shell'] = shell_tris

    # 3. Visor & Rails (Always drawn last so they stay on top)
    front_tris = []
    visor = np.zeros((17, 33, 3), np.float32)
    for i, y in enumerate(np.linspace(-42, 48, 17)):
        for j, x in enumerate(np.linspace(-70, 70, 33)):
            z = -68.0 + 21.0 * (x / 70.0) ** 2 + 2.5 * ((y - 6.0) / 44.0) ** 2
            visor[i, j] = (x, y, z)
    groups['visor'] = generate_grid_tris(visor, (31, 34, 39), VISOR_ALPHA)

    for y0, y1 in [(-47.0, -42.0), (48.0, 54.0)]:
        rail = np.zeros((2, 33, 3), np.float32)
        for i, y in enumerate((y0, y1)):
            for j, x in enumerate(np.linspace(-74, 74, 33)):
                z = -68.0 + 21.0 * (x / 74.0) ** 2
                rail[i, j] = (x, y, z - 1.0)
        front_tris.extend(generate_grid_tris(rail, (185, 189, 193), 1.0))

    groups['front'] = front_tris

    return groups


def head_rotation(roll, yaw, pitch):
    """Current shared head angles -> mesh rotation in OpenCV camera coordinates."""
    r, y, p = map(math.radians, (roll, -yaw, pitch))
    cr, sr, cy, sy, cp, sp = math.cos(r), math.sin(r), math.cos(y), math.sin(y), math.cos(p), math.sin(p)
    rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    return rz @ ry @ rx


class HelmetFilter:
    name = "Helmet"
    tracker = "mediapipe"

    def __init__(self):
        # Compile the prototype's geometry once, rather than rebuilding thousands
        # of vertices, colors and triangles for every person on every frame.
        groups = make_helmet_mesh_groups()
        self.groups = {}
        for key, tris in groups.items():
            vertices = np.array([t[:3] for t in tris], dtype=np.float64)
            colors = [tuple(float(c) for c in (*t[3], 255)) for t in tris]
            self.groups[key] = (vertices, colors)

    def apply(self, frame, face, opacity):
        values = (*face.nose, face.size, face.roll, face.yaw, face.pitch, opacity)
        if not all(math.isfinite(v) for v in values) or face.size <= 0 or opacity <= 0:
            return
        h, w = frame.shape[:2]
        focal = float(max(w, h)) * 1.15
        origin = np.array([w / 2, h / 2])
        rotation = head_rotation(face.roll, face.yaw, face.pitch)
        # Anchor the model's nose exactly to the shared current-frame nose.
        depth = MODEL_EYE_GAP * focal / face.size
        nose = rotation @ NOSE_ANCHOR
        translation = np.r_[(np.asarray(face.nose)-origin) * depth / focal, depth] - nose
        projected = {}
        for key, (vertices, colors) in self.groups.items():
            camera = vertices @ rotation.T + translation
            valid = (camera[:, :, 2] > 1).all(axis=1)
            if not valid.any():
                continue
            camera = camera[valid]
            xy = camera[:, :, :2] / camera[:, :, 2:3] * focal + origin
            projected[key] = (xy, camera[:, :, 2].mean(axis=1), np.flatnonzero(valid))
        if 'visor' not in projected:
            return
        all_points = np.concatenate([data[0].reshape(-1, 2) for data in projected.values()])
        low = np.maximum(np.floor(all_points.min(axis=0)), 0).astype(int)
        high = np.minimum(np.ceil(all_points.max(axis=0))+1, (w, h)).astype(int)
        if (high <= low).any():
            return
        x0, y0 = low
        x1, y1 = high
        canvas = np.zeros((y1-y0, x1-x0, 4), np.uint8)

        def draw_group(key):
            if key not in projected:
                return
            xy, depths, indices = projected[key]
            polygons = np.rint((xy-low)*4).astype(np.int32)
            colors = self.groups[key][1]
            # Keep Ishan's layer order; sort back-to-front within each layer.
            for idx in np.argsort(-depths, kind='stable'):
                cv2.fillConvexPoly(canvas, polygons[idx], colors[indices[idx]], cv2.LINE_AA, shift=2)

        draw_group('inner')
        draw_group('shell')
        # Open the opaque shell/core at the curved visor, then lay a slight tint
        # over the visitor's actual face. A single hull avoids triangle seams.
        visor = np.rint((projected['visor'][0].reshape(-1, 2)-low)*4).astype(np.int32)
        window = np.zeros(canvas.shape[:2], np.uint8)
        cv2.fillConvexPoly(window, cv2.convexHull(visor), 255, cv2.LINE_AA, shift=2)
        alpha = window.astype(np.float32) / 255
        canvas[:] = np.rint(canvas * (1-alpha[:, :, None])).astype(np.uint8)
        tint = np.array([31, 34, 39, 255]) * VISOR_ALPHA
        canvas[:] = np.clip(canvas + alpha[:, :, None]*tint, 0, 255).astype(np.uint8)
        draw_group('front')

        glints = np.array([[-40, -3, -58], [30, -12, -60], [-8, 16, -62]], float)
        camera = glints @ rotation.T + translation
        for point, axes in zip(camera, ((5, 15), (4, 10), (3, 5))):
            if point[2] <= 1:
                continue
            center = point[:2] / point[2] * focal + origin-low
            radii = np.maximum(np.rint(np.array(axes) * focal / point[2]), 1).astype(int)
            glint = np.zeros(canvas.shape[:2], np.uint8)
            cv2.ellipse(glint, tuple(np.rint(center).astype(int)), tuple(radii),
                        face.roll-12, 0, 360, 255, -1, cv2.LINE_AA)
            a = (glint.astype(np.float32) / 255 * alpha * 0.18)[:, :, None]
            canvas[:] = np.clip(canvas * (1-a) + np.array([125, 130, 136, 255])*a, 0, 255).astype(np.uint8)

        # Every pixel, including visor reflections, obeys the shared distance fade.
        # Colors are premultiplied by alpha; allocations never exceed the frame.
        strength = float(np.clip(opacity, 0, 1))
        a = canvas[:, :, 3:4].astype(np.float32) * (strength / 255)
        roi = frame[y0:y1, x0:x1]
        roi[:] = np.clip(roi*(1-a) + canvas[:, :, :3]*strength, 0, 255).astype(np.uint8)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from common.main import main
    sys.exit(main(["helmet", *sys.argv[1:]]))
