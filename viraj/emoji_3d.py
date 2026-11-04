"""
3D emojis in the style of Microsoft's Fluent 3D emojis, built in code (no model files needed).

Each emoji is a ball with a warm yellow -> orange -> pink gradient, and its features (eyes,
mouths, glasses, tears) are drawn as 2D shapes laid out like Microsoft's artwork from the front,
then lifted onto the ball as rounded 3D surfaces with their own gradients. The designs follow
https://github.com/microsoft/fluentui-emoji (MIT license); these are look-alikes, not their files.

Two ways to draw them:
  - GLRenderer:       uses the graphics chip through OpenGL (moderngl). Smooth gradients and
                      lighting on every pixel. Needs:  pip install moderngl
  - SoftwareRenderer: no extra packages. The ball itself is worked out per pixel (smooth), and
                      the features are drawn as triangles on top (slightly flatter look).

make_renderer() tries OpenGL first and falls back to the software one automatically.

Model coordinates match OpenCV's camera: x = right, y = down, z = away from the camera.
The ball has radius 1 and faces the camera (-z). Feature shapes use the same x/y as a
front-on picture of the ball (the circle of radius 1).
"""
import math
import os
import sys

import cv2
import numpy as np

EXTENT = 1.15             # the rendered square covers -1.15..1.15 (ball is radius 1; room for the sweat drop)


def rgb(r, g, b):
    """Colors are written as RGB here (easier to compare with the artwork) and stored as BGR 0..1."""
    return np.array([b, g, r], dtype=np.float32) / 255


# --- Fluent palette ---
FACE_TOP, FACE_MID, FACE_BOTTOM = rgb(255, 200, 58), rgb(255, 156, 36), rgb(255, 104, 150)
DARK = rgb(64, 42, 50)
EYE_WHITE_TOP, EYE_WHITE_BOTTOM = rgb(255, 255, 255), rgb(228, 222, 230)
MOUTH_TOP, MOUTH_BOTTOM = rgb(150, 36, 196), rgb(255, 78, 168)
TEETH = rgb(250, 248, 252)
TEAR_TOP, TEAR_BOTTOM = rgb(96, 176, 255), rgb(70, 112, 255)
SWEAT_TOP, SWEAT_BOTTOM = rgb(118, 176, 255), rgb(52, 106, 240)
LENS_TOP, LENS_BOTTOM = rgb(72, 54, 140), rgb(44, 30, 92)
SHADES_FRAME = rgb(112, 86, 196)
SHINE_SPOT = rgb(214, 204, 250)

# --- Lighting (the same math is used by both renderers) ---
LIGHT = np.array([-0.5, -0.7, -0.8]) / np.linalg.norm([-0.5, -0.7, -0.8])   # from upper left, in front
HALF = (LIGHT + np.array([0, 0, -1.0])) / np.linalg.norm(LIGHT + np.array([0, 0, -1.0]))
WRAP = 0.5                # soft light that wraps around the ball (the Fluent look is matte, not plastic)
RIM = rgb(255, 120, 170)  # warm pink glow around the edge


def shade(color, n, gloss):
    """color (..., 3) lit with normal n (..., 3) in camera coordinates. gloss 0..1 adds a sharp shine."""
    d = np.clip((n @ LIGHT + WRAP) / (1 + WRAP), 0, 1)[..., None]
    h = np.clip(n @ HALF, 0, None)[..., None]
    rim = (1 - np.clip(-n[..., 2], 0, 1))[..., None] ** 3
    out = color * (0.72 + 0.4 * d) + 0.1 * h ** 10 + np.asarray(gloss)[..., None] * 0.45 * h ** 60
    return np.clip(out * (1 - 0.3 * rim) + RIM * 0.3 * rim, 0, 1)


def face_color(m):
    """The ball's gradient at model points m (..., 3): yellow top left -> orange -> pink bottom right."""
    g = np.clip(0.5 + 0.5 * (0.35 * m[..., 0] + 0.95 * m[..., 1]), 0, 1)[..., None]
    low = FACE_TOP + (FACE_MID - FACE_TOP) * np.clip(g / 0.55, 0, 1)
    high = FACE_MID + (FACE_BOTTOM - FACE_MID) * np.clip((g - 0.55) / 0.45, 0, 1) ** 1.4
    return np.where(g < 0.55, low, high)


# --- Laying 2D shapes onto the ball ---
def on_ball(x, y):
    """Front-view picture point (x, y) -> point on the ball's surface (also its outward direction)."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    r = np.hypot(x, y)
    k = np.where(r > 0.985, 0.985 / np.maximum(r, 1e-9), 1.0)   # keep shapes just inside the outline
    x, y = x * k, y * k
    return np.stack([x, y, -np.sqrt(np.maximum(1 - x * x - y * y, 0))], -1)


def ellipse(cx, cy, rx, ry, n=48):
    t = np.linspace(0, 2 * math.pi, n, endpoint=False)
    return np.stack([cx + rx * np.cos(t), cy + ry * np.sin(t)], -1)


def grin(top, half_width, depth, n=48):
    """A D-shaped open mouth: flat top, round bottom."""
    xs = np.linspace(-1, 1, n)
    bottom = np.stack([xs[::-1] * half_width, top + depth * np.sqrt(np.clip(1 - xs[::-1] ** 2, 0, 1))], -1)
    flat = np.stack([xs * half_width, np.full(n, top)], -1)
    return np.concatenate([flat, bottom[1:-1]])


def band(top, height, half_width, mouth_half_width, mouth_depth, n=24):
    """The strip of teeth along the flat top of a grin, trimmed to the mouth's curve."""
    xs = np.linspace(-1, 1, n) * half_width
    lower = top + np.minimum(height, mouth_depth * np.sqrt(np.clip(1 - (xs / mouth_half_width) ** 2, 0, 1)))
    return np.concatenate([np.stack([xs, np.full(n, top)], -1), np.stack([xs[::-1], lower[::-1]], -1)])


def lens(cx, cy, half_w, half_h, n=56):
    """A sunglasses lens: a rounded square that narrows a little toward the bottom."""
    t = np.linspace(0, 2 * math.pi, n, endpoint=False)
    c, s = np.cos(t), np.sin(t)
    x = np.sign(c) * np.abs(c) ** 0.6 * half_w
    y = np.sign(s) * np.abs(s) ** 0.6 * half_h
    x *= 1 - 0.18 * (y / half_h + 1) / 2
    return np.stack([cx + x, cy + y], -1)


def stream(x, y_top, y_bottom, width_top, width_bottom, n=24):
    """A tear running down the cheek, getting a little wider as it goes."""
    ys = np.linspace(y_top, y_bottom, n)
    w = np.linspace(width_top, width_bottom, n) / 2
    right = np.stack([x + w, ys], -1)                    # down the right side
    left = np.stack([x - w, ys], -1)[::-1]               # back up the left side
    a = np.linspace(math.pi, 0, 11)[1:-1]                # rounded top, left to right
    cap = np.stack([x + width_top / 2 * np.cos(a), y_top - width_top / 2 * np.sin(a)], -1)
    return np.concatenate([right, left, cap])


def arc(cx, cy, half_width, rise, n=24):
    """Points along a curve: positive rise bulges up (a closed happy eye), negative bulges down (a smile)."""
    xs = np.linspace(-1, 1, n)
    return np.stack([cx + xs * half_width, cy - rise * (1 - xs * xs)], -1)


class Builder:
    def __init__(self, detail=1.0):
        self.detail = detail
        self.parts = []   # (verts, normals, colors, gloss, faces, layer)

    def _add(self, verts, normals, colors, gloss, faces, layer):
        self.parts.append((verts, normals, colors, np.full(len(verts), gloss), faces, layer))

    def ball(self):
        rings, segments = int(48 * self.detail), int(64 * self.detail)
        lat, lon = np.meshgrid(np.linspace(0, math.pi, rings + 1), np.linspace(0, 2 * math.pi, segments + 1), indexing="ij")
        v = np.stack([np.sin(lat) * np.cos(lon), np.cos(lat), np.sin(lat) * np.sin(lon)], -1).reshape(-1, 3)
        self._add(v, v, face_color(v), 0.0, grid_faces(rings + 1, segments + 1), 0)

    def blob(self, outline, top, bottom, height=0.03, lift=0.006, gloss=0.0, layer=1):
        """A filled shape (2D outline) lifted onto the ball as a gentle dome, with a top-to-bottom gradient."""
        outline = np.asarray(outline)
        center = outline.mean(axis=0)
        rings = max(4, int(8 * self.detail))
        s = np.linspace(0, 1, rings + 1)[:, None, None]
        flat = center + (outline - center)[None] * s                       # rings from center to edge
        dome = height * (1 - s[..., 0] ** 2.5) + lift
        verts = on_ball(flat[..., 0], flat[..., 1]) * (1 + dome)[..., None]
        ys = flat[..., 1]
        t = ((ys - ys.min()) / max(ys.max() - ys.min(), 1e-6))[..., None]
        colors = top + (bottom - top) * t
        normals = grid_normals(verts, closed_columns=True)
        verts, normals, colors = verts.reshape(-1, 3), normals.reshape(-1, 3), colors.reshape(-1, 3)
        self._add(verts, normals, colors, gloss, grid_faces(rings + 1, len(outline), wrap=True), layer)

    def stroke(self, points, width, color, height=0.035, lift=0.004, layer=1, gloss=0.0):
        """A rounded line lying on the ball (eyebrows, closed eyes, smiles, glasses frames)."""
        pts = np.asarray(points, dtype=np.float64)
        steps = max(6, int(len(pts) * self.detail))
        t = np.linspace(0, len(pts) - 1, steps)
        pts = np.stack([np.interp(t, np.arange(len(pts)), pts[:, i]) for i in range(2)], -1)
        c = on_ball(pts[:, 0], pts[:, 1])
        tangent = np.gradient(c, axis=0)
        tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
        side = np.cross(tangent, c)
        side /= np.linalg.norm(side, axis=1, keepdims=True)
        phi = np.linspace(0, math.pi, max(5, int(9 * self.detail)))
        cos, sin = np.cos(phi)[None, :, None], np.sin(phi)[None, :, None]
        base = c * (1 + lift)
        verts = base[:, None] + side[:, None] * width / 2 * cos + c[:, None] * height * sin
        normals = side[:, None] * cos / (width / 2) + c[:, None] * sin / height
        normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
        cols = np.tile(color, (verts.shape[0], verts.shape[1], 1))
        self._add(verts.reshape(-1, 3), normals.reshape(-1, 3), cols.reshape(-1, 3), gloss,
                  grid_faces(verts.shape[0], verts.shape[1]), layer)
        for i in (0, -1):   # round ends
            self.cap(base[i], c[i], tangent[i], side[i], width / 2, height, color, layer, gloss)

    def cap(self, center, out, tangent, side, radius, height, color, layer, gloss):
        rings, segs = max(3, int(5 * self.detail)), max(6, int(10 * self.detail))
        a, b = np.meshgrid(np.linspace(0, math.pi / 2, rings + 1), np.linspace(0, 2 * math.pi, segs + 1), indexing="ij")
        local = np.stack([np.sin(a) * np.cos(b) * radius, np.sin(a) * np.sin(b) * radius, np.cos(a) * height], -1)
        frame = np.stack([side, tangent, out])
        verts = center + local @ frame
        normals = np.stack([np.sin(a) * np.cos(b) / radius, np.sin(a) * np.sin(b) / radius, np.cos(a) / height], -1) @ frame
        normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
        self._add(verts.reshape(-1, 3), normals.reshape(-1, 3), np.tile(color, (verts.size // 3, 1)), gloss,
                  grid_faces(rings + 1, segs + 1), layer)

    def drop(self, x, y, rx, ry, top, bottom, forward=0.06):
        """The sweat drop: a small glossy capsule standing upright just in front of the ball's edge."""
        out = on_ball(x, y) + np.array([0.0, 0.0, -forward])
        frame = np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]])   # right, up, toward the camera
        rings, segs = max(6, int(12 * self.detail)), max(8, int(16 * self.detail))
        a, b = np.meshgrid(np.linspace(0, math.pi, rings + 1), np.linspace(0, 2 * math.pi, segs + 1), indexing="ij")
        unit = np.stack([np.sin(a) * np.cos(b), np.cos(a), np.sin(a) * np.sin(b)], -1)
        radii = np.array([rx, ry, rx * 0.8])
        verts = out + (unit * radii) @ frame
        normals = (unit / radii) @ frame
        normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
        t = ((unit[..., 1] + 1) / 2)[..., None]
        colors = top + (bottom - top) * (1 - t)
        self._add(verts.reshape(-1, 3), normals.reshape(-1, 3), colors.reshape(-1, 3), 0.8,
                  grid_faces(rings + 1, segs + 1), 1)

    def build(self) -> dict:
        verts, normals, colors, gloss, faces, layers = [], [], [], [], [], []
        count = 0
        for v, n, c, g, f, layer in self.parts:
            verts.append(v); normals.append(n); colors.append(c); gloss.append(g)
            faces.append(f + count); layers.append(np.full(len(f), layer))
            count += len(v)
        m = {"verts": np.concatenate(verts).astype(np.float32), "normals": np.concatenate(normals).astype(np.float32),
             "colors": np.concatenate(colors).astype(np.float32), "gloss": np.concatenate(gloss).astype(np.float32),
             "faces": np.concatenate(faces).astype(np.int32), "layers": np.concatenate(layers)}
        f = m["faces"]
        m["face_colors"] = m["colors"][f].mean(axis=1)
        m["face_gloss"] = m["gloss"][f].mean(axis=1)
        fn = m["normals"][f].mean(axis=1)
        m["face_normals"] = fn / np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-6)
        m["face_centers"] = m["verts"][f].mean(axis=1)
        return m


def grid_faces(rows, cols, wrap=False):
    """Two triangles per cell of a rows x cols grid of vertices (wrap = join the last column to the first)."""
    idx = np.arange(rows * cols).reshape(rows, cols)
    right = np.roll(idx, -1, axis=1) if wrap else idx[:, 1:]
    left = idx if wrap else idx[:, :-1]
    a, b, c, d = left[:-1], left[1:], right[1:], right[:-1]
    return np.concatenate([np.stack([a, b, c], -1).reshape(-1, 3), np.stack([a, c, d], -1).reshape(-1, 3)])


def grid_normals(verts, closed_columns=False):
    """Normals for a grid of vertices, pointing away from the ball's center."""
    du = np.gradient(verts, axis=0)
    dv = (np.roll(verts, -1, axis=1) - np.roll(verts, 1, axis=1)) if closed_columns else np.gradient(verts, axis=1)
    n = np.cross(du, dv)
    center_ring = np.linalg.norm(n, axis=-1) < 1e-9
    n[center_ring] = verts[center_ring]                              # the very middle of a dome faces straight out
    n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-9)
    return np.where((n * verts).sum(-1, keepdims=True) < 0, -n, n)


# --- The 6 emojis ---
def build_models(detail=1.0) -> dict:
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):   # macOS numpy prints false warnings
        return _build(detail)


def _build(detail) -> dict:
    models = {}

    def new():
        b = Builder(detail)
        b.ball()
        return b

    b = new()   # Smiling face with sunglasses
    for side in (-1, 1):
        shape = lens(side * 0.47, -0.13, 0.43, 0.34)
        b.blob(shape, LENS_TOP, LENS_BOTTOM, height=0.07, lift=0.02, gloss=0.9)
        b.stroke(np.vstack([shape, shape[:1]]), 0.075, SHADES_FRAME, height=0.06, lift=0.03, layer=2)
        b.blob(ellipse(side * 0.47 + 0.2, -0.3, 0.08, 0.055), SHINE_SPOT, SHINE_SPOT, height=0.0, lift=0.095, layer=3)
        b.stroke([(side * 0.88, -0.28), (side * 0.98, -0.26)], 0.07, SHADES_FRAME, height=0.04, lift=0.02, layer=2)
    b.stroke([(-0.06, -0.38), (0, -0.4), (0.06, -0.38)], 0.08, SHADES_FRAME, height=0.05, lift=0.03, layer=2)
    b.stroke(arc(0, 0.22, 0.33, -0.13), 0.07, DARK)
    models["Sunglasses Emoji"] = b.build()

    b = new()   # Nerd face
    for side in (-1, 1):
        b.blob(ellipse(side * 0.39, -0.2, 0.25, 0.25), EYE_WHITE_TOP, EYE_WHITE_BOTTOM, height=0.03)
        b.blob(ellipse(side * 0.37, -0.17, 0.14, 0.14), DARK, DARK, height=0.025, lift=0.035, gloss=0.6, layer=2)
        ring = ellipse(side * 0.39, -0.2, 0.28, 0.28, n=40)
        b.stroke(np.vstack([ring, ring[:1]]), 0.055, DARK, height=0.04, lift=0.03, layer=2, gloss=0.4)
        b.stroke([(side * 0.67, -0.24), (side * 0.95, -0.24)], 0.05, DARK, height=0.035, lift=0.02, layer=2)
    b.stroke([(-0.11, -0.25), (0, -0.28), (0.11, -0.25)], 0.05, DARK, height=0.04, lift=0.03, layer=2)
    b.blob(grin(0.17, 0.46, 0.6), MOUTH_TOP, MOUTH_BOTTOM, height=0.02)
    b.blob([(-0.1, 0.17), (0.11, 0.17), (0.11, 0.28), (-0.1, 0.28)], TEETH, TEETH, height=0.012, lift=0.026, layer=2)
    models["Nerd Emoji With Glasses"] = b.build()

    b = new()   # Winking face
    b.blob(ellipse(-0.37, -0.15, 0.24, 0.24), EYE_WHITE_TOP, EYE_WHITE_BOTTOM, height=0.03)
    b.blob(ellipse(-0.34, -0.13, 0.15, 0.15), DARK, DARK, height=0.025, lift=0.035, gloss=0.6, layer=2)
    b.stroke(arc(0.37, -0.1, 0.2, 0.1), 0.085, DARK)
    for side in (-1, 1):
        b.stroke(arc(side * 0.42, -0.55, 0.17, 0.06), 0.065, DARK)
    b.blob(grin(0.12, 0.54, 0.64), MOUTH_TOP, MOUTH_BOTTOM, height=0.02)
    b.blob(band(0.12, 0.13, 0.47, 0.54, 0.64), TEETH, TEETH, height=0.012, lift=0.026, layer=2)
    models["Wink Emoji"] = b.build()

    b = new()   # Face with open mouth (OMG)
    for side in (-1, 1):
        b.blob(ellipse(side * 0.35, -0.28, 0.24, 0.26), EYE_WHITE_TOP, EYE_WHITE_BOTTOM, height=0.03)
        b.blob(ellipse(side * 0.31, -0.26, 0.13, 0.14), DARK, DARK, height=0.025, lift=0.035, gloss=0.6, layer=2)
    b.blob(ellipse(0, 0.5, 0.27, 0.36), MOUTH_TOP, MOUTH_BOTTOM, height=0.025)
    models["OMG Face Emoji"] = b.build()

    b = new()   # Grinning face with sweat
    for side in (-1, 1):
        b.stroke(arc(side * 0.35, -0.22, 0.2, 0.11), 0.085, DARK)
    b.blob(grin(0.05, 0.6, 0.72), MOUTH_TOP, MOUTH_BOTTOM, height=0.02)
    b.blob(band(0.05, 0.13, 0.53, 0.6, 0.72), TEETH, TEETH, height=0.012, lift=0.026, layer=2)
    b.drop(0.62, -0.55, 0.16, 0.27, SWEAT_TOP, SWEAT_BOTTOM)
    models["Smiling with Sweat Emoji"] = b.build()

    b = new()   # Loudly crying face
    for side in (-1, 1):
        b.blob(stream(side * 0.5, -0.38, 0.97, 0.28, 0.34), TEAR_TOP, TEAR_BOTTOM, height=0.03, lift=0.01, gloss=0.5)
        b.stroke(arc(side * 0.38, -0.47, 0.21, 0.08), 0.085, DARK, layer=2)
    b.blob(ellipse(0, 0.06, 0.24, 0.33), MOUTH_TOP, MOUTH_BOTTOM, height=0.025)
    models["Loudly Crying Face Emoji"] = b.build()
    return models


# --- Renderer 1: OpenGL ---
class GLRenderer:
    name = "OpenGL"
    MAX = 512

    VERTEX = """
        #version 330
        uniform mat3 R; uniform float depth; uniform float extent;
        in vec3 in_pos; in vec3 in_normal; in vec3 in_color; in float in_gloss;
        out vec3 v_normal; out vec3 v_color; out float v_gloss;
        void main() {
            vec3 p = R * in_pos;
            float s = depth / (depth + p.z);
            gl_Position = vec4(p.x * s / extent, -p.y * s / extent, p.z / 4.0, 1.0);
            v_normal = R * in_normal; v_color = in_color; v_gloss = in_gloss;
        }"""
    FRAGMENT = """
        #version 330
        uniform vec3 light; uniform vec3 halfway; uniform float wrap; uniform vec3 rim_color;
        in vec3 v_normal; in vec3 v_color; in float v_gloss;
        out vec4 color;
        void main() {
            vec3 n = normalize(v_normal);
            float d = clamp((dot(n, light) + wrap) / (1.0 + wrap), 0.0, 1.0);
            float h = max(dot(n, halfway), 0.0);
            float rim = pow(1.0 - clamp(-n.z, 0.0, 1.0), 3.0);
            vec3 c = v_color * (0.72 + 0.4 * d) + 0.1 * pow(h, 10.0) + v_gloss * 0.45 * pow(h, 60.0);
            color = vec4(clamp(c * (1.0 - 0.3 * rim) + rim_color * 0.3 * rim, 0.0, 1.0), 1.0);
        }"""

    def __init__(self, models: dict):
        import moderngl
        if sys.platform.startswith("linux"):
            # The Raspberry Pi's graphics driver handles OpenGL 3.3 fine but doesn't advertise it
            os.environ.setdefault("MESA_GL_VERSION_OVERRIDE", "3.3")
            os.environ.setdefault("MESA_GLSL_VERSION_OVERRIDE", "330")
            try:
                self.ctx = moderngl.create_standalone_context(backend="egl")   # no screen needed
            except Exception:
                self.ctx = moderngl.create_standalone_context()
        else:
            self.ctx = moderngl.create_standalone_context()
        self.prog = self.ctx.program(vertex_shader=self.VERTEX, fragment_shader=self.FRAGMENT)
        for key, value in (("light", tuple(LIGHT)), ("halfway", tuple(HALF)), ("wrap", WRAP),
                           ("rim_color", tuple(RIM)), ("extent", EXTENT)):
            self.prog[key].value = value
        self.names = sorted(models)
        self.vaos = {}
        for name, m in models.items():
            data = np.hstack([m["verts"], m["normals"], m["colors"], m["gloss"][:, None]]).astype("f4")
            self.vaos[name] = self.ctx.vertex_array(
                self.prog, [(self.ctx.buffer(data.tobytes()), "3f 3f 3f 1f", "in_pos", "in_normal", "in_color", "in_gloss")],
                index_buffer=self.ctx.buffer(m["faces"].astype("i4").tobytes()))
        size = (self.MAX, self.MAX)
        try:   # smooth (anti-aliased) edges if the graphics chip supports it
            self.draw_fbo = self.ctx.framebuffer(self.ctx.renderbuffer(size, 4, samples=4),
                                                 self.ctx.depth_renderbuffer(size, samples=4))
        except Exception:
            self.draw_fbo = self.ctx.framebuffer(self.ctx.renderbuffer(size, 4), self.ctx.depth_renderbuffer(size))
        self.read_fbo = self.ctx.framebuffer(self.ctx.renderbuffer(size, 4))

    def render(self, name: str, rotation: np.ndarray, size: int, perspective: float) -> np.ndarray:
        """The emoji turned by `rotation`, as a size x size BGRA image (colors pre-multiplied by alpha)."""
        size = int(min(max(size, 8), self.MAX))
        self.prog["R"].write(rotation.T.astype("f4").tobytes())
        self.prog["depth"].value = 2 * perspective
        self.draw_fbo.use()
        self.draw_fbo.viewport = (0, 0, size, size)
        self.draw_fbo.clear(0, 0, 0, 0)
        self.ctx.enable(self.ctx.DEPTH_TEST)
        self.vaos[name].render()
        self.ctx.copy_framebuffer(self.read_fbo, self.draw_fbo)
        data = self.read_fbo.read(viewport=(0, 0, size, size), components=4)
        return np.flipud(np.frombuffer(data, np.uint8).reshape(size, size, 4))


# --- Renderer 2: numpy + OpenCV (fallback) ---
class SoftwareRenderer:
    """The ball is worked out per pixel (a line from the camera through each pixel, where it hits
    the ball, and that spot's gradient color and lighting), so it's perfectly smooth. The features
    are then drawn as triangles on top, far ones first, at 2x size for smooth edges."""
    name = "software (OpenCV)"
    SUPERSAMPLE = 2
    LAYER_PULL = 0.25   # features count as this much closer per layer, so they always land on top

    def __init__(self, models: dict):
        self.models = models
        self.names = sorted(models)
        self.grids = {}

    def render(self, name, rotation, size, perspective):
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):   # see build_models
            return self._render(name, rotation, size, perspective)

    def _hits(self, n, depth):
        """For an n x n grid of pixels: where each pixel's line from the camera hits the ball,
        and how far inside the ball's outline it is (in pixels, for a soft edge)."""
        if n not in self.grids:   # pixel positions only depend on the size
            px = ((np.arange(n) + 0.5) / n * 2 - 1).astype(np.float32)
            self.grids[n] = np.meshgrid(px * EXTENT, px * EXTENT)
        x, y = self.grids[n]
        v = np.stack([x, y, np.full_like(x, depth)], -1)
        o = np.array([0.0, 0.0, -depth], np.float32)
        a = (v * v).sum(-1)
        b = 2 * (v @ o)
        root = np.sqrt(np.maximum(b * b - 4 * a * (o @ o - 1), 0))
        hit = o + v * ((-b - root) / (2 * a))[..., None]
        return hit, root / (2 * np.sqrt(a)) * n / (2 * EXTENT)

    def _ball(self, rotation, size, depth):
        # The ball's colors are a smooth gradient, so they're worked out at half size and scaled up
        # (4x less work); its outline is worked out at full size so the edge stays smooth.
        hit, _ = self._hits(max(size // 2, 8), depth)
        color = shade(face_color(hit @ rotation.astype(np.float32)), hit, 0.0).astype(np.float32)
        color = cv2.resize(color, (size, size), interpolation=cv2.INTER_LINEAR)
        _, inside = self._hits(size, depth)
        alpha = np.clip(inside, 0, 1)
        return color * alpha[..., None], alpha            # pre-multiplied

    def _render(self, name, rotation, size, perspective):
        size = int(min(max(size, 8), 512))
        m = self.models[name]
        depth = 2 * perspective
        ball, ball_alpha = self._ball(rotation, size, depth)

        # Features: only the ones on the near side of the ball, drawn far-to-near
        features = m["layers"] > 0
        centers = m["face_centers"] @ rotation.T
        visible = features & ((centers / np.linalg.norm(centers, axis=1, keepdims=True))[:, 2] < 0.1)
        faces = m["faces"][visible]
        p = m["verts"] @ rotation.T
        s = depth / (depth + p[:, 2])
        big = size * self.SUPERSAMPLE
        pix = ((p[:, :2] * s[:, None] / EXTENT * 0.5 + 0.5) * big * 4).astype(np.int32)   # x4: sub-pixel (shift=2)
        order = np.argsort(-(centers[visible, 2] - self.LAYER_PULL * m["layers"][visible]))
        faces = faces[order]
        normals = (m["face_normals"][visible] @ rotation.T)[order]
        colors = shade(m["face_colors"][visible][order], normals, 0.4 * m["face_gloss"][visible][order]) * 255
        colors = np.hstack([colors, np.full((len(colors), 1), 255.0)]).tolist()
        canvas = np.zeros((big, big, 4), np.uint8)
        fill = cv2.fillConvexPoly
        for tri, color in zip(pix[faces], colors):
            fill(canvas, tri, color, cv2.LINE_8, 2)
        feat = cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255

        fa = feat[..., 3:4]
        out = np.empty((size, size, 4), np.uint8)
        out[..., :3] = np.clip((feat[..., :3] + ball * (1 - fa)) * 255, 0, 255)
        out[..., 3] = np.clip((fa[..., 0] + ball_alpha * (1 - fa[..., 0])) * 255, 0, 255)
        return out


def make_renderer(prefer: str = "auto"):
    """OpenGL when it works, otherwise the software one. prefer = "auto", "gl" or "software"."""
    if prefer in ("auto", "gl"):
        try:
            renderer = GLRenderer(build_models(detail=1.0))
            renderer.render(renderer.names[0], np.eye(3), 32, 3.0)   # make sure it actually draws
            print(f"[emoji] 3D emojis drawn with {renderer.name}")
            return renderer
        except Exception as e:
            print(f"[emoji] OpenGL not available ({str(e).splitlines()[0] if str(e) else type(e).__name__}); "
                  f"using the software renderer instead")
    renderer = SoftwareRenderer(build_models(detail=0.4))   # fewer triangles keeps it quick on the Pi
    print(f"[emoji] 3D emojis drawn with {renderer.name}")
    return renderer
