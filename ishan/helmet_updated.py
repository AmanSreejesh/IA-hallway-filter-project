"""Real-time procedural 3D astronaut helmet face filter with MediaPipe segmentation.

No helmet image is required. The helmet is a large, closed, wrap-around 3D mesh
that completely encompasses the user's head, using MediaPipe Selfie Segmentation
to remove underlying head pixels and replace them with an opaque helmet.

Install: pip install opencv-contrib-python mediapipe numpy
Press q to quit.
"""
import math
import os
import sys
import urllib.request

import cv2
import mediapipe as mp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "face_detection_yunet_2023mar.onnx")
MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)

# Camera / tracking settings
CAM_W, CAM_H = 640, 480
POSE_SMOOTHING = 0.82       # High damping for steady head replacement
MIN_EYE_DISTANCE = 8.0

# Five approximate 3D facial landmark locations corresponding to YuNet output.
FACE_MODEL_POINTS = np.array([
    [-31.5, -20.0,  0.0],
    [ 31.5, -20.0,  0.0],
    [  0.0,   0.0, -12.0],
    [-25.0,  25.0,  -2.0],
    [ 25.0,  25.0,  -2.0],
], dtype=np.float64)


def load_model():
    if os.path.isfile(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 100_000:
        return
    print("Downloading YuNet model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model downloaded.")


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
    for i, y in enumerate(np.linspace(-30, 44, 17)):
        for j, x in enumerate(np.linspace(-70, 70, 33)):
            z = -68.0 + 21.0 * (x / 70.0) ** 2 + 2.5 * ((y - 6.0) / 44.0) ** 2
            visor[i, j] = (x, y, z)
    front_tris.extend(generate_grid_tris(visor, (31, 34, 39), 1.0))

    for y0, y1 in [(-35.0, -30.0), (44.0, 50.0)]:
        rail = np.zeros((2, 33, 3), np.float32)
        for i, y in enumerate((y0, y1)):
            for j, x in enumerate(np.linspace(-74, 74, 33)):
                z = -68.0 + 21.0 * (x / 74.0) ** 2
                rail[i, j] = (x, y, z - 1.0)
        front_tris.extend(generate_grid_tris(rail, (185, 189, 193), 1.0))

    groups['front'] = front_tris

    return groups


def render_helmet(frame, mesh_groups, image_points, camera_matrix, dist_coeffs, previous_pose, segmenter):
    if np.linalg.norm(image_points[1] - image_points[0]) < MIN_EYE_DISTANCE:
        return frame, previous_pose

    ok, rvec, tvec = cv2.solvePnP(
        FACE_MODEL_POINTS, image_points.astype(np.float64),
        camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_EPNP
    )
    if not ok:
        return frame, previous_pose

    # Apply a baseline rotation offset to level the helmet perfectly at rest.
    # Adjust these values (radians) if it still leans slightly: [pitch, yaw, roll]
    resting_offset = np.array([0.0, 0.0, 0.0], dtype=np.float64).reshape(3, 1)
    rvec_mat, _ = cv2.Rodrigues(rvec)
    offset_mat, _ = cv2.Rodrigues(resting_offset)
    rvec, _ = cv2.Rodrigues(rvec_mat @ offset_mat)

    if previous_pose is not None:
        old_rvec, old_tvec = previous_pose
        rvec = (1.0 - POSE_SMOOTHING) * rvec + POSE_SMOOTHING * old_rvec
        tvec = (1.0 - POSE_SMOOTHING) * tvec + POSE_SMOOTHING * old_tvec

    h, w = frame.shape[:2]

    # MediaPipe segmentation mask
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = segmenter.process(rgb_frame)
    person_mask = (results.segmentation_mask > 0.5).astype(np.uint8) * 255 if results.segmentation_mask is not None else np.ones((h, w), dtype=np.uint8) * 255

    helmet_mask = np.zeros((h, w), dtype=np.uint8)

    # Render groups in strict layer order: inner -> shell -> front (visor)
    layer_order = ['inner', 'shell', 'front']
    for group_key in layer_order:
        tris = mesh_groups[group_key]
       
        flat_verts = []
        for t in tris:
            flat_verts.extend([t[0], t[1], t[2]])
        vertices = np.array(flat_verts, dtype=np.float32)
        colors = np.array([t[3] for t in tris], dtype=np.uint8)

        projected, _ = cv2.projectPoints(vertices, rvec, tvec, camera_matrix, dist_coeffs)
        points_2d = projected.reshape(-1, 3, 2)

        rotation, _ = cv2.Rodrigues(rvec)
        camera_vertices = (rotation @ vertices.T + tvec.reshape(3, 1)).T.reshape(-1, 3, 3)
        depths = camera_vertices[:, :, 2].mean(axis=1)
        order = np.argsort(depths)[::-1]  # back-to-front within group

        for idx in order:
            if depths[idx] <= 1:
                continue
            pts = points_2d[idx]
            if (pts[:, 0].max() < 0 or pts[:, 0].min() >= w or
                    pts[:, 1].max() < 0 or pts[:, 1].min() >= h):
                continue
            poly = np.rint(pts).astype(np.int32)
            color = tuple(int(v) for v in colors[idx])
            cv2.fillConvexPoly(frame, poly, color, lineType=cv2.LINE_AA)
            cv2.fillConvexPoly(helmet_mask, poly, 255, lineType=cv2.LINE_AA)

    # Glints on visor
    glints_3d = np.array([
        [-40.0, -3.0, -58.0], [30.0, -12.0, -60.0], [-8.0, 16.0, -62.0]
    ], dtype=np.float64)
    glints_2d, _ = cv2.projectPoints(glints_3d, rvec, tvec, camera_matrix, dist_coeffs)
    for p, axes in zip(glints_2d.reshape(-1, 2), [(5, 15), (4, 10), (3, 5)]):
        cv2.ellipse(frame, tuple(np.rint(p).astype(int)), axes, -12, 0, 360,
                    (125, 130, 136), -1, cv2.LINE_AA)

    return frame, (rvec, tvec)


def main():
    load_model()
    backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
    cap = cv2.VideoCapture(0, backend)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    detector = cv2.FaceDetectorYN.create(
        MODEL_PATH, "", (CAM_W, CAM_H), score_threshold=0.75,
        nms_threshold=0.3, top_k=5000
    )
   
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    segmenter = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

    mesh_groups = make_helmet_mesh_groups()
    previous_pose = None

    focal = float(CAM_W) * 1.15
    camera_matrix = np.array([
        [focal, 0, CAM_W / 2],
        [0, focal, CAM_H / 2],
        [0, 0, 1],
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    print("3D helmet running. Press q to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Error: Failed to grab webcam frame.")
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        if (w, h) != (CAM_W, CAM_H):
            detector.setInputSize((w, h))
            focal = float(w) * 1.15
            camera_matrix = np.array([
                [focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1]
            ], dtype=np.float64)

        _, faces = detector.detect(frame)
        if faces is not None and len(faces):
            face = max(faces, key=lambda f: f[2] * f[3])
            image_points = np.array([
                [face[4], face[5]], [face[6], face[7]], [face[8], face[9]],
                [face[10], face[11]], [face[12], face[13]],
            ], dtype=np.float64)
            frame, previous_pose = render_helmet(
                frame, mesh_groups, image_points, camera_matrix, dist_coeffs, previous_pose, segmenter
            )
        else:
            previous_pose = None

        cv2.imshow("3D Astronaut Helmet", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
