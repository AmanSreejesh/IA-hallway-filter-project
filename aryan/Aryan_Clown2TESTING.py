"""
Live webcam filter: puts glasses on every face using OpenCV's YuNet face detector.

Needs in the same folder as this script:
  - glasses.png  (a PNG with a transparent background; run make_glasses.py to make one)
  - the YuNet model file (downloaded automatically the first time)

Press 'q' to quit.
"""
import math
import os
import sys
import urllib.request

import cv2
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "face_detection_yunet_2023mar.onnx")
MODEL_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
             "face_detection_yunet/face_detection_yunet_2023mar.onnx")
NOSE_PATH = os.path.join(SCRIPT_DIR, "clown_nose.png")
NOSE_SCALE = 1.05   # size relative to eye distance

# Approximate calibration values
MAX_DISTANCE_FACE_WIDTH = 105   # px
MAX_FACE_WIDTH = 500            # px

SHOW_FILTER_ONLY_IN_RANGE = True

def face_passes_acceptance(face) -> bool:
    """
    Returns True only when the user is inside the allowed range
    and not excessively close to the camera.
    """

    face_width = float(face[2])

    # Outside 2.5-ft radius (face too small)
    if face_width < MAX_DISTANCE_FACE_WIDTH:
        return False

    # Face larger than allowed
    if face_width > MAX_FACE_WIDTH:
        return False

    return True

def facing_camera(face):
    left_eye_x = face[4]
    right_eye_x = face[6]
    nose_x = face[8]

    eye_mid = (left_eye_x + right_eye_x) / 2.0
    eye_dist = abs(right_eye_x - left_eye_x)

    offset_ratio = abs(nose_x - eye_mid) / eye_dist

    return offset_ratio < 0.35

def load_model() -> None:
    """Download the YuNet model the first time the script runs."""
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 100_000:
        return
    print("Downloading YuNet face model (one time only)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Done.")

def overlay_nose(frame: np.ndarray,
                 nose_png: np.ndarray,
                 nose_center: tuple,
                 face_width: float) -> np.ndarray:
    """
    Centers a clown nose PNG on the detected nose point.
    """
    size = int(face_width*0.475)

    resized = cv2.resize(
        nose_png,
        (size, size),
        interpolation=cv2.INTER_AREA
    )

    x = int(nose_center[0] - size / 2)
    y = int(nose_center[1] - size / 2 - 0.15*size)

    h, w = frame.shape[:2]

    if x >= w or y >= h or x + size <= 0 or y + size <= 0:
        return frame

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(w, x + size)
    y2 = min(h, y + size)

    nose_crop = resized[
        y1 - y:y2 - y,
        x1 - x:x2 - x
    ]

    alpha = nose_crop[:, :, 3:4].astype(np.float32) / 255.0

    roi = frame[y1:y2, x1:x2].astype(np.float32)

    frame[y1:y2, x1:x2] = (
        roi * (1 - alpha) +
        nose_crop[:, :, :3] * alpha
    ).astype(np.uint8)

    return frame

def main():
    load_model()

    nose_png = cv2.imread(NOSE_PATH, cv2.IMREAD_UNCHANGED)
    if nose_png is None:
        print(f"Error: Could not load {NOSE_PATH}")
        return
    if nose_png.shape[2] != 4:
        print("Error: clown_nose.png must have a transparent background.")
        return

    
    if sys.platform.startswith("win"):
        backend = cv2.CAP_DSHOW 
    else:
        backend = cv2.CAP_ANY
    cap = cv2.VideoCapture(0, backend)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    detector = cv2.FaceDetectorYN.create(MODEL_PATH, "", (640, 480),
                                         score_threshold=0.8)
    print("Running. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break
        frame = cv2.flip(frame, 1)   # mirror view, like a selfie camera

        h, w = frame.shape[:2]
        detector.setInputSize((w, h))
        _, faces = detector.detect(frame)

        # Each face row: [x, y, w, h, eye1_x, eye1_y, eye2_x, eye2_y,
        #                 nose_x, nose_y, mouth1_x, mouth1_y, mouth2_x, mouth2_y, score]
        if faces is not None:
            for face in faces:

                if face[2] < MAX_DISTANCE_FACE_WIDTH:
                    continue

                if not facing_camera(face):
                   continue

                if face[2] > MAX_FACE_WIDTH:
                    continue

                nose_center = (face[8], face[9])

                face_width = face[2]

                frame = overlay_nose(
                    frame,
                    nose_png,
                    nose_center,
                    face_width
                )

            cv2.imshow("Clown Nose Filter", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
