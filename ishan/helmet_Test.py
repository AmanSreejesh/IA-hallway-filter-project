"""
Live webcam filter: puts a helmet on every face using OpenCV's YuNet face detector.

Needs in the same folder as this script:
  - helmet.png (transparent PNG helmet)
  - the YuNet model file (downloaded automatically the first time)

Press 'q' to quit.
"""
global flag
flag=False
import math
import os
import sys
import urllib.request

import cv2
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    SCRIPT_DIR,
    "face_detection_yunet_2023mar.onnx"
)

MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)

HELMET_PATH = os.path.join(SCRIPT_DIR, "helmet.png")

# Tuning knobs
HELMET_SCALE = 3.8
FACE_Y_RATIO = 0.5
REFERENCE_DISTANCE=25
REFERENCE_EYE_WIDTH=41.7
VISIBLE_AT=25
INVISIBLE_AT=30

def load_model() -> None:
    """Download the YuNet model the first time the script runs."""
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 100000:
        return

    print("Downloading YuNet face model (one time only)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Done.")


def overlay_helmet(
    frame: np.ndarray,
    helmet: np.ndarray,
    left_eye: tuple,
    right_eye: tuple,
    nose: tuple,
    mouth_left: tuple,
    mouth_right: tuple,
) -> np.ndarray:
    """
    Scale, rotate, and place the helmet using all five facial landmarks.
    """
    
    # Head rotation from eye line
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle_deg = math.degrees(math.atan2(dy, dx))
    #if right_eye[1]-right_eye[0]<=6.2:
       # flag=True
    #else:
     #   flag=False
    # Face width estimate using eyes and mouth
    eye_width = abs(right_eye[0] - left_eye[0])
    mouth_width = abs(mouth_right[0] - mouth_left[0])
    face_width = max(eye_width, mouth_width)

    distance=REFERENCE_DISTANCE*REFERENCE_EYE_WIDTH/eye_width
    if distance <= VISIBLE_AT:
        opacity = 1.0
    elif distance >= INVISIBLE_AT:
        opacity = 0.0
    else:
        opacity = 1.0 - (
            (distance - VISIBLE_AT)
            / (INVISIBLE_AT - VISIBLE_AT)
        )
    # Average of all 5 landmarks
    center_x = (
        left_eye[0]
        + right_eye[0]
        + nose[0]
        + mouth_left[0]
        + mouth_right[0]
    ) / 5.0

    center_y = (
        left_eye[1]
        + right_eye[1]
        + nose[1]
        + mouth_left[1]
        + mouth_right[1]
    ) / 5.0

    h_img, w_img = helmet.shape[:2]

    scale = (face_width * HELMET_SCALE) / w_img

    # Point inside helmet where face should align
    anchor_x = w_img / 2
    anchor_y = h_img * FACE_Y_RATIO

    M = cv2.getRotationMatrix2D(
        (anchor_x, anchor_y),
        -angle_deg,
        scale
    )

    M[0, 2] += center_x - anchor_x
    M[1, 2] += center_y - anchor_y

    frame_h, frame_w = frame.shape[:2]

    warped = cv2.warpAffine(
        helmet,
        M,
        (frame_w, frame_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )

    alpha = warped[:, :, 3:4].astype(np.float32) / 255.0 * opacity

    blended = frame * (1 - alpha) + warped[:, :, :3] * alpha

    return blended.astype(np.uint8)


def main():
    #print(flag)
    load_model()

    WIDTH_SCALE = 0.5   # narrower
    HEIGHT_SCALE = 0.8  # taller

   

    helmet = cv2.imread(HELMET_PATH, cv2.IMREAD_UNCHANGED)
    helmet_adjusted = cv2.resize(
        helmet,
        (
            int(helmet.shape[1] * WIDTH_SCALE),
            int(helmet.shape[0] * HEIGHT_SCALE)
        ),
        interpolation=cv2.INTER_LINEAR
    )
    if helmet is None:
        print(f"Error: Could not load {HELMET_PATH}")
        return

    if helmet.shape[2] != 4:
        print("Error: helmet.png must have transparency (4 channels).")
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

    detector = cv2.FaceDetectorYN.create(
        MODEL_PATH,
        "",
        (640, 480),
        score_threshold=0.8,
    )

    print("Running. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to grab frame.")
            break

        frame = cv2.flip(frame, 1)

        h, w = frame.shape[:2]
        detector.setInputSize((w, h))

        _, faces = detector.detect(frame)

        if faces is not None:
            for face in faces:

                eye_a = (face[4], face[5])
                eye_b = (face[6], face[7])

                left_eye, right_eye = sorted(
                    [eye_a, eye_b],
                    key=lambda p: p[0]
                )

                nose = (face[8], face[9])
                mouth_left = (face[10], face[11])
                mouth_right = (face[12], face[13])

                frame = overlay_helmet(
                    frame,
                    helmet_adjusted,
                    left_eye,
                    right_eye,
                    nose,
                    mouth_left,
                    mouth_right,
                )

        cv2.imshow("Helmet Filter", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()