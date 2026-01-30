import os
import cv2
import mediapipe as mp
import numpy as np
import math
import minecraft_link as ml
import logging

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

CAM_INDEX = int(os.getenv("CAMERA_INDEX", 0))

emotions_convertion = {
    "Neutral": None,
    "Happy": "clear",
    "Sad": "rain",
    "Angry": "thunder",
}

# =====================
# MediaPipe setup
# =====================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(CAM_INDEX)

# =====================
# Smoothing parameters
# =====================
ALPHA = 0.3  # EMA smoothing factor

smile_width_s = None
corner_lift_s = None
brow_drop_s = None
last_emotion = "Neutral"

# =====================
# Helper functions
# =====================
def smooth(prev, curr, alpha=0.3):
    if prev is None:
        return curr
    return alpha * curr + (1 - alpha) * prev

def dist(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

# =====================
# Emotion detection
# =====================
def get_emotion(landmarks, w, h):
    """
    Detects the emotion of a person based on their facial landmarks. 
    - Happy (user is smiling)
    - Angry (user is not smiling and brow is lowered)
    - Sad (user is smiling down)
    
    :param landmarks: Landmarks of the face
    :param w: frame width
    :param h: frame height

    :return: (smile_width, corner_lift, brow_drop)
    """
    global smile_width_s, corner_lift_s, brow_drop_s, last_emotion

    pts = [(int(l.x * w), int(l.y * h)) for l in landmarks]

    # === Landmarks ===
    left_mouth, right_mouth = pts[61], pts[291]
    top_mouth, bottom_mouth = pts[13], pts[14]
    mouth_center = pts[13]

    left_eyebrow = pts[65]
    left_eye = pts[159]

    left_eye_corner = pts[33]
    right_eye_corner = pts[263]

    # === Distances ===
    mouth_width = dist(left_mouth, right_mouth)
    mouth_height = dist(top_mouth, bottom_mouth)
    eye_width = dist(left_eye_corner, right_eye_corner)

    eyebrow_eye = dist(left_eyebrow, left_eye)

    # === Metrics (normalized) ===
    smile_width = mouth_width / eye_width
    mouth_open = mouth_height / mouth_width

    avg_corner_y = (left_mouth[1] + right_mouth[1]) / 2
    corner_lift = mouth_center[1] - avg_corner_y  # + = up, - = down

    brow_drop = eyebrow_eye / eye_width

    # === Smooth ===
    smile_width_s = smooth(smile_width_s, smile_width)
    corner_lift_s = smooth(corner_lift_s, corner_lift)
    brow_drop_s = smooth(brow_drop_s, brow_drop)

    # === Emotion logic (very natural) ===
    if smile_width_s > 0.48 and corner_lift_s > 2:
        last_emotion = "Happy"

    elif corner_lift_s < -1 and mouth_open < 0.6:
        last_emotion = "Sad"

    elif brow_drop_s < 0.15 and smile_width_s < 0.6:
        last_emotion = "Angry"

    else:
        last_emotion = "Neutral"

    ml.weather(emotions_convertion[last_emotion])

    return (
        f"{last_emotion} | "
        f"sw={smile_width_s:.2f} cl={corner_lift_s:.1f} bd={brow_drop_s:.3f}"
    )

# =====================
# Main loop
# =====================
def recognize_emotions():

    tracking_enabled = True
    paused_text = "Tracking: ON"
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27:
            break

        if key == ord('p'):
            tracking_enabled = False
            paused_text = "Tracking: OFF"

        if key == ord('s'):
            tracking_enabled = True
            paused_text = "Tracking: ON"

        if tracking_enabled:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if results.multi_face_landmarks:
                h, w, _ = frame.shape
                for face_landmarks in results.multi_face_landmarks:
                    emotion = get_emotion(face_landmarks.landmark, w, h)

                    for lm in face_landmarks.landmark:
                        x, y = int(lm.x * w), int(lm.y * h)
                        cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

                    cv2.putText(frame, emotion, (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                (0, 0, 255), 2)

        # Status text (always visible)
        cv2.putText(frame, paused_text + "  [S=start | P=pause | Q=quit]",
                    (20, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2)

        cv2.imshow("Emotion Detection", frame)


    cap.release()
    cv2.destroyAllWindows()
