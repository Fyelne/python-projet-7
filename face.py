import os
import cv2
import mediapipe as mp
import math
import logging

from classes.recognizer import MPRecognizer
logger = logging.getLogger(__name__)

CAM_INDEX = int(os.getenv("CAMERA_INDEX", 0))

EMOTION_TO_WEATHER = {
    "Neutral": None,
    "Happy": "clear",
    "Sad": "rain",
    "Angry": "thunder",
}


class FaceRecognizer(MPRecognizer):
    def __init__(self):
        super().__init__()

        # =====================
        # MediaPipe setup
        # =====================
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.cap = cv2.VideoCapture(CAM_INDEX)

        # =====================
        # Runtime state
        # =====================
        self.running = False
        self.tracking_enabled = True
        self.last_emotion = "Neutral"

        # =====================
        # Smoothing (EMA)
        # =====================
        self.alpha = 0.3
        self.smile_width_s = None
        self.corner_lift_s = None
        self.brow_drop_s = None

    # =====================
    # Utility
    # =====================
    def smooth(self, prev, curr):
        if prev is None:
            return curr
        return self.alpha * curr + (1 - self.alpha) * prev

    def dist(self, p1, p2):
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    # =====================
    # Core logic
    # =====================
    def recognize(self, landmarks, w, h):
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
        pts = [(int(l.x * w), int(l.y * h)) for l in landmarks]

        left_mouth, right_mouth = pts[61], pts[291]
        top_mouth, bottom_mouth = pts[13], pts[14]
        mouth_center = pts[13]

        left_eyebrow = pts[65]
        left_eye = pts[159]

        left_eye_corner = pts[33]
        right_eye_corner = pts[263]

        mouth_width = self.dist(left_mouth, right_mouth)
        mouth_height = self.dist(top_mouth, bottom_mouth)
        eye_width = self.dist(left_eye_corner, right_eye_corner)

        eyebrow_eye = self.dist(left_eyebrow, left_eye)

        smile_width = mouth_width / eye_width
        mouth_open = mouth_height / mouth_width

        avg_corner_y = (left_mouth[1] + right_mouth[1]) / 2
        corner_lift = mouth_center[1] - avg_corner_y
        brow_drop = eyebrow_eye / eye_width

        self.smile_width_s = self.smooth(self.smile_width_s, smile_width)
        self.corner_lift_s = self.smooth(self.corner_lift_s, corner_lift)
        self.brow_drop_s = self.smooth(self.brow_drop_s, brow_drop)

        if self.smile_width_s > 0.48 and self.corner_lift_s > 2:
            self.last_emotion = "Happy"
        elif self.corner_lift_s < -1 and mouth_open < 0.6:
            self.last_emotion = "Sad"
        elif self.brow_drop_s < 0.15 and self.smile_width_s < 0.6:
            self.last_emotion = "Angry"
        else:
            self.last_emotion = "Neutral"

        return self.last_emotion

    def execute(self):
        self.control.execute(self.last_emotion)


    # =====================
    # Main loop
    # =====================
    def loop(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            self.quit()
            return
        if key == ord("p"):
            self.tracking_enabled = False
        if key == ord("s"):
            self.tracking_enabled = True

        if self.tracking_enabled:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb)

            if results.multi_face_landmarks:
                h, w, _ = frame.shape
                for face_landmarks in results.multi_face_landmarks:
                    emotion = self.recognize(face_landmarks.landmark, w, h)
                    self.execute()

                    for lm in face_landmarks.landmark:
                        x, y = int(lm.x * w), int(lm.y * h)
                        cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

                    cv2.putText(
                        frame,
                        f"{emotion}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 0, 255),
                        2
                    )

        status = "Tracking: ON" if self.tracking_enabled else "Tracking: OFF"
        cv2.putText(
            frame,
            f"{status}  [S=start | P=pause | Q=quit]",
            (20, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.imshow("Emotion Detection", frame)

    # =====================
    # Lifecycle
    # =====================
    def start(self):
        logger.info("FaceRecognizer started")
        self.running = True
        while self.running:
            self.loop()

    def stop(self):
        logger.info("FaceRecognizer stopped")
        self.running = False

    def quit(self):
        self.stop()
        self.cap.release()
        cv2.destroyAllWindows()

fr = FaceRecognizer()
fr.start()