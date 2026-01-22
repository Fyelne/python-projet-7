import time
import cv2
import numpy as np
import mediapipe as mp
from dollarpy import Recognizer, Template, Point

# ---------------- Réglages ----------------
CAM_INDEX = 0
W, H = 1280, 720

SAMPLE_MS = 20               # fréquence d'échantillonnage des points
MIN_POINTS = 30              # minimum pour reconnaître/valider un geste
PINCH_START_THRESH = 0.045   # distance normalisée (en fraction image) pour démarrer
PINCH_STOP_THRESH  = 0.060   # un peu plus grand (hystérésis) pour éviter le jitter

DRAW_THICKNESS = 6

# ------------------------------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(CAM_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)

canvas = np.zeros((H, W, 3), dtype=np.uint8)

templates = []
recognizer = Recognizer(templates)

recording = False
stroke_id = 1
points = []          # points du geste courant (dollarpy Points)
last_sample_t = 0.0

prev_xy = None       # pour dessiner une ligne sur le canvas
last_result = None   # (name, score) du dernier recognize


def clamp_xy(x, y):
    x = max(0, min(W - 1, x))
    y = max(0, min(H - 1, y))
    return x, y


def add_point(x, y):
    global points, prev_xy
    points.append(Point(x, y, stroke_id))
    if prev_xy is not None:
        cv2.line(canvas, prev_xy, (x, y), (255, 255, 255), DRAW_THICKNESS)
    prev_xy = (x, y)


print("Commandes: pinch pouce+index = start/stop | t=save template | c=clear | q=quit")

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_h, frame_w = frame.shape[:2]
        # si la caméra ne respecte pas W,H, on redimensionne le canvas à la taille réelle
        if frame_w != W or frame_h != H:
            W, H = frame_w, frame_h
            canvas = cv2.resize(canvas, (W, H), interpolation=cv2.INTER_NEAREST)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        pinch_dist = None
        index_tip_px = None
        thumb_tip_px = None

        if res.multi_hand_landmarks:
            hand = res.multi_hand_landmarks[0]
            lm = hand.landmark

            # Index tip = 8 ; Thumb tip = 4
            ix, iy = lm[8].x, lm[8].y
            tx, ty = lm[4].x, lm[4].y

            index_tip_px = clamp_xy(int(ix * W), int(iy * H))
            thumb_tip_px = clamp_xy(int(tx * W), int(ty * H))

            # distance normalisée (0..~1) en coordonnées image normalisées
            pinch_dist = ((ix - tx) ** 2 + (iy - ty) ** 2) ** 0.5

            # Dessiner les landmarks pour debug
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
            cv2.circle(frame, index_tip_px, 10, (0, 255, 0), -1)
            cv2.circle(frame, thumb_tip_px, 10, (0, 255, 255), -1)

            # Gestion start/stop avec hystérésis
            if not recording and pinch_dist is not None and pinch_dist < PINCH_START_THRESH:
                # START
                recording = True
                stroke_id = 1
                points = []
                prev_xy = None
                last_result = None
                last_sample_t = 0.0

            elif recording and pinch_dist is not None and pinch_dist > PINCH_STOP_THRESH:
                # STOP -> reconnaître
                recording = False
                prev_xy = None

                if len(points) >= MIN_POINTS and len(templates) > 0:
                    try:
                        last_result = recognizer.recognize(points)
                    except Exception as e:
                        last_result = ("error", 0.0)
                        print("Erreur recognize:", e)
                else:
                    # pas assez de points ou pas de templates
                    last_result = None

            # Enregistrement des points pendant recording
            if recording and index_tip_px is not None:
                now = time.time()
                if (now - last_sample_t) * 1000.0 >= SAMPLE_MS:
                    add_point(index_tip_px[0], index_tip_px[1])
                    last_sample_t = now

        # UI overlay
        overlay = cv2.addWeighted(frame, 1.0, canvas, 1.0, 0)

        status = "REC" if recording else "IDLE"
        cv2.putText(overlay, f"Status: {status} | points: {len(points)} | templates: {len(templates)}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        if pinch_dist is not None:
            cv2.putText(overlay, f"pinch_dist: {pinch_dist:.3f}",
                        (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        if last_result is not None:
            # souvent last_result = (name, score)
            try:
                name, score = last_result
                cv2.putText(overlay, f"Last: {name} ({score:.3f})",
                            (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            except Exception:
                cv2.putText(overlay, f"Last: {last_result}",
                            (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.putText(overlay, "Pinch start/stop | t=template | c=clear | q=quit",
                    (20, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow("Air Gesture Recognizer", overlay)
        cv2.imshow("Canvas", canvas)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        elif key == ord('c'):
            canvas[:] = 0
            points = []
            prev_xy = None
            last_result = None

        elif key == ord('t'):
            if len(points) < MIN_POINTS:
                print(f"⚠️ Pas assez de points ({len(points)}/{MIN_POINTS}). Dessine un peu plus longtemps.")
                continue
            name = input("Nom du template (ex: house, circle, arrow): ").strip()
            if not name:
                print("⚠️ Nom vide, annulé.")
                continue
            tmpl = Template(name, points.copy())
            templates.append(tmpl)
            recognizer = Recognizer(templates)
            print(f"✅ Template '{name}' ajouté ({len(points)} points).")
            # optionnel: reset canvas/points
            canvas[:] = 0
            points = []
            prev_xy = None
            last_result = None

cap.release()
cv2.destroyAllWindows()
