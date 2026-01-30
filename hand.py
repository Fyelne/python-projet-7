import time
import cv2
import numpy as np
import mediapipe as mp
from dollarpy import Recognizer, Template, Point
from collections import deque
import minecraft_link as ml

# ---------------- Réglages ----------------
CAM_INDEX = 0
#W, H = 1280, 720
#W, H = 360, 240
W, H = 640, 480  # Résolution augmentée pour meilleure détection


SAMPLE_MS = 8                # fréquence d'échantillonnage des points (réduit pour mouvements rapides)
MIN_POINTS = 30              # minimum pour reconnaître/valider un geste
PINCH_START_THRESH = 0.045   # distance normalisée (en fraction image) pour démarrer
PINCH_STOP_THRESH  = 0.060   # un peu plus grand (hystérésis) pour éviter le jitter

DRAW_THICKNESS = 6

# ---------------- Paramètres de lissage et stabilité ----------------
SMOOTHING_BUFFER_SIZE = 3       # Réduit pour plus de réactivité aux mouvements rapides
LOST_FRAMES_TOLERANCE = 8       # Nombre de frames sans détection avant d'arrêter le dessin
POSITION_SMOOTHING_ALPHA = 0.6  # Plus réactif (0.0=très lisse, 1.0=pas de lissage)
MIN_CONFIDENCE_THRESHOLD = 0.5  # Confiance minimale des landmarks pour les utiliser
MAX_POINT_DISTANCE = 50         # Distance max entre 2 points avant interpolation

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
last_build_igloo_t = 0.0  # timing pour build_igloo toutes les 30s

# ---------------- Variables pour le lissage et la stabilité ----------------
position_history = deque(maxlen=SMOOTHING_BUFFER_SIZE)  # Historique des positions
smoothed_position = None    # Position lissée courante
frames_without_detection = 0  # Compteur de frames sans détection
last_valid_pinch_dist = None  # Dernière distance de pinch valide


def clamp_xy(x, y):
    x = max(0, min(W - 1, x))
    y = max(0, min(H - 1, y))
    return x, y


def smooth_position(new_pos):
    """Applique un lissage exponentiel + moyenne mobile sur la position."""
    global smoothed_position, position_history
    
    if new_pos is None:
        return smoothed_position
    
    # Ajouter à l'historique
    position_history.append(new_pos)
    
    # Moyenne mobile sur l'historique
    if len(position_history) > 0:
        avg_x = sum(p[0] for p in position_history) / len(position_history)
        avg_y = sum(p[1] for p in position_history) / len(position_history)
        avg_pos = (int(avg_x), int(avg_y))
    else:
        avg_pos = new_pos
    
    # Lissage exponentiel
    if smoothed_position is None:
        smoothed_position = avg_pos
    else:
        smoothed_x = int(POSITION_SMOOTHING_ALPHA * avg_pos[0] + (1 - POSITION_SMOOTHING_ALPHA) * smoothed_position[0])
        smoothed_y = int(POSITION_SMOOTHING_ALPHA * avg_pos[1] + (1 - POSITION_SMOOTHING_ALPHA) * smoothed_position[1])
        smoothed_position = (smoothed_x, smoothed_y)
    
    return smoothed_position


def reset_smoothing():
    """Réinitialise les variables de lissage."""
    global smoothed_position, position_history, frames_without_detection
    smoothed_position = None
    position_history.clear()
    frames_without_detection = 0


def add_point(x, y):
    """Ajoute un point avec interpolation si le mouvement est trop rapide."""
    global points, prev_xy
    
    if prev_xy is not None:
        # Calculer la distance avec le point précédent
        dx = x - prev_xy[0]
        dy = y - prev_xy[1]
        distance = (dx**2 + dy**2) ** 0.5
        
        # Si la distance est trop grande, interpoler des points intermédiaires
        if distance > MAX_POINT_DISTANCE:
            num_intermediate = int(distance / MAX_POINT_DISTANCE)
            for i in range(1, num_intermediate + 1):
                t = i / (num_intermediate + 1)
                inter_x = int(prev_xy[0] + t * dx)
                inter_y = int(prev_xy[1] + t * dy)
                points.append(Point(inter_x, inter_y, stroke_id))
                # Dessiner le segment intermédiaire
                if i == 1:
                    cv2.line(canvas, prev_xy, (inter_x, inter_y), (255, 255, 255), DRAW_THICKNESS)
                else:
                    prev_inter_x = int(prev_xy[0] + (i-1) / (num_intermediate + 1) * dx)
                    prev_inter_y = int(prev_xy[1] + (i-1) / (num_intermediate + 1) * dy)
                    cv2.line(canvas, (prev_inter_x, prev_inter_y), (inter_x, inter_y), (255, 255, 255), DRAW_THICKNESS)
            
            # Dessiner le dernier segment vers le point final
            last_inter_x = int(prev_xy[0] + num_intermediate / (num_intermediate + 1) * dx)
            last_inter_y = int(prev_xy[1] + num_intermediate / (num_intermediate + 1) * dy)
            cv2.line(canvas, (last_inter_x, last_inter_y), (x, y), (255, 255, 255), DRAW_THICKNESS)
        else:
            # Distance normale, dessiner directement
            cv2.line(canvas, prev_xy, (x, y), (255, 255, 255), DRAW_THICKNESS)
    
    points.append(Point(x, y, stroke_id))
    prev_xy = (x, y)


def recognize_hand_gestures():
    """Fonction principale pour la reconnaissance de gestes avec la caméra"""
    global W, H, canvas, recording, stroke_id, points, prev_xy, last_result, last_sample_t, last_build_igloo_t
    global frames_without_detection, last_valid_pinch_dist
    
    print("Commandes: pinch pouce+index = start/stop | t=save template | c=clear | q=quit")

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.5,  # Réduit pour moins de pertes de détection
        min_tracking_confidence=0.5,   # Réduit pour meilleur suivi continu
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
            hand_detected = False

            if res.multi_hand_landmarks:
                hand = res.multi_hand_landmarks[0]
                lm = hand.landmark

                # Vérifier la confiance des landmarks (utiliser la visibilité si disponible)
                index_confidence = getattr(lm[8], 'visibility', 1.0) if hasattr(lm[8], 'visibility') else 1.0
                thumb_confidence = getattr(lm[4], 'visibility', 1.0) if hasattr(lm[4], 'visibility') else 1.0
                
                # Index tip = 8 ; Thumb tip = 4
                ix, iy = lm[8].x, lm[8].y
                tx, ty = lm[4].x, lm[4].y

                raw_index_pos = clamp_xy(int(ix * W), int(iy * H))
                thumb_tip_px = clamp_xy(int(tx * W), int(ty * H))
                
                # Appliquer le lissage à la position de l'index
                index_tip_px = smooth_position(raw_index_pos)

                # distance normalisée (0..~1) en coordonnées image normalisées
                pinch_dist = ((ix - tx) ** 2 + (iy - ty) ** 2) ** 0.5
                last_valid_pinch_dist = pinch_dist
                
                hand_detected = True
                frames_without_detection = 0

                # Dessiner les landmarks pour debug
                mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
                cv2.circle(frame, index_tip_px, 10, (0, 255, 0), -1)
                cv2.circle(frame, thumb_tip_px, 10, (0, 255, 255), -1)
                
            else:
                # Pas de main détectée - utiliser la dernière position lissée
                frames_without_detection += 1
                
                # Continuer avec la dernière position connue si on est en mode dessin
                if recording and smoothed_position is not None and frames_without_detection <= LOST_FRAMES_TOLERANCE:
                    index_tip_px = smoothed_position
                    hand_detected = True  # Simuler une détection pour continuer le dessin
                    # Afficher un indicateur visuel de "prédiction"
                    cv2.circle(frame, index_tip_px, 10, (0, 165, 255), 2)  # Orange pour position estimée

            # Gestion start/stop avec hystérésis
            if not recording and pinch_dist is not None and pinch_dist < PINCH_START_THRESH:
                # START
                recording = True
                stroke_id = 1
                points = []
                prev_xy = None
                last_result = None
                last_sample_t = 0.0
                reset_smoothing()

            elif recording:
                # Condition d'arrêt: pinch relâché OU trop de frames sans détection
                should_stop = (pinch_dist is not None and pinch_dist > PINCH_STOP_THRESH) or \
                              (frames_without_detection > LOST_FRAMES_TOLERANCE)
                
                if should_stop:
                    # STOP -> reconnaître
                    recording = False
                    prev_xy = None
                    reset_smoothing()

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
            detection_status = "OK" if frames_without_detection == 0 else f"LOST({frames_without_detection})"
            cv2.putText(overlay, f"Status: {status} | pts: {len(points)} | tpl: {len(templates)} | det: {detection_status}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            if pinch_dist is not None:
                now = time.time()
                if (now - last_build_igloo_t) >= 30.0:
                    ml.build_igloo()
                    last_build_igloo_t = now
                
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
