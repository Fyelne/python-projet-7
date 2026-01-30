import cv2
import mediapipe as mp
import math
import time
from collections import deque

# Initialisation
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

COOLDOWN = 0.03

# Fonction pour calculer la distance entre deux points
def distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)


def recognize_body_gestures(mc_connect):
    """Détection des gestes de la main"""
    cap = cv2.VideoCapture(1)
    
    # Vérifier si la caméra s'est bien ouverte
    if not cap.isOpened():
        print("[!] Erreur: Impossible d'ouvrir la caméra")
        return

    # Historique des distances pour détecter le pliage/dépliage
    distance_history = deque(maxlen=10)
    smoothed_distance_history = deque(maxlen=10)
    last_move_time = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                for handLms in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
                    
                    # Coordonnées des doigts
                    index_tip = handLms.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    middle_tip = handLms.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                    wrist = handLms.landmark[mp_hands.HandLandmark.WRIST]
                    palm = handLms.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]  # Base du majeur

                    # Distances entre doigts et poignet
                    index_distance = distance((index_tip.x, index_tip.y), (wrist.x, wrist.y))
                    middle_distance = distance((middle_tip.x, middle_tip.y), (wrist.x, wrist.y))
                    avg_distance = (index_distance + middle_distance) / 2
                    
                    # Ajouter la distance à l'historique
                    distance_history.append(avg_distance)
                    
                    # Calculer la moyenne lissée des 5 dernières frames
                    if distance_history:
                        smoothed_distance = sum(list(distance_history)[-5:]) / len(list(distance_history)[-5:])
                        smoothed_distance_history.append(smoothed_distance)
                    else:
                        smoothed_distance = 0
                    
                    # Vérifier que le poignet est vers le haut (Y du poignet < Y de la base du majeur)
                    wrist_up = wrist.y > palm.y
                    
                    # Détecter MARCHE si variation reste élevée
                    is_walking = False
                    current_time = time.time()
                    variation = (max(list(distance_history)) if distance_history else 0) - (min(list(distance_history)) if distance_history else 0)
                    if len(distance_history) >= 5 and wrist_up:
                        if variation > 0.2:
                            print("[👣] Marche détectée")
                            is_walking = True
                            print(current_time - last_move_time)
                            if mc_connect and (current_time - last_move_time) > COOLDOWN:
                                try:
                                    mc_connect.move_forward()
                                except Exception as e:
                                    print(f"[!] Erreur lors de l'exécution de move_forward: {e}")
                            last_move_time = current_time
                    
                    
                    # Afficher les valeurs
                    h, w, _ = frame.shape
                    cv2.putText(frame, f"Index Dist: {index_distance:.3f}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    cv2.putText(frame, f"Middle Dist: {middle_distance:.3f}", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    cv2.putText(frame, f"Distance lissée: {smoothed_distance:.3f}", (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.putText(frame, f"Variation: {(max(list(distance_history)) if distance_history else 0) - (min(list(distance_history)) if distance_history else 0):.3f}", (50, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    
                    # Afficher le statut du poignet
                    wrist_status = "POIGNET HAUT" if wrist_up else "POIGNET BAS"
                    wrist_color = (0, 255, 0) if wrist_up else (0, 0, 255)
                    cv2.putText(frame, wrist_status, (50, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.8, wrist_color, 2)
                    
                    # Afficher le statut MARCHE/IMMOBILE
                    status = "MARCHE" if is_walking else "IMMOBILE"
                    color = (0, 255, 0) if is_walking else (0, 0, 255)
                    cv2.putText(frame, status, (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
            cv2.imshow("Webcam", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # Esc pour quitter
                break

    except KeyboardInterrupt:
        print("\n[✓] Interruption détectée - Fermeture...")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
        print("[✓] Ressources libérées")
