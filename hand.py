# hand.py
# Dessin dans le vide avec pinch + reconnaissance $1 (dollarpy ou fallback)
#
# Dépendances:
#   pip install opencv-python mediapipe numpy
#   pip install dollarpy   (optionnel, sinon fallback intégré)
#
# Usage:
#   python hand.py
#
# Touches:
#   q : quitter
#   c : clear canvas
#   r : re-run recognition sur le dernier tracé
#   1 : enregistrer le dernier tracé comme template "epee"
#   2 : enregistrer le dernier tracé comme template "pioche"
#   3 : enregistrer le dernier tracé comme template "maison"
#   s : sauvegarder les templates dans templates.json
#   l : charger templates.json
#
# Astuce:
# - Dessine en gardant le pinch, relâche -> reconnaissance automatique.

import json
import math
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import cv2

import mediapipe as mp

# ---------------------------
# $1 Recognizer (via dollarpy si dispo, sinon fallback)
# ---------------------------

DOLLARPY_AVAILABLE = False
try:
    from dollarpy import Recognizer as DollarPyRecognizer  # type: ignore
    DOLLARPY_AVAILABLE = True
except Exception:
    DOLLARPY_AVAILABLE = False


Point = Tuple[float, float]


def _dist(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _path_length(points: List[Point]) -> float:
    if len(points) < 2:
        return 0.0
    return sum(_dist(points[i - 1], points[i]) for i in range(1, len(points)))


def _centroid(points: List[Point]) -> Point:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def _rotate(points: List[Point], angle_rad: float) -> List[Point]:
    c = _centroid(points)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    out = []
    for x, y in points:
        dx, dy = x - c[0], y - c[1]
        rx = dx * cos_a - dy * sin_a + c[0]
        ry = dx * sin_a + dy * cos_a + c[1]
        out.append((rx, ry))
    return out


def _bounding_box(points: List[Point]) -> Tuple[float, float, float, float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (min(xs), min(ys), max(xs), max(ys))


def _scale_to_square(points: List[Point], size: float = 250.0) -> List[Point]:
    minx, miny, maxx, maxy = _bounding_box(points)
    w = maxx - minx
    h = maxy - miny
    w = w if w > 1e-6 else 1e-6
    h = h if h > 1e-6 else 1e-6
    out = [((p[0] - minx) * (size / w), (p[1] - miny) * (size / h)) for p in points]
    return out


def _translate_to_origin(points: List[Point]) -> List[Point]:
    c = _centroid(points)
    return [(p[0] - c[0], p[1] - c[1]) for p in points]


def _resample(points: List[Point], n: int = 64) -> List[Point]:
    if len(points) == 0:
        return []
    if len(points) == 1:
        return points * n

    I = _path_length(points) / (n - 1)
    if I <= 1e-6:
        return [points[0]] * n

    D = 0.0
    new_points = [points[0]]
    i = 1
    while i < len(points):
        d = _dist(points[i - 1], points[i])
        if (D + d) >= I:
            t = (I - D) / d if d > 1e-6 else 0.0
            nx = points[i - 1][0] + t * (points[i][0] - points[i - 1][0])
            ny = points[i - 1][1] + t * (points[i][1] - points[i - 1][1])
            new_points.append((nx, ny))
            points.insert(i, (nx, ny))
            D = 0.0
            i += 1
        else:
            D += d
            i += 1

    while len(new_points) < n:
        new_points.append(points[-1])
    return new_points[:n]


def _indicative_angle(points: List[Point]) -> float:
    c = _centroid(points)
    return math.atan2(c[1] - points[0][1], c[0] - points[0][0])


def _path_distance(a: List[Point], b: List[Point]) -> float:
    return sum(_dist(a[i], b[i]) for i in range(len(a))) / len(a)


def _normalize_for_dollar(points: List[Point], n: int = 64, size: float = 250.0) -> List[Point]:
    pts = list(points)
    pts = _resample(pts, n=n)
    angle = _indicative_angle(pts)
    pts = _rotate(pts, -angle)
    pts = _scale_to_square(pts, size=size)
    pts = _translate_to_origin(pts)
    return pts


@dataclass
class Template:
    name: str
    points: List[Point]


class DollarFallbackRecognizer:
    def __init__(self, n_points: int = 64, square_size: float = 250.0):
        self.n_points = n_points
        self.square_size = square_size
        self.templates: List[Template] = []

    def add_template(self, name: str, points: List[Point]) -> None:
        norm = _normalize_for_dollar(points, n=self.n_points, size=self.square_size)
        self.templates.append(Template(name=name, points=norm))

    def recognize(self, points: List[Point]) -> Tuple[str, float]:
        if len(points) < 10 or not self.templates:
            return ("unknown", 0.0)

        cand = _normalize_for_dollar(points, n=self.n_points, size=self.square_size)

        angles = np.linspace(-0.5, 0.5, 13)
        best_name = "unknown"
        best_dist = float("inf")

        for tmpl in self.templates:
            for a in angles:
                rot = _rotate(cand, a)
                d = _path_distance(rot, tmpl.points)
                if d < best_dist:
                    best_dist = d
                    best_name = tmpl.name

        diag = math.sqrt((self.square_size ** 2) * 2)
        score = max(0.0, 1.0 - (best_dist / (0.5 * diag)))
        return (best_name, float(score))


class GestureRecognizer:
    def __init__(self):
        self.use_dollarpy = DOLLARPY_AVAILABLE
        self.fallback = DollarFallbackRecognizer()
        self._templates: Dict[str, List[Point]] = {}

        # Pour stabilité, on force fallback (évite les variations d'API dollarpy)
        self.use_dollarpy = False

    def add_template(self, name: str, points: List[Point]) -> None:
        self._templates[name] = list(points)
        self.fallback.add_template(name, points)

    def recognize(self, points: List[Point]) -> Tuple[str, float]:
        return self.fallback.recognize(points)

    def to_json(self) -> dict:
        out = {}
        for name, pts in self._templates.items():
            out[name] = [{"x": float(p[0]), "y": float(p[1])} for p in pts]
        return out

    def from_json(self, data: dict) -> None:
        self._templates = {}
        self.fallback = DollarFallbackRecognizer()
        for name, pts in data.items():
            points = [(float(p["x"]), float(p["y"])) for p in pts]
            self.add_template(name, points)

    def save(self, path: str = "templates.json") -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_json(), f, ensure_ascii=False, indent=2)

    def load(self, path: str = "templates.json") -> bool:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.from_json(data)
            return True
        except Exception:
            return False


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def lm_to_xy(lm, w: int, h: int) -> Point:
    return (lm.x * w, lm.y * h)


def pinch_strength(thumb_tip: Point, index_tip: Point, scale_ref: float) -> float:
    d = _dist(thumb_tip, index_tip)
    close = 0.18 * scale_ref
    far = 0.45 * scale_ref
    if d <= close:
        return 1.0
    if d >= far:
        return 0.0
    return float(1.0 - (d - close) / (far - close))


def smooth_point(prev: Optional[Point], curr: Point, alpha: float = 0.35) -> Point:
    if prev is None:
        return curr
    return (prev[0] * (1 - alpha) + curr[0] * alpha, prev[1] * (1 - alpha) + curr[1] * alpha)


def default_templates() -> GestureRecognizer:
    gr = GestureRecognizer()

    # epee
    epee = [
        (0, 100), (50, 100), (100, 100),
        (50, 100), (50, 60), (50, 20), (50, -20), (50, -60),
        (50, -80), (55, -90), (50, -100), (45, -90), (50, -80),
        (50, 120), (50, 160)
    ]

    # pioche
    pioche = [
        (0, 160), (0, 120), (0, 80), (0, 40), (0, 0),
        (-60, 0), (-90, 10), (-110, 25), (-120, 45),
        (-90, 20), (-60, 0), (60, 0), (90, 10), (110, 25), (120, 45),
        (90, 20), (60, 0), (0, 0), (0, 40)
    ]

    # maison (rectangle + toit triangulaire, tracé continu)
    # On fait: bas gauche -> bas droit -> haut droit -> sommet toit -> haut gauche -> bas gauche
    maison = [
        (-80, 80), (80, 80), (80, -20),   # base + mur droit
        (0, -100),                         # sommet du toit
        (-80, -20),                        # mur gauche haut
        (-80, 80),                         # retour bas gauche
    ]

    gr.add_template("epee", epee)
    gr.add_template("pioche", pioche)
    gr.add_template("maison", maison)
    return gr


_last_recognized_label: str = "unknown"
_last_score: float = 0.0
_last_stroke: List[Point] = []


def recognize_last_drawing() -> str:
    return _last_recognized_label


def main():
    global _last_recognized_label, _last_score, _last_stroke

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Impossible d'ouvrir la caméra (index 0).")

    canvas = None
    gr = default_templates()

    drawing = False
    stroke: List[Point] = []
    prev_smooth: Optional[Point] = None
    last_tip: Optional[Point] = None

    PINCH_ON = 0.72
    PINCH_OFF = 0.35
    MIN_POINT_DIST = 3.0
    last_recognition_time = 0.0

    with mp_hands.Hands(
        static_image_mode=False,
        model_complexity=1,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    ) as hands:

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            if canvas is None:
                canvas = np.zeros_like(frame)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)

            pinch = 0.0
            tip_xy: Optional[Point] = None

            if res.multi_hand_landmarks:
                hand_lms = res.multi_hand_landmarks[0]

                thumb_tip = lm_to_xy(hand_lms.landmark[4], w, h)
                index_tip = lm_to_xy(hand_lms.landmark[8], w, h)

                index_mcp = lm_to_xy(hand_lms.landmark[5], w, h)
                pinky_mcp = lm_to_xy(hand_lms.landmark[17], w, h)
                scale_ref = max(30.0, _dist(index_mcp, pinky_mcp))

                pinch = pinch_strength(thumb_tip, index_tip, scale_ref)
                tip_xy = index_tip

                mp_drawing.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

                cv2.putText(frame, f"pinch: {pinch:.2f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

            if tip_xy is not None:
                sm = smooth_point(prev_smooth, tip_xy, alpha=0.35)
                prev_smooth = sm

                if (not drawing) and pinch >= PINCH_ON:
                    drawing = True
                    stroke = []
                    last_tip = None

                if drawing:
                    if pinch >= PINCH_OFF:
                        if last_tip is None or _dist(last_tip, sm) >= MIN_POINT_DIST:
                            stroke.append(sm)
                            if last_tip is not None:
                                cv2.line(canvas,
                                         (int(last_tip[0]), int(last_tip[1])),
                                         (int(sm[0]), int(sm[1])),
                                         (255, 255, 255), 4, cv2.LINE_AA)
                            last_tip = sm
                    else:
                        drawing = False
                        prev_smooth = None
                        last_tip = None

                        _last_stroke = stroke

                        now = time.time()
                        if len(stroke) >= 10 and (now - last_recognition_time) > 0.2:
                            label, score = gr.recognize(stroke)
                            _last_recognized_label = label if score >= 0.65 else "unknown"
                            _last_score = score
                            last_recognition_time = now

            cv2.putText(frame, f"recognized: {_last_recognized_label} ({_last_score:.2f})",
                        (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (255, 255, 255), 2, cv2.LINE_AA)

            overlay = cv2.addWeighted(frame, 1.0, canvas, 0.9, 0)
            cv2.imshow("AirDraw (pinch) + $1 recognize", overlay)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("c"):
                canvas[:] = 0
                _last_recognized_label = "unknown"
                _last_score = 0.0
                _last_stroke = []
            elif key == ord("r"):
                if len(_last_stroke) >= 10:
                    label, score = gr.recognize(_last_stroke)
                    _last_recognized_label = label if score >= 0.65 else "unknown"
                    _last_score = score
            elif key == ord("1"):
                if len(_last_stroke) >= 10:
                    gr.add_template("epee", _last_stroke)
                    print("Template 'epee' mis à jour avec le dernier tracé.")
            elif key == ord("2"):
                if len(_last_stroke) >= 10:
                    gr.add_template("pioche", _last_stroke)
                    print("Template 'pioche' mis à jour avec le dernier tracé.")
            elif key == ord("3"):
                if len(_last_stroke) >= 10:
                    gr.add_template("maison", _last_stroke)
                    print("Template 'maison' mis à jour avec le dernier tracé.")
            elif key == ord("s"):
                gr.save("templates.json")
                print("Templates sauvegardés dans templates.json")
            elif key == ord("l"):
                if gr.load("templates.json"):
                    print("Templates chargés depuis templates.json")
                else:
                    print("Impossible de charger templates.json (fichier absent ou invalide).")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
