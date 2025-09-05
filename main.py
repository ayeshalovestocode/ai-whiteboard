import cv2
import mediapipe as mp
import numpy as np
import math

# ----- Mediapipe setup -----
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    model_complexity=1,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ----- Video / canvas -----
cap = cv2.VideoCapture(0)
canvas = None

# ----- Drawing settings -----
YELLOW = (0, 255, 255)     # Sharp yellow
PEN_THICK = 8
ERASER_RADIUS = 30

# Smoothing / stability
alpha = 0.30
sx, sy = None, None
px, py = None, None
dead_zone = 2.0

last_mode = "idle"
stable_mode = "idle"
mode_stability_frames = 0
REQUIRED_STABLE = 2

def fingers_up(hand_landmarks):
    """Return [thumb, index, middle, ring, pinky] as 1/0 (up/down)."""
    lm = hand_landmarks.landmark
    fingers = []

    # Thumb (x comparison, works for mirrored webcam)
    fingers.append(1 if lm[4].x < lm[3].x else 0)

    tip_ids = [8, 12, 16, 20]
    pip_ids = [6, 10, 14, 18]
    for tip, pip in zip(tip_ids, pip_ids):
        fingers.append(1 if lm[tip].y < lm[pip].y else 0)

    return fingers

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    if canvas is None:
        canvas = np.zeros_like(frame)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    mode = "idle"
    cx, cy = None, None

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        lm = hand_landmarks.landmark

        rx = int(lm[8].x * w)
        ry = int(lm[8].y * h)

        # Clamp so it doesn’t cut off at bottom
        rx = max(0, min(w - 1, rx))
        ry = max(0, min(h - 1, ry))

        if sx is None or sy is None:
            sx, sy = rx, ry
        else:
            sx = int(alpha * rx + (1 - alpha) * sx)
            sy = int(alpha * ry + (1 - alpha) * sy)

        f = fingers_up(hand_landmarks)
        index_up = f[1] == 1
        middle_up = f[2] == 1
        pinky_up = f[4] == 1  # baby finger

        # Pinky finger visible = STOP (no matter what else is up)
        if pinky_up:
            mode = "stop"
        elif index_up and not middle_up:
            mode = "draw"
        elif index_up and middle_up:
            mode = "erase"
        else:
            mode = "idle"

        # Stability check
        if mode == last_mode:
            mode_stability_frames += 1
        else:
            mode_stability_frames = 1
            last_mode = mode

        if mode_stability_frames >= REQUIRED_STABLE:
            stable_mode = mode

        cx, cy = sx, sy
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # ----- Actions -----
    if stable_mode == "draw" and cx is not None and cy is not None:
        if px is None or py is None:
            px, py = cx, cy
        if math.hypot(cx - px, cy - py) >= dead_zone:
            cv2.line(canvas, (px, py), (cx, cy), YELLOW, PEN_THICK, lineType=cv2.LINE_AA)
            px, py = cx, cy
    elif stable_mode == "erase" and cx is not None and cy is not None:
        cv2.circle(canvas, (cx, cy), ERASER_RADIUS, (0, 0, 0), -1, lineType=cv2.LINE_AA)
        px, py = None, None
    else:
        px, py = None, None  # Reset when idle or stop

    # ----- Output -----
    out = cv2.addWeighted(frame, 0.7, canvas, 0.3, 0)

    # Show current mode
    if stable_mode == "stop":
        cv2.putText(out, "STOP ✋ (PINKY)", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)
    else:
        cv2.putText(out, f"Mode: {stable_mode.upper()}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("AI Whiteboard", out)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
