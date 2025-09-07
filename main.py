# improved_ai_whiteboard.py
import cv2
import mediapipe as mp
import numpy as np
import math
import time
import os

# ----- Mediapipe setup -----
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    model_complexity=1,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# ----- Video / canvas -----
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera")

canvas = None

# ----- Palette (BGR tuples) -----
PALETTE = [
    (0, 0, 255),     # Red
    (0, 255, 0),     # Green
    (255, 0, 0),     # Blue
    (0, 255, 255),   # Yellow
    (255, 255, 255), # White
    (0, 128, 255),   # Orange-ish
    (180, 0, 180),   # Purple-ish
    (0, 200, 200),   # Cyan-ish
    (50, 50, 200),   # Dark blue
    (20, 200, 20),   # Light green
    (0, 0, 0),       # Black
    (200, 200, 100)  # Tan-ish
]
PALETTE_COLS = 6
PALETTE_SW = 48
PALETTE_GAP = 10
PALETTE_LEFT = 12
PALETTE_TOP = 12

# ----- Drawing settings -----
pen_thickness = 8
ERASER_RADIUS = 36
alpha_smooth = 0.30   # smoothing for fingertip
dead_zone = 2.0

# ----- State -----
sx = sy = None        # smoothed client coords
px = py = None        # previous draw point
last_mode = "idle"
stable_mode = "idle"
mode_stability_frames = 0
REQUIRED_STABLE = 2   # require this many frames of same detected mode
current_color = PALETTE[3]  # default Yellow
hover_idx = None
hover_start = 0.0
HOVER_REQUIRE = 0.35   # seconds to hold over swatch to select

# recording
is_recording = False
video_writer = None
record_fps = 20.0

def fingers_up(hand_landmarks):
    """Return [thumb, index, middle, ring, pinky] as 1/0 (up/down)."""
    lm = hand_landmarks.landmark
    fingers = []
    # Thumb: compare x (works if image is mirrored)
    fingers.append(1 if lm[4].x < lm[3].x else 0)
    tip_ids = [8, 12, 16, 20]
    pip_ids = [6, 10, 14, 18]
    for tip, pip in zip(tip_ids, pip_ids):
        fingers.append(1 if lm[tip].y < lm[pip].y else 0)
    return fingers  # [thumb,index,middle,ring,pinky]

def palette_hit_test(x, y):
    """x,y are client coordinates (pixel) relative to frame (not normalized). Returns palette index or None."""
    ox = PALETTE_LEFT
    oy = PALETTE_TOP
    sw = PALETTE_SW
    gap = PALETTE_GAP
    cols = PALETTE_COLS
    rows = (len(PALETTE) + cols - 1) // cols
    for i in range(len(PALETTE)):
        r = i // cols
        c = i % cols
        left = ox + c * (sw + gap)
        top = oy + r * (sw + gap)
        right = left + sw
        bottom = top + sw
        if x >= left and x <= right and y >= top and y <= bottom:
            return i
    return None

def start_recording(w, h):
    global is_recording, video_writer
    ts = int(time.time())
    fname = f"whiteboard_RECORD_{ts}.mp4"
    # try mp4v first
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(fname, fourcc, record_fps, (w, h))
    if not video_writer.isOpened():
        # try XVID fallback
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_writer = cv2.VideoWriter(fname.replace('.mp4', '.avi'), fourcc, record_fps, (w, h))
    if not video_writer.isOpened():
        print("⚠️ Recording failed: no suitable codec / writer. Try installing codecs or use a different fourcc.")
        video_writer = None
        is_recording = False
    else:
        is_recording = True
        print("Recording started:", fname)

def stop_recording():
    global is_recording, video_writer
    if video_writer:
        video_writer.release()
        print("Recording saved.")
    video_writer = None
    is_recording = False

def take_snapshot(img):
    ts = int(time.time())
    fname = f"whiteboard_SNAP_{ts}.png"
    cv2.imwrite(fname, img)
    print("Snapshot saved:", fname)

print("Controls: r=record, s=snapshot, c=clear, +/- pen size, q=quit")
last_color_change_msg_t = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # mirror for natural interaction
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    if canvas is None:
        canvas = np.zeros_like(frame)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    mode = "idle"
    cx = cy = None

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        lm = hand_landmarks.landmark

        # compute fingertip (index tip) client coords
        rx = lm[8].x * w
        ry = lm[8].y * h

        # clamp
        rx = max(0.0, min(w - 1.0, rx))
        ry = max(0.0, min(h - 1.0, ry))

        # smoothing
        if sx is None or sy is None:
            sx, sy = rx, ry
        else:
            sx = alpha_smooth * rx + (1 - alpha_smooth) * sx
            sy = alpha_smooth * ry + (1 - alpha_smooth) * sy

        f = fingers_up(hand_landmarks)
        index_up = f[1] == 1
        middle_up = f[2] == 1
        pinky_up = f[4] == 1
        any_up = sum(f)

        # Gesture mapping requested:
        # - Index only -> draw
        # - Index + Middle -> erase
        # - Index + Pinky -> hover/select
        # - Else -> idle
        if index_up and pinky_up:
            mode = "hover"
        elif index_up and middle_up:
            mode = "erase"
        elif index_up and not middle_up and not pinky_up:
            mode = "draw"
        else:
            mode = "idle"

        # stability filter
        if mode == last_mode:
            mode_stability_frames += 1
        else:
            mode_stability_frames = 1
            last_mode = mode

        if mode_stability_frames >= REQUIRED_STABLE:
            stable_mode = mode

        cx, cy = int(sx), int(sy)

        # draw landmarks on preview
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    else:
        # no hand visible
        px = py = None
        cx = cy = None

    # ----- Actions -----
    if stable_mode == "draw" and cx is not None and cy is not None:
        if px is None or py is None:
            px, py = cx, cy
        if math.hypot(cx - px, cy - py) >= dead_zone:
            cv2.line(canvas, (px, py), (cx, cy), current_color, pen_thickness, lineType=cv2.LINE_AA)
            px, py = cx, cy

        # reset hover state
        hover_idx = None
        hover_start = 0

    elif stable_mode == "erase" and cx is not None and cy is not None:
        cv2.circle(canvas, (cx, cy), ERASER_RADIUS, (0, 0, 0), -1, lineType=cv2.LINE_AA)
        px = py = None
        hover_idx = None
        hover_start = 0

    elif stable_mode == "hover" and cx is not None and cy is not None:
        # show tiny cursor
        px = py = None
        idx = palette_hit_test(cx, cy)
        now = time.time()
        if idx is not None:
            if hover_idx == idx:
                # been hovering same square: check duration
                if now - hover_start >= HOVER_REQUIRE:
                    # select color
                    if current_color != PALETTE[idx]:
                        current_color = PALETTE[idx]
                        last_color_change_msg_t = now
                        print(f"Selected color index {idx}, color={current_color}")
                        # give a small visual flash (handled in UI drawing below)
                    # keep hover_idx so we don't keep re-selecting
            else:
                hover_idx = idx
                hover_start = now
        else:
            hover_idx = None
            hover_start = 0

    else:
        # idle or nothing
        px = py = None
        hover_idx = None
        hover_start = 0

    # ----- Compose output ----- 
    alpha_blend = 0.7
    # Make a copy of frame to draw UI on
    out = cv2.addWeighted(frame, 1.0 - alpha_blend, canvas, alpha_blend, 0)

    # Draw palette UI
    for i, col in enumerate(PALETTE):
        r = i // PALETTE_COLS
        c = i % PALETTE_COLS
        left = PALETTE_LEFT + c * (PALETTE_SW + PALETTE_GAP)
        top = PALETTE_TOP + r * (PALETTE_SW + PALETTE_GAP)
        right = left + PALETTE_SW
        bottom = top + PALETTE_SW
        cv2.rectangle(out, (left, top), (right, bottom), (180,180,180), 1)
        cv2.rectangle(out, (left+2, top+2), (right-2, bottom-2), col, -1)
        # highlight currently selected
        if col == current_color:
            cv2.rectangle(out, (left-3, top-3), (right+3, bottom+3), (0,255,0), 2)
        # highlight hovered
        if hover_idx == i:
            cv2.rectangle(out, (left-6, top-6), (right+6, bottom+6), (0,200,255), 2)

    # Draw cursor
    if cx is not None and cy is not None:
        # small circle showing fingertip
        cv2.circle(out, (cx, cy), 6, (0,0,0), -1)
        cv2.circle(out, (cx, cy), 4, current_color, -1)

    # Mode & status text
    cv2.putText(out, f"Mode: {stable_mode.upper()}", (10, h - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (240,240,240), 2, cv2.LINE_AA)
    cv2.putText(out, f"Pen: {pen_thickness}    Hold over swatch to select", (10, h - 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1, cv2.LINE_AA)

    # record indicator
    if is_recording:
        cv2.putText(out, "● REC", (w - 120, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2, cv2.LINE_AA)

    # small feedback text on color change
    if time.time() - last_color_change_msg_t < 1.2:
        cv2.putText(out, "Color changed", (w - 220, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)

    cv2.imshow("AI Whiteboard (hand-controlled)", out)

    # write recording if enabled
    if is_recording and video_writer:
        video_writer.write(out)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        if not is_recording:
            start_recording(w, h)
        else:
            stop_recording()
    elif key == ord('s'):
        take_snapshot(out)
    elif key == ord('c'):
        canvas[:] = 0
        print("Canvas cleared")
    elif key == ord('+') or key == ord('='):
        pen_thickness = min(80, pen_thickness + 2)
        print("Pen thickness:", pen_thickness)
    elif key == ord('-') or key == ord('_'):
        pen_thickness = max(2, pen_thickness - 2)
        print("Pen thickness:", pen_thickness)

# cleanup
if is_recording:
    stop_recording()
cap.release()
cv2.destroyAllWindows()
