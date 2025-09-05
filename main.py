import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)

cap = cv2.VideoCapture(0)
canvas = None
prev_x, prev_y = None, None  # To store last fingertip position

# Pink color (BGR format)
PINK = (255, 0, 255)

try:
    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if canvas is None:
            canvas = np.zeros_like(frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                h, w, c = frame.shape
                x1 = int(hand_landmarks.landmark[8].x * w)
                y1 = int(hand_landmarks.landmark[8].y * h)
                x2 = int(hand_landmarks.landmark[12].x * w)
                y2 = int(hand_landmarks.landmark[12].y * h)

                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                index_up = hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y
                middle_up = hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y

                if index_up and not middle_up:
                    # Drawing mode → smooth pink line
                    if prev_x is not None and prev_y is not None:
                        cv2.line(canvas, (prev_x, prev_y), (x1, y1), PINK, 4)
                    prev_x, prev_y = x1, y1
                else:
                    prev_x, prev_y = None, None

        # Opaque drawing (no transparency)
        combined = cv2.add(frame, canvas)

        cv2.imshow("AI Whiteboard", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # 'q' or ESC
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
