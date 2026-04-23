import cv2
import pyautogui
import math
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

game_mode = False

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")

options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)

detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

screen_w, screen_h = pyautogui.size()

prev_x, prev_y = 0, 0
smoothening = 4

frame_margin = 100

connections = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
]

clicking = False
click_threshold = 40

right_clicking = False

sensitivity = 1.3

alpha = 0.2
threshold = 10

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    result = detector.detect(mp_image)

    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:

            for idx, lm in enumerate(hand_landmarks):
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

            for c in connections:
                x1, y1 = int(hand_landmarks[c[0]].x * w), int(
                    hand_landmarks[c[0]].y * h
                )
                x2, y2 = int(hand_landmarks[c[1]].x * w), int(
                    hand_landmarks[c[1]].y * h
                )
                cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            index = hand_landmarks[8]
            cx, cy = int(index.x * w), int(index.y * h)

            cv2.circle(frame, (cx, cy), 10, (255, 0, 255), -1)

            x_sens = 1.5
            y_sens = 2.2

            mapped_x = int(cx * screen_w / w * x_sens)
            mapped_y = int((h - cy) * screen_h / h * y_sens)

            mapped_x = max(0, min(screen_w, mapped_x))
            mapped_y = max(0, min(screen_h, mapped_y))

            dx = mapped_x - prev_x
            dy = mapped_y - prev_y

            dist = math.hypot(dx, dy)

            if dist < 12:
                curr_x, curr_y = prev_x, prev_y
            else:
                curr_x = alpha * mapped_x + (1 - alpha) * prev_x
                curr_y = alpha * mapped_y + (1 - alpha) * prev_y

            if game_mode:
                move_x = curr_x - prev_x
                move_y = curr_y - prev_y
                pyautogui.moveRel(move_x, move_y, duration=0)
            else:
                pyautogui.moveTo(curr_x, curr_y, duration=0.01)

            prev_x, prev_y = curr_x, curr_y

            thumb = hand_landmarks[4]

            tx, ty = int(thumb.x * w), int(thumb.y * h)
            ix, iy = cx, cy

            distance = ((tx - ix) ** 2 + (ty - iy) ** 2) ** 0.5

            wrist = hand_landmarks[0]
            middle_base = hand_landmarks[9]

            wx, wy = int(wrist.x * w), int(wrist.y * h)
            mx, my = int(middle_base.x * w), int(middle_base.y * h)

            hand_size = ((wx - mx) ** 2 + (wy - my) ** 2) ** 0.5
            if hand_size == 0:
                hand_size = 1

            ratio = distance / hand_size

            if ratio < 0.4 and not clicking:
                pyautogui.click()
                clicking = True

            if ratio > 0.5:
                clicking = False

            ring = hand_landmarks[16]
            pinky = hand_landmarks[20]

            rx, ry = int(ring.x * w), int(ring.y * h)
            px, py = int(pinky.x * w), int(pinky.y * h)

            cv2.line(frame, (rx, ry), (px, py), (255, 255, 0), 2)

            distance_rp = ((rx - px) ** 2 + (ry - py) ** 2) ** 0.5

            ratio_rp = distance_rp / hand_size

            if ratio_rp < 0.35 and not right_clicking:
                pyautogui.rightClick()
                right_clicking = True

            if ratio_rp > 0.5:
                right_clicking = False

    cv2.imshow("Air Mouse", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
