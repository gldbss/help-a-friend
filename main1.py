import cv2
import mediapipe as mp
import pyautogui
import math

pyautogui.FAILSAFE = False
screen_w, screen_h = pyautogui.size()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Use built-in webcam (usually device index 0)
cap = cv2.VideoCapture(0)
click_threshold = 40
right_click_triggered = False
left_click_triggered = False

def get_pixel_distance(p1,p2,w,h):
    return math.hypot((p1.x-p2.x)*w,math.hypot(p1.y-p2.y)*h)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            index_finger = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            middle_finger = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

            screen_x = int(index_finger.x * screen_w)
            screen_y = int(index_finger.y * screen_h)
            pyautogui.moveTo(screen_x, screen_y)

            thumb_dist = get_pixel_distance(index_finger,thumb_tip,w,h)
            middle_dist = get_pixel_distance(index_finger,middle_finger,w,h)

            if thumb_dist < click_threshold and not left_click_triggered:
                pyautogui.click()
                left_click_triggered = True
                cv2.putText(frame, "Left Click", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            elif thumb_dist >= click_threshold:
                left_click_triggered = False

            if middle_dist < click_threshold and not right_click_triggered:
                pyautogui.click(button='right')
                right_click_triggered = True
                cv2.putText(frame, "Right Click", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            elif middle_dist >= click_threshold and right_click_triggered:
                right_click_triggered = False



    cv2.imshow("Gesture Control", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()