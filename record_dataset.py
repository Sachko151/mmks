import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

label = input("Gesture label (example: fist, palm, thumb): ")

data = []

while True:

    ret, frame = cap.read()
    frame = cv2.flip(frame,1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)

    if result.multi_hand_landmarks:

        for hand_landmarks in result.multi_hand_landmarks:

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = []

            for lm in hand_landmarks.landmark:
                landmarks.append(lm.x)
                landmarks.append(lm.y)

            landmarks.append(label)

            data.append(landmarks)

    cv2.imshow("Recording dataset", frame)

    key = cv2.waitKey(1)

    if key == ord("s"):
        print("Sample saved")

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()

df = pd.DataFrame(data)

df.to_csv("gesture_dataset.csv", mode='a', header=False, index=False)

print("Dataset saved")