import cv2
import mediapipe as mp
import numpy as np



print(mp.__version__)


mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)

cap = cv2.VideoCapture(0)


def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def get_features(landmarks):
    wrist = landmarks[0]
    thumb = landmarks[4]
    index = landmarks[8]
    middle = landmarks[12]
    ring = landmarks[16]
    pinky = landmarks[20]

    features = []

    features.append(distance(wrist, thumb))
    features.append(distance(wrist, index))
    features.append(distance(wrist, middle))
    features.append(distance(wrist, ring))
    features.append(distance(wrist, pinky))

    features.append(distance(thumb, index))
    features.append(distance(index, middle))
    features.append(distance(middle, ring))
    features.append(distance(ring, pinky))

    return np.array(features)


def classify_gesture(landmarks):

    index_tip = landmarks[8]
    index_pip = landmarks[6]

    middle_tip = landmarks[12]
    middle_pip = landmarks[10]

    ring_tip = landmarks[16]
    ring_pip = landmarks[14]

    pinky_tip = landmarks[20]
    pinky_pip = landmarks[18]

    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]

    fingers = []

    fingers.append(index_tip[1] < index_pip[1])
    fingers.append(middle_tip[1] < middle_pip[1])
    fingers.append(ring_tip[1] < ring_pip[1])
    fingers.append(pinky_tip[1] < pinky_pip[1])

    if all(fingers):
        return "OPEN PALM"

    if not any(fingers):
        return "FIST"

    if fingers[0] and not fingers[1]:
        return "POINT"

    if thumb_tip[0] > thumb_ip[0]:
        return "THUMB UP"

    return "UNKNOWN"


while True:

    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)

    if result.multi_hand_landmarks:

        for hand_landmarks in result.multi_hand_landmarks:

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = []

            for lm in hand_landmarks.landmark:
                h, w, c = frame.shape
                landmarks.append((int(lm.x * w), int(lm.y * h)))

            gesture = classify_gesture(landmarks)

            cv2.putText(frame,
                        gesture,
                        (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2)

    cv2.imshow("Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()