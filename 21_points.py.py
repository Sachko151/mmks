import numpy as np

def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def extract_features(landmarks):

    wrist = landmarks[0]
    thumb = landmarks[4]
    index = landmarks[8]
    middle = landmarks[12]
    ring = landmarks[16]
    pinky = landmarks[20]

    features = [

        distance(wrist,index),
        distance(wrist,middle),
        distance(wrist,ring),
        distance(wrist,pinky),

        distance(thumb,index),
        distance(index,middle),
        distance(middle,ring),
        distance(ring,pinky)

    ]

    return np.array(features)