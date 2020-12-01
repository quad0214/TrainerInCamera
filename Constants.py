FACIAL_LANDMARK_DETECTOR = "./shape_predictor_68_face_landmarks.dat"
HAARCASCADE_EYE_DETECTOR = "./haarcascade_eye.xml"
RIGHT_EYE_INDEXES = [36, 37, 38, 39, 40, 41]
LEFT_EYE_INDEXES = [42, 43, 44, 45, 46, 47]

EYE_MARGIN = {'y': 7, 'x': 2}
EYE_THRESHOLD = 110

DISTANCE_PIVOT_INDEXES = [36, 45, 30, 48, 54, 8]
DISTANCE_PIVOT_POSES = [
    (-7, 10, 0),
    (7, 10, 0),
    (0, 0, 10),
    (-5, -5, 2),
    (5, -5, 2),
    (0, -10, 0)
]

CAMERA_MATRIX = [
    [641.790778, 0.0, 320.0],
    [0, 641.790778, 240.0],
    [0, 0, 1]
]

