import numpy as np

def euclidean_dist(a, b):
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def eye_aspect_ratio(eye):
    # print("EYE =====>", eye)
    alpha = 1.5
    beta = 1.5
    A = euclidean_dist(eye[1], eye[5])
    B = euclidean_dist(eye[2], eye[4])
    C = euclidean_dist(eye[0], eye[3])

    ear = (alpha * A + beta * B) / (2.0 * C)

    left_ear = alpha * A / C
    right_ear = beta * B / C

    return (ear, left_ear, right_ear)

def final_ear(shape):
    leftEye = shape[36:42]
    rightEye = shape[42:48]

    leftEAR, left_, _ = eye_aspect_ratio(leftEye)
    rightEAR, _, right_ = eye_aspect_ratio(rightEye)
    ear = (leftEAR + rightEAR) / 2.0
    return (ear, leftEye, rightEye, left_, right_)


class Eye:
    def __init__(self, drowsy_nmax=10):
        self.EAR = 0
        self.EARs = []
        self.drowsy = False
        self.drowsy_nmax = drowsy_nmax

    def update_EAR(self, lmks):
        """
        keep last N EAR and check drowsiness
        """
        self.EAR = final_ear(lmks)[0]
        self.EARs.append(self.EAR)
        while len(self.EARs) > self.drowsy_nmax:
            self.EARs.pop(0)

        self.check_eye_drowsy()

    def check_eye_drowsy(self):
        """
        EARs -> drowsy[T/F]
        """
        if len(self.EARs) < self.drowsy_nmax:
            self.drowsy = False
        else:
            # self.EAR_thres * self.this_eye_size:
            if np.mean(self.EARs) < 0.3:
                self.drowsy = True
            else:
                self.drowsy = False

    def reset_EAR(self):
        self.EAR_min = 1
        self.EAR_max = 0
        self.EARs = []
        self.drowsy = False