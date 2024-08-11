import numpy as np
from scipy.spatial.distance import pdist

def std2d(x, y):
    return np.sqrt(np.var(x) + np.var(y))

def gini_coefficient_2d(points):
    # Calculate pairwise distances between points
    pairwise_distances = pdist(points, 'euclidean')
    
    # Sort distances
    sorted_distances = np.sort(pairwise_distances)
    
    n = len(sorted_distances)
    cumulative_sum = np.cumsum(sorted_distances)
    
    # Gini coefficient calculation
    gini = (2 / n) * np.sum((np.arange(1, n+1) * sorted_distances)) / cumulative_sum[-1] - (n + 1) / n
    
    return gini
    #return max(0, gini) 

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

def final_ear(lmks):
    leftEye = np.array([lmks[0][36:42],
                        lmks[1][36:42]]).T
    rightEye = np.array([lmks[0][42:48],
                        lmks[1][42:48]]).T

    leftEAR, left_, _ = eye_aspect_ratio(leftEye)
    rightEAR, _, right_ = eye_aspect_ratio(rightEye)
    ear = (leftEAR + rightEAR) / 2.0
    return (ear, leftEye, rightEye, left_, right_)


class Eye:
    def __init__(self, drowsy_nmax=20):
        self.EAR = 0
        self.EARs = []
        self.drowsy_val = 0
        self.drowsy_nmax = drowsy_nmax

    def update_EAR(self, lmks):
        """
        keep last N EAR and check drowsiness
        """
        self.EAR = float(final_ear(lmks)[0])
        # print("EYE", self.EAR)
        self.EARs.append(self.EAR)
        while len(self.EARs) > self.drowsy_nmax:
            self.EARs.pop(0)

        # print("len(self.EARs)", len(self.EARs))
        if len(self.EARs) > 5:
            self.check_eye_drowsy()

    def check_eye_drowsy(self):
        """
        EARs -> drowsy[T/F]
        """
        # TODO | OverflowError: cannot convert float infinity to integer
        # try:
        self.drowsy_val = (1 - (min(20, int(20 * np.mean(self.EARs) / 0.4)) / 20))*6
    
    def reset_EAR(self):
        self.EAR_min = 1
        self.EAR_max = 0
        self.EARs = []
        self.drowsy_val = 0