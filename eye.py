import numpy as np
from scipy.spatial.distance import pdist
import cv2

def std2d(x, y):
    return np.sqrt(np.var(x) + np.var(y))

def euclidean_dist(a, b):
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def eye_aspect_ratio(eye):
    # print("EYE =====>", eye)
    alpha = 1.7
    beta = 1.7
    A = euclidean_dist(eye[1], eye[5])
    B = euclidean_dist(eye[2], eye[4])
    C = euclidean_dist(eye[0], eye[3])
    if C==0 or C < A or C < B or np.isfinite(A)*np.isfinite(B)*np.isfinite(C) == False:
        return 0.4, 0.4, 0.4
    
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
    ear = max((leftEAR + rightEAR) / 2.0 - 0.15, 0.01)
    return (ear, leftEye, rightEye, left_, right_)


class Eye:
    def __init__(self, drowsy_nmax=60):
        self.EAR = 0
        self.EARs = []
        self.drowsy_val = 0
        self.drowsy_nmax = drowsy_nmax
        self.alpha = 0.3
        self._mean_ear = 0.5
        self._mean_arr = []
        self.mean_nmax = 60

    def reset(self):
        self.EAR = 0
        self.EARs = []
        self.drowsy_val = 0
        self._mean_ear = 0.5
        self._mean_arr = []

    def update_EAR(self, lmks):
        """
        keep last N EAR and check drowsiness
        """
        old_EAR = self.EAR
        self.EAR = self.alpha * float(final_ear(lmks)[0]) + (1 - self.alpha) * old_EAR
        
        if self.EAR > 0.2 and len(self._mean_arr) < self.mean_nmax:
            self._mean_arr.append(self.EAR)
        if len(self._mean_arr) == self.mean_nmax:
            self._mean_ear = np.percentile(self._mean_arr, 80)
        # print("SELF.MEAN", self._mean_ear)
        # print("EYE", self.EAR)
        self.EARs.append(self.EAR)
        while len(self.EARs) > self.drowsy_nmax:
            self.EARs.pop(0)
            
        # while len(self._mean_arr) > 30:
        #     self._mean_arr.pop(0)

        # print("len(self.EARs)", len(self.EARs))
        if len(self.EARs) > 5:
            self.check_eye_drowsy()

    def check_eye_drowsy(self):
        """
        EARs -> drowsy[T/F]
        """
        # TODO | OverflowError: cannot convert float infinity to integer
        # try:
        self.drowsy_val = (1 - (min(self.drowsy_nmax, int(self.drowsy_nmax * np.mean(self.EARs) / self._mean_ear)) / self.drowsy_nmax))*6
    
    def reset_EAR(self):
        self.EAR_min = 1
        self.EAR_max = 0
        self.EARs = []
        self.drowsy_val = 0
        
        
class Face():
    def __init__(self, 
                 n_initial=20,
                 fov_v=91,
                 image_width = 1024,
                 image_height = 1024):
        self._initial_guess_face_h = np.zeros(n_initial)
        self._initial_guess_face_w = np.zeros(n_initial)
        self._initial_count = 0
        self._guess_max = 15
        self.face_hr = None
        self.face_wr = None
        self.face_cnt = 0
        self.dist_face = 0
        # for 1920 x 1080
        self.camera_matrix = np.array([[1141.7483, 0., 981.12228],
                                        [0., 858.629385, 551.762145],
                                        [0., 0., 1.]])
        self.P = np.array([[379.9881*3, 0., 326.52974*3, 0.],
                           [0., 380.81802*2.25, 244.71673*2.25, 0.],
                           [0., 0., 1., 0.]])
        self.dist_coeffs = np.array([-0.330314, 0.130840, 0.000384, 0.000347, -0.026249])
        self._reliable = False
        self.alpha = 0.05
        self.fov_v = fov_v
        self.img_height = image_height
        self.img_width = image_width
        self.deg_per_pix = self.fov_v/self.img_height
        self.set_undistort()

    def set_undistort(self):
        self.undistort_top = self.undistort_points(self.img_width/2, 0)
        self.undistort_bottom = self.undistort_points(self.img_width/2, self.img_height)
        self.undistort_scale = np.abs(self.undistort_bottom[1] - self.undistort_top[1])

    def undistort_points(self, x, y):
        points = np.array([[x, y]], dtype=np.float32).reshape(-1, 1, 2)
        undistorted_points = cv2.undistortPoints(points, self.camera_matrix, self.dist_coeffs, P=self.P)
        return tuple(undistorted_points[0][0])
    
    def undistort_normed(self, x, y):
        """
        Undistorting causes points to move out of the scene (if central part's pix/angle is retained)
        or points' coordinates to shrink (if the top/bottom part's pix/angle is retained)
        So, undistorted coordinates need to be normalized.
        """
        #_new_p1 = self.undistort_points(x, y)
        #points = np.array([[x, y]], dtype=np.float32).reshape(-1, 1, 2)
        points = np.vstack([x, y]).T.reshape(-1, 1, 2)
        _new_p1 = cv2.undistortPoints(points, self.camera_matrix, self.dist_coeffs, P=self.P)
        return (_new_p1[:,0,0], (_new_p1[:,0,1]-self.undistort_top[1])/self.undistort_scale*self.img_height)    
        
    def update_face_wh(self, flmk_x, flmk_y):
        #undistorted_ = np.array([self.undistort_normed(p) for p in keypoints_2d.T]).T
        flmk_x, flmk_y = self.undistort_normed(flmk_x, flmk_y)
        self._face_width = np.mean(flmk_x[14:17] - flmk_x[:3])
        if self._face_width < 60:
            self._reliable = False
        else:
            self._face_height = 0.5*(np.linalg.norm(flmk_y[0] - flmk_y[9])+
                            np.linalg.norm(flmk_y[16] - flmk_y[9]))
            if self._face_height < 60:
                self._reliable = False
            else:
                self._reliable = True
            
    def update_face_dist(self, flmk_x, flmk_y):
        # if self._reliable == False:
        #     return
        # print(f"Ratio {eye_dist_ratio:.2f}")
        # It's ratio, scale-invariant. No need to worry about undistort
        # To some degrees, at least.
        eye_dist_ratio = self.eye_ratio(flmk_x, flmk_y)
        if not eye_dist_ratio:
            return False
        # D1 * h1 = D2 * h2  ->  D2 = D1 * h1 / h2
        dist_fh = self.face_hr / self._face_height
        dist_fw = self.face_wr / self._face_width
        
        # print("dist_fh", dist_fh, "dist_fw", dist_fw)
        # Magic ratio == ad-hoc
        _dist_now = max(min(eye_dist_ratio * dist_fh + (1-eye_dist_ratio)*dist_fw, 3.0), 0.1)
        self.dist_face = self.alpha * _dist_now + (1 - self.alpha) * self.dist_face
         #= eye_dist_ratio * dist_fh + (1-eye_dist_ratio)*dist_fw 
            
    def update(self, dist, flmk_x, flmk_y):
        if self._reliable:
            if self._initial_count < self._guess_max:
                face_wr = self._face_width * dist # pixel / meter (distorted)
                face_hr = self._face_height * dist
                self._initial_guess_face_h[self._initial_count] = face_hr
                self._initial_guess_face_w[self._initial_count] = face_wr
                self._initial_count += 1
                # print("Guess added", face_hr, face_wr)
            elif self._initial_count == self._guess_max:
                self.fix_face_size()
                # print("Face size fixed")
            else:
                success = self.update_face_dist(flmk_x, flmk_y)
                # print("Face dist updated", success)
                
            
    def fix_face_size(self):
        # if self._initial_count < self._guess_max:
        #     self.face_wr = None
        #     self.face_hr = None
        #     return
        self.face_wr = np.percentile(self._initial_guess_face_w, 90)
        self.face_hr = np.percentile(self._initial_guess_face_h, 90)
        self._initial_count = 0
        
    @staticmethod
    def eye_ratio(flmk_x, flmk_y):
        std_both = std2d(flmk_x[36:48], flmk_y[36:48])
        std_single = max([std2d(flmk_x[36:42], flmk_y[36:42]), 
                        std2d(flmk_x[42:48], flmk_y[42:48])])
        if std_both == 0 or std_single == 0:
            return False
        
        return std_single/std_both * 2 # ~ 0.5
    