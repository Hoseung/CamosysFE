import math
import numpy as np
import cv2

class DistanceTriangle():
    """
    All calculations in meter unit.
    """
    def __init__(self, 
                 camera_height = 1.5, 
                 fov_v=91,
                 horizontal_fov_deg=166, 
                 image_width = 1920,
                 image_height = 1080, 
                 camera_pitch = -20):
        
        # For 640x480
        # self.camera_matrix = np.array([[380.5828, 0., 327.04076],
        #                                [0., 381.61306, 245.22762],
        #                                [0., 0., 1.]])
        # self.P = np.array([[379.9881, 0., 326.52974, 0.],
        #                    [0., 380.81802, 244.71673, 0.],
        #                    [0., 0., 1., 0.]])
        
        # for 1920 x 1080
        self.camera_matrix = np.array([[1141.7483, 0., 981.12228],
                                        [0., 858.629385, 551.762145],
                                        [0., 0., 1.]])
        self.P = np.array([[379.9881*3, 0., 326.52974*3, 0.],
                           [0., 380.81802*2.25, 244.71673*2.25, 0.],
                           [0., 0., 1., 0.]])
        self.dist_coeffs = np.array([-0.330314, 0.130840, 0.000384, 0.000347, -0.026249])
        
        self.fov_v = fov_v
        self.img_height = image_height
        self.img_width = image_width
        self.deg_per_pix = self.fov_v/self.img_height
        self.camera_height = camera_height
        self.camera_pitch = camera_pitch
        self.u_l_ratio = 1.2
        self.u_l_alpha = 0.1
        self.set_undistort()
        self.foot_ind = 0

    def set_undistort(self):
        self.undistort_top = self.undistort_points(self.img_width/2, 0)
        self.undistort_bottom = self.undistort_points(self.img_width/2, self.img_height)
        self.undistort_scale = np.abs(self.undistort_bottom[1] - self.undistort_top[1])

    def undistort_points(self, x, y):
        points = np.array([[x, y]], dtype=np.float32).reshape(-1, 1, 2)
        undistorted_points = cv2.undistortPoints(points, self.camera_matrix, self.dist_coeffs, P=self.P)
        return tuple(undistorted_points[0][0])
    
    def undistort_normed(self, p):
        """
        Undistorting causes points to move out of the scene (if central part's pix/angle is retained)
        or points' coordinates to shrink (if the top/bottom part's pix/angle is retained)
        So, undistorted coordinates need to be normalized.
        """
        _new_p1 = self.undistort_points(*p)
        return (_new_p1[0], (_new_p1[1]-self.undistort_top[1])/self.undistort_scale*self.img_height)
    
    def deg_img_vert(self, y_p):
        # angle between optical axis and vertical point
        return (self.img_height/2 - (self.img_height - y_p)) * self.deg_per_pix
    
    def deg_cam_foot_camroot(self, y_p):
        return self.camera_pitch + self.deg_img_vert(y_p)
    
    def dist_camroot_to_foot(self,y_p):
        #del_theta = self.deg_img_vert(y_p)
        return self.camera_height / np.tan(np.deg2rad(self.deg_cam_foot_camroot(y_p)))

    def deg_foot_cam_head(self, yp1, yp2):
        return (yp2-yp1) * self.deg_per_pix

    def solve_triangle_ABc(self, A,B,c):
        # Calculate the third angle
        C = 180 - A - B
        
        # Convert angles to radians for trigonometric functions
        A_rad = math.radians(A)
        B_rad = math.radians(B)
        C_rad = math.radians(C)
        
        # Use the Law of Sines to calculate the other sides
        a = c * math.sin(A_rad) / math.sin(C_rad)
        b = c * math.sin(B_rad) / math.sin(C_rad)
        
        return {
            'angles': {'A': A, 'B': B, 'C': C},
            'sides': {'a': a, 'b': b, 'c': c}
        }

    def dist_cam_to_foot(self, y_p):
        theta = self.deg_cam_foot_camroot(y_p)
        return self.camera_height / np.sin(np.deg2rad(theta))
    
    def get_obj_size(self, p1, p2, verbose=False):
        """
        p1 = (x1, y1), p2 = (x2, y2)
        
        A: deg  head - cam - foot
        B: deg  cam - foot - head 
        C: deg cam - head - foot
        
        a: dist head - foot == height
        b: dist cam - head
        c: dist cam - foot
        """
        deg_A = self.deg_foot_cam_head(p2[1], p1[1])
        deg_B = 90 - self.deg_cam_foot_camroot(p1[1]) # Assume upright pose
    
        side_c = self.dist_cam_to_foot(p1[1]) # from cam
        result = self.solve_triangle_ABc(deg_A, deg_B, side_c)
        if verbose: print(result)
        return result['sides']['a']
    
    def cal_body_size(self, keypoints_2d):
        p_head = keypoints_2d[:,0]
        feet = [keypoints_2d[:,11], keypoints_2d[:,12]]
        height = self.get_obj_size(feet[self.foot_ind], p_head, verbose=False)
        dist = 0.5*(self.dist_camroot_to_foot(feet[0][1]) + \
            self.dist_camroot_to_foot(feet[1][1]))
        # self.foot_ind = 0
            # print("[cal_body_size] Height", height, "Dist", dist, "MEAN")
        
        return height, dist
    
    def height_taken(self, keypoints_2d, take_frac = 0.86):
        keypoints_2d[1,:] += 56
        undistorted = np.array([self.undistort_normed(p) for p in keypoints_2d.T]).T
        height_pixel, bone_sum_pixel, lower = self.get_bone_length_sum(undistorted)
        new_u_l_ratio = (height_pixel - lower) / lower
        #new_u_l_ratio = self.upper_lower_ratio(undistorted)
        if abs(new_u_l_ratio - self.u_l_ratio) / self.u_l_ratio > 0.2:
            # print("[ignore] U-L ratio changed", new_u_l_ratio, self.u_l_ratio)
            # Probably not in straight pose
            return False, False
        else: 
            self.update_upper_lower_ratio(new_u_l_ratio)
            if take_frac * height_pixel > bone_sum_pixel:
                # print("[take] U-L ratio", new_u_l_ratio, self.u_l_ratio)
                # print("[height_taken] Measured height", height_pixel, "bone_sum", bone_sum_pixel)
                height, dist = self.cal_body_size(undistorted)
                return height, dist
            else:
                return False, False
        
    def update_upper_lower_ratio(self, new_u_l_ratio):
        self.u_l_ratio = self.u_l_alpha * new_u_l_ratio + (1 - self.u_l_alpha) * self.u_l_ratio
            
    def upper_lower_ratio(self, keypoints_2d):
        """
        keypoints_2d: 13x2
        """
        if self.foot_ind == 0:
            lower_body = np.linalg.norm(keypoints_2d[:,7] - keypoints_2d[:,11])
        else:
            lower_body = np.linalg.norm(keypoints_2d[:,8] - keypoints_2d[:,12])
            
        print("Lower body", lower_body)
        
        upper_body = take_avg_or_larger(np.linalg.norm(keypoints_2d[:,0] - keypoints_2d[:,7]),
                                        np.linalg.norm(keypoints_2d[:,0] - keypoints_2d[:,8]))
        print("Upper body", upper_body)

        return upper_body / lower_body
    
    def get_bone_length_sum(self, keypoints_2d):
        # in pixels
        #lc = length_connections_2d
        if self.foot_ind == 0:
            torso = np.linalg.norm(keypoints_2d[:,1] - keypoints_2d[:,7])
            thigh = np.linalg.norm(keypoints_2d[:,7] - keypoints_2d[:,9])
            calf  = np.linalg.norm(keypoints_2d[:,9] - keypoints_2d[:,11])
            lower = np.linalg.norm(keypoints_2d[:,7] - keypoints_2d[:,11])
            height_pixel = np.linalg.norm(keypoints_2d[:,0] - keypoints_2d[:,11])
        elif self.foot_ind == 1:
            torso = np.linalg.norm(keypoints_2d[:,2] - keypoints_2d[:,8])
            thigh = np.linalg.norm(keypoints_2d[:,8] - keypoints_2d[:,10])
            calf  = np.linalg.norm(keypoints_2d[:,10] - keypoints_2d[:,12])
            lower = np.linalg.norm(keypoints_2d[:,8] - keypoints_2d[:,12])
            height_pixel = np.linalg.norm(keypoints_2d[:,0] - keypoints_2d[:,12])    
        # print("torso, thigh, calf", torso, thigh, calf)
        bone_sum_pixel = torso + thigh + calf
        
        return height_pixel, bone_sum_pixel, lower


def take_avg_or_larger(v1, v2, threshold=1.2):
    if max((v1,v2)) > threshold*min(v1,v2):
        return np.max((v1,v2))
    else:
        return np.mean((v1,v2))
            
