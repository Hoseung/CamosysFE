import math
import numpy as np
#import yaml
import cv2


class DistanceTriangle():
    """
    All calculations in meter unit.
    """
    def __init__(self, 
                 camera_height = 1.5, 
                 fov_v=91,
                 horizontal_fov_deg=166, 
                 image_width = 640,
                 image_height = 480, 
                 camera_pitch = -24):
        self.camera_matrix = np.array([[380.5828, 0., 327.04076],
                                       [0., 381.61306, 245.22762],
                                       [0., 0., 1.]])
        self.dist_coeffs = np.array([-0.330314, 0.130840, 0.000384, 0.000347, -0.026249])
        self.P = np.array([[379.9881, 0., 326.52974, 0.],
                           [0., 380.81802, 244.71673, 0.],
                           [0., 0., 1., 0.]])
        
        self.fov_v = fov_v
        self.img_height = image_height
        self.img_width = image_width
        self.deg_per_pix = self.fov_v/self.img_height
        self.camera_height = camera_height
        self.camera_pitch = camera_pitch
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
        heights = np.array([self.get_obj_size(foot, p_head, verbose=False) 
                            for foot in feet])

        if max(heights) > 1.2 * min(heights):
            # If uneven, take larger one
            self.foot_ind = np.argmax(heights)
            height = heights[self.foot_ind]
            dist = self.dist_camroot_to_foot(feet[self.foot_ind][1])
        else:
            height = np.mean(heights)
            dist = 0.5*(self.dist_camroot_to_foot(feet[0][1]) + \
                self.dist_camroot_to_foot(feet[1][1]))
            self.foot_ind = 0
        
        return height, dist
    
    def height_taken(self, keypoints_2d, take_frac = 0.86):
        height_pixel, bone_sum_pixel = get_bone_length_sum(keypoints_2d)
        height, dist = self.cal_body_size(keypoints_2d)
        if take_frac * height_pixel > bone_sum_pixel:
            return height, dist
        else:
            return height, False

def take_avg_or_larger(v1, v2, threshold=1.2):
    if max((v1,v2)) > threshold*min(v1,v2):
        return np.max((v1,v2))
    else:
        return np.mean((v1,v2))
            
def get_bone_length_sum(keypoints_2d):
    # in pixels
    #lc = length_connections_2d
    l_torso = np.linalg.norm(keypoints_2d[:,1] - keypoints_2d[:,7])
    l_thigh = np.linalg.norm(keypoints_2d[:,7] - keypoints_2d[:,9])
    l_calf  = np.linalg.norm(keypoints_2d[:,9] - keypoints_2d[:,11])
    l_height = l_torso + l_thigh + l_calf
    r_torso = np.linalg.norm(keypoints_2d[:,2] - keypoints_2d[:,8])
    r_thigh = np.linalg.norm(keypoints_2d[:,8] - keypoints_2d[:,10])
    r_calf  = np.linalg.norm(keypoints_2d[:,10] - keypoints_2d[:,12])
    r_height = r_torso + r_thigh + r_calf

    head_to_lfoot = np.linalg.norm(keypoints_2d[:,0] - keypoints_2d[:,11])
    head_to_rfoot = np.linalg.norm(keypoints_2d[:,0] - keypoints_2d[:,12])
    
    height_pixel = take_avg_or_larger(head_to_lfoot, head_to_rfoot)
    bone_sum_pixel = take_avg_or_larger(l_height, r_height)
    
    return height_pixel, bone_sum_pixel