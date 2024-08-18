import math
import yaml
import cv2
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt 
from typing import List, Tuple
from constants import connections_2d, connections_3d

def to_world(key_3d, scale):
    key3d = np.array(key_3d) * scale
    # Pelvis center == world center
    key3d -= 0.5 * (key3d[1,:] + key3d[2,:])
    return key3d

def rotate_3d_points(points, angles):
    """
    Rotate 3D points by given angles around the x, y, and z axes.

    Parameters:
    - points (numpy.ndarray): An Nx3 array of points to be rotated.
    - angles (tuple): A tuple (alpha, beta, gamma) representing the rotation angles in degrees
                      around the x, y, and z axes, respectively.

    Returns:
    - numpy.ndarray: The rotated Nx3 array of points.
    """
    alpha, beta, gamma = np.radians(angles)  # Convert angles to radians
    
    # Rotation matrix for rotation around the x-axis
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(alpha), -np.sin(alpha)],
        [0, np.sin(alpha), np.cos(alpha)]
    ])

    # Rotation matrix for rotation around the y-axis
    R_y = np.array([
        [np.cos(beta), 0, np.sin(beta)],
        [0, 1, 0],
        [-np.sin(beta), 0, np.cos(beta)]
    ])

    # Rotation matrix for rotation around the z-axis
    R_z = np.array([
        [np.cos(gamma), -np.sin(gamma), 0],
        [np.sin(gamma), np.cos(gamma), 0],
        [0, 0, 1]
    ])

    # Combined rotation matrix
    R = R_z @ R_y @ R_x
    
    # Apply rotation to all points
    rotated_points = points @ R.T

    return rotated_points

def estimate_upper_body(keypoints_3d, scale=100):
    """_summary_

    Obsolete.

    Args:
        keypoints_3d (_type_): _description_
        scale (int, optional): _description_. Defaults to 100.

    Returns:
        _type_: _description_
    """
    upper_body_l = np.sqrt((keypoints_3d[1, 0] - keypoints_3d[7, 0])**2 +
                           (keypoints_3d[1, 1] - keypoints_3d[7, 1])**2 +
                           (keypoints_3d[1, 2] - keypoints_3d[7, 2])**2)
    upper_body_r = np.sqrt((keypoints_3d[2, 0] - keypoints_3d[8, 0])**2 +
                           (keypoints_3d[2, 1] - keypoints_3d[8, 1])**2 +
                           (keypoints_3d[2, 2] - keypoints_3d[8, 2])**2)
    upper_body = 0.5 * (upper_body_l + upper_body_r) * scale

    l_s = 0.5 * (np.sqrt((keypoints_3d[1, 2] - keypoints_3d[7, 2])**2) +
                 np.sqrt((keypoints_3d[2, 2] - keypoints_3d[8, 2])**2))
    l_l = 0.5 * (np.sqrt((keypoints_3d[1, 1] - keypoints_3d[7, 1])**2) +
                 np.sqrt((keypoints_3d[2, 1] - keypoints_3d[8, 1])**2))
    # theta = np.rad2deg(math.atan(l_s / l_l))
    # print("THETA", theta)
    # upper_body /= np.cos(np.deg2rad(theta))
    return upper_body


def angle_between_vectors(p0, p1, p2, p3):
    # Calculate the vectors
    vectorA = [p1[0] - p0[0], p1[1] - p0[1]]
    vectorB = [p3[0] - p2[0], p3[1] - p2[1]]
    
    # Calculate the dot product of vectors A and B
    dot_product = vectorA[0] * vectorB[0] + vectorA[1] * vectorB[1]
    
    # Calculate the magnitudes of vectors A and B
    magnitudeA = math.sqrt(vectorA[0]**2 + vectorA[1]**2)
    magnitudeB = math.sqrt(vectorB[0]**2 + vectorB[1]**2)
    
    # Calculate the cosine of the angle
    cos_theta = dot_product / (magnitudeA * magnitudeB)
    
    # Ensure the value is within the valid range for arccos to avoid numerical errors
    cos_theta = max(-1.0, min(1.0, cos_theta))
    
    angle_radians = math.acos(cos_theta)
    angle_degrees = math.degrees(angle_radians)
    
    return angle_degrees


class Visualizer:
    def __init__(self, elev=130, azim=90):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')

        self.ax.set_xlim(-50, 50)
        self.ax.set_ylim(-50, 50)
        self.ax.set_zlim(-50, 50)
        self.ax.view_init(elev=elev, azim=azim)
        
        self.n_points = 15
        self.init_plot()
        # Set plot title
        self.ax.set_title('3D Keypoints with Connections')

    def init_plot(self):
        self.scatters = self.ax.scatter(np.random.random(self.n_points) - 0.5, 
                                       np.random.random(self.n_points) - 0.5, 
                                       np.random.random(self.n_points) - 0.5, 
                                       marker='o')

        # Plot lines between the connected keypoints
        self.lines = []
        self.texts = []
        for connection in connections_3d:
            x_values = [0.01, 0.02]
            y_values = [0.01, 0.02]
            z_values = [0.01, 0.02]
            line = self.ax.plot(x_values, y_values, z_values, 'r-')
            self.lines.append(line[0])
        
            point1, point2 = connection
            #x_values = [kpts[point1, 0], kpts[point2, 0]]
            #y_values = [kpts[point1, 1], kpts[point2, 1]]
            #z_values = [kpts[point1, 2], kpts[point2, 2]]        
            # text on plot line
            self.ax.text(np.mean(x_values), np.mean(y_values), np.mean(z_values), \
                         f'{int(np.linalg.norm(0.02 - 0.01)*100)} cm', \
                            size=10, zorder=1, color='b')
          
    def visualize_and_save_keypoints(self, kpts: np.ndarray, connections: List[Tuple[int, int]],
                                     plot_filename: str = None, show_text=False):
        """
        Visualizes 3D keypoints with connections and saves the plot and keypoints data.

        :param keypoints: A numpy array of shape (n, 3) representing 3D keypoints.
        :param connections: A list of tuples where each tuple contains two indices of keypoints to be connected.
        :param plot_filename: The filename to save the plot (e.g., '3d_keypoints_with_connections.png').
        :param data_filename: The filename to save the keypoints data (e.g., 'keypoints.npy').
        """
        # Visualize Keypoints and Connections
        self.scatters._offsets3d = (kpts[:, 0], kpts[:, 1], kpts[:, 2])

        for line, (point1, point2) in zip(self.lines, connections):
            line.set_data_3d((kpts[point1, 0], kpts[point2, 0]),
                             (kpts[point1, 1], kpts[point2, 1]),
                             (kpts[point1, 2], kpts[point2, 2]))
            
        if show_text:
            for connection in connections:
                point1, point2 = connection
                x_values = [kpts[point1, 0], kpts[point2, 0]]
                y_values = [kpts[point1, 1], kpts[point2, 1]]
                z_values = [kpts[point1, 2], kpts[point2, 2]]        
                # text on plot line
                self.ax.text(np.mean(x_values), np.mean(y_values), np.mean(z_values), \
                            f'{int(np.linalg.norm(kpts[point1] - kpts[point2])*100)} cm', \
                                size=10, zorder=1, color='b')

        #fig.canvas.draw_idle()
        plt.pause(0.001)
        if plot_filename is not None:
            plt.savefig(plot_filename)

def image_to_world_coordinates(u, v, Z, K_inv):
    uv1 = np.array([u, v, 1])
    XYZ = Z * np.dot(K_inv, uv1)
    return XYZ

class RealScaler():
    def __init__(self, cam_yml):
        with open(cam_yml, 'r') as f:
            yy = yaml.safe_load(f)
    
        self.cam_mtx = np.array(yy['camera_matrix']['data']).reshape(3,3)
        self.distor_coeff = np.array(yy['distortion_coefficients']['data'])
        self.focal_length = np.sqrt(self.cam_mtx[0,0]**2 + self.cam_mtx[1,1]**2)
        self.K_inv = np.linalg.inv(self.cam_mtx)

    def undistort_points(self, points):
        """
        왜곡된 좌표를 보정하는 함수
        :param points: 왜곡된 좌표
        :param camera_matrix: 카메라 매트릭스
        :param dist_coeffs: 왜곡 계수
        :return: 보정된 좌표
        """
        points = np.array(points, dtype=np.float32)
        points = points.reshape(-1, 1, 2)
        undistorted_points = cv2.undistortPoints(points, self.cam_mtx, self.distor_coeff, None, self.cam_mtx)
        return undistorted_points.reshape(-1, 2)

    def get_3d_length(self, p1, p2):
        """
        p1 = [u,v,z] 
        for example, [280, 156, 1.02]
        """
        xyz1 = image_to_world_coordinates(*p1, self.K_inv)
        xyz2 = image_to_world_coordinates(*p2, self.K_inv)
        return np.linalg.norm(xyz2-xyz1)
    
    def get_real_length(self, p1, p2, projection_angle=21, z_dist = 1.02):
        undistorted_points = self.undistort_points([p1, p2])
        undistorted_length = np.linalg.norm(undistorted_points[1]-undistorted_points[0])
        # print("original length", np.linalg.norm(np.array(p2)-np.array(p1)))
        # print("undistorted length", undistorted_length)
        real_length_in_px = undistorted_length/np.cos(np.deg2rad(projection_angle))
        real_length = real_length_in_px/self.focal_length * z_dist
        return real_length
    
class StateTracker:
    def __init__(self, update_threshold=5):
        self.previous_state = -1
        self.current_count = 0
        self.current_measurement = -1
        self.update_threshold = update_threshold

    def update_state(self, new_measurement):
        if new_measurement == self.current_measurement:
            self.current_count += 1
        else:
            self.current_measurement = new_measurement
            self.current_count = 1

        # Change state only if we have more than 5 consistent measurements in a row
        if self.current_count > self.update_threshold:
            self.previous_state = self.current_measurement

        return self.previous_state