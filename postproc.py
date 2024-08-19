import numpy as np
from ruler import DistanceTriangle, take_avg_or_larger
from eye import Eye, Face

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
class PostProcessor:
    def __init__(self, image_width=1024, image_height=1024,
                 camera_height=1.4, camera_pitch=10, height_factor=1.2):
        self.height_factor = height_factor
        self.camera_height=camera_height
        self.dt = DistanceTriangle(fov_v=90,
                                   image_width=image_width,
                                   image_height=image_height,
                                   camera_height=self.camera_height,
                                   camera_pitch=camera_pitch)
    
        self.cam_loc = np.array([0,0,self.dt.camera_height])
    
        # 2d keypoint smoothing parameter
        self.alpha_2d = 0.4
        self.smoothed_keypoints_2d = None

        # upper body running mean
        self.alpha = 0.01
        self.cnt_initial = 15
                
        self.foot_ind3d = [3,6]
        self.area_lmin = 10
        self.area_rmax = image_width -10
        self.class_tracker = StateTracker()

    def is_face_in_roi(self, box):
        return box is not None and box[0] > self.area_lmin and box[3] < self.area_rmax
                
    def run(self, label_array):
        initial_guess = []
        running_avg = 0
        # smoothed_keypoints_2d = None
        z_dist_foot = 120
        dist = None
        i = 0
        cnt = 0
        scale3d = 100.0
        no_person = 0
        conf_threshold = 50
        
        eye = Eye()
        face = Face()
        label_array['passenger'][0] = 0
        
        while True:
            # Update incoming values with new ones sent to the generator
            label_array = yield 
            # box = label_array['face_bounding_box'][0]
            keypoints_2d = label_array['body_keypoints2d'][0].astype(np.float32)
            keypoints_2d_conf = label_array['body_keypoints2d_conf'][0]
            key3d = label_array['body_keypoints3d'][0].astype(np.float32)
            flmk_x = label_array['face_landmarks_x'][0].astype(np.float32)
            flmk_y = label_array['face_landmarks_y'][0].astype(np.float32)
            # passenger = label_array['passenger']
            
            i += 1
            # cnt += 1
            # FACE            
            no_face = any(flmk_x == -1) or any(flmk_y == -1)
            empty = any(keypoints_2d[0] == -1)
            # Face detected in the AOI
            if not no_face:
                eye.update_EAR((flmk_x, flmk_y))
                face.update_face_wh(flmk_x, flmk_y)
                # print("updated face", face.face_hr, face.face_wr)
                if face.face_hr is not None and face.face_wr is not None:
                    face.update_face_dist(flmk_x, flmk_y)
                
            if not empty:
                # print("Full body visible")
                if keypoints_2d_conf[0] > conf_threshold and keypoints_2d_conf[11] > conf_threshold or keypoints_2d_conf[12] > conf_threshold:
                    ind_ok = np.argmax([keypoints_2d_conf[11], keypoints_2d_conf[12]])
                    self.dt.foot_ind = ind_ok
                    # print("Foot index", self.dt.foot_ind)
                    height, dist = self.dt.height_taken(keypoints_2d, take_frac = 0.85)
                    
                    if dist:
                        #takes.append(True)
                        cnt += 1
                        face.update(dist, flmk_x, flmk_y)
                        if cnt > self.cnt_initial:
                            if running_avg == 0 and len(initial_guess) > 1:
                                running_avg = np.percentile(initial_guess, 90)
                            else:
                                running_avg += (height - running_avg) / cnt

                                if np.abs(height - running_avg) <= 0.05 * running_avg:
                                    running_avg = self.alpha * height + (1 - self.alpha) * running_avg
                                
                                # Update 3D Scale
                                l_head_to_foot = np.linalg.norm(key3d[:,8]-key3d[:,3])
                                r_head_to_foot = np.linalg.norm(key3d[:,8]-key3d[:,6])
                                head_to_foot3d = take_avg_or_larger(l_head_to_foot, r_head_to_foot)
                                scale3d = running_avg / head_to_foot3d * 100
                                # print("Body size", running_avg*100)
                        elif cnt < self.cnt_initial: 
                            # print(f"Taking {height*100:.2f}")
                            initial_guess.append(height)
                            
                            # Estimating the passenger class
                            passenger_class = -1
                            
                        elif cnt == self.cnt_initial: 
                            running_avg = np.percentile(initial_guess, 90)
                            # print("Initial guess", running_avg*100)
                    else:
                        # print("Unreliable Height measurement222")
                        pass
                            
                    # update only when the person is detected
                    # dist_neck = np.linalg.norm(key3d[:, 7] - self.cam_loc)
                    if dist is not None:    
                        z_dist_foot = 0.3 * (dist * 100) + 0.7 * z_dist_foot
                        label_array['body_keypoints3d'][0][2,:] = (z_dist_foot - (label_array['body_keypoints3d'][0][2,:] * scale3d)).astype(np.int16)
                        label_array['body_keypoints3d'][0][1,:] -= min(label_array['body_keypoints3d'][0][1,:])
                else:
                    no_person += 1

            # when detected person is gone
            else:# running_avg > 0:
                no_person += 1
            if no_person > 10:
                # print("No person detected!")
                passenger_class = 0
                label_array['height'][0] = 0
                cnt = 0
                running_avg = 0
                no_person = 0
                initial_guess = []
                label_array['distance'] = -1
                label_array['eye_openness'][0] = 0
                label_array['drowsiness'][0] = 6
                eye.reset()
                    # empty = True
            if not empty:
                # print("SETTING HEIGHT - VALID")
                if running_avg*self.height_factor > 0.40 and running_avg*self.height_factor < 1.20:
                    passenger_class = 1
                elif running_avg*self.height_factor >= 1.2 and running_avg*self.height_factor < 1.60:
                    passenger_class = 2
                elif running_avg*self.height_factor >= 1.60 and running_avg*self.height_factor < 1.85:
                    passenger_class = 3
                elif running_avg*self.height_factor >= 1.85 and running_avg*self.height_factor < 2.3:
                    passenger_class = 4
                else:
                    # Something wrong
                    passenger_class = -1
            # print("Distance to camera", dist_face)
            
            # print("RUNNING AVG", running_avg)
                label_array['height'][0] = min(running_avg*100*self.height_factor, 999)
                label_array['distance'][0] = min(face.dist_face*100, 9999)
                label_array['eye_openness'][0] = min(int(eye.EAR/0.5*100), 100)
                label_array['drowsiness'][0] = eye.drowsy_val
            else:
                # print("SETTING HEIGHT - INVALID")
                passenger_class = 0
                label_array['height'][0] = 0
                label_array['distance'][0] = -1
                label_array['eye_openness'][0] = 0
                label_array['drowsiness'][0] = 6
            
            label_array['passenger'][0] = self.class_tracker.update_state(passenger_class)
            