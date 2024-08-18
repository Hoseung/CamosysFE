import numpy as np
from ruler import DistanceTriangle, take_avg_or_larger
from eye import Eye, std2d

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
        self.cnt_initial = 20
                
        self.foot_ind3d = [3,6]
        self.area_lmin = 10
        self.area_rmax = image_width -10
        
    def run(self, label_array):
        initial_guess = []
        initial_guess_face_h = []
        initial_guess_face_w = []
        running_avg = 0
        # smoothed_keypoints_2d = None
        z_dist_foot = 1
        i = 0
        cnt = 0
        scale3d = 1.0
        no_person = 0
        conf_threshold = 60
        
        eye = Eye()
        
        label_array['passenger'][0] = 0
        height_class_before = label_array['passenger'][0]
        while True:
            # Update incoming values with new ones sent to the generator
            label_array = yield 
            box = label_array['face_bounding_box'][0]
            keypoints_2d = label_array['body_keypoints2d'][0].astype(np.float32)
            keypoints_2d_conf = label_array['body_keypoints2d_conf'][0]
            key3d = label_array['body_keypoints3d'][0].astype(np.float32)
            flmk_x = label_array['face_landmarks_x'][0].astype(np.float32)
            flmk_y = label_array['face_landmarks_y'][0].astype(np.float32)
            # passenger = label_array['passenger']
            
            i += 1
            # cnt += 1
            # FACE
            face_wr_final = 90 
            face_hr_final = 100
            dist_face = 0
            # Valid face
            # print("CNT", cnt)
            # Face detected in the AOI
            empty = True
            if box is not None and box[0] > self.area_lmin and box[3] < self.area_rmax:
                empty = False    
                # If Face width not measurable
                if any(flmk_x[36:42] == 0) or any(flmk_y[36:42] == 0):
                    print("Invalid face landmarks")
                    print(flmk_x)
                    print(flmk_y)
                    continue
                eye.update_EAR((flmk_x, flmk_y))
                face_width = np.mean(flmk_x[14:17] - flmk_x[:3]) 
                face_height = 0.5*(np.linalg.norm(flmk_y[0] - flmk_y[9])+
                                np.linalg.norm(flmk_y[16] - flmk_y[9]))
                
                if face_hr_final is not None:
                    std_single = max([std2d(flmk_x[36:42], flmk_y[36:42]), 
                                    std2d(flmk_x[42:48], flmk_y[42:48])])
                    
                    std_both = std2d(flmk_x[36:48], flmk_y[36:48])
                    # print(f"std_single {std_single:.2f} std_both {std_both:.2f}")
                    eye_dist_ratio = std_single/std_both * 2 # ~ 0.5
                    # print(f"Ratio {eye_dist_ratio:.2f}")
                    
                    dist_fh = face_hr_final / face_height
                    dist_fw = face_wr_final / face_width
                    dist_face = (eye_dist_ratio * dist_fh + (1-eye_dist_ratio)*dist_fw) * 0.7
                    # print(face_height, face_width, dist_fh, dist_fw, dist_face)
                
                if keypoints_2d_conf[0] > conf_threshold and keypoints_2d_conf[11] > conf_threshold and keypoints_2d_conf[12] > conf_threshold:
                    # print("Full body visible")
                    height, dist = self.dt.height_taken(keypoints_2d, take_frac = 0.95)
                    if dist:
                        #takes.append(True)
                        cnt += 1
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
                                scale3d = running_avg / head_to_foot3d
                                # print("Body size", running_avg*100)
                        elif cnt < self.cnt_initial: 
                            initial_guess.append(height)
                            
                            # check for head size
                            # print("GUESSING HEAD SIZE")
                            face_wr = face_width / dist
                            face_hr = face_height / dist
                            # print(f"face width and height {face_wr:.2f} {face_hr:.2f}")
                            # print(f"at dist {dist:.2f}")                            
                            initial_guess_face_h.append(face_hr)
                            initial_guess_face_w.append(face_wr)
                            
                            # Estimating the passenger class
                            label_array['passenger'][0] = -1
                            
                        elif cnt == self.cnt_initial: 
                            running_avg = np.percentile(initial_guess, 90)
                            # print("face_w", initial_guess_face_w)
                            # print("face_h", initial_guess_face_h)
                            face_wr_final = np.percentile(initial_guess_face_w, 90)
                            face_hr_final = np.percentile(initial_guess_face_h, 90)
                            #running_std = np.std(initial_guess)
                            print("Initial guess", running_avg*100)
                            
                        z_dist_foot = dist
                        
                    else:
                        # print("Unreliable Height measurement222")
                        pass
                        
                    # scale
                    # print("key3d", key3d[2,:])
                    key3d[2,:] *= scale3d
                    # print("key3d after scale", key3d[2,:])
                    # translation``
                    key3d[2,:] += (z_dist_foot - key3d[2, self.foot_ind3d[self.dt.foot_ind]])
                    # print("key3d after translation", key3d[2,:])

                    # update only when the person is detected
                    # dist_neck = np.linalg.norm(key3d[:, 7] - self.cam_loc)
                else:
                    pass
                    # dist_neck = 0
                # print(f"Height  {running_avg*100}")
                # if cnt % 10 == 1 and running_avg > 0:

            # when detected person is gone
            elif running_avg > 0:
                no_person += 1
                if no_person > 10:
                    # print("No person detected!")
                    label_array['passenger'][0] = 0
                    label_array['height'][0] = 0
                    cnt = 0
                    running_avg = 0
                    print("RESETTING HEIGHT because NO_PERSON")
                    no_person = 0
                    initial_guess = []
                    label_array['distance'] = -1
                    empty = True
            else:
                empty = True # or pass?
                    
            #if running_avg > 0:
            if not empty:
                # print("SETTING HEIGHT - VALID")
                if running_avg*self.height_factor > 0.40 and running_avg*self.height_factor < 1.20:
                    label_array['passenger'][0] = 1
                elif running_avg*self.height_factor >= 1.2 and running_avg*self.height_factor < 1.60:
                    print("SETTING HEIGHT - 2")
                    print("running_avg*self.height_factor", running_avg*self.height_factor)
                    label_array['passenger'][0] = 2
                elif running_avg*self.height_factor >= 1.60 and running_avg*self.height_factor < 1.85:
                    label_array['passenger'][0] = 3
                elif running_avg*self.height_factor >= 1.85 and running_avg*self.height_factor < 2.3:
                    label_array['passenger'][0] = 4
                else:
                    # Something wrong
                    label_array['passenger'][0] = -1
            # print("Distance to camera", dist_face)
            
            # print("RUNNING AVG", running_avg)
                label_array['height'][0] = min(running_avg*100*self.height_factor, 999)
                label_array['distance'][0] = min(dist_face*100, 9999)
                label_array['eye_openness'][0] = min(int(eye.EAR/0.4*100), 100)
                label_array['drowsiness'][0] = eye.drowsy_val
            
            else:
                # print("SETTING HEIGHT - INVALID")
                label_array['passenger'][0] = 0
                label_array['height'][0] = 0
                label_array['distance'][0] = -1
                label_array['eye_openness'][0] = 0
                label_array['drowsiness'][0] = 0
            
            
            # DEBUG
            if height_class_before != label_array['height'][0]:
                print("HEIGHT CLASS Changed", label_array['passenger'][0])
                height_class_before = label_array['height'][0]
            # print("HEIGHT CLASS this time", label_array['passenger'][0])
            # yield dist_neck