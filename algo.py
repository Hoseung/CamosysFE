import numpy as np
from constants import *
# import matplotlib.gridspec as gridspec
from utils import Visualizer#, RealScaler, to_world, rotate_3d_points
from ruler import DistanceTriangle, take_avg_or_larger#, get_bone_length_sum


class Postprocess():
    def __init__(self, camera_height=1.6,
                      camera_pitch=20):
        self.dt = DistanceTriangle("./ost2.yaml", 
                      fov_v=90,
                      camera_height=camera_height,
                      camera_pitch=camera_pitch)
    
        self.cam_loc = np.array([0,0,self.dt.camera_height])
    
        # 2d keypoint smoothing parameter
        self.alpha_2d = 0.4
        self.smoothed_keypoints_2d = None

        # upper body running mean
        self.alpha = 0.01
        self.cnt_initial = 100
        self.initial_guess = []
        
        # running_std = None

        

        # debug = True
        
        # if debug:
        #     f2d = open('2d.txt', 'w')
        #     f3d = open('3d.txt', 'w')

        running_avg = None
        z_dist_foot = 1
        too_close = 0
        
        # foot_ind2d = [11,12]
        self.foot_ind3d = [3,6]
        
    
    def run(self, box, keypoints_2d, key3d, flmk, label_values):
        area_lmin = 180
        area_rmax = 460
        i = 0

        # if args.vis:
        #     vis = Visualizer(elev=130, azim=90)
        no_person = 0
        
        heights = []
        takes = []
        
        scale3d = 1.0
        while True:
            i += 1

            if box is not None and box[0] > area_lmin and box[3] < area_rmax:

                # smoothing 2d keypoints
                if smoothed_keypoints_2d is None:
                    smoothed_keypoints_2d = keypoints_2d[:,:2]
                
                else:
                    smoothed_keypoints_2d = self.alpha_2d * keypoints_2d[:,:2] + (1 - self.alpha_2d) * smoothed_keypoints_2d
                    keypoints_2d[:,:2] = smoothed_keypoints_2d
                
                # full body visible?
                if keypoints_2d[0,2] > 0.8 and keypoints_2d[11,2] > 0.8 and keypoints_2d[12,2] > 0.8:
                
                    height, dist = self.dt.height_taken(smoothed_keypoints_2d, take_frac = 0.88)
                    if dist:
                        #takes.append(True)
                        cnt += 1
                        if cnt > self.cnt_initial:
                            old_avg = running_avg
                            #print(height, running_avg, cnt)
                            running_avg += (height - running_avg) / cnt

                            if np.abs(height - running_avg) <= 0.05 * running_avg:
                                running_avg = self.alpha * height + (1 - self.alpha) * running_avg
                            
                            # Update 3D Scale
                            l_head_to_foot = np.linalg.norm(key3d[8]-key3d[3])
                            r_head_to_foot = np.linalg.norm(key3d[8]-key3d[6])
                            head_to_foot3d = take_avg_or_larger(l_head_to_foot, r_head_to_foot)
                            scale3d = running_avg / head_to_foot3d
                        elif cnt < self.cnt_initial: 
                            self.initial_guess.append(height)
                        elif cnt == self.cnt_initial: 
                            running_avg = np.percentile(self.initial_guess, 90)
                            #running_std = np.std(initial_guess)
                            print("Initial guess", running_avg*100)
                            
                        z_dist_foot = dist
                    else:
                        continue
                        #print("Unreliable Height measurement222")    
                    # scale
                    key3d *= scale3d

                    # translation``
                    print(f"z_dist_to_foot {z_dist_foot:.2f}")
                    key3d[:,2] += (z_dist_foot - key3d[self.foot_ind3d[self.dt.foot_ind],2])
                
                if cnt % 100 == 1 and running_avg is not None:
                    if running_avg < 1.58:
                        print("Body size: AF05")
                        print(f"Running Average: {running_avg*100:.1f} cm")
                    elif running_avg < 1.85:
                        print("Body size: AM50")
                        print(f"Running Average: {running_avg*100:.1f} cm")
                    elif running_avg < 2.05:
                        print("Body size: AM95")
                        print(f"Running Average: {running_avg*100:.1f} cm")
                    else:
                        print("Body size: TOO BIG. Check the measurement!")
                        print(f"Running Average: {running_avg*100:.1f} cm")
                # Distance between wheel and head ([4])
                # But, I will use [3], neck. 
                # Beacuse when head is too close, head is not seen in the image
                
                # dist_neck to camera_root
                if cnt > self.cnt_initial and cnt % 10 == 1:
                    dist_neck = np.linalg.norm(key3d[7] - self.cam_loc) 
                    print(f"Distance to camera: {dist_neck:.3f} m")
                    if dist_neck < 0.5:
                        too_close = min(50, too_close + 1)
                    else:
                        too_close = max(0, too_close - 1)
                    if too_close > 0.2:
                        print("Too close to the Camera!")
                
                #Or, if upper body is leaning towards the wheel by more than XX degrees
            
            else: # when no person is detected
                no_person += 1
                if no_person > 20:
                    print("No person detected!")
                    cnt = 0
                    too_close = 0 
                    running_avg = None
                    smoothed_keypoints_2d = None
                    no_person = 0
            label_values