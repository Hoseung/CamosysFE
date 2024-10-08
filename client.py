import pickle
import queue
from datagenerator import CameraDataGenerator
from postproc import PostProcessor
import socket
import struct
import numpy as np
import cv2
from glob import glob
import time
class Client:
    def __init__(self, server_ip, port=65432, 
                 frame_width=1024, 
                 frame_height=1024,
                 camera_height=1.4,
                 camera_pitch=10,
                 height_factor=1.0
                 ):
        self.server_address = (server_ip, port)
        self.sock = None
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.frame_queue = queue.Queue(maxsize=10)
        self.label_data_queue = queue.Queue(maxsize=10)
        self.label_size = struct.calcsize(
            '=h b b b b B b 68H 68H 15H 15H 15H 13H 13H 13B 4H')
        self.running = True
        
        self.image_generator = CameraDataGenerator(camera_index=2, 
                                                   crop=(56,1080, 448, 1472))
        self.post_processor = PostProcessor(semi_undistort=True,
                                            image_width=frame_width,
                                            image_height=frame_height,
                                            camera_height = camera_height, 
                                            camera_pitch = camera_pitch,
                                            height_factor=height_factor)
        self.postproc_gen = None
        
        self.label_dtype = np.dtype([
            ("distance", np.int16),
            ("eye_openness", np.int8),
            ("drowsiness", np.int8),
            ("phoneuse", np.int8),
            ("phone_use_conf", np.int8),
            ("height", np.uint8), 
            ("passenger", np.int8), 
            ("face_landmarks_x", np.int16, (68)),  # 68 elements of uint16
            ("face_landmarks_y", np.int16, (68)),  # 68 elements of uint16
            ("body_keypoints3d", np.int16, (3, 15)),  # 15 elements of uint16 in 3D
            ("body_keypoints2d", np.int16, (2, 13)),  # 13 elements of uint16 in 2D
            ("body_keypoints2d_conf", np.uint8, (13,)),  # 13 elements of uint16 in 1D
            ("face_bounding_box", np.uint16, (4,))# 4 elements of uint16
            ])

    def setup_socket(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, True)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2 ** 20)
        self.sock.bind(self.server_address)
        self.sock.listen(1)
        print("Server listening on {}:{}".format(*self.server_address)) 
        # self.sock.connect(self.server_address)
        # print("Connected to server at {}:{}".format(*self.server_address))

    def accept_connection(self):
        self.conn, addr = self.sock.accept()
        print("Connection from", addr)
        self.conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, True)
        self.conn.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2**20)

    def cleanup(self):
        if self.conn:
            self.conn.close()
        if self.sock:
            self.sock.close()

    def receive_frame_(self):
        flist = glob("../CamosysFE_data/test_9cm/frame*.jpg")
        flist.sort()
        labels = pickle.load(open("../CamosysFE_data/test_9cm/label_data.pkl", "rb"))

        for i, label_array in enumerate(labels):
            time.sleep(0.03)
            frame = cv2.imread(flist[i])
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            frame = cv2.copyMakeBorder(
                            frame,
                            56,0, 448, 448,
                            cv2.BORDER_CONSTANT,
                            value=[0, 0, 0]  # Black padding; change to other values for different colors
                        )
            # label_array = lablels[i]
            
            if self.postproc_gen is None:
                self.postproc_gen = self.post_processor.run(label_array)
                    # Prime the generator (advance to the first yield)
                next(self.postproc_gen)
                # dist_neck, body_size = 0,0
            else:
                # Get the next result from the generator
                self.postproc_gen.send(label_array)
                # print(dist_neck)  # Handle the output from the generator
            
            ### Finally update the view
            if not self.frame_queue.full():
                self.frame_queue.put(frame)

            if not self.label_data_queue.full():
                self.label_data_queue.put(label_array)
            # self.frame_queue.put(frame)
            # self.label_data_queue.put(label_array)
        print("DONE")

    def receive_frame(self):
        save = []
        cnt = 0
        bad = 0
        while self.running:
            # try:
            if True:
                frame, frame_org = next(self.image_generator)
                
                cv2.imwrite(f"../CamosysFE_data/frame{cnt:04d}.jpg", frame)
                # print("Fame sent size", frame.shape)
                self.conn.sendall(frame.tobytes())
            
                label_data = self.conn.recv(self.label_size)
                if not label_data:
                    bad += 1
                    if bad == 30:
                        bad = 0
                        break
                    continue
                # print("Label data size:", len(label_data))
                
                label_array = np.frombuffer(label_data, dtype=self.label_dtype).copy()
                save.append(label_array)
                cnt += 1
                # Consider the main person.
                # Initialize the generator if it's the first run
                if self.postproc_gen is None:
                    self.postproc_gen = self.post_processor.run(label_array)
                        # Prime the generator (advance to the first yield)
                    next(self.postproc_gen)
                    # dist_neck, body_size = 0,0
                else:
                    # Get the next result from the generator
                    self.postproc_gen.send(label_array)
                
                ### Finally update the view
                if not self.frame_queue.full():
                    self.frame_queue.put(frame_org)

                if not self.label_data_queue.full():
                    self.label_data_queue.put(label_array)
                
                # print(f"Count -- {cnt}")
                if cnt % 100 == 0:
                    pickle.dump(save, open("../CamosysFE_data/label_data.pkl", "wb"))
            else:        
            # except Exception as e:
                pickle.dump(save, open("../CamosysFE_data/label_data.pkl", "wb"))
                print(f"Error receiving data: {e}")
                print("_____________________")
                print("Pickle saved")
                self.cleanup()
                break

        self.cleanup()

    def cleanup(self):
        if self.sock:
            self.sock.close()
        self.postproc_gen = None
        self.running = False
