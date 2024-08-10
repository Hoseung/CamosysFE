import queue
from datagenerator import CameraDataGenerator
from postproc import PostProcessor
import socket
import struct
import numpy as np
class Client:
    def __init__(self, server_ip, port=65432, frame_width=1080, frame_height=1080):
        self.server_address = (server_ip, port)
        self.sock = None
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.frame_queue = queue.Queue(maxsize=10)
        self.label_data_queue = queue.Queue(maxsize=10)
        self.label_size = struct.calcsize(
            '=h b b b b b B 68H 68H 42H 28H 14B 4H')
        self.running = True
        
        self.image_generator = CameraDataGenerator(camera_index=2, 
                                                   crop=(0,1080, 420, 1500))
        self.post_processor = PostProcessor()
        self.postproc_gen = None
        
        self.label_dtype = np.dtype([
            ("distance", np.int16),
            ("eye_openness", np.int8),
            ("drowsiness", np.int8),
            ("phoneuse", np.int8),
            ("phone_use_conf", np.int8),
            ("height", np.uint8), 
            ("passenger", np.int8), 
            ("face_landmarks_x", np.uint16, (68)),  # 68 elements of uint16
            ("face_landmarks_y", np.uint16, (68)),  # 68 elements of uint16
            ("body_keypoints3d", np.uint16, (14, 3)),  # 14 elements of uint16 in 3D
            ("body_keypoints2d", np.uint16, (14, 2)),  # 14 elements of uint16 in 2D
            ("body_keypoints2d_conf", np.uint8, (14,)),  # 14 elements of uint16 in 1D
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

    def receive_frame(self):
        while self.running:
            # try:
                frame = next(self.image_generator)
                self.conn.sendall(frame.tobytes())
            
                label_data = self.conn.recv(self.label_size)
                # print("Label data received")
                if not label_data:
                    break
                # print("Label data size:", len(label_data))
                
                label_array = np.frombuffer(label_data, dtype=self.label_dtype).copy()

                # Consider the main person.
                # Initialize the generator if it's the first run
                if self.postproc_gen is None:
                    self.postproc_gen = self.post_processor.run(label_array)
                        # Prime the generator (advance to the first yield)
                    next(self.postproc_gen)
                    dist_neck, body_size = 0,0
                else:
                    # Get the next result from the generator
                    self.postproc_gen.send(label_array)

                    print(dist_neck)  # Handle the output from the generator
                
                ### Finally update the view
                if not self.frame_queue.full():
                    self.frame_queue.put(frame)

                label_array['distance'] = np.int16(dist_neck * 100)
                #print("Distance:", label_array['distance'])
                print(label_array['body_keypoints2d'])
                if not self.label_data_queue.full():
                    self.label_data_queue.put(label_array)
                

            # except Exception as e:
            #     print(f"Error receiving data: {e}")
            #     break

        self.cleanup()

    def cleanup(self):
        if self.sock:
            self.sock.close()
        self.postproc_gen = None
        self.running = False
