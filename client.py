import socket
import struct
import numpy as np
import cv2
import threading
import queue


# def decode_label_data(data):
#     """
#     Decode label data from the received packed data.

#     Args:
#     data (bytes): The packed label data received from the server.

#     Returns:
#     tuple: The decoded label values.
#     """
#     label_format = (
#         '=b e b ? e ? ? ? ? ? ' + # Int8, float16, Int8, bool, float16, 5*bool,
#         '68H 68H ' + # 68 * uint16 (face landmark X and Y)
#         '14H 14H 14H ' +  # 3*14*uint16 (3D body keypoints)
#         '10b ' +  # 10*int8 Joint length
#         '4H'  # 4*uint16 Face bounding box (x1, y1, x2, y2)
#     )

#     label_size = struct.calcsize(label_format)
#     label_values = struct.unpack(label_format, data[:label_size])
#     return label_values

# Define the dtype for the recarray
label_dtype = np.dtype([
    ('distance', np.int8),
    ('eye_openness', np.float16),
    ('drowsiness', np.int8),
    ('phoneuse', np.bool_),
    ('phone_use_conf', np.float16),
    ('passenger', np.bool_, (5,)),
    ('face_landmarks_x', np.uint16, (68,)),
    ('face_landmarks_y', np.uint16, (68,)),
    ('body_keypoints_x', np.uint16, (14,)),
    ('body_keypoints_y', np.uint16, (14,)),
    ('body_keypoints_z', np.uint16, (14,)),
    ('joint_lengths', np.int8, (10,)),
    ('face_bounding_box', np.uint16, (4,))
])

# def create_recarray():
#     # Create an empty recarray
#     return np.recarray(1, dtype=label_dtype)

# def update_recarray(label_recarray, label_data):
#     # Ensure the length of label_data matches the size of the recarray's item
#     assert len(label_data) == label_recarray.itemsize, "Data size mismatch"
#     # Directly copy the data into the recarray's buffer
#     label_recarray.data[:] = label_data

class Client:
    def __init__(self, server_ip, port=65432, frame_width=640, frame_height=480):
        self.server_address = (server_ip, port)
        self.sock = None
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.frame_queue = queue.Queue(maxsize=10)  # Buffer with max size
        self.label_size = struct.calcsize(
                    '=b e b ? e ? ? ? ? ? ' + '68H 68H ' + '14H 14H 14H ' + '10b ' + '4H')
        self.running = True
        self.label_array = None

    def setup_socket(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, True)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2**20)
        self.sock.connect(self.server_address)
        print("Connected to server at {}:{}".format(*self.server_address))

    def receive_frame(self):
        while self.running:
            try:
                # Receive the frame size
                frame_size_data = self.sock.recv(4)
                if not frame_size_data:
                    break
                frame_size = struct.unpack('!I', frame_size_data)[0]

                # Receive the frame data
                frame_data = b''
                while len(frame_data) < frame_size:
                    packet = self.sock.recv(frame_size - len(frame_data))
                    if not packet:
                        break
                    frame_data += packet

                if len(frame_data) != frame_size:
                    break

                # Convert the byte data to a numpy array
                frame = np.frombuffer(frame_data, dtype=np.uint8).reshape((self.frame_height, self.frame_width, 3))

                # Put the frame in the queue if space is available
                if not self.frame_queue.full():
                    self.frame_queue.put(frame)

                # Convert the byte data to a numpy array
                frame = np.frombuffer(frame_data, dtype=np.uint8).reshape((self.frame_height, self.frame_width, 3))
                self.frame_queue.put(frame)

                label_data = self.sock.recv(self.label_size)
                if not label_data:
                    break
                
                self.label_array = np.frombuffer(label_data, dtype=label_dtype)
                
                #print("Received label values:", self.label_array)
                
            except Exception as e:
                print(f"Error receiving data: {e}")
                break

        self.cleanup()

    def display_frame(self):
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=1)
                cv2.imshow('Received Frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False
                    break
            except queue.Empty:
                continue

        self.cleanup()

    def run(self):
        self.setup_socket()

        # Create and start the threads
        receive_thread = threading.Thread(target=self.receive_frame)
        display_thread = threading.Thread(target=self.display_frame)

        receive_thread.start()
        display_thread.start()

        receive_thread.join()
        display_thread.join()

    def cleanup(self):
        if self.sock:
            self.sock.close()
        cv2.destroyAllWindows()
        self.running = False
