import socket
import cv2
import struct
import numpy as np
from datagenerator import DataGenerator, ImageDataGenerator, fake_label_data_generator


def pack_data(image, label_values):
    """
    Pack image and label data for sending over a socket.

    Args:
    image (np.ndarray): Image data in RGB888 format with 640 x 480 resolution.
    label_values (tuple): Tuple containing the label values in the specified format.

    Returns:
    bytes: Packed data ready to be sent over a socket.
    """
    # Ensure the image is in the correct shape and type
    assert image.shape == (480, 640, 3), "Image must be in shape (480, 640, 3)"
    assert image.dtype == np.uint8, "Image must be in RGB888 format (dtype=np.uint8)"

    # Pack the image data
    image_data = image.tobytes()

    # Define the format for the label data
    label_format = (
        'b e b ? e ? ? ? ? ? ' + # Int8, float16, Int8, bool, float16, 5*bool,
        '68H 68H ' + # 68 * uint16 (face landmark X and Y)
        '14H 14H 14H ' +  # 3*14*uint16 (3D body keypoints)
        '10b ' +  # 10*int8 Joint length
        '4H'  # 4*uint16 Face bounding box (x1, y1, x2, y2)
    )

    # Pack the label data
    packed_labels = struct.pack(label_format, *label_values)

    # Combine the image data and label data
    packed_data = struct.pack('!I', len(image_data)) + image_data + packed_labels

    return packed_data


class Server:
    def __init__(self, host_ip, img_dir, label_path, 
                 port=65432, 
                 frame_width=640, 
                 frame_height=480):
        self.server_address = (host_ip, port)
        self.img_dir = img_dir
        self.label_path = label_path
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.sock = None
        self.conn = None
        self.label_generator = fake_label_data_generator()
        self.image_generator = ImageDataGenerator(self.img_dir)
    
    def set_generator(self):    
        self.generator = DataGenerator(self.img_dir, self.label_path)

    def setup_socket(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, True)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 2**20)
        self.sock.bind(self.server_address)
        self.sock.listen(1)
        print("Server listening on {}:{}".format(*self.server_address))

    def accept_connection(self):
        self.conn, addr = self.sock.accept()
        print("Connection from", addr)
        self.conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, True)
        self.conn.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2**20)

    # def send_frame(self, frame):
    #     resized = cv2.resize(frame, (self.frame_width, self.frame_height), interpolation=cv2.INTER_LINEAR)
    #     frame_data = resized.tobytes()
    #     frame_size = len(frame_data)
    #     self.conn.sendall(struct.pack('!I', frame_size))
    #     self.conn.sendall(frame_data)

    # def send_byte_array(self):
    #     int8_values = np.random.randint(-128, 127, 10, dtype=np.int8)
    #     float16_values = np.random.uniform(-1, 1, 10).astype(np.float16)
    #     byte_array = struct.pack('!10b10e', *(int8_values.tolist() + float16_values.tolist()))
    #     self.conn.sendall(byte_array)
        
    def send_data(self, image, label_values):
        packed_data = pack_data(image, label_values)
        self.conn.sendall(packed_data)

    def run(self):
        self.setup_socket()
        self.accept_connection()
        try:
            while True:
                image = next(self.image_generator)
                label_values = next(self.label_generator)
                self.send_data(image, label_values)
        finally:
            self.cleanup()

    def cleanup(self):
        if self.conn:
            self.conn.close()
        if self.sock:
            self.sock.close()

# Example usage
if __name__ == "__main__":
    server = Server(host_ip='127.0.0.1')
    server.run()
