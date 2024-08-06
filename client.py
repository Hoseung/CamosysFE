import socket
import struct
import numpy as np
import cv2

class Client:
    def __init__(self, server_ip, port=65432, 
                 frame_width=640, 
                 frame_height=480):
        self.server_address = (server_ip, port)
        self.sock = None
        self._frame_width = frame_width
        self._frame_height = frame_height

    def setup_socket(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, True)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2**20)
        self.sock.connect(self.server_address)
        print("Connected to server at {}:{}".format(*self.server_address))

    def receive_frame(self):
        # Receive the frame size
        frame_size_data = self.sock.recv(4)
        if not frame_size_data:
            return None
        frame_size = struct.unpack('!I', frame_size_data)[0]

        # Receive the frame data
        frame_data = b''
        while len(frame_data) < frame_size:
            packet = self.sock.recv(frame_size - len(frame_data))
            if not packet:
                return None
            frame_data += packet

        # Convert the byte data to a numpy array
        frame = np.frombuffer(frame_data, dtype=np.uint8).reshape((self._frame_height, self._frame_width, 3))

        return frame

    def receive_byte_array(self):
        # Receive the byte array (20 bytes: 10 int8 + 10 float16)
        byte_array_data = self.sock.recv(30)
        if not byte_array_data:
            return None
        byte_array = struct.unpack('!10b10e', byte_array_data)
        return byte_array

    def run(self):
        self.setup_socket()
        try:
            while True:
                frame = self.receive_frame()
                if frame is None:
                    print("Error: Could not receive frame")
                    break

                # Display the received frame
                cv2.imshow('Received Frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                byte_array = self.receive_byte_array()
                if byte_array is None:
                    print("Error: Could not receive byte array")
                    break

                print("Received byte array:", byte_array)

        finally:
            self.cleanup()

    def cleanup(self):
        if self.sock:
            self.sock.close()
        cv2.destroyAllWindows()