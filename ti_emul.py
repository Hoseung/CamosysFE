import socket
import struct
import numpy as np
import cv2
import threading
import queue
from datagenerator import FakeLabelGenerator

class TiEmulator:
    """
    Receive frames and emit label data. 
    This class is used to emulate the behavior of the TI device.
    """
    def __init__(self, server_ip, port=65432, frame_width=1080, frame_height=1080):
        self.server_address = (server_ip, port)
        self.sock = None
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.frame_queue = queue.Queue(maxsize=10)  # Buffer with max size
        self.frame_size = frame_width*frame_height # 1 MB
        
        #struct.calcsize(
        #            '=b e b ? e ? ? ? ? ? ' + '68H 68H ' + '14H 14H 14H ' + '10b ' + '4H')
        self.running = True
        self.label_generator = FakeLabelGenerator()

    def setup_socket(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        #self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, True)
        #self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2**20)
        self.sock.connect(self.server_address)
        print("Connected to server at {}:{}".format(*self.server_address))

    # def send_data(self, label_values):
    #     #self.conn.sendall(pack_data(label_values))
    #     self.sock.sendall(pack_data(label_values))

    def receive_frame(self):
        while self.running:
            try:
                # Receive the frame size
                frame_data = b''
                while len(frame_data) < self.frame_size:
                    packet = self.sock.recv(self.frame_size - len(frame_data))
                    if not packet:
                        break
                    frame_data += packet
                    #print("RECEIVING DATA", len(frame_data), self.frame_size)

                if len(frame_data) != self.frame_size:
                    break
                print("Received frame data")
                # Convert the byte data to a numpy array
                frame = np.frombuffer(frame_data, dtype=np.uint8)
                frame = frame.reshape((self.frame_height, self.frame_width))

                print("Frame shape:", frame.shape)
                
                # Put the frame in the queue if space is available
                if not self.frame_queue.full():
                    self.frame_queue.put(frame)

                label_data_bytes = next(self.label_generator)
                print("label_data size:", len(label_data_bytes))
                self.sock.sendall(label_data_bytes)
                
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


if __name__ == '__main__':
    host_ip = ['169.254.31.226', '169.254.59.105', '169.254.244.73', '127.0.0.1'][2]
    emulator = TiEmulator(host_ip, frame_height=1080, frame_width=1080)
    emulator.run()
