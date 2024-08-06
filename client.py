import socket
import struct
import numpy as np
import cv2
import threading
import queue

class Client:
    def __init__(self, server_ip, port=65432):
        self.server_address = (server_ip, port)
        self.sock = None
        self.frame_queue = queue.Queue()
        self.running = True

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
                frame = np.frombuffer(frame_data, dtype=np.uint8).reshape((480, 640, 3))

                # Put the frame in the queue
                self.frame_queue.put(frame)

                # Receive and print the byte array
                byte_array_data = self.sock.recv(30)
                if not byte_array_data:
                    break
                byte_array = struct.unpack('!10b10e', byte_array_data)
                print("Received byte array:", byte_array)

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
