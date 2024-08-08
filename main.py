import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import numpy as np
import socket
import struct
import queue
import threading


def my_exception_hook(exctype, value, traceback):
    # Print the error and traceback
    print(exctype, value, traceback)
    # Call the normal Exception hook after
    sys._excepthook(exctype, value, traceback)


# Back up the reference to the exceptionhook
sys._excepthook = sys.excepthook

# Set the exception hook to our wrapping function
sys.excepthook = my_exception_hook


class MainWindow(QWidget):
    def __init__(self, client, use, *args, **kwargs):
        super(QWidget, self).__init__(*args, **kwargs)
        self.client = client
        self.use = use

        self.setWindowTitle("DEEPINSIGHT")
        self.setGeometry(100, 100, 1920, 1080)
        # 윈도우 프레임영역 제거하려면 아래 주석 제거
        # self.setWindowFlags(Qt.FramelessWindowHint)

        vboxmain = QVBoxLayout()
        vboxmain.setSpacing(0)
        vboxmain.setContentsMargins(0, 0, 0, 0)

        hbox2 = QHBoxLayout()
        hbox2.setSpacing(0)

        vbox2_1 = QVBoxLayout()
        vbox2_1.setSpacing(0)

        vbox2_1_1 = QVBoxLayout()
        vbox2_1_1.setContentsMargins(30, 30, 30, 30)

        cont2_1_1 = QWidget()
        cont2_1_1.setStyleSheet("""
                background-color: rgb(23, 23, 23);
                border-radius: 20px;
            """)

        vbox2_1_1_1 = QVBoxLayout()
        vbox2_1_1_1.setSpacing(0)
        cont2_1_1.setLayout(vbox2_1_1_1)

        lbl_txt_2_1_1_1 = QLabel("Distance")
        lbl_txt_2_1_1_1.setAlignment(Qt.AlignBottom)
        lbl_txt_2_1_1_1.setContentsMargins(20, 0, 0, 0)
        lbl_txt_2_1_1_1.setStyleSheet(
            """
            color: rgb(105, 105, 105);
            font-size: 20px;
            font-weight: bold;
            """
        )
        vbox2_1_1_1.addWidget(lbl_txt_2_1_1_1)
        self.lbl_txt_2_1_1_2 = QLabel("30cm")
        self.lbl_txt_2_1_1_2.setAlignment(Qt.AlignTop)
        self.lbl_txt_2_1_1_2.setContentsMargins(20, 0, 0, 0)
        self.lbl_txt_2_1_1_2.setStyleSheet(
            """
            color: rgb(255, 255, 255);
            font-size: 40px;
            font-weight: bold;
            """
        )
        vbox2_1_1_1.addWidget(self.lbl_txt_2_1_1_2)
        lbl_txt_2_1_1_3 = QLabel("Eye Openness")
        lbl_txt_2_1_1_3.setAlignment(Qt.AlignBottom)
        lbl_txt_2_1_1_3.setContentsMargins(20, 0, 0, 0)
        lbl_txt_2_1_1_3.setStyleSheet(
            """
            color: rgb(105, 105, 105);
            font-size: 20px;
            font-weight: bold;
            """
        )
        vbox2_1_1_1.addWidget(lbl_txt_2_1_1_3)
        self.lbl_txt_2_1_1_4 = QLabel("64%")
        self.lbl_txt_2_1_1_4.setAlignment(Qt.AlignTop)
        self.lbl_txt_2_1_1_4.setContentsMargins(20, 0, 0, 0)
        self.lbl_txt_2_1_1_4.setStyleSheet(
            """
            color: rgb(255, 255, 255);
            font-size: 40px;
            font-weight: bold;
            """
        )
        vbox2_1_1_1.addWidget(self.lbl_txt_2_1_1_4)
        lbl_txt_2_1_1_5 = QLabel("Body Size")
        lbl_txt_2_1_1_5.setAlignment(Qt.AlignBottom)
        lbl_txt_2_1_1_5.setContentsMargins(20, 0, 0, 0)
        lbl_txt_2_1_1_5.setStyleSheet(
            """
            color: rgb(105, 105, 105);
            font-size: 20px;
            font-weight: bold;
            """
        )
        vbox2_1_1_1.addWidget(lbl_txt_2_1_1_5)
        self.lbl_txt_2_1_1_6 = QLabel("46cm")
        self.lbl_txt_2_1_1_6.setAlignment(Qt.AlignTop)
        self.lbl_txt_2_1_1_6.setContentsMargins(20, 0, 0, 0)
        self.lbl_txt_2_1_1_6.setStyleSheet(
            """
            color: rgb(255, 255, 255);
            font-size: 40px;
            font-weight: bold;
            """
        )
        vbox2_1_1_1.addWidget(self.lbl_txt_2_1_1_6)

        vbox2_1.addLayout(vbox2_1_1, stretch=5)
        vbox2_1_1.addWidget(cont2_1_1)

        vbox2_1_2 = QVBoxLayout()
        self.lbl_img_2_1_2 = QLabel()
        self.lbl_img_2_1_2.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.lbl_img_2_1_2.setAlignment(Qt.AlignCenter)
        img2_1_2_1 = QPixmap('icon/Property 1=0, Selected=Off.png')
        img2_1_2_2 = QPixmap('icon/Property 1=0, Selected=On.png')
        self.lbl_img_2_1_2.setPixmap(img2_1_2_1)
        vbox2_1_2.addWidget(self.lbl_img_2_1_2, stretch=3)
        lbl_txt_2_1_2 = QLabel("Drowsiness")
        lbl_txt_2_1_2.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        lbl_txt_2_1_2.setAlignment(Qt.AlignHCenter)
        lbl_txt_2_1_2.setStyleSheet(
            """
            /* color: rgb(105, 105, 105); */
            color: rgb(255, 255, 255);
            font-size: 20px;
            font-weight: bold;
            """
        )
        flg2_1_2 = [1]
        # self.timer2_1_2 = QTimer()
        # self.timer2_1_2.timeout.connect(lambda: self.switch_image(self.lbl_img_2_1_2, lbl_txt_2_1_2, img2_1_2_1, img2_1_2_2, flg2_1_2))
        # self.timer2_1_2.start(1000)
        vbox2_1_2.addWidget(lbl_txt_2_1_2, stretch=1)
        vbox2_1.addLayout(vbox2_1_2, stretch=3)

        self.img_lbl = QLabel()
        self.img_lbl.setAlignment(Qt.AlignCenter)

        vbox2_2 = QVBoxLayout()
        # btn2_2 = QPushButton("이미지 및 메타데이터 실시간 출력")
        # btn2_2.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        vbox2_2.addWidget(self.img_lbl, stretch=2)

        hbox2.addLayout(vbox2_1, stretch=2)
        hbox2.addLayout(vbox2_2, stretch=9)

        hbox3 = QHBoxLayout()
        hbox3.setSpacing(0)

        vbox3_1 = QVBoxLayout()
        self.lbl_img_3_1 = QLabel()
        self.lbl_img_3_1.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.lbl_img_3_1.setAlignment(Qt.AlignCenter)
        img3_1_1 = QPixmap('icon/Property 1=Phone use (90%), Selected=Off.png')
        img3_1_2 = QPixmap('icon/Property 1=Phone use (90%), Selected=On.png')
        self.lbl_img_3_1.setPixmap(img3_1_1)
        vbox3_1.addWidget(self.lbl_img_3_1, stretch=3)
        self.lbl_txt_3_1 = QLabel("Phone use (90%)")
        self.lbl_txt_3_1.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.lbl_txt_3_1.setAlignment(Qt.AlignHCenter)
        self.lbl_txt_3_1.setStyleSheet(
            """
            /* color: rgb(105, 105, 105); */
            color: rgb(255, 255, 255);
            font-size: 20px;
            font-weight: bold;
            """
        )
        flg3_1 = [1]
        # self.timer3_1 = QTimer()
        # self.timer3_1.timeout.connect(lambda: self.switch_image(self.lbl_img_3_1, self.lbl_txt_3_1, img3_1_1, img3_1_2, flg3_1))
        # self.timer3_1.start(1000)
        vbox3_1.addWidget(self.lbl_txt_3_1, stretch=1)
        hbox3.addLayout(vbox3_1, stretch=10)

        vbox3_2 = QVBoxLayout()
        self.lbl_img_3_2 = QLabel()
        self.lbl_img_3_2.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.lbl_img_3_2.setAlignment(Qt.AlignCenter)
        img3_2_1 = QPixmap('icon/Property 1=Empty, Selected=Off.png')
        img3_2_2 = QPixmap('icon/Property 1=Empty, Selected=On.png')
        self.lbl_img_3_2.setPixmap(img3_2_1)
        vbox3_2.addWidget(self.lbl_img_3_2, stretch=3)
        self.lbl_txt_3_2 = QLabel("Empty")
        self.lbl_txt_3_2.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.lbl_txt_3_2.setAlignment(Qt.AlignHCenter)
        self.lbl_txt_3_2.setStyleSheet(
            """
            color: rgb(105, 105, 105);
            font-size: 20px;
            font-weight: bold;
            """
        )
        flg3_2 = [1]
        # self.timer3_2 = QTimer()
        # self.timer3_2.timeout.connect(lambda: self.switch_image(self.lbl_img_3_2, self.lbl_txt_3_2, img3_2_1, img3_2_2, flg3_2))
        # self.timer3_2.start(1000)
        vbox3_2.addWidget(self.lbl_txt_3_2, stretch=1)
        hbox3.addLayout(vbox3_2, stretch=9)

        vbox3_3 = QVBoxLayout()
        self.lbl_img_3_3 = QLabel()
        self.lbl_img_3_3.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.lbl_img_3_3.setAlignment(Qt.AlignCenter)
        img3_3_1 = QPixmap('icon/Property 1=Age 1~6, Selected=Off.png')
        img3_3_2 = QPixmap('icon/Property 1=Age 1~6, Selected=On.png')
        self.lbl_img_3_3.setPixmap(img3_3_1)
        vbox3_3.addWidget(self.lbl_img_3_3, stretch=3)
        self.lbl_txt_3_3 = QLabel("Age 1~6")
        self.lbl_txt_3_3.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.lbl_txt_3_3.setAlignment(Qt.AlignHCenter)
        self.lbl_txt_3_3.setStyleSheet(
            """
            color: rgb(105, 105, 105);
            font-size: 20px;
            font-weight: bold;
            """
        )
        flg3_3 = [1]
        # self.timer3_3 = QTimer()
        # self.timer3_3.timeout.connect(lambda: self.switch_image(self.lbl_img_3_3, self.lbl_txt_3_3, img3_3_1, img3_3_2, flg3_3))
        # self.timer3_3.start(1000)
        vbox3_3.addWidget(self.lbl_txt_3_3, stretch=1)
        hbox3.addLayout(vbox3_3, stretch=9)

        vbox3_4 = QVBoxLayout()
        self.lbl_img_3_4 = QLabel()
        self.lbl_img_3_4.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.lbl_img_3_4.setAlignment(Qt.AlignCenter)
        img3_4_1 = QPixmap('icon/Property 1=AF05, Selected=Off.png')
        img3_4_2 = QPixmap('icon/Property 1=AF05, Selected=On.png')
        self.lbl_img_3_4.setPixmap(img3_4_1)
        vbox3_4.addWidget(self.lbl_img_3_4, stretch=3)
        self.lbl_txt_3_4 = QLabel("AF05")
        self.lbl_txt_3_4.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.lbl_txt_3_4.setAlignment(Qt.AlignHCenter)
        self.lbl_txt_3_4.setStyleSheet(
            """
            color: rgb(105, 105, 105);
            font-size: 20px;
            font-weight: bold;
            """
        )
        flg3_4 = [1]
        # self.timer3_4 = QTimer()
        # self.timer3_4.timeout.connect(lambda: self.switch_image(self.lbl_img_3_4, self.lbl_txt_3_4, img3_4_1, img3_4_2, flg3_4))
        # self.timer3_4.start(1000)
        vbox3_4.addWidget(self.lbl_txt_3_4, stretch=1)
        hbox3.addLayout(vbox3_4, stretch=9)

        vbox3_5 = QVBoxLayout()
        self.lbl_img_3_5 = QLabel()
        self.lbl_img_3_5.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.lbl_img_3_5.setAlignment(Qt.AlignCenter)
        img3_5_1 = QPixmap('icon/Property 1=AM50, Selected=Off.png')
        img3_5_2 = QPixmap('icon/Property 1=AM50, Selected=On.png')
        self.lbl_img_3_5.setPixmap(img3_5_1)
        vbox3_5.addWidget(self.lbl_img_3_5, stretch=3)
        self.lbl_txt_3_5 = QLabel("AM50")
        self.lbl_txt_3_5.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.lbl_txt_3_5.setAlignment(Qt.AlignHCenter)
        self.lbl_txt_3_5.setStyleSheet(
            """
            color: rgb(105, 105, 105);
            font-size: 20px;
            font-weight: bold;
            """
        )
        flg3_5 = [1]
        # self.timer3_5 = QTimer()
        # self.timer3_5.timeout.connect(lambda: self.switch_image(self.lbl_img_3_5, self.lbl_txt_3_5, img3_5_1, img3_5_2, flg3_5))
        # self.timer3_5.start(1000)
        vbox3_5.addWidget(self.lbl_txt_3_5, stretch=1)
        hbox3.addLayout(vbox3_5, stretch=9)

        vbox3_6 = QVBoxLayout()
        self.lbl_img_3_6 = QLabel()
        self.lbl_img_3_6.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.lbl_img_3_6.setAlignment(Qt.AlignCenter)
        img3_6_1 = QPixmap('icon/Property 1=AM95, Selected=Off.png')
        img3_6_2 = QPixmap('icon/Property 1=AM95, Selected=On.png')
        self.lbl_img_3_6.setPixmap(img3_6_1)
        vbox3_6.addWidget(self.lbl_img_3_6, stretch=3)
        self.lbl_txt_3_6 = QLabel("AM95")
        self.lbl_txt_3_6.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.lbl_txt_3_6.setAlignment(Qt.AlignHCenter)
        self.lbl_txt_3_6.setStyleSheet(
            """
            color: rgb(105, 105, 105);
            font-size: 20px;
            font-weight: bold;
            """
        )
        flg3_6 = [1]
        # self.timer3_6 = QTimer()
        # self.timer3_6.timeout.connect(lambda: self.switch_image(self.lbl_img_3_6, self.lbl_txt_3_6, img3_6_1, img3_6_2, flg3_6))
        # self.timer3_6.start(1000)
        vbox3_6.addWidget(self.lbl_txt_3_6, stretch=1)
        hbox3.addLayout(vbox3_6, stretch=9)

        vboxmain.addLayout(hbox2, stretch=8)
        vboxmain.addLayout(hbox3, stretch=3)

        self.setLayout(vboxmain)

        # Start the client threads
        self.client_thread = threading.Thread(target=self.client.receive_frame)
        self.client_thread.start()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_view)
        self.timer.start(30)  # Update every 30 ms

    def update_view(self):
        if not self.client.frame_queue.empty() and not self.client.label_data_queue.empty():
            frame = self.client.frame_queue.get()
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            qimg = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
            pixmap = QPixmap.fromImage(qimg)
            self.img_lbl.setPixmap(pixmap)

            label_data = self.client.label_data_queue.get()
            # for name in label_data.dtype.names:
            #     print(name, label_data[name])
            # print(label_data["distance"], type(label_data["distance"]))
            self.lbl_txt_2_1_1_2.setText(str(label_data["distance"][0]))
            self.lbl_txt_2_1_1_4.setText(str(label_data["eye_openness"][0]*100)+"%")
            # bodysize
            self.lbl_img_2_1_2.setPixmap(QPixmap(f'icon/Property 1={str(label_data["drowsiness"][0])}, Selected=Off.png'))
            self.lbl_img_3_1.setPixmap(QPixmap(f'icon/Property 1=Phone use (90%), Selected={"On" if label_data["phoneuse"][0] else "Off"}.png'))
            self.lbl_txt_3_1.setText("Phone use ("+str(label_data["phone_use_conf"][0] * 100) + "%)")

            self.lbl_img_3_2.setPixmap(QPixmap(f'icon/Property 1=Empty, Selected={"On" if label_data["passenger"][0][0] else "Off"}.png'))
            self.lbl_txt_3_2.setStyleSheet(
                f"""
                {"color: rgb(17, 94, 255);" if label_data["passenger"][0][0] else "color: rgb(105, 105, 105);"}
                font-size: 20px;
                font-weight: bold;
                """
            )

            self.lbl_img_3_3.setPixmap(QPixmap(f'icon/Property 1=Age 1~6, Selected={"On" if label_data["passenger"][0][1] else "Off"}.png'))
            self.lbl_txt_3_3.setStyleSheet(
                f"""
                            {"color: rgb(17, 94, 255);" if label_data["passenger"][0][1] else "color: rgb(105, 105, 105);"}
                            font-size: 20px;
                            font-weight: bold;
                            """
            )

            self.lbl_img_3_4.setPixmap(QPixmap(f'icon/Property 1=AF05, Selected={"On" if label_data["passenger"][0][2] else "Off"}.png'))
            self.lbl_txt_3_4.setStyleSheet(
                f"""
                            {"color: rgb(17, 94, 255);" if label_data["passenger"][0][2] else "color: rgb(105, 105, 105);"}
                            font-size: 20px;
                            font-weight: bold;
                            """
            )

            self.lbl_img_3_5.setPixmap(QPixmap(f'icon/Property 1=AM50, Selected={"On" if label_data["passenger"][0][3] else "Off"}.png'))
            self.lbl_txt_3_5.setStyleSheet(
                f"""
                            {"color: rgb(17, 94, 255);" if label_data["passenger"][0][3] else "color: rgb(105, 105, 105);"}
                            font-size: 20px;
                            font-weight: bold;
                            """
            )

            self.lbl_img_3_6.setPixmap(QPixmap(f'icon/Property 1=AM95, Selected={"On" if label_data["passenger"][0][4] else "Off"}.png'))
            self.lbl_txt_3_6.setStyleSheet(
                f"""
                            {"color: rgb(17, 94, 255);" if label_data["passenger"][0][4] else "color: rgb(105, 105, 105);"}
                            font-size: 20px;
                            font-weight: bold;
                            """
            )

    def closeEvent(self, event):
        self.client.running = False
        self.client_thread.join()
        event.accept()

    def switch_image(self, lbl_img, lbl_txt, img1, img2, flag):
        if flag[0] == 0:
            lbl_img.setPixmap(img1)
            lbl_txt.setStyleSheet(
                """
                color: rgb(105, 105, 105);
                font-size: 20px;
                font-weight: bold;
                """
            )
            flag[0] = 1
        else:
            lbl_img.setPixmap(img2)
            lbl_txt.setStyleSheet(
                """
                color: rgb(17, 94, 255);
                font-size: 20px;
                font-weight: bold;
                """
            )
            flag[0] = 0


class Client:
    def __init__(self, server_ip, port=65432, frame_width=640, frame_height=480):
        self.server_address = (server_ip, port)
        self.sock = None
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.frame_queue = queue.Queue(maxsize=10)
        self.label_data_queue = queue.Queue(maxsize=10)
        self.label_size = struct.calcsize(
            '=b e b ? e ? ? ? ? ? ' + '68H 68H ' + '14H 14H 14H ' + '10b ' + '4H')
        self.running = True
        self.label_array = None

    def setup_socket(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, True)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2 ** 20)
        self.sock.connect(self.server_address)
        print("Connected to server at {}:{}".format(*self.server_address))

    def receive_frame(self):
        while self.running:
            try:
                frame_size_data = self.sock.recv(4)
                if not frame_size_data:
                    break
                frame_size = struct.unpack('!I', frame_size_data)[0]
                frame_data = b''
                while len(frame_data) < frame_size:
                    packet = self.sock.recv(frame_size - len(frame_data))
                    if not packet:
                        break
                    frame_data += packet

                if len(frame_data) != frame_size:
                    break

                frame = np.frombuffer(frame_data, dtype=np.uint8).reshape((self.frame_height, self.frame_width, 3))
                if not self.frame_queue.full():
                    self.frame_queue.put(frame)

                label_data = self.sock.recv(self.label_size)
                if not label_data:
                    break

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
                self.label_array = np.frombuffer(label_data, dtype=label_dtype)
                # print("Received label values:", self.label_array)
                # for name in self.label_array.dtype.names:
                #     print(name, self.label_array[name])

                if not self.label_data_queue.full():
                    self.label_data_queue.put(self.label_array)


            except Exception as e:
                print(f"Error receiving data: {e}")
                break

        self.cleanup()

    def cleanup(self):
        if self.sock:
            self.sock.close()
        self.running = False


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet("""
                QWidget{
                    font-family: 'Malgun Gothic';
                    background-color: rgb(30, 30, 30);
                }
                """)

    client = Client(server_ip='127.0.0.1')
    client.setup_socket()

    use = ["distance", "eye_openness", "drowsiness", "phoneuse", "phone_use_conf", "passenger", "face_landmarks_x", "face_landmarks_y",
           "body_keypoints_x", "body_keypoints_y", "body_keypoints_z", "joint_lengths", "face_bounding_box"]

    window = MainWindow(client, use)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
