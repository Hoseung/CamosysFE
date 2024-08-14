import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import numpy as np
import threading
import cv2
import argparse
from client import Client

def my_exception_hook(exctype, value, traceback):
    # Print the error and traceback
    print(exctype, value, traceback)
    # Call the normal Exception hook after
    sys._excepthook(exctype, value, traceback)


# Back up the reference to the exceptionhook
sys._excepthook = sys.excepthook

# Set the exception hook to our wrapping function
sys.excepthook = my_exception_hook


connections_2d = [
    (0, 2), (0, 1), (2, 4), (4, 6),
    (1, 3), (3, 5), (2, 8), (1, 2),
    (1, 7), (7, 8), (8, 10), (10, 12),
    (7, 9), (9, 11)
]

sort2d_to_3d = (8,9,12,10,13,11,14,4,1,5,2,6,3)

class MainWindow(QWidget):
    def __init__(self, client, use, *args, **kwargs):
        super(QWidget, self).__init__(*args, **kwargs)
        self.client = client
        self.use = use
        # self.post = Postprocess()

        self.setWindowTitle("Ti DEMO")
        self.setGeometry(100, 100, 1920, 1080)
        # 윈도우 프레임영역 제거하려면 아래 주석 제거
        self.setWindowFlags(Qt.FramelessWindowHint)

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
        self.lbl_txt_2_1_1_2 = QLabel("100cm")
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
        self.lbl_txt_2_1_1_6 = QLabel("146cm")
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
            frame_width_resize = 1333 # 1333 -> 16:9
            frame_height_resize = 750
            
            fhd_shift_x = 448
            fhd_shift_y = 56
            draw_conf_thr = 50

            frame = self.client.frame_queue.get()
            if frame_width_resize != frame.shape[1] or frame_height_resize != frame.shape[0]:
                frame_width_resize_ratio = frame_width_resize / frame.shape[1]
                frame_height_resize_ratio = frame_height_resize / frame.shape[0]
                frame = cv2.resize(frame, (frame_width_resize, frame_height_resize))
            
            label_data = self.client.label_data_queue.get()

            h, w = frame.shape
            bytes_per_line = w * 3
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            
            
            area_lmin = 100
            area_rmax = 1024 - 100
            
            box = label_data["face_bounding_box"][0]
            cv2.rectangle(frame, (int((fhd_shift_x+area_lmin)*frame_width_resize_ratio),
                                  int(fhd_shift_y*frame_height_resize_ratio)),
                          (int((1920 - fhd_shift_x - area_lmin)*frame_width_resize_ratio),
                                  int(frame_height_resize)), 
                          (105, 105, 105), 3)
            if box is not None and box[0] > area_lmin and box[3] < area_rmax:
                    
                # FHD 이미지를 resize 할때 쓴 ratio를 얻었으므로, 
                # 좌표로 FHD 기준으로 바꿔준 뒤 ratio 사용. 
                
                # body_keypoints
                bk2d_x = np.round((label_data["body_keypoints2d"][0][0] + fhd_shift_x) * frame_width_resize_ratio).astype(int)
                bk2d_y = np.round((label_data["body_keypoints2d"][0][1] + fhd_shift_y) * frame_height_resize_ratio).astype(int)
                # body_keypoints_z = np.round(label_data["body_keypoints_z"][0] * frame_z_resize_ratio).astype(int)
                # print("after2", bk2d_x)
                bk_3dx = label_data["body_keypoints3d"][0][0]
                bk_3dy = label_data["body_keypoints3d"][0][1]
                bk_3dz = label_data["body_keypoints3d"][0][2]
                
                color = (24, 24, 244)  # BGR
                color2 = (46, 234, 255)
                radius = 5

                for connection, conf in zip(connections_2d, label_data["body_keypoints2d_conf"][0]):
                    if conf > draw_conf_thr:
                        cv2.line(frame, (bk2d_x[connection[0]], bk2d_y[connection[0]]),
                                        (bk2d_x[connection[1]], bk2d_y[connection[1]]), 
                                        (0, 255, 0), 2)

                # Draw the keypoints
                for i in range(len(bk2d_x)):
                    if label_data["body_keypoints2d_conf"][0][i] > draw_conf_thr:
                        cv2.circle(frame, (bk2d_x[i], bk2d_y[i]), radius, color2, -1, cv2.LINE_AA)
                        cv2.putText(frame, f"{bk_3dx[sort2d_to_3d[i]]}  {bk_3dy[sort2d_to_3d[i]]}  {bk_3dz[sort2d_to_3d[i]]}", 
                                    (bk2d_x[i], bk2d_y[i]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

                # face_landmarks
                face_landmarks_x = np.round((label_data["face_landmarks_x"][0] + fhd_shift_x) * frame_width_resize_ratio).astype(int)
                face_landmarks_y = np.round((label_data["face_landmarks_y"][0] + fhd_shift_y) * frame_height_resize_ratio).astype(int)

                radius = 2

                for i in range(len(face_landmarks_x)):
                    cv2.circle(frame, (face_landmarks_x[i], face_landmarks_y[i]), radius, color, -1, cv2.LINE_AA)
                bbox = label_data["face_bounding_box"][0].astype(int)
                ptl = np.array([(bbox[0] + fhd_shift_x)* frame_width_resize_ratio,
                                (bbox[1] + fhd_shift_y)* frame_height_resize_ratio]).astype(int)
                pbr = np.array([(bbox[2] + fhd_shift_x)* frame_width_resize_ratio,
                                (bbox[3] + fhd_shift_y)* frame_height_resize_ratio]).astype(int)
                cv2.rectangle(frame, ptl, pbr, (46, 234, 255), 5)

            qimg = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
            #qimg = QImage(frame.data, w, h, bytes_per_line, QImage.Format_Grayscale8).rgbSwapped()

            pixmap = QPixmap.fromImage(qimg)
            self.img_lbl.setPixmap(pixmap)

            distance_value = label_data["distance"][0]
            self.lbl_txt_2_1_1_2.setText(str(distance_value)+ "cm")
            
            if distance_value < 1:
                color = "rgb(105, 105, 105)"
            elif distance_value < 50:
                color = "red"
            else:
                color = "rgb(255,255,255)"

            # Set the color using a stylesheet
            self.lbl_txt_2_1_1_2.setStyleSheet(f"color: {color}; font-size: 40px; font-weight: bold;")
            self.lbl_txt_2_1_1_4.setText(str(label_data["eye_openness"][0]) +"%")
            self.lbl_txt_2_1_1_6.setText(str(label_data['height'][0]))
            # bodysize
            self.lbl_img_2_1_2.setPixmap(QPixmap(f'icon/Property 1={str(label_data["drowsiness"][0])}, Selected=Off.png'))
            self.lbl_img_3_1.setPixmap(QPixmap(f'icon/Property 1=Phone use (90%), Selected={"On" if label_data["phoneuse"][0] else "Off"}.png'))
            self.lbl_txt_3_1.setText("Phone use (" + str(round(label_data["phone_use_conf"][0], 2)) + "%)")
            self.lbl_img_3_2.setPixmap(QPixmap(f'icon/Property 1=Empty, Selected={"On" if label_data["passenger"][0] == 0 else "Off"}.png'))
            self.lbl_txt_3_2.setStyleSheet(
                f"""
                {"color: rgb(17, 94, 255);" if label_data["passenger"][0] == 0 else "color: rgb(105, 105, 105);"}
                font-size: 20px;
                font-weight: bold;
                """
            )

            self.lbl_img_3_3.setPixmap(QPixmap(f'icon/Property 1=Age 1~6, Selected={"On" if label_data["passenger"][0] == 1 else "Off"}.png'))
            self.lbl_txt_3_3.setStyleSheet(
                f"""
                            {"color: rgb(17, 94, 255);" if label_data["passenger"][0] == 1 else "color: rgb(105, 105, 105);"}
                            font-size: 20px;
                            font-weight: bold;
                            """
            )

            self.lbl_img_3_4.setPixmap(QPixmap(f'icon/Property 1=AF05, Selected={"On" if label_data["passenger"][0] == 2 else "Off"}.png'))
            self.lbl_txt_3_4.setStyleSheet(
                f"""
                            {"color: rgb(17, 94, 255);" if label_data["passenger"][0] == 2 else "color: rgb(105, 105, 105);"}
                            font-size: 20px;
                            font-weight: bold;
                            """
            )

            self.lbl_img_3_5.setPixmap(QPixmap(f'icon/Property 1=AM50, Selected={"On" if label_data["passenger"][0] == 3 else "Off"}.png'))
            self.lbl_txt_3_5.setStyleSheet(
                f"""
                            {"color: rgb(17, 94, 255);" if label_data["passenger"][0] == 3 else "color: rgb(105, 105, 105);"}
                            font-size: 20px;
                            font-weight: bold;
                            """
            )

            self.lbl_img_3_6.setPixmap(QPixmap(f'icon/Property 1=AM95, Selected={"On" if label_data["passenger"][0] == 4 else "Off"}.png'))
            self.lbl_txt_3_6.setStyleSheet(
                f"""
                            {"color: rgb(17, 94, 255);" if label_data["passenger"][0] == 4 else "color: rgb(105, 105, 105);"}
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

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("height", type=float, default=1.4)
    parser.add_argument("pitch", type=float, default=-10)
    args=parser.parse_args()
    
    app = QApplication(sys.argv)
    app.setStyleSheet("""
                QWidget{
                    font-family: 'Malgun Gothic';
                    background-color: rgb(30, 30, 30);
                }
                """)
    host_ip = ['169.254.244.73','169.254.31.226'][0]
    client = Client(server_ip = host_ip, 
                    camera_height=args.height, 
                    camera_pitch=args.pitch) # 
    # client.setup_socket()
    # client.accept_connection()

    use = ["distance", "eye_openness", "drowsiness", "phoneuse", "phone_use_conf", "passenger", "face_landmarks_x", "face_landmarks_y",
           "body_keypoints_x", "body_keypoints_y", "body_keypoints_z", "joint_lengths", "face_bounding_box"]

    window = MainWindow(client, use)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

