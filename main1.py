import sys
from PyQt5.QtWidgets import QLabel, QWidget, QVBoxLayout, QHBoxLayout, QApplication, QSizePolicy
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer, Qt
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
style_105_20 = """
            color: rgb(105, 105, 105);
            font-size: 20px;
            font-weight: bold;
            """
style_255_40 = """
            color: rgb(255, 255, 255);
            font-size: 60px;
            font-weight: bold;
            """
style_255_20 = """
            /* color: rgb(105, 105, 105); */
            color: rgb(255, 255, 255);
            font-size: 20px;
            font-weight: bold;
            """

class MainWindow(QWidget):
    def __init__(self, client, use, *args, **kwargs):
        super(QWidget, self).__init__(*args, **kwargs)
        self.client = client
        self.use = use
        self.draw_alpha = 0.3
        self.box_old = None
        self.bk2d_x_old = None
        self.bk2d_y_old = None
        self.phone_use = None
        
        self.frame_width_resize = 1333 # 1333 -> 16:9
        self.frame_height_resize = 750
        self.fhd_shift_x = 448
        self.fhd_shift_y = 56
        self.draw_conf_thr = 50
        
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
        vbox2_1_1.setContentsMargins(30, 120, 30, 120)

        cont2_1_1 = QWidget()
        cont2_1_1.setStyleSheet("""
                background-color: rgb(23, 23, 23);
                border-radius: 20px;
            """)

        vbox2_1_1_1 = QVBoxLayout()
        vbox2_1_1_1.setSpacing(0)
        cont2_1_1.setLayout(vbox2_1_1_1)

        def add_label_to_layout(layout, text, alignment, margins, style):
            label = QLabel(text)
            label.setAlignment(alignment)
            label.setContentsMargins(*margins)
            label.setStyleSheet(style)
            layout.addWidget(label)
            return label

        # Add the labels to the layout
        add_label_to_layout(vbox2_1_1_1, "Distance", Qt.AlignBottom, (20, 0, 0, 0), style_105_20)
        self.lbl_txt_2_1_1_2 = add_label_to_layout(vbox2_1_1_1, "100cm", Qt.AlignTop, (20, 0, 0, 0), style_255_40)
        add_label_to_layout(vbox2_1_1_1, "Eye Openness", Qt.AlignBottom, (20, 0, 0, 0), style_105_20)
        self.lbl_txt_2_1_1_4 = add_label_to_layout(vbox2_1_1_1, "64%", Qt.AlignTop, (20, 0, 0, 0), style_255_40)
        add_label_to_layout(vbox2_1_1_1, "Body Size", Qt.AlignBottom, (20, 0, 0, 0), style_105_20)
        self.lbl_txt_2_1_1_6 = add_label_to_layout(vbox2_1_1_1, "146cm", Qt.AlignTop, (20, 0, 0, 0), style_255_40)

        vbox2_1.addLayout(vbox2_1_1, stretch=5)
        vbox2_1_1.addWidget(cont2_1_1)

        self.img_lbl = QLabel()
        self.img_lbl.setAlignment(Qt.AlignCenter)

        vbox2_2 = QVBoxLayout()
        vbox2_2.addWidget(self.img_lbl, stretch=2)

        hbox2.addLayout(vbox2_1, stretch=2)
        hbox2.addLayout(vbox2_2, stretch=9)

        def create_label_vbox(img_path, text, text_style):
            vbox = QVBoxLayout()

            lbl_img = QLabel()
            lbl_img.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            lbl_img.setAlignment(Qt.AlignCenter)
            lbl_img.setPixmap(QPixmap(img_path))
            vbox.addWidget(lbl_img, stretch=3)

            lbl_txt = QLabel(text)
            lbl_txt.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            lbl_txt.setAlignment(Qt.AlignHCenter)
            lbl_txt.setStyleSheet(text_style)
            vbox.addWidget(lbl_txt, stretch=1)

            return vbox, lbl_img, lbl_txt

        hbox3 = QHBoxLayout()
        hbox3.setSpacing(0)

        # List of tuples containing image path, text, and stylesheet
        label_data = [
            ('icon/PhoneUseOff.png', 'Phone use (90%)', style_255_20),
            ('icon/PassengerEmptyOff.png', 'Empty', style_105_20),
            ('icon/PassengerKidsOff.png', 'Age 1~6', style_105_20),
            ('icon/PassengerAF05Off.png', 'AF05', style_105_20),
            ('icon/PassengerAM50Off.png', 'AM50', style_105_20),
            ('icon/PassengerAM95Off.png', 'AM95', style_105_20),
        ]

        # Initialize a list to keep track of the text labels
        self.lbl_txt_list = []
        self.lbl_img_list = []

        # Add each label's vbox to hbox3 and store the label in the list
        for img_path, text, text_style in label_data:
            vbox, lbl_img, lbl_txt = create_label_vbox(img_path, text, text_style)
            hbox3.addLayout(vbox, stretch=9)
            self.lbl_txt_list.append(lbl_txt)
            self.lbl_img_list.append(lbl_img)

        vboxmain.addLayout(hbox2, stretch=8)
        vboxmain.addLayout(hbox3, stretch=3)

        self.setLayout(vboxmain)

        # Start the client threads
        self.client_thread = threading.Thread(target=self.client.receive_frame)
        self.client_thread.start()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_view)
        self.timer.start(33)  # Update every 30 ms

    def update_view(self):
        if not self.client.frame_queue.empty() and not self.client.label_data_queue.empty():
            frame = self.client.frame_queue.get()
            if self.frame_width_resize != frame.shape[1] or self.frame_height_resize != frame.shape[0]:
                frame_width_resize_ratio = self.frame_width_resize / frame.shape[1]
                frame_height_resize_ratio = self.frame_height_resize / frame.shape[0]
                frame = cv2.resize(frame, (self.frame_width_resize, self.frame_height_resize))
            
            label_data = self.client.label_data_queue.get()

            h, w = frame.shape
            bytes_per_line = w * 3
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            
            area_lmin = 200
            area_rmax = 1024 -200
            
            # ROI
            cv2.rectangle(frame, (int((self.fhd_shift_x+area_lmin)*frame_width_resize_ratio),
                                  int(self.fhd_shift_y*frame_height_resize_ratio)),
                          (int((1920 - self.fhd_shift_x - area_lmin)*frame_width_resize_ratio),
                                  int(self.frame_height_resize)), 
                          (105, 105, 105), 5)
            
            if all(label_data["body_keypoints2d"][0][0] != -1):
                # body_keypoints
                bk2d_x = np.round((label_data["body_keypoints2d"][0][0] + self.fhd_shift_x) * frame_width_resize_ratio).astype(int)
                bk2d_y = np.round((label_data["body_keypoints2d"][0][1] + self.fhd_shift_y) * frame_height_resize_ratio).astype(int)
                
                if self.bk2d_x_old is None:
                    self.bk2d_x_old = bk2d_x
                    self.bk2d_y_old = bk2d_y
                
                bk2d_x = np.round(self.draw_alpha * bk2d_x + (1 - self.draw_alpha) * self.bk2d_x_old).astype(int)
                bk2d_y = np.round(self.draw_alpha * bk2d_y + (1 - self.draw_alpha) * self.bk2d_y_old).astype(int)
                
                self.bk2d_x_old = bk2d_x
                self.bk2d_y_old = bk2d_y
                # body_keypoints_z = np.round(label_data["body_keypoints_z"][0] * frame_z_resize_ratio).astype(int)
                # print("after2", bk2d_x)
                bk_3dx, bk_3dy, bk_3dz = label_data["body_keypoints3d"][0]
                
                color = (24, 24, 244)  # BGR
                color2 = (46, 234, 255)
                radius = 5

                for connection in connections_2d:
                    if label_data["body_keypoints2d_conf"][0][connection[0]] > self.draw_conf_thr and \
                        label_data["body_keypoints2d_conf"][0][connection[1]] > self.draw_conf_thr:
                        cv2.line(frame, (bk2d_x[connection[0]], bk2d_y[connection[0]]),
                                        (bk2d_x[connection[1]], bk2d_y[connection[1]]), 
                                        (0, 255, 0), 2)

                # Draw the keypoints
                for i in range(len(bk2d_x)):
                    if label_data["body_keypoints2d_conf"][0][i] > self.draw_conf_thr:
                        cv2.circle(frame, (bk2d_x[i], bk2d_y[i]), radius, color2, -1, cv2.LINE_AA)
                        cv2.putText(frame, f"{bk_3dx[sort2d_to_3d[i]]}  {bk_3dy[sort2d_to_3d[i]]}  {bk_3dz[sort2d_to_3d[i]]}", 
                                    (bk2d_x[i], bk2d_y[i]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            # Face bounding box
            bbox = label_data["face_bounding_box"][0]
            if bbox is not None and bbox[0] > area_lmin and bbox[3] < area_rmax:
                if self.box_old is None:
                    self.box_old = bbox
                bbox = self.draw_alpha * bbox + (1 - self.draw_alpha) * self.box_old
                self.box_old = bbox
                
            if not label_data["face_landmarks_x"][0][0] == -1:
                face_landmarks_x = np.round((label_data["face_landmarks_x"][0] + self.fhd_shift_x) * frame_width_resize_ratio).astype(int)
                face_landmarks_y = np.round((label_data["face_landmarks_y"][0] + self.fhd_shift_y) * frame_height_resize_ratio).astype(int)

                # radius = 2
                for i in range(len(face_landmarks_x)):
                    cv2.circle(frame, (face_landmarks_x[i], face_landmarks_y[i]), 2, (24, 24, 244), -1, cv2.LINE_AA)

                # bbox = label_data["face_bounding_box"][0].astype(int)
                ptl = np.array([(bbox[0] + self.fhd_shift_x)* frame_width_resize_ratio,
                                (bbox[1] + self.fhd_shift_y)* frame_height_resize_ratio]).astype(int)
                pbr = np.array([(bbox[2] + self.fhd_shift_x)* frame_width_resize_ratio,
                                (bbox[3] + self.fhd_shift_y)* frame_height_resize_ratio]).astype(int)
                cv2.rectangle(frame, ptl, pbr, (46, 234, 255), 5)
                
                self.lbl_txt_2_1_1_4.setText(str(label_data["eye_openness"][0]) +"%")
                # self.lbl_img_2_1_2.setPixmap(QPixmap(f'icon/Drowsiness{str(label_data["drowsiness"][0])}ON.png'))
                
                # Phone use
                if self.phone_use is None:
                    self.phone_use = 1 if label_data["phoneuse"][0] == 1 else 0
                _phone_use = 1 if label_data["phoneuse"][0] == 1 else 0
                self.phone_use = self.draw_alpha * _phone_use + (1 - self.draw_alpha) * self.phone_use
                is_using_phone = self.phone_use > 0.8
                self.lbl_txt_list[0].setPixmap(QPixmap(f'icon/PhoneUse{"On" if is_using_phone else "Off"}.png'))
                # self.lbl_img_3_1.setPixmap(QPixmap(f'icon/PhoneUse{"On" if is_using_phone else "Off"}.png'))
                if is_using_phone:
                    phone_conf = label_data["phone_use_conf"][0]
                else:
                    phone_conf = 0
                # self.lbl_txt_3_1.
                self.lbl_txt_list[0].setText("Phone use (" + str(round(phone_conf,2)) + "%)")                  
                
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
            self.lbl_txt_2_1_1_2.setStyleSheet(f"color: {color}; font-size: 60px; font-weight: bold;")
            self.lbl_txt_2_1_1_6.setText(str(label_data['height'][0]))
            # bodysize
                        
            self.update_labels(label_data["passenger"][0])
                
    def set_label(self, label_img, label_txt, icon_name, condition):
        label_img.setPixmap(QPixmap(f'icon/{icon_name}{"On" if condition else "Off"}.png'))
        label_txt.setStyleSheet(
            f"""
            {"color: rgb(17, 94, 255);" if condition else "color: rgb(105, 105, 105);"}
            font-size: 20px;
            font-weight: bold;
            """
        )

    def update_labels(self, passenger_status):
        labels_info = [
            ('PassengerEmpty',passenger_status == 0),
            ('PassengerKids', passenger_status == 1),
            ('PassengerAF05', passenger_status == 2),
            ('PassengerAM50', passenger_status == 3),
            ('PassengerAM95', passenger_status == 4),
        ]
        
        for label_img, label_txt, (icon_name, condition) in zip(self.lbl_img_list[1:], 
                                                                self.lbl_img_list[1:], labels_info):
            self.set_label(label_img, label_txt, icon_name, condition)

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
    host_ip = '192.168.100.3'
    client = Client(server_ip = host_ip, 
                    camera_height=args.height, 
                    camera_pitch=args.pitch,
                    height_factor=1.0) #
    # client.setup_socket()
    # client.accept_connection()

    use = ["distance", "eye_openness", "drowsiness", "phoneuse", "phone_use_conf", "passenger", "face_landmarks_x", "face_landmarks_y",
           "body_keypoints_x", "body_keypoints_y", "body_keypoints_z", "joint_lengths", "face_bounding_box"]

    window = MainWindow(client, use)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

