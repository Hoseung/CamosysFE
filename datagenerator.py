import os
from PIL import Image
import numpy as np
import cv2

class ImageDataGenerator:
    def __init__(self, directory):
        self.directory = directory
        self.files = [f for f in os.listdir(directory) if f.endswith('.png')]
        if not self.files:
            raise Exception("No images found in the directory")
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.files):
            self.index = 0  # Loop over the dataset

        image_path = os.path.join(self.directory, self.files[self.index])
        image = Image.open(image_path)
        image = image.resize((640, 480))
        image_data = np.array(image)
        self.index += 1

        return image_data

class CameraDataGenerator:
    def __init__(self, camera_index=0):
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise Exception("Could not open video device")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def __iter__(self):
        return self

    def __next__(self):
        ret, frame = self.cap.read()
        if not ret:
            raise Exception("Could not read frame from camera")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (640, 480))
        return frame

    def release(self):
        self.cap.release()

def fake_label_data_generator():
    while True:
        distance = np.random.randint(1, 127, dtype=np.int8)
        eye_openness = np.float16(np.random.uniform(0, 1))
        drowsiness = np.random.randint(0, 6, dtype=np.int8)
        phoneuse = bool(np.random.randint(0, 2))
        phone_use_conf = np.float16(np.random.uniform(0, 1))
        passenger = [bool(np.random.randint(0, 2)) for _ in range(5)]
        face_landmarks_x = np.random.randint(0, 640, 68, dtype=np.uint16)
        face_landmarks_y = np.random.randint(0, 480, 68, dtype=np.uint16)
        body_keypoints = [np.random.randint(50, 600, 14, dtype=np.uint16) for _ in range(3)]
        joint_lengths = np.random.randint(5, 50, 10, dtype=np.int8)
        face_bounding_box = np.random.randint(10, 200, 4, dtype=np.uint16)
        face_bounding_box[2:] = face_bounding_box[:2] + np.random.randint(10, 20, 2, dtype=np.uint16)

        label_values = (
            distance, eye_openness, drowsiness, phoneuse, phone_use_conf, *passenger,
            *face_landmarks_x, *face_landmarks_y,
            *body_keypoints[0], *body_keypoints[1], *body_keypoints[2],
            *joint_lengths,
            *face_bounding_box
        )

        yield label_values