import os
from PIL import Image
import numpy as np
import cv2
import struct

data_format = { 
    "distance": np.int16,
    "eye_openness": np.int8,
    "drowsiness": np.int8,
    "phoneuse": np.int8,
    "phone_use_conf": np.int8,
    "height": np.uint8,
    "passenger": np.int8, 
    "face_landmarks_x": (np.uint16, 68),  # 68 elements of uint16
    "face_landmarks_y": (np.uint16, 68),  # 68 elements of uint16
    "body_keypoints3d": (np.uint16, 14, 3),  # 14 elements of uint16 in 3D
    "body_keypoints2d": (np.uint16, 14, 2),  # 14 elements of uint16 in 3D
    "body_keypoints2d_conf": (np.uint8, 14),  # 14 elements of uint16 in 3D
    "face_bounding_box": (np.uint16, 4),  # 4 elements of uint16
}

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

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    def __iter__(self):
        return self

    def __next__(self):
        ret, frame = self.cap.read()
        if not ret:
            raise Exception("Could not read frame from camera")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (1920, 1080))
        return frame

    def release(self):
        self.cap.release()


class FakeLabelGenerator():
    def __init__(self):
        # Define the data format as a dictionary
        self.data_format = data_format
        self.type_map = {
            np.int8: 'b',
            np.uint8: 'B',
            np.int16: 'h',
            np.uint16: 'H',
            np.int32: 'i',
            np.uint32: 'I',
            np.float32: 'f',
            np.float64: 'd',
        }
        self.label_format = self.create_format_string(self.data_format)
    
    # Calculate the byte size directly from the data format dictionary
    def calculate_byte_size(self, data_format):
        total_size = 0
        for key, value in self.data_format.items():
            if isinstance(value, tuple):
                dtype = value[0]
                num_elements = np.prod(value[1:])  # Multiply all dimensions to get the total number of elements
            else:
                dtype = value
                num_elements = 1
            
            element_size = np.dtype(dtype).itemsize  # Get the byte size of the data type
            total_size += element_size * num_elements

        return total_size

    def create_format_string(self, data_format):
        format_string = '='  # Start with '=' for standard size and alignment
        for key, value in self.data_format.items():
            if isinstance(value, tuple):  # Array of elements
                dtype = value[0]
                num_elements = np.prod(value[1:])  # Multiply all dimensions to get the total number of elements
                format_string += f'{num_elements}{self.type_map[dtype]} '
            else:  # Single element
                dtype = value
                format_string += f'{self.type_map[dtype]} '
        return format_string.strip()
        

    def __iter__(self):
        return self  # Return the iterator object itself

    def __next__(self):
        data = (
            np.int16(np.random.randint(1, 127, dtype=self.data_format["distance"])),
            np.int8(np.random.uniform(0, 1)), # eye_openness
            np.int8(np.random.randint(0, 6, dtype=self.data_format["drowsiness"])),
            np.int8(np.random.randint(0, 2, dtype=self.data_format["phoneuse"])),
            np.int8(np.random.uniform(0, 1)), # phone_use_conf
            np.uint8(np.random.randint(0, 200)), # height
            np.int8(np.random.randint(0, 2)), # passenger
            *np.random.randint(0, 640, self.data_format["face_landmarks_x"][1], dtype=self.data_format["face_landmarks_x"][0]),
            *np.random.randint(0, 480, self.data_format["face_landmarks_y"][1], dtype=self.data_format["face_landmarks_y"][0]),
            *np.random.randint(50, 600, self.data_format["body_keypoints3d"][1], dtype=self.data_format["body_keypoints3d"][0]),
            *np.random.randint(50, 600, self.data_format["body_keypoints3d"][1], dtype=self.data_format["body_keypoints3d"][0]),
            *np.random.randint(50, 600, self.data_format["body_keypoints3d"][1], dtype=self.data_format["body_keypoints3d"][0]),
            *np.random.randint(50, 600, self.data_format["body_keypoints2d"][1], dtype=self.data_format["body_keypoints2d"][0]),
            *np.random.randint(50, 600, self.data_format["body_keypoints2d"][1], dtype=self.data_format["body_keypoints2d"][0]),
            *np.random.randint(0, 100, self.data_format["body_keypoints2d_conf"][1], dtype=self.data_format["body_keypoints2d"][0]),
            *np.random.randint(10, 600, self.data_format["face_bounding_box"][1], dtype=self.data_format["face_bounding_box"][0]),
        )

        return self.pack_data(data)
            
    def pack_data(self, label_values):
        """
        Pack image and label data for sending over a socket.

        Args:
        image (np.ndarray): Image data in GRAYSCALE format with 1080 x 1080 resolution.
        label_values (tuple): Tuple containing the label values in the specified format.

        Returns:
        bytes: Packed data ready to be sent over a socket.
        """

        # Pack the label data
        packed_labels = struct.pack(self.label_format, *label_values)

        return packed_labels