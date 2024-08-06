import os
from PIL import Image
import numpy as np

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
class DataGenerator:
    def __init__(self, img_dir, label_path):
        self._img_dir = img_dir
        self._label_path = label_path
        self._label = None
        self.img_generator = ImageDataGenerator(self._img_dir)
        self.init_label()
        
    def init_label(self):
        self._label = self.label_generator(self._label_path)
        fields_2d = [('cnt', 'i4')]
        for part in ['head', 'lshoulder', 'rshoulder', 'lelbow', 'relbow', 'lhand', 'rhand', 'lpelvis', 'rpelvis', 'lknee', 'rknee']:
            fields_2d.append((f"{part}_x", 'f4'))
            fields_2d.append((f"{part}_y", 'f4'))
            fields_2d.append((f"{part}_conf", 'f4'))
            
    def generate(self):
        return (self.image_generator, self.label_generator)
                
    def label_generator(self, label_path):
        """
        Generator that yields labels one by one from a specified file.

        Args:
        label_path (str): Path to the file containing labels.

        Yields:
        str: A label.
        """
        with open(label_path, 'r') as file:
            for line in file:
                yield line.strip()
                
                