import sys
import cv2
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget
from PyQt5.QtGui import QImage, QPixmap

class ImageViewer(QWidget):
    def __init__(self, image_path):
        super().__init__()
        self.setWindowTitle("OpenCV and PyQt Test")

        # Load image using OpenCV
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Image not found or unable to load.")

        # Convert the image to RGB format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Get image dimensions
        height, width, channel = image.shape
        bytes_per_line = 3 * width

        # Convert the image to a QImage
        q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)

        # Set the QImage in a QLabel
        image_label = QLabel(self)
        image_label.setPixmap(QPixmap.fromImage(q_image))

        # Set up the layout
        layout = QVBoxLayout()
        layout.addWidget(image_label)
        self.setLayout(layout)

if __name__ == "__main__":
    # Create the application
    app = QApplication(sys.argv)

    # Create the main window with an image path
    image_path = "../captured_pngs/frame_1.png"  # Replace with the path to your image file
    viewer = ImageViewer(image_path)
    viewer.show()

    # Execute the application
    sys.exit(app.exec_())
