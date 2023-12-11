import sys
import cv2
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget
import numpy as np
###  多线程
class VideoThread(QThread):
    image_data = pyqtSignal(np.ndarray)

    def __init__(self, rtsp_url):
        super().__init__()
        self.rtsp_url = rtsp_url
        self.cap = cv2.VideoCapture(self.rtsp_url)

    def run(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.image_data.emit(frame)
        self.cap.release()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)

        central_widget = QWidget()
        central_layout = QVBoxLayout()
        central_layout.addWidget(self.video_label)
        central_widget.setLayout(central_layout)

        self.setCentralWidget(central_widget)

        self.video_thread = VideoThread(rtsp_url='rtsp://admin:a1234567@10.112.89.10:554/h264/ch1/main/av_stream')
        self.video_thread.image_data.connect(self.update_video_frame)
        self.video_thread.start()

    def update_video_frame(self, frame):
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(q_image)
        self.video_label.setPixmap(pixmap)

    def closeEvent(self, event):
        self.video_thread.quit()
        self.video_thread.wait()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.setGeometry(100, 100, 800, 600)
    window.show()
    sys.exit(app.exec_())
