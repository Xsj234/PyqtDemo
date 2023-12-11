# coding=gbk //放到文件首行

import sys, math
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import numpy as np
import cv2
import os
import time
import base64
import copy
from PIL import Image, ImageDraw, ImageFont
from queue import LifoQueue
import threading

Decode2Play = LifoQueue()


# lock = threading.Lock()

class cvDecode(QThread):
    def __init__(self):
        super(cvDecode, self).__init__()
        self.threadFlag = 0  # 控制线程退出
        self.rtsp = "rtsp://admin:a1234567@10.112.89.10:554/h264/ch1/main/av_stream"
        self.cap = cv2.VideoCapture(self.rtsp)

    def run(self):
        # print("当前线程cvDecode: self.threadFlag:{}".format(self.threadFlag))
        while self.threadFlag:
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                time.sleep(0.001)  # 控制读取录像的时间，连实时视频的时候改成time.sleep(0.001)，多线程的情况下最好加上，否则不同线程间容易抢占资源

                if ret:
                    # lock.acquire()
                    Decode2Play.put(frame)  # 解码后的数据放到队列中
                    # lock.release()
                del frame  # 释放资源


class play_Work(QThread):
    def __init__(self):
        super(play_Work, self).__init__()
        self.threadFlag = 0  # 控制线程退出
        self.playLabel = QLabel()  # 初始化QLabel对象
        # cv2.namedWindow("test")
        # cv2.resizeWindow("test", 640, 480)

    def run(self):
        while self.threadFlag:
            if not Decode2Play.empty():
                frame = Decode2Play.get()
                while not Decode2Play.empty():
                    Decode2Play.get()

                frame = cv2.resize(frame, (800, 600), cv2.INTER_LINEAR)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                qimg = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)  # 在这里可以对每帧图像进行处理，
                self.playLabel.setPixmap(QPixmap.fromImage(qimg))  # 图像在QLabel上展示
                # cv2.imshow('test', frame)

            time.sleep(0.001)


class MyWindow(QWidget):
    # opencam_complete_signal = pyqtSignal(str)  # 子线程返回摄像头图像到主线程的信号
    def __init__(self):
        super().__init__()
        self.setGeometry(0, 0, 840, 800)
        # self.setFixedSize(1600, 960)

        self.init_ui()

    def init_ui(self):
        self.cam_view = QLabel(self)
        self.cam_view.setStyleSheet("border:2px solid black")
        self.cam_view.setGeometry(20, 20, 800, 600)

        self.open_cam_btn = QPushButton(self)
        # self.pushButton.setStyleSheet("border:2px solid black")
        self.open_cam_btn.setText('播放')
        self.open_cam_btn.setGeometry(380, 700, 100, 30)
        self.open_cam_btn.clicked.connect(self.init_work)

    def init_work(self):
        self.decodework = cvDecode()
        self.decodework.threadFlag = 1
        self.decodework.start()

        self.playwork = play_Work()
        self.playwork.threadFlag = 1
        self.playwork.playLabel = self.cam_view
        self.playwork.start()

    def closeEvent(self, event):
        print("关闭线程")
        # Qt需要先退出循环才能关闭线程
        if self.decodework.isRunning():
            self.decodework.threadFlag = 0
            self.decodework.quit()
        if self.playwork.isRunning():
            self.playwork.threadFlag = 0
            self.playwork.quit()


if __name__ == '__main__':
    app = QApplication(sys.argv)  # 创建对象

    w = MyWindow()
    # 展示窗口
    w.show()

    # 程序进行循环等待状态
    app.exec_()
