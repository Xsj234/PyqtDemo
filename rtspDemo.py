# coding=gbk //放到文件首行
import sys
import time
from queue import LifoQueue
import cv2
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from rtspUI import Ui_MainWindow

# 使用了queue模块中的 Last-In-First-Out，LIFO后进先出的队列
# queue提供了在多线程情况下进行模块线程安全的队列操作的工具。
# LifoQueue是queue模块中的一种队列类型，表示后进先出的队列，和栈（stack）的行为。
# Decode2Play是一个队列对象，你可以用它进行队列操作，比如放入元素（put）和取出元素（get）等。
Decode2Play = LifoQueue()

class cvDecode(QThread):
    def __init__(self):
        # 自动调用这个方法来初始化该对象
        super(cvDecode, self).__init__()
        self.threadFlag = 0  # 控制线程退出
        self.rtsp = "rtsp://admin:buptkyl338@10.112.89.82/h264/ch1/main/av_stream"
        self.cap = cv2.VideoCapture(self.rtsp)

    def run(self):

        while self.threadFlag:
            if self.cap.isOpened():
                ret, frame = self.cap.read()  # ret返回True or Flase frame返回numpy数组
                time.sleep(0.001)  # 控制读取录像的时间，连实时视频的时候改成time.sleep(0.001)，多线程的情况下最好加上，否则不同线程间容易抢占资源

                if ret:
                    # lock.acquire()
                    Decode2Play.put(frame)  # 解码后的数据放到队列中
                    print(Decode2Play.qsize()) #查看
                    # lock.release()
                del frame  # 释放资源



class play_Work(QThread):
    def __init__(self):
        super(play_Work, self).__init__()
        self.threadFlag = 0  # 控制线程退出
        self.playLabel = QLabel()  # 初始化QLabel对象

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
            time.sleep(0.001)


class MyMainWindow(QMainWindow, Ui_MainWindow):  # 继承 QMainWindow类和 Ui_MainWindow界面类

    def __init__(self, parent=None):
        super(MyMainWindow, self).__init__(parent)  # 初始化父类
        self.setupUi(self)  # 继承 Ui_MainWindow 界面类

    def init_work(self):
        print("打开线程")
        self.decodework = cvDecode()
        self.decodework.threadFlag = 1
        self.decodework.start()

        self.playwork = play_Work()
        self.playwork.threadFlag = 1
        self.playwork.playLabel = self.label
        self.playwork.start()

    def closeEvent(self, event):
        print("关闭线程")
        # Qt需要先退出循环才能关闭线程
        if self.decodework.isRunning():
            self.decodework.threadFlag = 0
            self.decodework.quit()
            print("写入结束")
        if self.playwork.isRunning():
            self.playwork.threadFlag = 0
            self.playwork.quit()
            print("读取结束")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MyMainWindow()
    window.show()
    sys.exit(app.exec_())
