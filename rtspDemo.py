# coding=gbk //�ŵ��ļ�����
import sys
import time
from queue import LifoQueue
import cv2
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from rtspUI import Ui_MainWindow

# ʹ����queueģ���е� Last-In-First-Out��LIFO����ȳ��Ķ���
# queue�ṩ���ڶ��߳�����½���ģ���̰߳�ȫ�Ķ��в����Ĺ��ߡ�
# LifoQueue��queueģ���е�һ�ֶ������ͣ���ʾ����ȳ��Ķ��У���ջ��stack������Ϊ��
# Decode2Play��һ�����ж���������������ж��в������������Ԫ�أ�put����ȡ��Ԫ�أ�get���ȡ�
Decode2Play = LifoQueue()

class cvDecode(QThread):
    def __init__(self):
        # �Զ����������������ʼ���ö���
        super(cvDecode, self).__init__()
        self.threadFlag = 0  # �����߳��˳�
        self.rtsp = "rtsp://admin:buptkyl338@10.112.89.82/h264/ch1/main/av_stream"
        self.cap = cv2.VideoCapture(self.rtsp)

    def run(self):

        while self.threadFlag:
            if self.cap.isOpened():
                ret, frame = self.cap.read()  # ret����True or Flase frame����numpy����
                time.sleep(0.001)  # ���ƶ�ȡ¼���ʱ�䣬��ʵʱ��Ƶ��ʱ��ĳ�time.sleep(0.001)�����̵߳��������ü��ϣ�����ͬ�̼߳�������ռ��Դ

                if ret:
                    # lock.acquire()
                    Decode2Play.put(frame)  # ���������ݷŵ�������
                    print(Decode2Play.qsize()) #�鿴
                    # lock.release()
                del frame  # �ͷ���Դ



class play_Work(QThread):
    def __init__(self):
        super(play_Work, self).__init__()
        self.threadFlag = 0  # �����߳��˳�
        self.playLabel = QLabel()  # ��ʼ��QLabel����

    def run(self):
        while self.threadFlag:
            if not Decode2Play.empty():
                frame = Decode2Play.get()
                while not Decode2Play.empty():
                    Decode2Play.get()

                frame = cv2.resize(frame, (800, 600), cv2.INTER_LINEAR)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                qimg = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)  # ��������Զ�ÿ֡ͼ����д���
                self.playLabel.setPixmap(QPixmap.fromImage(qimg))  # ͼ����QLabel��չʾ
            time.sleep(0.001)


class MyMainWindow(QMainWindow, Ui_MainWindow):  # �̳� QMainWindow��� Ui_MainWindow������

    def __init__(self, parent=None):
        super(MyMainWindow, self).__init__(parent)  # ��ʼ������
        self.setupUi(self)  # �̳� Ui_MainWindow ������

    def init_work(self):
        print("���߳�")
        self.decodework = cvDecode()
        self.decodework.threadFlag = 1
        self.decodework.start()

        self.playwork = play_Work()
        self.playwork.threadFlag = 1
        self.playwork.playLabel = self.label
        self.playwork.start()

    def closeEvent(self, event):
        print("�ر��߳�")
        # Qt��Ҫ���˳�ѭ�����ܹر��߳�
        if self.decodework.isRunning():
            self.decodework.threadFlag = 0
            self.decodework.quit()
            print("д�����")
        if self.playwork.isRunning():
            self.playwork.threadFlag = 0
            self.playwork.quit()
            print("��ȡ����")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MyMainWindow()
    window.show()
    sys.exit(app.exec_())
