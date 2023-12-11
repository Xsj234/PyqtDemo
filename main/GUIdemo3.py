import sys
import traceback

import cv2
import time
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox, QFileDialog
import multiprocessing as mp

from GUI.OCR import Ui_MainWindow  # 导入 uiDemo4.py 中的 Ui_MainWindow 界面类


# from main.rstp import image_put, image_get


class MyMainWindow(QMainWindow, Ui_MainWindow):  # 继承 QMainWindow类和 Ui_MainWindow界面类
    save_path = ''

    def __init__(self, parent=None):
        super(MyMainWindow, self).__init__(parent)  # 初始化父类
        self.setupUi(self)  # 继承 Ui_MainWindow 界面类

    def save_txt(self):
        global save_path
        fileName, ok = QFileDialog.getSaveFileName(None, "文件保存", "H:/")  # 弹出保存图框
        print(fileName)  # 打印保存文件的全部路径（包括文件名和后缀名）
        save_path = fileName
        # self.save_path_text.setText(_translate("Form", save_path))
        if save_path is not None:
            with open(file=save_path + '.txt', mode='w+', encoding='utf-8') as file:
                file.write(self.textEdit.toPlainText())
            print('已保存！')

    def save_txt2(self):
        global save_path
        fileName2, ok2 = QFileDialog.getSaveFileName(None, "文件保存", "H:/")  # 弹出保存图框
        print(fileName2)  # 打印保存文件的全部路径（包括文件名和后缀名）
        save_path = fileName2
        # self.save_path_text.setText(_translate("Form", save_path))
        if save_path is not None:
            with open(file=save_path + '.txt', mode='w+', encoding='utf-8') as file:
                file.write(self.textEdit_3.toPlainText())
            print('已保存！')
            # self.textEdit.clear()  # 清屏

    # *****************************************************
    def image_put(self, q, name, pwd, ip):
        cap = cv2.VideoCapture("rtsp://%s:%s@%s:554/h264/ch1/main/av_stream" % (name, pwd, ip))
        while True:
            q.put(cap.read()[1])
            q.get() if q.qsize() > 1 else time.sleep(0.01)

    def image_get(self, q, window_name):
        cv2.namedWindow(window_name, flags=cv2.WINDOW_FREERATIO)
        while True:
            frame = q.get()
            cv2.imshow(window_name, frame)
            cv2.waitKey(1)

    def run_multi_camera(self):
        # user_name, user_pwd = "admin", "password"
        user_name, user_pwd = "admin", "a1234567"
        camera_ip_l = [
            "10.112.89.10",  # ipv4
        ]

        mp.set_start_method(method='spawn')  # init
        queues = [mp.Queue(maxsize=2) for _ in camera_ip_l]

        processes = []
        for queue, camera_ip in zip(queues, camera_ip_l):
            # 多个摄像头
            processes.append(mp.Process(target=self.image_put, args=(queue, user_name, user_pwd, camera_ip)))
            processes.append(mp.Process(target=self.image_get, args=(queue, camera_ip)))

        for process in processes:
            process.daemon = True
            process.start()
        for process in processes:
            process.join()


if __name__ == '__main__':
    app = QApplication(sys.argv)  # 在 QApplication 方法中使用，创建应用程序对象
    myWin = MyMainWindow()  # 实例化 MyMainWindow 类，创建主窗口
    myWin.show()  # 在桌面显示控件 myWin
    sys.exit(app.exec_())  # 结束进程，退出程序
