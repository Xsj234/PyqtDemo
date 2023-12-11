import sys
import traceback

import cv2
import time
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox, QFileDialog
import multiprocessing as mp

from GUI.OCR import Ui_MainWindow  # 导入 uiDemo4.py 中的 Ui_MainWindow 界面类


# from main.rstp import image_put, image_get
#
# 程序作用是
#
class MyMainWindow(QMainWindow, Ui_MainWindow):  # 继承 QMainWindow类和 Ui_MainWindow界面类
    save_path = ''
    Time = ''

    def __init__(self, parent=None):
        super(MyMainWindow, self).__init__(parent)  # 初始化父类
        self.setupUi(self)  # 继承 Ui_MainWindow 界面类
        self.pushButton_run.clicked.connect(self.save_txt)
        self.pushButton_run_2.clicked.connect(self.save_txt2)

    def save_txt(self):
        global save_path

        global Time
        Time = "11月27日 11:55:23"  # +a+ Time +

        fileName, ok = QFileDialog.getSaveFileName(None, "文件保存", f"H:/屏幕识别{Time}.txt")  # 改变地址
        print(fileName)  # 打印保存文件的全部路径（包括文件名和后缀名）
        save_path = fileName

        if save_path is not None:
            with open(file=fileName, mode='w+', encoding='utf-8') as file:
                file.write(self.textEdit.toPlainText())
            print('已保存！')

    def save_txt2(self):
        global save_path
        global Time
        Time = "11月27日 11:55:23"  # +a+ Time +

        fileName2, ok2 = QFileDialog.getSaveFileName(None, "文件保存", f"H:/弹窗识别{Time}.txt")  # 弹出保存图框
        print(fileName2)  # 打印保存文件的全部路径（包括文件名和后缀名）
        save_path = fileName2

        if save_path is not None:
            with open(file=save_path, mode='w+', encoding='utf-8') as file:
                file.write(self.textEdit_3.toPlainText())
            print('已保存！')
            # self.textEdit.clear()  # 清屏


if __name__ == '__main__':
    app = QApplication(sys.argv)  # 在 QApplication 方法中使用，创建应用程序对象
    myWin = MyMainWindow()  # 实例化 MyMainWindow 类，创建主窗口
    myWin.show()  # 在桌面显示控件 myWin
    sys.exit(app.exec_())  # 结束进程，退出程序
