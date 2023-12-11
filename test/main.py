# 2023年12月2日
# 徐胜杰
# 登录界面主程序
from InterfaceUi import *
from LoginUi import *
import webbrowser
from PyQt5.QtWidgets import QApplication, QMainWindow
import sys

user_now = None

ok = ""

class LoginWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.win = None
        self.ui = Ui_LoginWindow()
        self.ui.setupUi(self)
        # 隐藏window窗口
        self.setWindowFlag(QtCore.Qt.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        # 设置阴影
        self.shadow = QtWidgets.QGraphicsDropShadowEffect(self)
        self.shadow.setOffset(0, 0)
        self.shadow.setBlurRadius(10)
        self.shadow.setColor(QtCore.Qt.black)
        self.ui.frame.setGraphicsEffect(self.shadow)
        # self.ui.frame_2.setGraphicsEffect(self.shadow)
        # 连接函数
        self.ui.pushButton_Login.clicked.connect(lambda: self.ui.stackedWidget_2.setCurrentIndex(0))
        self.ui.pushButton_Register.clicked.connect(lambda: self.ui.stackedWidget_2.setCurrentIndex(1))
        # 登录界面跳转主界面
        self.ui.pushButton_L_sure.clicked.connect(self.loginIn)
        self.show()

    def loginIn(self):
        # 判断账号密码 报警和跳转
        account = self.ui.lineEdit_L_account.text()
        password = self.ui.lineEdit_L_password.text()
        if account == 'admin' and password == "123456":
            self.win = MainWindow()
            self.close()
        else:
            print("wrong")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.login = None
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        # 隐藏window窗口
        self.setWindowFlag(QtCore.Qt.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        # 设置阴影
        self.shadow = QtWidgets.QGraphicsDropShadowEffect(self)
        self.shadow.setOffset(0, 0)
        self.shadow.setBlurRadius(10)
        self.shadow.setColor(QtCore.Qt.black)
        self.ui.frame_6.setGraphicsEffect(self.shadow)
        # 连接函数
        self.ui.pushButton_home.clicked.connect(lambda: self.ui.stackedWidget.setCurrentIndex(0))
        self.ui.pushButton_web.clicked.connect(self.goWeb)
        self.ui.pushButton_my.clicked.connect(lambda: self.ui.stackedWidget.setCurrentIndex(2))
        self.ui.pushButton_logout.clicked.connect(self.logout)

        self.show()

    def goWeb(self):
        # 网址推荐
        self.ui.stackedWidget.setCurrentIndex(1)
        self.ui.pushButton_BIBI.clicked.connect(lambda: webbrowser.open("https://www.bilibili.com/"))
        self.ui.pushButton_APPLE.clicked.connect(lambda: webbrowser.open("https://www.apple.com.cn/"))
        self.ui.pushButton_CSDN.clicked.connect(lambda: webbrowser.open("https://blog.csdn.net/"))
        self.ui.pushButton_TV.clicked.connect(lambda: webbrowser.open("https://v.qq.com/"))

    def logout(self):
        global user_now
        self.login = LoginWindow()
        self.close()
        user_now = ''

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = LoginWindow()
    sys.exit(app.exec_())
