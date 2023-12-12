# 2023年12月2日
# 徐胜杰
# 登录界面主程序
import psycopg2

from InterfaceUi import *
from LoginUi import *
import webbrowser
from PyQt5.QtWidgets import QApplication, QMainWindow
import sys

user_now = None


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

        self.ui.pushButton_R_sure.clicked.connect(self.registerIn)
        self.show()

    def loginIn(self):
        # 判断账号密码 报警和跳转
        account = self.ui.lineEdit_L_account.text()
        password = self.ui.lineEdit_L_password.text()
        account_list = []
        password_list = []
        conn = psycopg2.connect(database="DataMy", user="postgres", password="123456", port="5432")
        cur = conn.cursor()
        # 数据库语言
        cur.execute("select * from users;")
        rows = cur.fetchall()
        for row in rows:
            account_list.append(row[0])
            password_list.append(row[1])
        print(account_list, password_list)
        # 上传
        conn.commit()
        # 关闭
        conn.close()

        for i in range(len(account_list)):
            if len(account) == 0 or len(password) == 0:
                self.ui.stackedWidget.setCurrentIndex(1)
            elif account == account_list[i] and password == password_list[i]:
                global user_now
                user_now = account
                print(user_now)
                self.win = MainWindow()
                self.close()
            else:
                self.ui.stackedWidget.setCurrentIndex(2)

    def registerIn(self):
        account = self.ui.lineEdit_R_account.text()
        password = self.ui.lineEdit__R_password_1.text()
        password2 = self.ui.lineEdit__R_password_2.text()
        if len(account)==0 or len(password)==0 or len(password2)==0:
            self.ui.stackedWidget.setCurrentIndex(1)
        elif password!=password2:
            self.ui.stackedWidget.setCurrentIndex(3)
        else:
            self.ui.stackedWidget.setCurrentIndex(4)
            conn = psycopg2.connect(database="DataMy", user="postgres", password="123456", port="5432")
            cur = conn.cursor()
            # 数据库语言
            cur.execute(f"insert into users values('{account}','{password}');")
            # 上传
            conn.commit()
            # 关闭
            conn.close()


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
        self.ui.pushButton_M_sure.clicked.connect(self.changePassword)
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

    def changePassword(self):
        global user_now
        password = self.ui.lineEdit_L_M_pass1.text()
        if len(self.ui.lineEdit_L_M_pass1.text()) == 0 or len(self.ui.lineEdit_L_M_pass2.text()) == 0:
            self.ui.stackedWidget_2.setCurrentIndex(1)
        elif self.ui.lineEdit_L_M_pass1.text() == self.ui.lineEdit_L_M_pass2.text():
            conn = psycopg2.connect(database="DataMy", user="postgres", password="123456", port="5432")
            cur = conn.cursor()
            # 数据库语言
            cur.execute(f"update users set passwords='{password}' where accounts = '{user_now}';")
            conn.commit()
            # 关闭
            conn.close()
            self.ui.stackedWidget_2.setCurrentIndex(3)
        else:
            self.ui.stackedWidget_2.setCurrentIndex(2)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = LoginWindow()
    sys.exit(app.exec_())
