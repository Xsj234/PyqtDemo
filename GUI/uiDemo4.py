# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'uiDemo4.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("C:/Users/lenovo/.designer/image/youcans.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.pushButton_1 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_1.setGeometry(QtCore.QRect(110, 120, 75, 23))
        self.pushButton_1.setObjectName("pushButton_1")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(110, 190, 75, 23))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(110, 260, 75, 23))
        self.pushButton_3.setObjectName("pushButton_3")
        self.checkBox_4 = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_4.setGeometry(QtCore.QRect(110, 330, 75, 23))
        self.checkBox_4.setObjectName("checkBox_4")
        self.lineEdit_1 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_1.setGeometry(QtCore.QRect(270, 120, 113, 20))
        self.lineEdit_1.setObjectName("lineEdit_1")
        self.lineEdit_2 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_2.setGeometry(QtCore.QRect(270, 190, 113, 20))
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.lineEdit_3 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_3.setGeometry(QtCore.QRect(270, 260, 113, 20))
        self.lineEdit_3.setObjectName("lineEdit_3")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 22))
        self.menubar.setObjectName("menubar")
        self.menu = QtWidgets.QMenu(self.menubar)
        self.menu.setGeometry(QtCore.QRect(242, 117, 114, 101))
        self.menu.setObjectName("menu")
        self.menumenuQuit = QtWidgets.QMenu(self.menubar)
        self.menumenuQuit.setObjectName("menumenuQuit")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.toolBar = QtWidgets.QToolBar(MainWindow)
        self.toolBar.setObjectName("toolBar")
        MainWindow.addToolBar(QtCore.Qt.TopToolBarArea, self.toolBar)
        self.actionOpen = QtWidgets.QAction(MainWindow)
        self.actionOpen.setObjectName("actionOpen")
        self.actionSave = QtWidgets.QAction(MainWindow)
        self.actionSave.setObjectName("actionSave")
        self.actionClose = QtWidgets.QAction(MainWindow)
        self.actionClose.setObjectName("actionClose")
        self.actionQuit = QtWidgets.QAction(MainWindow)
        self.actionQuit.setObjectName("actionQuit")
        self.actionactionSetup = QtWidgets.QAction(MainWindow)
        self.actionactionSetup.setObjectName("actionactionSetup")
        self.actionactionHelp = QtWidgets.QAction(MainWindow)
        self.actionactionHelp.setObjectName("actionactionHelp")
        self.menu.addAction(self.actionOpen)
        self.menu.addAction(self.actionSave)
        self.menu.addAction(self.actionClose)
        self.menumenuQuit.addAction(self.actionQuit)
        self.menubar.addAction(self.menu.menuAction())
        self.menubar.addAction(self.menumenuQuit.menuAction())
        self.toolBar.addAction(self.actionactionSetup)
        self.toolBar.addAction(self.actionactionHelp)
        self.toolBar.addAction(self.actionSave)
        self.toolBar.addAction(self.actionOpen)
        self.toolBar.addAction(self.actionClose)
        self.toolBar.addAction(self.actionQuit)

        self.retranslateUi(MainWindow)
        self.actionQuit.triggered.connect(MainWindow.close) # type: ignore
        self.pushButton_1.clicked.connect(self.lineEdit_1.clear) # type: ignore
        self.pushButton_2.clicked.connect(MainWindow.click_pushButton_2) # type: ignore
        self.pushButton_3.clicked.connect(MainWindow.click_pushButton_3) # type: ignore
        self.checkBox_4.clicked['bool'].connect(self.checkBox_4.setChecked) # type: ignore
        self.actionactionHelp.triggered.connect(MainWindow.trigger_actHelp) # type: ignore
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "数字图像处理"))
        self.pushButton_1.setText(_translate("MainWindow", "1# 按钮"))
        self.pushButton_2.setText(_translate("MainWindow", "2# 按钮"))
        self.pushButton_3.setText(_translate("MainWindow", "3# 按钮"))
        self.checkBox_4.setText(_translate("MainWindow", "4# 按钮"))
        self.lineEdit_1.setText(_translate("MainWindow", "文本编辑行-1"))
        self.lineEdit_2.setText(_translate("MainWindow", "文本编辑行-2"))
        self.lineEdit_3.setText(_translate("MainWindow", "文本编辑行-3"))
        self.menu.setTitle(_translate("MainWindow", "文件"))
        self.menumenuQuit.setTitle(_translate("MainWindow", "退出"))
        self.toolBar.setWindowTitle(_translate("MainWindow", "toolBar"))
        self.actionOpen.setText(_translate("MainWindow", "打开"))
        self.actionSave.setText(_translate("MainWindow", "保存"))
        self.actionClose.setText(_translate("MainWindow", "关闭"))
        self.actionQuit.setText(_translate("MainWindow", "退出"))
        self.actionactionSetup.setText(_translate("MainWindow", "actionSetup"))
        self.actionactionSetup.setIconText(_translate("MainWindow", "设置"))
        self.actionactionSetup.setToolTip(_translate("MainWindow", "actionSetup"))
        self.actionactionHelp.setText(_translate("MainWindow", "actionHelp"))
        self.actionactionHelp.setIconText(_translate("MainWindow", "帮助"))
