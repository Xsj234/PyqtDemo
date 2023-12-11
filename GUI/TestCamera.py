# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'TestCamera.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(797, 336)
        self.layoutWidget = QtWidgets.QWidget(Form)
        self.layoutWidget.setGeometry(QtCore.QRect(30, 20, 741, 292))
        self.layoutWidget.setObjectName("layoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.layoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.Camera1 = QtWidgets.QLabel(self.layoutWidget)
        self.Camera1.setMinimumSize(QtCore.QSize(241, 141))
        self.Camera1.setMaximumSize(QtCore.QSize(241, 141))
        self.Camera1.setAutoFillBackground(True)
        self.Camera1.setText("")
        self.Camera1.setScaledContents(True)
        self.Camera1.setObjectName("Camera1")
        self.gridLayout.addWidget(self.Camera1, 0, 0, 1, 1)
        self.Camera2 = QtWidgets.QLabel(self.layoutWidget)
        self.Camera2.setMinimumSize(QtCore.QSize(241, 141))
        self.Camera2.setMaximumSize(QtCore.QSize(241, 141))
        self.Camera2.setAutoFillBackground(True)
        self.Camera2.setText("")
        self.Camera2.setScaledContents(True)
        self.Camera2.setObjectName("Camera2")
        self.gridLayout.addWidget(self.Camera2, 0, 1, 1, 1)
        self.Camera3 = QtWidgets.QLabel(self.layoutWidget)
        self.Camera3.setMinimumSize(QtCore.QSize(241, 141))
        self.Camera3.setMaximumSize(QtCore.QSize(241, 141))
        self.Camera3.setAutoFillBackground(True)
        self.Camera3.setText("")
        self.Camera3.setScaledContents(True)
        self.Camera3.setObjectName("Camera3")
        self.gridLayout.addWidget(self.Camera3, 0, 2, 1, 1)
        self.Camera4 = QtWidgets.QLabel(self.layoutWidget)
        self.Camera4.setMinimumSize(QtCore.QSize(241, 141))
        self.Camera4.setMaximumSize(QtCore.QSize(241, 141))
        self.Camera4.setAutoFillBackground(True)
        self.Camera4.setText("")
        self.Camera4.setScaledContents(True)
        self.Camera4.setObjectName("Camera4")
        self.gridLayout.addWidget(self.Camera4, 1, 0, 1, 1)
        self.Camera5 = QtWidgets.QLabel(self.layoutWidget)
        self.Camera5.setMinimumSize(QtCore.QSize(241, 141))
        self.Camera5.setMaximumSize(QtCore.QSize(241, 141))
        self.Camera5.setAutoFillBackground(True)
        self.Camera5.setText("")
        self.Camera5.setScaledContents(True)
        self.Camera5.setObjectName("Camera5")
        self.gridLayout.addWidget(self.Camera5, 1, 1, 1, 1)
        self.Camera6 = QtWidgets.QLabel(self.layoutWidget)
        self.Camera6.setMinimumSize(QtCore.QSize(241, 141))
        self.Camera6.setMaximumSize(QtCore.QSize(241, 141))
        self.Camera6.setAutoFillBackground(True)
        self.Camera6.setText("")
        self.Camera6.setScaledContents(True)
        self.Camera6.setObjectName("Camera6")
        self.gridLayout.addWidget(self.Camera6, 1, 2, 1, 1)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))

