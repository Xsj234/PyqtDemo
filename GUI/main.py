import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLineEdit, QFileDialog


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("示例界面")
        self.setGeometry(100, 100, 300, 200)

        self.text_input = QLineEdit(self)
        self.text_input.setGeometry(100, 30, 200, 30)

        self.export_button = QPushButton("导出", self)
        self.export_button.setGeometry(100, 80, 100, 30)
        self.export_button.clicked.connect(self.export_data)

    def export_data(self):
        # text = self.text_input.text()  # 获取输入框的文本
        # print("导出文本:", text)
        fileName, ok = QFileDialog.getSaveFileName(None, "文件保存", "H:/")  # 弹出保存图框
        print(fileName)  # 打印保存文件的全部路径（包括文件名和后缀名）
        save_path = fileName
        # self.save_path_text.setText(_translate("Form", save_path))
        if save_path is not None:
            with open(file=save_path, mode='w+', encoding='utf-8') as file:
                file.write(self.text_input.toPlainText())
            print('已保存！')
        # 在这里执行实际的导出操作，可以将文本写入文件、发送网络请求等


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())

