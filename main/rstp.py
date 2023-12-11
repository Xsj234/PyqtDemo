#####  多进程

from threading import Thread

import cv2

import time
import multiprocessing as mp

from PyQt5.QtGui import QImage, QPixmap


class MyMainWindow():
    def image_put(self, q, name, pwd, ip):
        cap = cv2.VideoCapture("rtsp://%s:%s@%s:554/h264/ch1/main/av_stream" % (name, pwd, ip))
        while True:
            q.put(cap.read()[1])
            q.get() if q.qsize() > 1 else time.sleep(0.01)

    def image_get(self, q, window_name):
        cv2.namedWindow(window_name, flags=cv2.WINDOW_FREERATIO)
        cv2.resizeWindow(window_name, 500, 300)
        while True:
            frame = q.get()

            # frame = cv2.resize(frame, (800, 600), cv2.INTER_LINEAR)
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # qimg = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)  # 在这里可以对每帧图像进行处理，
            # self.playLabel.setPixmap(QPixmap.fromImage(qimg))  # 图像在QLabel上展示

            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()

    def thread_rtsp(self):
        # user_name, user_pwd = "admin", "password"
        user_name, user_pwd = "admin", "a1234567"
        camera_ip_l = [
            "10.112.89.10",  # ipv4
            # "10.112.89.11" #ipv4进程
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

    def thread_rtsp_start(self):
        thread_rtsp = Thread(target=self.thread_rtsp)
        thread_rtsp.setDaemon(True)  # ==========================================================
        thread_rtsp.start()
        thread_rtsp.join()
        # 不行就多睡会
        # time.sleep(3)


if __name__ == '__main__':
    myWin = MyMainWindow()
    myWin.thread_rtsp()
