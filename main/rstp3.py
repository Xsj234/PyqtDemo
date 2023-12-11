import os
import cv2
import gc
import time
import numpy as np
from PIL import Image

import multiprocessing
from multiprocessing import Process, Manager


top = 100

# 向共享缓冲栈中写入数据:
def write(stack, cam, top: int) -> None:
    """
    :param cam: 摄像头参数
    :param stack: Manager.list对象
    :param top: 缓冲栈容量
    :return: None
    """
    print('Process to write: %s' % os.getpid())
    cap = cv2.VideoCapture(cam)
    while True:
        _, img = cap.read()
        if _:
            stack.append(img)
            # 每到一定容量清空一次缓冲栈
            # 利用gc库，手动清理内存垃圾，防止内存溢出
            if len(stack) >= top:
                del stack[:]
                gc.collect()


# 在缓冲栈中读取数据:
def read(stack) -> None:
    print('Process to read: %s' % os.getpid())
    # 开始时间
    t1 = time.time()
    # 图片计数
    count = 0

    while True:
        if len(stack) != 0:
            # 开始图片消耗
            print("stack的长度", len(stack))
            if len(stack) != 100 and len(stack) != 0:
                value = stack.pop()
            else:
                pass

            if len(stack) >= top:
                del stack[:]
                gc.collect()

            # 格式转变，BGRtoRGB
            frame = cv2.cvtColor(value, cv2.COLOR_BGR2RGB)

            # 转变成Image
            frame = Image.fromarray(np.uint8(frame))
            # cv2.imshow(frame)
            print("*" * 100)

            count += 1
            print("数量为：", count)

            t2 = time.time()
            print("时间差：", int(t2 - t1))

            if int(t2 - t1) == 600:
                # 记录 消耗的图片数量
                with open('count.txt', 'ab') as f:
                    f.write(str(count).encode() + "\n".encode())
                    f.flush()

                # count = 0  # 不要置零，计算总数
                t1 = t2
            else:
                cv2.imshow("Real-time Image Display", cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)



if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    # 父进程创建缓冲栈，并传给各个子进程：
    q = Manager().list()
    # 创建一个进程去运行write函数
    pw = Process(target=write, args=(q, "rtsp://admin:a1234567@10.112.89.10:554/h264/ch1/main/av_stream", top))
    pr = Process(target=read, args=(q,))
    # 启动子进程pw，写入:
    pw.start()
    # 启动子进程pr，读取:
    pr.start()
    # 等待pr结束:
    pr.join()
    # pw进程里是死循环，无法等待其结束，只能强行终止:
    pw.terminate()
    # pr.join()
