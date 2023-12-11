from multiprocessing import Process

import time


def f(name):
    time.sleep(1)
    print('hello', name, time.time())


def direct_call_process():
    p_list = []
    for i in range(3):
        p = Process(target=f, args=('process:%s' % i,))
        p_list.append(p)
        p.start()
    for p in p_list:
        p.join()
    print('end')


# --------------------------------------------


class myProcess(Process):
    def __init__(self):
        super(myProcess, self).__init__()

    def run(self):
        print('hello', self.name, time.time())
        time.sleep(1)


def inherit_call_process():
    p_list = []

    for i in range(3):
        p = myProcess()
        p.start()
        p_list.append(p)

    for p in p_list:
        p.join()

    print('end')


if __name__ == '__main__':
    direct_call_process()
    print('------------------------------------')
    inherit_call_process()
