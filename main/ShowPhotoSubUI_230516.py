##大家好,这里是一个好程序
#可以实现12路视频的拉流和显示(但是开始有掉帧现象)
#如果不显示12路，拉流和检测都是正常的
#检测两个模型轮着，就是慢。。。
import argparse
import random
import signal
import threading
from datetime import datetime
import torch
from torch.backends import cudnn
from src.ui.sub_ui.ILSP_1 import Islongstandperson
from src.ui.sub_ui.ICAR_1 import IsredcarpetInredcar
from src.ui.sub_ui.false_work import Islongfalsework

from src.ui.sub_ui.OrdinaryViolation import isOrdinaryViolation


import models.common
from UI.sub_UI import showphoto0527

from src.ui.sub_ui.setmessagesubui import Setmessage
from src.ui.sub_ui.set_alarm_param_sub_ui import Set_alarm_param
import cv2
from threading import Thread
import time
from PyQt5 import QtCore, QtGui, QtWidgets
import traceback
import queue
import numpy as np
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import scale_coords, non_max_suppression, check_img_size
from utils.torch_utils import select_device
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Process
import multiprocessing
import socket
from time import strftime
from src.ui.sub_ui.Database0517 import mydatabase
from datetime import datetime
import datetime
from utils.plots import *
from multiprocessing import Process,Manager
from UI import setmessage
from src.ui.sub_ui.socket_server import SocketServer
names_all_80 = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
                'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
                'cell phone',
                'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                'teddy bear',
                'hair drier', 'toothbrush']

names_all_27 = ['head',
             'cloth_deepblue',
             'cloth_lightblue',
             'cloth_deepblue_unnormal',
             'cloth_lightblue_unnormal',
             'cloth_white',
             'cloth_white_false',
             'cloth_false',
             'phone',
             'shoes_normal_1',
             'shoes_cover',
             'shoes_false',
             'cigarette',
             'box_hing',
             'fall_down',
             'safety_hat',
             'tricolor_light_on',
             'flip_box_open',
             'flip_box_close',
             'tricolor_light_off',
             'cloth_lightblue_1',
             'cloth_lightblue_unnormal_1',
            "fire",
            "smog",
            "car_1",
            "car_red",
            "red_carpet",
            "hand_ball",
            "hand_ball_1",
            "table_yes",
            "table_no",
            "grounding",
             ]

login_flag = False



class ly_Camera(showphoto0527.Ui_MainWindow):
    def __init__(self, child_conn, Rtsp_all, frame_queue, number_process):
        self.cam1 = None
        self.Rtsp_all = Rtsp_all
        self.frame_queue_all = frame_queue
        # self.frame_queue_detect = frame_queue_detect
        self.is_running = True  # 状态标签
        self.fps = 0.0  # 实时帧率
        self._t_last = time.time() * 1000
        self._data = {}
        # self.detect_flag = detect_flag
        self.number_process = int(number_process)

        self.lock = threading.RLock()

        self.run()
        # self.number_process = number_process

    def capture_queue(self):
        # 捕获图像
        self.cam = []
        self._t_last = time.time() * 1000
        frame = []


        for index1 in range(len(self.Rtsp_all)):
            self.cam1 = cv2.VideoCapture(self.Rtsp_all[index1][0])
            # print("buffer:<><><><><><>><>><><><> ",self.cam1.get(cv2.CAP_PROP_BUFFERSIZE))

            self.cam.append(self.cam1)
        try:
            while self.is_running:
                # print("------------self.number_process----------------", self.number_process/2)
                # print("buffer:<><><><><><>><>><><><> ", self.cam1.get(cv2.CAP_PROP_BUFFERSIZE))
                for indexa in range(len(self.Rtsp_all)):

                    self.lock.acquire()
                    if self.cam[indexa].isOpened():
                        self.cam[indexa].grab()
                        ret, frame = self.cam[indexa].read()
                        if not ret:
                            self.cam[indexa] = cv2.VideoCapture(self.Rtsp_all[indexa][0])
                            time.sleep(0.05)
                            continue
                        if self.frame_queue_all[self.number_process + indexa].qsize() < 10:

                            # 当队列中的图像都被消耗完后，再压如新的图像
                            t = time.time() * 1000
                            t_span = t - self._t_last
                            self.fps = int(1000.0 / t_span)
                            self._data["image"] = frame
                            self._data["fps"] = self.fps

                            self.frame_queue_all[self.number_process + indexa].put(self._data)
                            self._data = {}
                            self._t_last = t
                    else:
                        self.cam[indexa] = cv2.VideoCapture(self.Rtsp_all[indexa][0])
                        time.sleep(0.05)
                        continue
                    self.lock.release()

                time.sleep(0.01)
        except:
            traceback.print_exc()

    def run(self):

        self.is_running = True
        capture_queue_thread = Thread(target=self.capture_queue)
        capture_queue_thread.start()
        capture_queue_thread.join()


class pic_detect(showphoto0527.Ui_MainWindow):
    def __init__(self, child_conn, queueq, result_list, obj_pos, number_rtsp, process_index, frame_queu_save, load_model):
        self.queueq = queueq
        self.child_conn = child_conn
        self.result_list = result_list
        self.obj_pos = obj_pos
        self.number_rtsp = len(queueq)
        self.process_index = process_index
        self.frame_queue_save = frame_queu_save
        self.load_model = load_model

        self.result = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
                       [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]

        self.start = True

        self.detect_data = [False, False, False, False, False, False,
                            False, False, False, False, False, False,
                            False, False, False, False]
        self.detect_time = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.video = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        parser = argparse.ArgumentParser()
        # parser.add_argument('--weights', nargs='+', type=str, default='./weights/best_60epoch_13.pt',
        #                    help='model.pt path(s)')

        parser.add_argument('--weights', nargs='+', type=str, default='./0716_best.pt', help='model.pt path(s)')
        parser.add_argument('--weights1', nargs='+', type=str, default='./yolov5s.pt', help='model.pt path(s)')
        parser.add_argument('--source', type=str, default=0, help='source')  # file/folder, 0 for webcam
        parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
        parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
        parser.add_argument('--conf-thres1', type=float, default=0.5, help='object confidence threshold')
        parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
        parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--view-img', action='store_true', help='display results')
        parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
        parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
        parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
        parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
        parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
        parser.add_argument('--augment', action='store_true', help='augmented inference')
        parser.add_argument('--update', action='store_true', help='update all models')
        parser.add_argument('--project', default='runs/detect', help='save results to project/name')
        parser.add_argument('--name', default='exp', help='save results to project/name')
        parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
        self.opt = parser.parse_args()
        # print(self.opt)

        source, weights, view_img, save_txt, imgsz = self.opt.source, self.opt.weights, self.opt.view_img, self.opt.save_txt, self.opt.img_size
        weights1 = self.opt.weights1
        self.device = select_device(self.opt.device)
        cudnn.benchmark = True
        # Load model
        self.model = attempt_load(weights, map_location=self.device)  # load FP32 model
        self.model.half()  # to FP16frame_queu_save
        stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(imgsz, s=stride)  # check img_size
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names

        self.model1 = attempt_load(weights1, map_location=self.device)  # load FP32 model
        self.model1.half()  # to FP16
        self.names1 = self.model1.module.names if hasattr(self.model1, 'module') else self.model1.names
        self.load_model = True
        # print("-----self.process_index0-----------", self.process_index)
        self.run()



    def run(self):

        model_yolo = True
        # self.action_num_list = [0, 0, 0, ]
        detect_flag = True
        cycle_number = 0
        # list_temp_pos = {}
        # for i in range(4):
        #     list_temp_pos[i] = {'person_list': [], "car_per_list": []}

        while detect_flag:

            s_time = time.time()

            # print("-----self.process_index1-----------", self.process_index)
            for index in range(self.number_rtsp):
                # print("process:", self.process_index, "+:1-1")
                # yolov5
                # self.result_list[index]['person'] = 0
                names_1 = names_all_80
                self.model_new = self.model1
                # print("process:", self.process_index, "+:1-2")
                if not self.queueq[index].empty():
                    data = self.queueq[index].get()
                else:
                    continue

                img = data["image"]
                img_1 = img
                # print(datetime.datetime.now())
                showimg = img
                # print("process:", self.process_index, "+:1-3")
                # print("----------img---------------", img)

                if self.frame_queue_save[index].qsize() > 4:
                    while not self.frame_queue_save[index].empty():
                        self.frame_queue_save[index].get()
                self.frame_queue_save[index].put(img)  # 用于保存的队列

                s = ''
                with torch.no_grad():
                    img = letterbox(img, new_shape=self.opt.img_size)[0]
                    # Convert
                    # BGR to RGB, to 3x416x416
                    img = img[:, :, ::-1].transpose(2, 0, 1)
                    img = np.ascontiguousarray(img)
                    img = torch.from_numpy(img).to(self.device)
                    img = img.half()  # uint8 to fp16/32
                    img /= 255.0  # 0 - 255 to 0.0 - 1.0
                    if img.ndimension() == 3:
                        img = img.unsqueeze(0)
                    # model_new current model
                    pred = self.model_new(img, augment=self.opt.augment)[0]

                    # Apply NMS
                    pred = non_max_suppression(pred, self.opt.conf_thres1, self.opt.iou_thres,
                                               classes=self.opt.classes,
                                               agnostic=self.opt.agnostic_nms)
                    for i, det in enumerate(pred):
                        # if det is None or len(det) == 0:
                        #     self.result_list[index]["camera_index"] = index
                        #     self.result_list[index]['person'] = 0
                        #     self.obj_pos[index]['person_list'] = 0
                        self.result_list[index]["camera_index"] = index
                        self.result_list[index]['person'] = 0
                        self.obj_pos[index]['person_list'] = []
                        self.obj_pos[index]['phone1'] = []
                        if len(det) != 0:
                            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], showimg.shape).round()
                            for c in det[:, -1].unique():
                                n = (det[:, -1] == c).sum()  # detections per class
                                # print("dadsfasfdsf",n,c)
                                s += f"{n} {names_1[int(c)]}{'s' * (n > 1)}, "  # add to string
                                if names_1[int(c)] == "person":  # yolo detect person
                                    self.result_list[index]['camera_index'] = index
                                    # self.result_list[index]['person'] = 0
                                    # print("person detect num: ",int((det[:, -1] == 0).sum()))
                                    self.result_list[index]['person'] = int((det[:, -1] == 0).sum())
                            list_temp_pos = []
                            list_phone1 = []
                            for *xyxy, conf, cls in reversed(det):
                                if names_1[int(cls)] == 'person':
                                    list_temp_pos.append([int(xyxy[0]) + (int(xyxy[2]) - int(xyxy[0])) // 2,
                                                          int(xyxy[1]) + (int(xyxy[3]) - int(xyxy[1])) // 2,
                                                          int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])])

                                if names_1[int(cls)] == 'cell phone':
                                    list_phone1.append([int(xyxy[0]) + (int(xyxy[2]) - int(xyxy[0])) // 2,
                                                          int(xyxy[1]) + (int(xyxy[3]) - int(xyxy[1])) // 2,
                                                          int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])])
                                if names_1[int(cls)] == 'remote':
                                    list_phone1.append([int(xyxy[0]) + (int(xyxy[2]) - int(xyxy[0])) // 2,
                                                          int(xyxy[1]) + (int(xyxy[3]) - int(xyxy[1])) // 2,
                                                          int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])])

                            self.obj_pos[index]['person_list'] = list_temp_pos
                            self.obj_pos[index]['phone1'] = list_phone1

                            # if self.process_index == 3:
                            #     print("-----------------self.obj_pos[index]['person_list']------------------", index, self.obj_pos[index]['person_list'])

                names_1 = names_all_27
                self.model_new = self.model
                for name in names_1:
                    self.result_list[index][name] = 0
                showing_1 = img_1
                s = ''
                with torch.no_grad():
                    img_1 = letterbox(img_1, new_shape=self.opt.img_size)[0]
                    # Convert
                    # BGR to RGB, to 3x416x416
                    img_1 = img_1[:, :, ::-1].transpose(2, 0, 1)
                    img_1 = np.ascontiguousarray(img_1)
                    img_1 = torch.from_numpy(img_1).to(self.device)
                    img_1 = img_1.half()  # uint8 to fp16/32
                    img_1 /= 255.0  # 0 - 255 to 0.0 - 1.0
                    if img_1.ndimension() == 3:
                        img_1 = img_1.unsqueeze(0)
                    # model_new current model
                    pred = self.model_new(img_1, augment=self.opt.augment)[0]

                    # Apply NMS
                    pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres,
                                               classes=self.opt.classes,
                                               agnostic=self.opt.agnostic_nms)
                    for i, det in enumerate(pred):

                        self.result_list[index]["camera_index"] = index
                        self.obj_pos[index]['car_per_list'] = []
                        self.obj_pos[index]['red_car_per_list'] = []
                        self.obj_pos[index]['red_carpet'] = []
                        self.obj_pos[index]['uniforms_vio'] = []
                        self.obj_pos[index]['cloth_false'] = []
                        self.obj_pos[index]['shoes_false'] = []
                        self.obj_pos[index]['phone'] = []
                        self.obj_pos[index]['head'] = []
                        self.obj_pos[index]['tricolor_light_on'] = []
                        self.obj_pos[index]['fall_down'] = []
                        self.obj_pos[index]['fire_smog'] = []
                        self.obj_pos[index]['box_hing'] = []
                        self.obj_pos[index]['hand_ball'] = []
                        self.obj_pos[index]['hand_ball_1'] = []
                        self.obj_pos[index]['tongdao'] = []
                        self.obj_pos[index]['table_yes'] = []
                        self.obj_pos[index]['grounding'] = []

                        for name in names_all_27:
                            self.result_list[index][name] = 0

                        if len(det) != 0:
                            det[:, :4] = scale_coords(img_1.shape[2:], det[:, :4], showing_1.shape).round()
                            for c in det[:, -1].unique():
                                n = (det[:, -1] == c).sum()  # detections per class

                                s += f"{n} {names_1[int(c)]}{'s' * (n > 1)}, "  # add to string
                                self.result_list[index][names_all_27[int(c)]] = int((det[:, -1] == c).sum())
                            list_temp_pos = []
                            list_temp_pos1 = []
                            list_temp_pos2 = []
                            list_uniforms_vio = []
                            list_cloth_false = []
                            list_shoes_false = []
                            list_phone = []
                            list_head = []
                            list_tricolor_light = []
                            list_fall_down = []
                            list_fire_smog = []
                            list_box_hing = []
                            list_hand_ball = []
                            list_tongdao = []
                            list_table_yes = []
                            list_hand_ball_1 = []
                            list_grounding = []


                            for *xyxy, conf, cls in reversed(det):

                                if names_all_27[int(cls)] == 'grounding':
                                    list_grounding.append([int(xyxy[0]) + (int(xyxy[2]) - int(xyxy[0])) // 2,
                                                          int(xyxy[1]) + (int(xyxy[3]) - int(xyxy[1])) // 2,
                                                          int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])])
                                if names_all_27[int(cls)] == 'table_yes':
                                    list_table_yes.append([int(xyxy[0]) + (int(xyxy[2]) - int(xyxy[0])) // 2,
                                                          int(xyxy[1]) + (int(xyxy[3]) - int(xyxy[1])) // 2,
                                                          int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])])
                                if names_all_27[int(cls)] == 'car_1':
                                    list_temp_pos.append([int(xyxy[0]) + (int(xyxy[2]) - int(xyxy[0])) // 2,
                                                          int(xyxy[1]) + (int(xyxy[3]) - int(xyxy[1])) // 2,
                                                          int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])])
                                    list_tongdao.append([int(xyxy[0]) + (int(xyxy[2]) - int(xyxy[0])) // 2,
                                                          int(xyxy[1]) + (int(xyxy[3]) - int(xyxy[1])) // 2,
                                                          int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])])
                                if names_all_27[int(cls)] == 'car_red':
                                    list_temp_pos1.append([int(xyxy[0]) + (int(xyxy[2]) - int(xyxy[0])) // 2,
                                                          int(xyxy[1]) + (int(xyxy[3]) - int(xyxy[1])) // 2,
                                                          int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])])
                                    list_tongdao.append([int(xyxy[0]) + (int(xyxy[2]) - int(xyxy[0])) // 2,
                                                         int(xyxy[1]) + (int(xyxy[3]) - int(xyxy[1])) // 2,
                                                         int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])])
                                if names_all_27[int(cls)] == 'red_carpet':
                                    list_temp_pos2.append([int(xyxy[0]) + (int(xyxy[2]) - int(xyxy[0])) // 2,
                                                          int(xyxy[1]) + (int(xyxy[3]) - int(xyxy[1])) // 2,
                                                          int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])])

                                    list_tongdao.append([int(xyxy[0]) + (int(xyxy[2]) - int(xyxy[0])) // 2,
                                                         int(xyxy[1]) + (int(xyxy[3]) - int(xyxy[1])) // 2,
                                                         int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])])

                                if names_all_27[int(cls)] == 'cloth_deepblue_unnormal':
                                    list_uniforms_vio.append([int(xyxy[0]) + (int(xyxy[2]) - int(xyxy[0])) // 2,
                                                          int(xyxy[1]) + (int(xyxy[3]) - int(xyxy[1])) // 2,
                                                          int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])])
                                if names_all_27[int(cls)] == 'cloth_lightblue_unnormal':
                                    list_uniforms_vio.append([int(xyxy[0]) + (int(xyxy[2]) - int(xyxy[0])) // 2,
                                                          int(xyxy[1]) + (int(xyxy[3]) - int(xyxy[1])) // 2,
                                                          int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])])
                                if names_all_27[int(cls)] == 'cloth_white_false':
                                    list_uniforms_vio.append([int(xyxy[0]) + (int(xyxy[2]) - int(xyxy[0])) // 2,
                                                          int(xyxy[1]) + (int(xyxy[3]) - int(xyxy[1])) // 2,
                                                          int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])])
                                if names_all_27[int(cls)] == 'cloth_lightblue_unnormal_1':
                                    list_uniforms_vio.append([int(xyxy[0]) + (int(xyxy[2]) - int(xyxy[0])) // 2,
                                                          int(xyxy[1]) + (int(xyxy[3]) - int(xyxy[1])) // 2,
                                                          int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])])

                                if names_all_27[int(cls)] == 'cloth_false':
                                    list_cloth_false.append([int(xyxy[0]) + (int(xyxy[2]) - int(xyxy[0])) // 2,
                                                          int(xyxy[1]) + (int(xyxy[3]) - int(xyxy[1])) // 2,
                                                          int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])])

                                if names_all_27[int(cls)] == 'shoes_false':
                                    list_shoes_false.append([int(xyxy[0]) + (int(xyxy[2]) - int(xyxy[0])) // 2,
                                                          int(xyxy[1]) + (int(xyxy[3]) - int(xyxy[1])) // 2,
                                                          int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])])

                                if names_all_27[int(cls)] == 'phone':
                                    list_phone.append([int(xyxy[0]) + (int(xyxy[2]) - int(xyxy[0])) // 2,
                                                          int(xyxy[1]) + (int(xyxy[3]) - int(xyxy[1])) // 2,
                                                          int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])])

                                if names_all_27[int(cls)] == 'head':
                                    list_head.append([int(xyxy[0]) + (int(xyxy[2]) - int(xyxy[0])) // 2,
                                                          int(xyxy[1]) + (int(xyxy[3]) - int(xyxy[1])) // 2,
                                                          int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])])

                                if names_all_27[int(cls)] == 'tricolor_light_on':
                                    list_tricolor_light.append([int(xyxy[0]) + (int(xyxy[2]) - int(xyxy[0])) // 2,
                                                          int(xyxy[1]) + (int(xyxy[3]) - int(xyxy[1])) // 2,
                                                          int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])])

                                if names_all_27[int(cls)] == 'fall_down':
                                    list_fall_down.append([int(xyxy[0]) + (int(xyxy[2]) - int(xyxy[0])) // 2,
                                                                int(xyxy[1]) + (int(xyxy[3]) - int(xyxy[1])) // 2,
                                                                int(xyxy[0]), int(xyxy[1]), int(xyxy[2]),
                                                                int(xyxy[3])])

                                if names_all_27[int(cls)] == 'fire':
                                    list_fire_smog.append([int(xyxy[0]) + (int(xyxy[2]) - int(xyxy[0])) // 2,
                                                                int(xyxy[1]) + (int(xyxy[3]) - int(xyxy[1])) // 2,
                                                                int(xyxy[0]), int(xyxy[1]), int(xyxy[2]),
                                                                int(xyxy[3])])
                                if names_all_27[int(cls)] == 'smog':
                                    list_fire_smog.append([int(xyxy[0]) + (int(xyxy[2]) - int(xyxy[0])) // 2,
                                                                int(xyxy[1]) + (int(xyxy[3]) - int(xyxy[1])) // 2,
                                                                int(xyxy[0]), int(xyxy[1]), int(xyxy[2]),
                                                                int(xyxy[3])])

                                if names_all_27[int(cls)] == 'box_hing':
                                    list_box_hing.append([int(xyxy[0]) + (int(xyxy[2]) - int(xyxy[0])) // 2,
                                                                int(xyxy[1]) + (int(xyxy[3]) - int(xyxy[1])) // 2,
                                                                int(xyxy[0]), int(xyxy[1]), int(xyxy[2]),
                                                                int(xyxy[3])])
                                    list_tongdao.append([int(xyxy[0]) + (int(xyxy[2]) - int(xyxy[0])) // 2,
                                                         int(xyxy[1]) + (int(xyxy[3]) - int(xyxy[1])) // 2,
                                                         int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])])

                                if names_all_27[int(cls)] == 'hand_ball':
                                    list_hand_ball.append([int(xyxy[0]) + (int(xyxy[2]) - int(xyxy[0])) // 2,
                                                                int(xyxy[1]) + (int(xyxy[3]) - int(xyxy[1])) // 2,
                                                                int(xyxy[0]), int(xyxy[1]), int(xyxy[2]),
                                                                int(xyxy[3])])
                                if names_all_27[int(cls)] == 'hand_ball_1':
                                    list_hand_ball_1.append([int(xyxy[0]) + (int(xyxy[2]) - int(xyxy[0])) // 2,
                                                                int(xyxy[1]) + (int(xyxy[3]) - int(xyxy[1])) // 2,
                                                                int(xyxy[0]), int(xyxy[1]), int(xyxy[2]),
                                                                int(xyxy[3])])
                                if names_all_27[int(cls)] == 'flip_box_open':
                                    list_tongdao.append([int(xyxy[0]) + (int(xyxy[2]) - int(xyxy[0])) // 2,
                                                         int(xyxy[1]) + (int(xyxy[3]) - int(xyxy[1])) // 2,
                                                         int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])])
                                if names_all_27[int(cls)] == 'flip_box_close':
                                    list_tongdao.append([int(xyxy[0]) + (int(xyxy[2]) - int(xyxy[0])) // 2,
                                                         int(xyxy[1]) + (int(xyxy[3]) - int(xyxy[1])) // 2,
                                                         int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])])
                                if names_all_27[int(cls)] == 'safety_hat':
                                    list_tongdao.append([int(xyxy[0]) + (int(xyxy[2]) - int(xyxy[0])) // 2,
                                                         int(xyxy[1]) + (int(xyxy[3]) - int(xyxy[1])) // 2,
                                                         int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])])


                            self.obj_pos[index]['car_per_list'] = list_temp_pos
                            self.obj_pos[index]['red_car_per_list'] = list_temp_pos1
                            self.obj_pos[index]['red_carpet'] = list_temp_pos2
                            self.obj_pos[index]['uniforms_vio'] = list_uniforms_vio
                            self.obj_pos[index]['cloth_false'] = list_cloth_false
                            self.obj_pos[index]['shoes_false'] = list_shoes_false
                            self.obj_pos[index]['phone'] = list_phone
                            self.obj_pos[index]['head'] = list_head
                            self.obj_pos[index]['tricolor_light_on'] = list_tricolor_light
                            self.obj_pos[index]['fall_down'] = list_fall_down
                            self.obj_pos[index]['fire_smog'] = list_fire_smog
                            self.obj_pos[index]['box_hing'] = list_box_hing
                            self.obj_pos[index]['hand_ball'] = list_hand_ball
                            self.obj_pos[index]['hand_ball_1'] = list_hand_ball_1
                            self.obj_pos[index]['tongdao'] = list_tongdao
                            self.obj_pos[index]['table_yes'] = list_table_yes
                            self.obj_pos[index]['grounding'] = list_grounding
            e_time = time.time()

            # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>", e_time-s_time)
            time.sleep(0.02)



class show_phtotSubUI(QtWidgets.QMainWindow, showphoto0527.Ui_MainWindow):

    def __init__(self, parent=None):
        super(show_phtotSubUI, self).__init__(parent)

        self.setupUi(self)
        self.setWindowTitle("视频显示")
        self.open_flag = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True,
                          True]

        self.socket_server = SocketServer()
        self.socket_aserver.setDaemon(True)
        self.socket_server.start()




        self.set_alarm_pushButton.clicked.connect(lambda: self.set_alarm())
        self.stop_alarm_pushButton.clicked.connect(lambda: self.stop_alarm())

        self.thread_rtsp_start()
        self.thread_detect_start()

        self.detect_flag = True

        self.ILSP_test = Islongstandperson()
        self.ILSP_test.set_camera_num(15) #she zhi she xiang tou ge shu

        self.False_work = Islongfalsework()
        self.False_work.set_camera_num(15)

        self.ICAR_test = IsredcarpetInredcar()
        self.ICAR_test.set_camera_num(15)


        self.set_room1_pushButton.clicked.connect(lambda: self.set_message(0))
        self.set_room2_pushButton.clicked.connect(lambda: self.set_message(1))
        self.set_room3_pushButton.clicked.connect(lambda: self.set_message(2))
        self.set_room4_pushButton.clicked.connect(lambda: self.set_message(3))
        self.set_room5_pushButton.clicked.connect(lambda: self.set_message(4))
        self.set_room6_pushButton.clicked.connect(lambda: self.set_message(5))

        self.room_dict = {
            0: 0,
            1: 0,
            2: 0,
            3: 0,
            4: 1,
            5: 1,
            6: 1,
            7: 2,
            8: 2,
            9: 3,
            10: 1,
            11: 4,
            12: 5,
            13: 6,
            14: 7,
        }

        self.camera_numbers = 15

        self.vio_names = ["perroom_person_overcount", "uniforms_vio", "cloth_false",
                           "shoes_false", "phone", "head", "fall_down", "fire_smog",
                          "box_hing","car_and_carpet","car_long_people","false_work", "hand_ball",'tongdao',"table_yes","grounding"]
        self.alarm_message_dict = {"人数超员": 0, "着装不当": 0, "衣服错误": 0, "鞋子不当": 0, "使用手机": 0, "安全帽": 0, "摔倒": 0,
             "烟雾火焰": 0, "开箱": 0, "红垫子": 0, "小车站人": 0, "炮弹未接地": 0, "未除静电": 0,"单人违规作业": 0,'通道占用':0}

        self.vio_carema_list = [[0] * self.camera_numbers, [0] * self.camera_numbers, [0] * self.camera_numbers,
                                  [0] * self.camera_numbers, [0] * self.camera_numbers, [0] * self.camera_numbers,
                                  [0] * self.camera_numbers, [0] * self.camera_numbers, [0] * self.camera_numbers,
                                [0] * self.camera_numbers,[0] * self.camera_numbers,[0] * self.camera_numbers,
                                [0] * self.camera_numbers,[0] * self.camera_numbers,[0] * self.camera_numbers,[0] * self.camera_numbers]

        self.vio_flag = dict(zip(self.vio_names, self.vio_carema_list))
        self.alarm_or_not = True
        self.mydatabase = mydatabase()

        self.OrdinaryViolation_test = isOrdinaryViolation()
        self.OrdinaryViolation_test.set_camera_num(15)  # she zhi she xiang tou ge shu
        self.OrdinaryViolation_test.set_continue_num(30)
        self.OrdinaryViolation_test.set_total_num(14)


        self.OrdinaryViolation_test.set_room_dict(self.room_dict)

        self.start_time = 0
        self.per_start_time = 0

        # parent_conn, child_conn = multiprocessing.Pipe()
        #
        # process_restart = multiprocessing.Process(target=self.process_restart, args=(child_conn, self.restart_state))
        # process_restart.daemon = True
        # process_restart.start()





    # 一键开启函数
    def thread_rtsp_start(self):
        thread_rtsp = Thread(target=self.thread_rtsp)
        thread_rtsp.setDaemon(True)  # ==========================================================
        thread_rtsp.start()
        # thread_rtsp.join()

        # 不行就多睡会

        time.sleep(3)

    def thread_detect_start(self):

        thread_detect = Thread(target=self.thread_pic_detect)
        thread_detect.setDaemon(True)  # ====================================================
        thread_detect.start()
        # thread_detect.join()

    def thread_rtsp(self):

        self.rtsp_number = 15

        self.Rtsp_all = [
            ["rtsp://admin:a1234567@192.168.80.192/h264/ch1/main/av_stream", 100],  # z1西
            ["rtsp://admin:a1234567@192.168.80.200/h264/ch1/main/av_stream", 100],  # z1东
            ["rtsp://admin:a1234567@192.168.80.204/h264/ch1/main/av_stream", 100],  # z1zcj1
            ["rtsp://admin:a1234567@192.168.80.198/h264/ch1/main/av_stream", 100],  # z1zcj2
            ["rtsp://admin:a1234567@192.168.80.196/h264/ch1/main/av_stream", 100],  # z2东
            ["rtsp://admin:a1234567@192.168.80.195/h264/ch1/main/av_stream", 100],  # z2西
            ["rtsp://admin:a1234567@192.168.80.194/h264/ch1/main/av_stream", 100],  # z2内
            ["rtsp://admin:a1234567@192.168.80.193/h264/ch1/main/av_stream", 100],  # z3西
            ["rtsp://admin:a1234567@192.168.80.205/h264/ch1/main/av_stream", 100],  # z3东
            ["rtsp://admin:a1234567@192.168.80.202/h264/ch1/main/av_stream", 100],  # z4
            ["rtsp://admin:a1234567@192.168.80.186/h264/ch1/main/av_stream", 100],  # z2内shdd
            ["rtsp://admin:a1234567@192.168.80.199/h264/ch1/main/av_stream", 100],  # wllclj
            ["rtsp://admin:a1234567@192.168.80.201/h264/ch1/main/av_stream", 100],  # 陀螺仪
            ["rtsp://admin:a1234567@192.168.80.188/h264/ch1/main/av_stream", 100],  # zld
            ["rtsp://admin:a1234567@192.168.80.191/h264/ch1/main/av_stream", 100],  # 火工区西门
        ]

        self.frame_queue = []
        for i in range(self.rtsp_number):
            # print(i)
            queue_rtsp = multiprocessing.Queue()
            self.frame_queue.append(queue_rtsp)
            # time.sleep(0.1)

        parent_conn, child_conn = multiprocessing.Pipe()

        processes = []
        process = multiprocessing.Process(target=ly_Camera, args=(child_conn, self.Rtsp_all[0:2], self.frame_queue, 0))
        process1 = multiprocessing.Process(target=ly_Camera, args=(child_conn, self.Rtsp_all[2:4], self.frame_queue, 2))
        process2 = multiprocessing.Process(target=ly_Camera, args=(child_conn, self.Rtsp_all[4:6], self.frame_queue, 4))
        process3 = multiprocessing.Process(target=ly_Camera, args=(child_conn, self.Rtsp_all[6:8], self.frame_queue, 6))
        process4 = multiprocessing.Process(target=ly_Camera, args=(child_conn, self.Rtsp_all[8:10], self.frame_queue, 8))
        process5 = multiprocessing.Process(target=ly_Camera, args=(child_conn, self.Rtsp_all[10:12], self.frame_queue, 10))
        process6 = multiprocessing.Process(target=ly_Camera, args=(child_conn, self.Rtsp_all[12:14], self.frame_queue, 12))
        process7 = multiprocessing.Process(target=ly_Camera, args=(child_conn, self.Rtsp_all[14:16], self.frame_queue, 14))

        processes.append(process)
        processes.append(process1)
        processes.append(process2)
        processes.append(process3)
        processes.append(process4)
        processes.append(process5)
        processes.append(process6)
        processes.append(process7)

        # 启动子进程
        for process in processes:
            process.daemon = True  # 守护进程
            process.start()
        for process in processes:
            process.join()


    def thread_pic_detect(self):
        # queuq是传输图片
        # pipe是传输index，并传回给主进程
        # queuq是传输图片
        # pipe是传输index，并传回给主进程
        # def stop_thread_pic_detect():
        #     pass
        self.result_list = []
        for i in range(20):
            dict = {}
            dict = Manager().dict()
            dict["camera_index"] = 0
            dict["person"] = 0
            for name in names_all_27:
                dict[name] = 0
            self.result_list.append(dict)

        self.obj_pos = []
        for i in range(20):
            dict = Manager().dict()
            dict["person_list"] = []
            dict["car_per_list"] = []
            dict["red_car_per_list"] = []
            dict["red_carpet"] = []
            dict["uniforms_vio"] = []
            dict["cloth_false"] = []
            dict["shoes_false"] = []
            dict["phone"] = []
            dict["phone1"] = []
            dict["head"] = []
            dict["tricolor_light_on"] = []
            dict["fall_down"] = []
            dict["fire_smog"] = []
            dict["box_hing"] = []
            dict["hand_ball"] = []
            dict["hand_ball_1"] = []
            dict["tongdao"] = []
            dict["table_yes"] = []
            dict["grounding"] = []
            self.obj_pos.append(dict)

        self.frame_queue_save = []
        for i in range(self.rtsp_number):
            # print(i)
            queue_rtsp = multiprocessing.Queue()
            self.frame_queue_save.append(queue_rtsp)
            # time.sleep(0.1)

        self.load_model = []

        # print("pic_queue_size:+++++++++++++++++++++++++", len(self.frame_queue))


        for i in range(4):
            dict = Manager().dict()
            dict["flag"] = True
            self.load_model.append(dict)


        parent_conn, child_conn = multiprocessing.Pipe()
        # for frame in self.frame_queue:


        for i in range(4):
            if i != 3:

                process = multiprocessing.Process(target=pic_detect, args=(child_conn, self.frame_queue[i*4:(i+1)*4],
                                                                       self.result_list[i*4:(i+1)*4], self.obj_pos[i*4:(i+1)*4], 4, i,
                                                                       self.frame_queue_save[i*4:(i+1)*4],
                                                                       self.load_model[0]["flag"]))
            else:
                process = multiprocessing.Process(target=pic_detect, args=(child_conn, self.frame_queue[12:15],
                                                                           self.result_list[12:15], self.obj_pos[12:15],
                                                                           3, 3, self.frame_queue_save[12:15],
                                                                           self.load_model[3]["flag"]))
            process.daemon = True  # 守护进程
            process.start()

        # process = multiprocessing.Process(target=pic_detect, args=(child_conn, self.frame_queue[0:4],
        #                                                            self.result_list[0:4], self.obj_pos[0:4], 4, 0, self.frame_queue_save[0:4], self.load_model[0]["flag"]))
        # process1 = multiprocessing.Process(target=pic_detect, args=(child_conn, self.frame_queue[4:8],
        #                                                             self.result_list[4:8], self.obj_pos[4:8], 4, 1, self.frame_queue_save[4:8], self.load_model[1]["flag"]))
        # process2 = multiprocessing.Process(target=pic_detect, args=(child_conn, self.frame_queue[8:12],
        #                                                             self.result_list[8:12], self.obj_pos[8:12], 4, 2, self.frame_queue_save[8:12], self.load_model[2]["flag"]))
        # process3 = multiprocessing.Process(target=pic_detect, args=(child_conn, self.frame_queue[12:15],
        #                                                             self.result_list[12:15], self.obj_pos[12:15], 3, 3, self.frame_queue_save[12:15], self.load_model[3]["flag"]))
        # process4 = multiprocessing.Process(target=pic_detect, args=(child_conn, self.frame_queue[12:15],
        #                                                             self.result_list[12:15], self.obj_pos[12:15], 3, 4,
        #                                                             self.frame_queue_save[12:15],
        #                                                             self.load_model[3]["flag"]))
        #
        #
        # # 启动子进程
        # process.daemon = True  # 守护进程
        # process1.daemon = True  # 守护进程
        # process2.daemon = True  # 守护进程
        # process3.daemon = True  # 守护进程
        # # process4.daemon = True  # 守护进程
        # # process5.daemon = True  # 守护进程
        # # process6.daemon = True  # 守护进程
        #
        # process.start()
        # process1.start()
        # process2.start()
        # process3.start()

        # process.join()
        # process1.join()
        # process2.join()
        # process3.join()


        # self.result_list是列表，里面有16个摄像头的结果，是16个字典
        # 每个字典里面有28个类的结果，存储的是检测到的物体的数量（人+我们模型的类）
        # 比如人的数量是：self.result_list[0]['person']
        # 头的数量是：self.result_list[0]['head']

        while self.load_model[0]["flag"] and self.load_model[1]["flag"] and self.load_model[2]["flag"] and self.load_model[3]["flag"]:
            list_obj_dict = []
            # luchenkai congzhekaishi
            #15是摄像头数量,记得修改
            for i in range(15):
                list_obj_dict.append({"person_list": self.obj_pos[i]["person_list"],
                                  "car_per_list": self.obj_pos[i]["car_per_list"],
                                  "red_car_per_list": self.obj_pos[i]["red_car_per_list"],
                                  "red_carpet": self.obj_pos[i]["red_carpet"],
                                  "uniforms_vio": self.obj_pos[i]["uniforms_vio"],
                                  "cloth_false": self.obj_pos[i]["cloth_false"],
                                  "shoes_false": self.obj_pos[i]["shoes_false"],
                                  "phone": self.obj_pos[i]["phone"],
                                  "phone1": self.obj_pos[i]["phone1"],
                                  "head": self.obj_pos[i]["head"],
                                  "tricolor_light_on": self.obj_pos[i]["tricolor_light_on"],
                                  "fall_down": self.obj_pos[i]["fall_down"],
                                  "fire_smog": self.obj_pos[i]["fire_smog"],
                                  "box_hing": self.obj_pos[i]["box_hing"],
                                  "hand_ball": self.obj_pos[i]["hand_ball"],
                                  "hand_ball_1": self.obj_pos[i]["hand_ball_1"],
                                  "tongdao":self.obj_pos[i]["tongdao"],
                                  "table_yes": self.obj_pos[i]["table_yes"],
                                  "grounding": self.obj_pos[i]["grounding"]
                                  })
                # print("---------------------self.obj_pos[i]person_list]--------------------", i,self.obj_pos[i]["person_list"])
            sum_person_overcount_list = self.OrdinaryViolation_test.is_sum_person_overcount(list_obj_dict)
            perroom_person_num = self.OrdinaryViolation_test.perroom_person(list_obj_dict)


            # set label !!!!!!!!!!!!!!!!!!!!!!!
            self.set_perroom_person(perroom_person_num[0: -1])
            self.set_sum_person(str(sum_person_overcount_list[0]))


            perroom_person_overcount_roomlist, perroom_person_overcount_caremalist = self.OrdinaryViolation_test.is_perroom_person_overcount(perroom_person_num)
            uniforms_vio_roomlist, uniforms_vio_caremalist = self.OrdinaryViolation_test.is_uniforms_vio(list_obj_dict)
            cloth_false_roomlist, cloth_false_caremalist = self.OrdinaryViolation_test.is_cloth_false(list_obj_dict)
            shoes_false_roomlist, shoes_false_caremalist = self.OrdinaryViolation_test.is_shoes_false(list_obj_dict)
            phone_roomlist, phone_caremalist = self.OrdinaryViolation_test.is_phone(list_obj_dict)
            safety_hat_roomlist, safety_hat_caremalist = self.OrdinaryViolation_test.is_safety_hat(list_obj_dict)
            fall_down_roomlist, fall_down_caremalist = self.OrdinaryViolation_test.is_fall_down(list_obj_dict)
            fire_smog_roomlist, fire_smog_caremalist = self.OrdinaryViolation_test.is_fire_smog(list_obj_dict)
            box_hing_roomlist, box_hing_caremalist = self.OrdinaryViolation_test.is_box_hing(list_obj_dict)
            tongdao_roomlist, tongdao_caremalist = self.OrdinaryViolation_test.is_tongdao(list_obj_dict)
            table_yes_roomlist , table_yes_caremalist = self.OrdinaryViolation_test.is_table_yes(list_obj_dict)
            ICAR_list, ICAR_list1 = self.ICAR_test.is_redcarpet_under_car(list_obj_dict, 200, 300)
            ILSP_list, ILSP_list1 = self.ILSP_test.is_person_longstand_car(list_obj_dict, 200, 300)  # 小车是否长时间站人
            False_list , False_list1 = self.False_work.is_person_false_work(list_obj_dict,200,300)
            hand_ball_roomint, hand_ball_caremalist = self.OrdinaryViolation_test.is_hand_ball(list_obj_dict)
            grounding_list,grounding_caremalist = self.OrdinaryViolation_test.is_grounding(list_obj_dict)

            self.year_label.setText(time.strftime("%Y-%m-%d %H-%M-%S", time.localtime()).split(' ')[0])
            self.time_label.setText(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()).split(' ')[1])

            end_time = time.time()
            if self.per_start_time == 0:
                # mydatabase
                res3 = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())
                self.mydatabase.insert_person_num("0", res3, perroom_person_num[0])
                self.mydatabase.insert_person_num("1", res3, perroom_person_num[1])
                self.mydatabase.insert_person_num("2", res3, perroom_person_num[2])
                self.mydatabase.insert_person_num("3", res3, perroom_person_num[3])
                self.mydatabase.insert_person_num("4", res3, perroom_person_num[4])
                self.mydatabase.insert_person_num("5", res3, perroom_person_num[5])

                self.per_start_time = end_time


            if 600 < end_time - self.per_start_time < 601:
                res3 = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())
                self.mydatabase.insert_person_num("0", res3, perroom_person_num[0])
                self.mydatabase.insert_person_num("1", res3, perroom_person_num[1])
                self.mydatabase.insert_person_num("2", res3, perroom_person_num[2])
                self.mydatabase.insert_person_num("3", res3, perroom_person_num[3])
                self.mydatabase.insert_person_num("4", res3, perroom_person_num[4])
                self.mydatabase.insert_person_num("5", res3, perroom_person_num[5])

                self.per_start_time = end_time

            if self.alarm_message_dict['人数超员'] == 0:
                sum_person_overcount_list[1] = 0
                perroom_person_overcount_roomlist = [0]*len(perroom_person_overcount_roomlist)
            if self.alarm_message_dict['着装不当'] == 0:
                uniforms_vio_roomlist = [0]*len(uniforms_vio_roomlist)
            if self.alarm_message_dict['衣服错误'] == 0:
                cloth_false_roomlist = [0]*len(cloth_false_roomlist)
            if self.alarm_message_dict['鞋子不当'] == 0:
                shoes_false_roomlist = [0]*len(shoes_false_roomlist)
            if self.alarm_message_dict['使用手机'] == 0:
                phone_roomlist = [0]*len(phone_roomlist)
            if self.alarm_message_dict['安全帽'] == 0:
                safety_hat_roomlist = [0]*len(safety_hat_roomlist)
            if self.alarm_message_dict['摔倒'] == 0:
                fall_down_roomlist = [0]*len(fall_down_roomlist)
            if self.alarm_message_dict['烟雾火焰'] == 0:
                fire_smog_roomlist = [0]*len(fire_smog_roomlist)
            if self.alarm_message_dict['开箱'] == 0:
                box_hing_roomlist = [0]*len(box_hing_roomlist)
            if self.alarm_message_dict['红垫子'] == 0:
                ICAR_list = [0]*len(ICAR_list)
            if self.alarm_message_dict['小车站人'] == 0:
                ILSP_list = [0]*len(ILSP_list)
            if self.alarm_message_dict['单人违规作业'] == 0:
                False_list = [0]*len(False_list)
            if self.alarm_message_dict['炮弹未接地'] == 0:
                grounding_list = [0]*len(grounding_list)
            if self.alarm_message_dict['未除静电'] == 0:
                hand_ball_roomint = 0

            room_vio_str = self.OrdinaryViolation_test.room_list_or(perroom_person_overcount_roomlist, uniforms_vio_roomlist, cloth_false_roomlist,
                                                                     shoes_false_roomlist, phone_roomlist, safety_hat_roomlist, fall_down_roomlist,
                                                                     fire_smog_roomlist, box_hing_roomlist,ICAR_list,ILSP_list,False_list,grounding_list)
            perroom_person_num_list = ",".join(map(str, perroom_person_num[0:6]))
            vio_result = "@" + perroom_person_num_list+ ',' + str(sum_person_overcount_list[0]) + '@' + room_vio_str + str(sum_person_overcount_list[1] or hand_ball_roomint)+ '@'

            if sum_person_overcount_list[1] or hand_ball_roomint == 1:
                print("------------------------", vio_result)

            # self.set_perroom_person(perroom_person_num)
            # self.set_sum_person(str(sum_person_overcount_list[0]))

            # socket发送数据
            if self.alarm_or_not:
                end_time = time.time()
                if self.start_time == 0:
                    self.socket_server.send_alarm_massage(vio_result)
                    self.start_time = end_time
                if 3 < end_time - self.start_time < 3.5:

                    self.socket_server.send_alarm_massage(vio_result)
                    self.start_time = end_time
            else:
                end_time = time.time()
                if self.start_time == 0:
                    self.socket_server.send_alarm_massage('@0,0,0,0,0,0,0@0000000@')
                    self.start_time = end_time
                if 3 < end_time - self.start_time < 3.5:

                    self.socket_server.send_alarm_massage('@0,0,0,0,0,0,0@0000000@')
                    self.start_time = end_time



            self.save_record_delay(perroom_person_overcount_caremalist, uniforms_vio_caremalist, cloth_false_caremalist,
                                       shoes_false_caremalist, phone_caremalist, safety_hat_caremalist, fall_down_caremalist,
                                   fire_smog_caremalist, box_hing_caremalist,ICAR_list1, ILSP_list1,False_list1,
                                   tongdao_caremalist,table_yes_caremalist,hand_ball_caremalist,grounding_caremalist)


            list_room_vio = [0,0,0,0,0,0]
            for i in range(6):
                if perroom_person_overcount_roomlist[i] == 1:
                    list_room_vio[i] = 1
                if uniforms_vio_roomlist[i] == 1:
                    list_room_vio[i] = 1
                if cloth_false_roomlist[i] == 1:
                    list_room_vio[i] = 1
                if shoes_false_roomlist[i] == 1:
                    list_room_vio[i] = 1
                if phone_roomlist[i] == 1:
                    list_room_vio[i] = 1
                if safety_hat_roomlist[i] == 1:
                    list_room_vio[i] = 1
                if fall_down_roomlist[i] == 1:
                    list_room_vio[i] = 1
                if fire_smog_roomlist[i] == 1:
                    list_room_vio[i] = 1
                if box_hing_roomlist[i] == 1:
                    list_room_vio[i] = 1


    def set_alarm(self):
        set_alarm = Set_alarm_param()
        if set_alarm.exec_() == QtWidgets.QDialog.Accepted:
            total_person_limit, self.alarm_message_dict = set_alarm.get_alarmmessage()
            self.total_person_label_4.setText(str(total_person_limit))
            self.OrdinaryViolation_test.set_total_num(total_person_limit)

        return self.alarm_message_dict

    def stop_alarm(self):
        if self.stop_alarm_pushButton.text().split("\n")[0] == "停 止":
            self.alarm_or_not = False
            self.stop_alarm_pushButton.setText("恢 复\n报 警")
        else:
            self.alarm_or_not = True
            self.stop_alarm_pushButton.setText("停 止\n报 警")

    def save_record_delay(self, list0, list1, list2, list3, list4, list5, list6, list7, list8 ,list9 ,list10,list11,list12,list13,list14,list15):

        # 15是摄像头号
        for i in range(15):
            # print(list0[i])
            if list0[i] == 1:
                if self.vio_flag["perroom_person_overcount"][i] == 0:
                    print("perroom_person_overcount保存图片")
                    self.vio_flag["perroom_person_overcount"][i] = 1
                    vio_name = strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
                    pic_name = str(vio_name) + "-" +str(i)

                    mydatabase().mysql_write_violate(self.room_dict[i], vio_name, "0", pic_name)

                    self.save_picture(pic_name, i,self.room_dict[i])
                    time.sleep(0.08)
            else:
                self.vio_flag["perroom_person_overcount"][i] = 0

            if list1[i] == 1:
                if self.vio_flag["uniforms_vio"][i] == 0:
                    print("uniforms_vio保存图片")
                    self.vio_flag["uniforms_vio"][i] = 1
                    vio_name = strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
                    pic_name = str(vio_name) + "-" +str(i)
                    mydatabase().mysql_write_violate(self.room_dict[i], vio_name, "1", pic_name)
                    self.save_picture(pic_name, i, self.room_dict[i])
                    time.sleep(0.08)
            else:
                self.vio_flag["uniforms_vio"][i] = 0

            if list2[i] == 1:
                if self.vio_flag["cloth_false"][i] == 0:
                    print("cloth_false保存图片")
                    self.vio_flag["cloth_false"][i] = 1
                    vio_name = strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
                    pic_name = str(vio_name) + "-" +str(i)
                    mydatabase().mysql_write_violate(self.room_dict[i], vio_name, "2", pic_name)
                    self.save_picture(pic_name, i, self.room_dict[i])
                    time.sleep(0.08)
            else:
                self.vio_flag["cloth_false"][i] = 0

            if list3[i] == 1:
                if self.vio_flag["shoes_false"][i] == 0:
                    print("shoes_false保存图片")
                    self.vio_flag["shoes_false"][i] = 1
                    vio_name = strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
                    pic_name = str(vio_name) + "-" +str(i)
                    mydatabase().mysql_write_violate(self.room_dict[i], vio_name, "3", pic_name)
                    self.save_picture(pic_name, i, self.room_dict[i])
                    time.sleep(0.08)
            else:
                self.vio_flag["shoes_false"][i] = 0

            if list4[i] == 1:
                if self.vio_flag["phone"][i] == 0:
                    print("phone保存图片")
                    self.vio_flag["phone"][i] = 1
                    vio_name = strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
                    pic_name = str(vio_name) + "-" +str(i)
                    mydatabase().mysql_write_violate(self.room_dict[i], vio_name, "4", pic_name)
                    self.save_picture(pic_name, i, self.room_dict[i])
                    time.sleep(0.08)
            else:
                self.vio_flag["phone"][i] = 0

            if list5[i] == 1:
                if self.vio_flag["head"][i] == 0:
                    print("head保存图片")
                    self.vio_flag["head"][i] = 1
                    vio_name = strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
                    pic_name = str(vio_name) + "-" +str(i)
                    mydatabase().mysql_write_violate(self.room_dict[i], vio_name, "5", pic_name)
                    self.save_picture(pic_name, i, self.room_dict[i])
                    time.sleep(0.08)
            else:
                self.vio_flag["head"][i] = 0

            if list6[i] == 1:
                if self.vio_flag["fall_down"][i] == 0:
                    print("fall_down保存图片")
                    self.vio_flag["fall_down"][i] = 1
                    vio_name = strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
                    pic_name = str(vio_name) + "-" +str(i)
                    mydatabase().mysql_write_violate(self.room_dict[i], vio_name, "6", pic_name)
                    self.save_picture(pic_name, i, self.room_dict[i])
                    time.sleep(0.08)
            else:
                self.vio_flag["fall_down"][i] = 0

            if list7[i] == 1:
                if self.vio_flag["fire_smog"][i] == 0:
                    print("fire_smog保存图片")
                    self.vio_flag["fire_smog"][i] = 1
                    vio_name = strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
                    pic_name = str(vio_name) + "-" +str(i)
                    mydatabase().mysql_write_violate(self.room_dict[i], vio_name, "7", pic_name)
                    self.save_picture(pic_name, i, self.room_dict[i])
                    time.sleep(0.08)
            else:
                self.vio_flag["fire_smog"][i] = 0

            if list8[i] == 1:
                if self.vio_flag["box_hing"][i] == 0:
                    print("box_hing保存图片")
                    self.vio_flag["box_hing"][i] = 1
                    vio_name = strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
                    pic_name = str(vio_name) + "-" +str(i)
                    mydatabase().mysql_write_violate(self.room_dict[i], vio_name, "8", pic_name)
                    self.save_picture(pic_name, i, self.room_dict[i])
                    time.sleep(0.08)
            else:
                self.vio_flag["box_hing"][i] = 0

            if list9[i] == 1:
                if self.vio_flag["car_and_carpet"][i] == 0:
                    print("hongdianzi保存图片")
                    self.vio_flag["car_and_carpet"][i] = 1
                    vio_name = strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
                    pic_name = str(vio_name) + "-" +str(i)
                    mydatabase().mysql_write_violate(self.room_dict[i], vio_name, "9", pic_name)
                    self.save_picture(pic_name, i, self.room_dict[i])
                    time.sleep(0.08)
            else:
                self.vio_flag["car_and_carpet"][i] = 0

            if list10[i] == 1:
                if self.vio_flag["car_long_people"][i] == 0:
                    print("paodan保存图片")
                    self.vio_flag["car_long_people"][i] = 1
                    vio_name = strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
                    pic_name = str(vio_name) + "-" +str(i)
                    mydatabase().mysql_write_violate(self.room_dict[i], vio_name, "10", pic_name)
                    self.save_picture(pic_name, i, self.room_dict[i])
                    time.sleep(0.08)
            else:
                self.vio_flag["car_long_people"][i] = 0

            if list11[i] == 1:
                if self.vio_flag["false_work"][i] == 0:
                    print("false_work保存图片")
                    self.vio_flag["false_work"][i] = 1
                    vio_name = strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
                    pic_name = str(vio_name) + "-" +str(i)
                    mydatabase().mysql_write_violate(self.room_dict[i], vio_name, "11", pic_name)
                    self.save_picture(pic_name, i, self.room_dict[i])
                    time.sleep(0.08)
            else:
                self.vio_flag["false_work"][i] = 0

            if list12[i] == 1:
                if self.vio_flag["tongdao"][i] == 0:
                    print("tongdao保存图片")
                    self.vio_flag["tongdao"][i] = 1
                    vio_name = strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
                    pic_name = str(vio_name) + "-" +str(i)
                    mydatabase().mysql_write_violate(self.room_dict[i], vio_name, "12", pic_name)
                    self.save_picture(pic_name, i, self.room_dict[i])
                    time.sleep(0.08)
            else:
                self.vio_flag["tongdao"][i] = 0

            if list13[i] == 1:
                if self.vio_flag["table_yes"][i] == 0:
                    print("table_yes保存图片")
                    self.vio_flag["table_yes"][i] = 1
                    vio_name = strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
                    pic_name = str(vio_name) + "-" +str(i)
                    mydatabase().mysql_write_violate(self.room_dict[i], vio_name, "13", pic_name)
                    self.save_picture(pic_name, i, self.room_dict[i])
                    time.sleep(0.08)
            else:
                self.vio_flag["table_yes"][i] = 0

            if list14[i] == 1:
                if self.vio_flag["hand_ball"][i] == 0:
                    print("hand_ball保存图片")
                    self.vio_flag["hand_ball"][i] = 1
                    vio_name = strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
                    pic_name = str(vio_name) + "-" +str(i)
                    mydatabase().mysql_write_violate(self.room_dict[i], vio_name, "14", pic_name)
                    self.save_picture(pic_name, i, self.room_dict[i])
                    time.sleep(0.08)
            else:
                self.vio_flag["hand_ball"][i] = 0

            if list15[i] == 1:
                if self.vio_flag["grounding"][i] == 0:
                    print("grounding保存图片")
                    self.vio_flag["grounding"][i] = 1
                    vio_name = strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
                    pic_name = str(vio_name) + "-" +str(i)
                    mydatabase().mysql_write_violate(self.room_dict[i], vio_name, "15", pic_name)
                    self.save_picture(pic_name, i, self.room_dict[i])
                    time.sleep(0.08)
            else:
                self.vio_flag["grounding"][i] = 0

    def save_picture(self,pic_name,camera_index,room_index):
        Flag = False
        try:
            # print("----------self.frame_queue_save[index].qsize()-----------",self.frame_queue_save[camera_index].qsize())
            img = self.frame_queue_save[camera_index].get(timeout=1)

            Flag = True
        except:
            print("no image saved")

        if Flag:
            path_picture = "pictureSave/room"+ str(room_index) + "/"
            filename = path_picture + pic_name + ".jpg"
            # print("保存图片的路径是",filename)
            cv2.imwrite(filename, img)
            time.sleep(0.05)


    #当违规事件发生时，在前端改变label，用图片显示

    def show_picture_list(self, index, color):
        if color == "unnormal":
            img1 = cv2.imread("peoplepic/wrong.png")
        elif color == "normal":
            img1 = cv2.imread("peoplepic/right.png")
        else:
            img1 = cv2.imread("peoplepic/right.jpg")
        self.show_pic(img1, self.label_status[index], self.label_status[index].width(), self.label_status[index].height())
        time.sleep(0.01)
    def show_pic(self, img, label, width, height):
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img2 = cv2.resize(img2, (width, height))
        image = QtGui.QImage(img2.data, img2.shape[1], img2.shape[0], img2.shape[1] * 3, QtGui.QImage.Format_RGB888)
        jpg = QtGui.QPixmap(image)
        label.setPixmap(jpg)
        QtWidgets.QApplication.processEvents()

    def set_message(self, pushbutton_index):
        set_message = Setmessage()
        num2room = {0: "装配间1", 1: "装配间2", 2: "装配间3", 3: "装配间4", 4: "物理量测量间", 5: "陀螺仪装配间"}
        set_message.setCurrentIndex(num2room[pushbutton_index])
        set_message.center()
        message_list = []
        if set_message.exec_() == QtWidgets.QDialog.Accepted:
            self.work_content, self.power_amount, self.limit_peo = set_message.get_message()

            self.OrdinaryViolation_test.number_per_room_list[pushbutton_index] = int(self.limit_peo)
            if pushbutton_index == 0:
                self.room1_work_content_label.setText(str(self.work_content))
                self.room1_power_amount_label.setText(str(self.power_amount))
                self.room1_limit_people_count_label.setText(str(self.limit_peo))
            if pushbutton_index == 1:
                self.room2_work_content_label.setText(str(self.work_content))
                self.room2_power_amount_label.setText(str(self.power_amount))
                self.room2_limit_people_count_label.setText(str(self.limit_peo))
            if pushbutton_index == 2:
                self.room3_work_content_label.setText(str(self.work_content))
                self.room3_power_amount_label.setText(str(self.power_amount))
                self.room3_limit_people_count_label.setText(str(self.limit_peo))
            if pushbutton_index == 3:
                self.room4_work_content_label.setText(str(self.work_content))
                self.room4_power_amount_label.setText(str(self.power_amount))
                self.room4_limit_people_count_label.setText(str(self.limit_peo))
            if pushbutton_index == 4:
                self.room5_work_content_label.setText(str(self.work_content))
                self.room5_power_amount_label.setText(str(self.power_amount))
                self.room5_limit_people_count_label.setText(str(self.limit_peo))
            if pushbutton_index == 5:
                self.room6_work_content_label.setText(str(self.work_content))
                self.room6_power_amount_label.setText(str(self.power_amount))
                self.room6_limit_people_count_label.setText(str(self.limit_peo))

            message_list.append(str(pushbutton_index))
            message_list.append(str(num2room[pushbutton_index]))
            message_list.append(str(self.limit_peo))
            message_list.append(str(self.work_content))
            message_list.append(str(self.power_amount))

            message_result = "#" + "#".join(message_list) + "#"

            print("message_result", message_result)

            # time.sleep(2.2)
            # self.socket_server.send_alarm_massage(message_result)
            for i in range(3):
                time.sleep(1)
                self.socket_server.send_alarm_massage(message_result)

