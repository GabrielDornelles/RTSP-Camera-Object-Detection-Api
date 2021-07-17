# -*- coding: utf-8 -*-
import torch
from torchvision import transforms
import torchvision
import numpy as np
import cv2
import time
import random
import os

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

USER = os.getenv("USER")
PASSWORD = os.getenv("PASSWORD")
CAMERA_IP = os.getenv("CAMERA_IP")
RTSP_PORT = os.getenv("RTSP_PORT", "554")
CUSTOM_URL = os.getenv("CUSTOM_URL",None)

class ObjectDetection:
    """
    Class with object detection and RTPS.
    """

    def __init__(self, channel ,gpu: bool = False):
        self.gpu = gpu
        self.channel = str(channel)
        self.device = torch.device("cuda") if self.gpu else torch.device("cpu")
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.to(self.device)
        self.model.eval()
        self.categories =  [
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
        'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
        'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

    def get_prediction_gpu(self,img, threshold=0.8):
        transform = transforms.Compose([transforms.ToTensor()])
        img = transform(img)
        img = img.cuda()
        pred = self.model([img]) 
        pred_class = [self.categories[i] for i in list(pred[0]['labels'].cpu().numpy())] 
        pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].cpu().detach().numpy())] 
        pred_score = list(pred[0]['scores'].cpu().detach().numpy())
        pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]
        pred_box = pred_boxes[:pred_t+1]
        pred_class = pred_class[:pred_t+1]
        return pred_box, pred_class
    
    def get_prediction_cpu(self,img, threshold=0.8):
        transform = transforms.Compose([transforms.ToTensor()])
        img = transform(img)
        pred = self.model([img]) 
        pred_class = [self.categories[i] for i in list(pred[0]['labels'].numpy())] 
        pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())] 
        pred_score = list(pred[0]['scores'].detach().numpy())
        pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]
        pred_box = pred_boxes[:pred_t+1]
        pred_class = pred_class[:pred_t+1]
        return pred_box, pred_class
    
    def object_detection(self,img,threshold=0.8, rect_th=3, text_size=2, text_th=2):
        #img = cv2.resize(img,None,fx=1, fy=0.7, interpolation=cv2.INTER_CUBIC)
        boxes, pred_clas = self.get_prediction_gpu(img, threshold=threshold) if self.gpu else self.get_prediction_cpu(img, threshold=threshold)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for i in range(len(boxes)):
            r, g, b = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) # Random Color
            #if any(x in pred_clas[i] for x in right_class):
            cv2.rectangle(img, boxes[i][0], boxes[i][1], color=(r, g, b), thickness=rect_th) # Draw Rectangle with the coordinates
            cv2.putText(img, pred_clas[i], boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, text_size, (r, g, b), thickness=text_th)
        return img

    def display_camera(self,framerate: int = 1, threshold: int = 0.65):
        prev = 0
        rtsp_url = f"rtsp://{USER}:{PASSWORD}@{CAMERA_IP}:{RTSP_PORT}/cam/realmonitor?channel={self.channel}&subtype=0" if CUSTOM_URL is None else CUSTOM_URL
        vcap = cv2.VideoCapture(rtsp_url)
        while(1):
            time_elapsed = time.time() - prev
            ret, frame = vcap.read()
            if time_elapsed > 1./framerate:
                prev= time.time()
                try:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = self.object_detection(frame, threshold=threshold)
                    ret, jpeg =  cv2.imencode(".jpg", frame)
                    yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
                            bytearray(jpeg) + b'\r\n')
                except Exception as ex:
                    print(f"exception: {ex}")
    
