# -*- coding: utf-8 -*-
import torch
import numpy as np
import cv2
import time
import random
from rich.console import Console
import sys
import os
from PIL import Image
import torchvision.transforms as T
from dotenv import load_dotenv, find_dotenv
from models.detrdemo import DETRdemo

load_dotenv(find_dotenv())
USER = os.getenv("USER")
PASSWORD = os.getenv("PASSWORD")
CAMERA_IP = os.getenv("CAMERA_IP")
RTSP_PORT = os.getenv("RTSP_PORT", "554")
CUSTOM_URL = os.getenv("CUSTOM_URL",None)
console = Console()

class ObjectDetectionDETR:
    """
    Class with object detection and RTPS using DETR.
    """

    def __init__(self, channel ,gpu: bool = False):
        self.gpu = gpu
        self.transform = T.Compose([
        #T.Resize(800),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.channel = str(channel)
        self.device = torch.device("cuda") if self.gpu else torch.device("cpu")
        self.model = DETRdemo(num_classes=91)
        state_dict = torch.hub.load_state_dict_from_url(url='https://dl.fbaipublicfiles.com/detr/detr_demo-da2a99e9.pth',check_hash=True)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        self.categories = [
            'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
            'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
            'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
            'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
            'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
            'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
            'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
            'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
            'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
            'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
            'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]

    #@staticmethod
    def box_cxcywh_to_xyxy(self,x):
        x_c, y_c, w, h = x.unbind(1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
            (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=1)

    #@staticmethod
    def rescale_bboxes(self,out_bbox, size):
        img_w, img_h = size
        b = self.box_cxcywh_to_xyxy(out_bbox)
        b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
        return b
    
    def forward(self, im, threshold=0.7):
        """
        Inputs: PIL Image
        Outputs: Tensors: probabilities higher than threshold, boxes coordinates x1,y1,x2,y2
        """
        # mean-std normalize the input image (batch-size: 1)
        img = self.transform(im).unsqueeze(0)

        # demo model only support by default images with aspect ratio between 0.5 and 2
        # if you want to use images with an aspect ratio outside this range
        # rescale your image so that the maximum size is at most 1333 for best results
        assert img.shape[-2] <= 1600 and img.shape[-1] <= 1600, 'demo model only supports images up to 1600 pixels on each side'

        # propagate through the model
        if self.gpu: img = img.cuda()
        outputs = self.model(img)
        # keep only predictions with 0.7+ confidence
        probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
        keep = probas.max(-1).values > threshold

        # convert boxes from [0; 1] to image scales
        boxes = outputs['pred_boxes'][0, keep].cpu() if self.gpu else outputs['pred_boxes'][0, keep]
        bboxes_scaled = self.rescale_bboxes(boxes, im.size)
        return probas[keep], bboxes_scaled
    
    def draw_frame(self,img,boxes,scores,rect_th=1, text_size=1, text_th=2):
        boxes = boxes.int() # cv2 expects integer coordinates
        boxes = boxes.cpu().detach().numpy() # boxes are returned as a tensor, cast to np array
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for i in range(len(boxes)):
            r, g, b = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) # Random Color
            xmin, ymin, xmax, ymax = boxes[i]
            cv2.rectangle(img,(xmin,ymin),(xmax,ymax),color=(r,g,b), thickness=rect_th)
            object_class = scores[i].argmax()
            object_score = scores[i][object_class].detach()
            cv2.putText(img, f"{self.categories[object_class]}-{object_score:.2f}", (xmin,ymin), cv2.FONT_HERSHEY_SIMPLEX, text_size, (r, g, b), thickness=text_th)
        return img
    
    def object_detection(self,img):
        scores, boxes = self.forward(img)
        frame = self.draw_frame(img, boxes, scores)
        return frame

    def display_camera(self,framerate: int = 4, threshold: int = 0.65):
        prev = 0
        rtsp_url = f"rtsp://{USER}:{PASSWORD}@{CAMERA_IP}:{RTSP_PORT}/cam/realmonitor?channel={self.channel}&subtype=0" if CUSTOM_URL is None else CUSTOM_URL
        vcap = cv2.VideoCapture(rtsp_url)
        while(1):
            time_elapsed = time.time() - prev
            ret, frame = vcap.read()
            if time_elapsed > 1./framerate:
                prev= time.time()
                try:
                    frame = Image.fromarray(frame)
                    frame = self.object_detection(frame)
                    ret, jpeg =  cv2.imencode(".jpg", frame)
                    
                    yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
                            bytearray(jpeg) + b'\r\n')
                except Exception:
                    console.print_exception()
                    sys.exit()

                
                    
                    
                    
              

  
