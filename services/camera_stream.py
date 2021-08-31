# -*- coding: utf-8 -*-
import numpy as np
import cv2
import os
import time
from dotenv import load_dotenv, find_dotenv
from rich.console import Console
import sys

load_dotenv(find_dotenv())
USER = os.getenv("USER",None)
PASSWORD = os.getenv("PASSWORD",None)
CAMERA_IP = os.getenv("CAMERA_IP",None)
RTSP_PORT = os.getenv("RTSP_PORT", "554")
CUSTOM_URL = os.getenv("CUSTOM_URL",None)
USB_CHANNEL = int(os.getenv("USB_CHANNEL",0)) #1 is the index of the camera on the system, like /dev/video1. my camera often freezes and switch between 0 and 1, but default is 0
console = Console()

class CameraStream:
    """
    Class with simple Camera Stream through USB or IP Camera using RTSP.
    """

    def __init__(self, channel=5 ,gpu: bool = False):
        self.gpu = gpu
        self.channel = str(channel)
    
    def display_usb_camera(self):
        """
        Usb Camera device Streaming.
        params:
        crop: list with x1,y1,x2,y2
        """
        cam = cv2.VideoCapture(USB_CHANNEL) 
        #x1,x2,y1,y2 = crop
        while(1):
            try:
                ret, frame = cam.read()
                #frame = frame[y1:y2, x1:x2]
                ret, jpeg =  cv2.imencode(".jpg", frame)
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
                        bytearray(jpeg) + b'\r\n')
            except Exception as ex:
                console.print_exception()
                break
       

    def display_ip_camera(self,crop:list = None,framerate: int = -1):
        """
        IP Camera Streaming.
        params:
        framerate: default is -1, which means capture the entire buffer. Can be limited to a desired value.
        crop: list with x1,y1,x2,y2
        """
        prev = 0
        rtsp_url = f"rtsp://{USER}:{PASSWORD}@{CAMERA_IP}:{RTSP_PORT}/cam/realmonitor?channel={self.channel}&subtype=0" if CUSTOM_URL is None else CUSTOM_URL
        vcap = cv2.VideoCapture(rtsp_url)
        if crop is not None: 
            x1,x2,y1,y2 = crop
        try:
            if framerate == -1:
                while(1):
                    ret, frame = vcap.read()
                    frame = frame[y1:y2, x1:x2]
                    ret, jpeg =  cv2.imencode(".jpg", frame)
                    yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
                            bytearray(jpeg) + b'\r\n')         
            else: # built there so we have less instructions if framerate is not important to be limited.
                while(1):
                    time_elapsed = time.time() - prev
                    if time_elapsed > 1./framerate:
                        prev= time.time()
                        ret, frame = vcap.read()
                        frame = frame[y1:y2, x1:x2]
                        ret, jpeg =  cv2.imencode(".jpg", frame)

                        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
                                bytearray(jpeg) + b'\r\n')
        except Exception:
            console.print_exception()
    
    def usb_camera_background_subtract(self):
        """
        returns: frame buffer with the difference of frames for segmentation.
        """
        cam = cv2.VideoCapture(USB_CHANNEL) 
        fgbg = cv2.createBackgroundSubtractorMOG2() # no params where requested in the test
        while(1):
            try:
                ret, frame = cam.read()
                fgmask = fgbg.apply(frame)
                ret, jpeg =  cv2.imencode(".jpg", fgmask)
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
                        bytearray(jpeg) + b'\r\n')
            except Exception as ex:
                console.print_exception()
                break
    
    def usb_camera_binarize(self, k:int):
        """
        apply kernel on a hsv mask where
        Hue ranges from 0 to 180 , being k the max value
        Saturation ranges from 0 to 255 (unchanged)
        Value ranges from 0 to 255 (unchanged)
        returns: frame buffer with the masked image
        """
        cam = cv2.VideoCapture(USB_CHANNEL) 
        min_ = np.array([0,0,0], dtype = "uint16")
        k = np.array([k,255,255], dtype = "uint16") 
        while(1):
            try:
                ret, frame = cam.read()
                masked_image = cv2.inRange(frame, min_, k)
                ret, jpeg =  cv2.imencode(".jpg", masked_image)
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
                        bytearray(jpeg) + b'\r\n')
            except Exception as ex:
                console.print_exception()
                break
    

    
           
    