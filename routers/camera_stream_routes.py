import os
from dotenv import load_dotenv, find_dotenv
from fastapi_utils.cbv import cbv
from fastapi_utils.inferring_router import InferringRouter
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from services.camera_stream import CameraStream
from fastapi.templating import Jinja2Templates


templates = Jinja2Templates(directory="templates")
router = InferringRouter()
load_dotenv(find_dotenv())

@cbv(router)
class CameraStreamRoutes:

    @router.get(
    '/usbcam',
    tags=["Camera"],
    summary="Camera usb.",
    description="Abre a Camera Usb",
    response_description="Frame Buffer")
    def display_usb_camera(self):
        #crop = None#[0,100,0,100]
        return StreamingResponse(CameraStream().display_usb_camera(), media_type="multipart/x-mixed-replace;boundary=frame")
    
    @router.get(
    '/usbcam-subtract',
    tags=["Camera"],
    summary="Camera usb utilizando segmentacao por subtracao de pixels",
    description="Segmentacao por diferenciacao",
    response_description="Frame Buffer")
    def display_usb_camera_subtract(self):
        return StreamingResponse(CameraStream().usb_camera_background_subtract(), media_type="multipart/x-mixed-replace;boundary=frame")
    
    @router.get(
    '/usbcam-binarize',
    tags=["Camera"],
    summary="Camera usb com binarizacao com valor k",
    description="Params: ?k",
    response_description="Frame Buffer")
    def binarize(self,k: int = 180):
        return StreamingResponse(CameraStream().usb_camera_binarize(k=k), media_type="multipart/x-mixed-replace;boundary=frame")
    
    @router.get(
    '/ipcam',
    tags=["Camera"],
    summary="Camera IP no canal desejado.",
    description="Abre a Camera IP no canal desejado e detecta objetos na cena.\nParams: ?channel",
    response_description="Frame Buffer")
    def display_camera(self,channel: int = 1):
        return StreamingResponse(CameraStream(channel=channel).display_ip_camera(), media_type="multipart/x-mixed-replace;boundary=frame")
    
    @router.get(
    '/home',
    tags=["home"],
    summary="Pagina principal da API",
    description="Html base para controle",
    response_description="html")
    def home(self,request: Request):
        return templates.TemplateResponse("home.html", {
             "request": request,
        })
    
        