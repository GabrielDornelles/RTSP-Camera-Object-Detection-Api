from fastapi_utils.cbv import cbv
from fastapi_utils.inferring_router import InferringRouter
from fastapi import APIRouter, Request
from services.object_detection import ObjectDetection
from fastapi.responses import StreamingResponse
router = InferringRouter()

@cbv(router)
class RTSPObjectDetection:

    @router.get(
    '/status',
    tags=["Status"],
    summary="Status da API",
    description="Rota para validação",
    response_description="Retorna OK se a API estiver up")
    def status(self):
        return {"status": "OK"}
     
    @router.get(
    '/cam',
    tags=["Camera"],
    summary="Camera no canal desejado.",
    description="Abre a Camera IP no canal desejado e detecta objetos na cena.\nParams: ?channel",
    response_description="Frame Buffer")
    def display_camera(self,channel: int = 0):
        return StreamingResponse(ObjectDetection(channel=channel,gpu=True).display_camera(), media_type="multipart/x-mixed-replace;boundary=frame")
        