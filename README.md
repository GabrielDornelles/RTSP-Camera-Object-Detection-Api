# Teste mvisia

Demonstracao completa(video com instalacao, funcionamento e exemplos descritos neste repositorio): https://youtu.be/LYrI6xg8zP8 

# Instalacao 

## (linux)

```
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
```

## (windows)

```
python -m venv venv
\venv\Scripts\activate.bat
pip install -r requirements.txt
```

# Run
```
>>> uvicorn app:app
```

## Rotas adicionadas:
- Home: http://127.0.0.1:8000/home
- Camera normal: http://127.0.0.1:8000/usbcam
- Camera normal com binarizacao: http://127.0.0.1:8000/usbcam-binarize?k=param {param: int 0-180 (Hue)}
- Camera normal com subtracao (segmentacao): http://127.0.0.1:8000/usbcam-subtract
- Camera IP por canal: http://127.0.0.1:8000/ipcamera?channel=param {param: int}
- Crop implementado no modelo x1,y1,x2,y2. Visto que um post seria um tanto nao intuitivo e nao conheco muito de front-end, ficou implementado apenas no metodo em: services/camera_stream_routes.py - CameraStreamRoutes().display_usb_camera

## Rotas Extra teste:
- Deteccao de objetos em camera IP: 127.0.0.1:8000/cam?channel=param {param: int (numero da camera)}
- Documentacao dos metodos e descricao: 127.0.0.1:8000/docs

## Tracebacks
Tracebacks com highlight para todos os metodos implementados.

![image](https://user-images.githubusercontent.com/56324869/131551491-03075de2-ca86-4e59-9108-580320f600d3.png)

## variaveis de ambiente necessarias para a camera usb comum (.env)
- USB_CHANNEL=0

Segue a documentacao padrao da API para a deteccao de objetos.

# RTPS-Camera-Object-Detection-Api
An API for IP camera video streaming and object detection using [DETR(DEtection TRansformer)](https://github.com/facebookresearch/detr) or Faster-rcnn. Both of them uses resnet50 as backbone. DETR is about 4x faster but Faster-RCNN is more precise on real scenes.
Built with FastAPI and PyTorch.
# Install
```
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
```
# Run
```
>>> uvicorn app:app
```

## Demo:
![ezgif com-gif-maker (7)](https://user-images.githubusercontent.com/56324869/126047793-9cfb412a-9822-4b7a-b575-c225487cacb0.gif)
# Environment variables
This API was built using Intelbras cameras, this variables are related to the rtsp url, if you use a different type of camera, there's a variable **CUSTOM_URL** where you can just pass the entire url for your cameras at rtsp.

If you use Intelbras Cameras, you can just use the following
Create a .env file with the following variables:
```env
USER= 
PASSWORD=
CAMERA_IP=
RTSP_PORT=
PORT=
ENVIRONMENT_NAME=
MODEL=
GPU=
```
- USER: Your user within your camera system.
- PASSWORD: Your password within the system.
- CAMERA_IP: Your camera IP.
- RTSP_PORT: 554 as default, but you can change if you changed in your system.
- PORT: Api port, 8000 as default.
- ENVIRONMENT_NAME: Will disable swagger and redocs for FastAPI if starts with "production". Default is None.
- GPU: Default is false, 1 to use cuda.
- MODEL: default is DETR, FASTER-RCNN available.

# Routes
- /cam

Params: channel

Description: open the camera x (1,2,3,4...).

Example: 127.0.0.1:8000/cam?channel=5

# Models
- Demo DETR: 
    * learned positional encoding (instead of sine)
    * positional encoding is passed at input (instead of attention)
    * fc bbox predictor (instead of MLP)
- A faster-rcnn with a resnet50 backbone as the cnn, available at torchvision and trained using COCO dataset.

# TODO:
- Custom set of categories on detection.
- Track desired objects in a specific space of time.
- Interface (now its just buffering the camera frames into the model and displaying at the cam route).
- Implement best DETR and see if its still worse than faster-rcnn on accuracy.
