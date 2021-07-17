import os
import uvicorn
from dotenv import load_dotenv, find_dotenv
from fastapi import FastAPI
from routers.rtsp_object_detection_routes import router

load_dotenv(find_dotenv())
ENVIRONMENT_NAME = os.getenv("ENVIRONMENT_NAME", None)

if ENVIRONMENT_NAME != None and ENVIRONMENT_NAME.startswith("production"):
    # Disable swagger and redoc on production enviroment.
    app = FastAPI(docs_url=None, redoc_url=None)
else:
    app = FastAPI(
        title="API de streaming de video com deteccao de objetos.",
        description="API para visualizacao e deteccao de objetos em Cameras IP pelo protocolo RTSP",
        version="0.0.1")

app.include_router(router)

if __name__ == '__main__':
    uvicorn.run(app=app, port=int(os.getenv('PORT', 8000)))
