import uvicorn
import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile
from herod import helpers

api = FastAPI()


@api.post("/search_image")
async def search_image(
    collection: str,
    file: UploadFile = File(...),
    search_list: int = 16,
    limit: int = 100,
):
    buf = await file.read()
    img = np.frombuffer(buf, dtype=np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_GRAYSCALE)
    return helpers.search_image(collection, img, search_list, limit)


def start_server():
    uvicorn.run(api, host="0.0.0.0", port=8000)
