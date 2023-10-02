import uvicorn
import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile
from herod import indexer
from herod.feature import Extractor, Filter

api = FastAPI()

INDEXER = {}


@api.post("/load_collection")
async def load_collection(
    collection: str,
    search: bool,
    extractor: Extractor = Extractor.SURF,
    filter: Filter = Filter.FUFP,
):
    INDEXER[collection] = indexer.Indexer(
        collection, extractor=extractor, filter=filter, search=search
    )


@api.post("/unload_collection")
async def unload_collection(collection: str):
    del INDEXER[collection]


@api.post("/add_image")
async def add_image(
    collection: str,
    name: str,
    file: UploadFile = File(...),
    limit: int = 500,
):
    if collection not in INDEXER:
        INDEXER[collection] = indexer.Indexer(collection)
    buf = await file.read()
    INDEXER[collection].add_image_raw(buf, name, limit)


@api.post("/search_image")
async def search_image(
    collection: str,
    file: UploadFile = File(...),
    search_list: int = 16,
    limit: int = 100,
):
    if collection not in INDEXER:
        INDEXER[collection] = indexer.Indexer(collection, search=True)
    buf = await file.read()
    img = np.frombuffer(buf, dtype=np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_GRAYSCALE)
    return INDEXER[collection].search_image(collection, img, search_list, limit)[:20]


def start_server(host: str, port: int):
    uvicorn.run(api, host=host, port=port, reload=False)
