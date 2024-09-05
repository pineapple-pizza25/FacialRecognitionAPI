from fastapi import FastAPI, HTTPException, File, Form, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
from deepface import DeepFace
import os
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient
import bson.binary
import PIL
from PIL import Image, ImageDraw, ImageFont
import io
import numpy as np
import tensorflow as tf
from tempfile import NamedTemporaryFile
import logging
from deepface.modules import verification
from io import BytesIO

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()

UPLOAD_DIR = "uploaded_images"
os.makedirs(UPLOAD_DIR, exist_ok=True)

origins = ["http://localhost:3000"]

client = MongoClient("mongodb://localhost:27017/")
db = client['facial_recognition']
collection = db['images']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def getImage():
    return {"Welcome"}

@app.post("/storeface")
async def storeImageEmbedding(file:UploadFile = File(...)):
    
    raw_file = await file.read()
    image = file_to_image(raw_file)
    npArray = pil_to_numpy(image)

    try:
        embedding = DeepFace.represent(npArray)

        collection.insert_one({
            "image_name": "image1", 
            "embedding": embedding
            })

        return {
            "message": "Face embedding stored successfully",
            "embedding": embedding
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail="There was an error:"+str(e))


@app.post("/facialrecognition")
async def getFacialRecognition(file:UploadFile = File(...)):

    raw_file = await file.read()
    image = file_to_image(raw_file)
    npArray = pil_to_numpy(image)

    image1_path = "C:/Users/snmis/OneDrive/Pictures/Camera Roll/WIN_20240903_14_52_19_Pro.jpg"

    try:
        if DeepFace.verify(npArray, image1_path,threshold=0.6)['verified']:
            return {"message":"This is the same dude"}
        else: 
            return{"message":"this is not the same dude"}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Something went wrong: {str(e)}")




@app.post("/verifyface")
async def postVerifyFace(file:UploadFile = File(...)):

    raw_file = await file.read()
    image = file_to_image(raw_file)
    npArray = pil_to_numpy(image)

    img1_representation = DeepFace.represent(npArray)
    img2_representation = DeepFace.represent(npArray2)

    distance = verification.find_euclidean_distance(img1_representation, img2_representation)
    threshold = verification.find_threshold('VGG-face', 'euclidean')

    if distance <= threshold:
        return {"message":"This is the same dude"}
    else:
        return {"message":"This is not the same dude"}
    



def file_to_image(file):
  image = Image.open(BytesIO(file)).convert("RGB")
  return image



def pil_to_numpy(img: Image.Image) -> np.ndarray:
    return np.array(img)




def load_image_into_numpy_array(data):
    return np.array(Image.open(BytesIO(data)))



