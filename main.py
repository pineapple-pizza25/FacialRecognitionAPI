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
import settings
import cv2
from fastapi.encoders import jsonable_encoder
import json
import base64


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()

UPLOAD_DIR = "uploaded_images"
os.makedirs(UPLOAD_DIR, exist_ok=True)

origins = ["http://localhost:8000"]

client = MongoClient('mongodb+srv://st10066487:debtduty96@apds.sw61z.mongodb.net/?retryWrites=true&w=majority&appName=APDS')
db = client['facial_recognition']
collection = db['images']

class NumpyArrayPayload(BaseModel):
    array: list

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/storeface")
async def storeFace(file:UploadFile = File(...)):
    
    raw_file = await file.read()
    image = file_to_image(raw_file)

    try:
        base64_image = base64.b64encode(raw_file).decode("utf-8")
        image_document = {
            "name": "image1",
            "data": base64_image
        }
        collection.insert_one(image_document)
        print(f"Image stored in MongoDB.")

        return {
            "message": "Face embedding stored successfully",
            "embedding": base64_image
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail="There was an error:"+str(e))


#verifies uploaded image against images stored in db
@app.post("/facialrecognition1")
async def facialrecognition1(file: UploadFile = File(...)):
    #converts image to a numpy array
    raw_file = await file.read()
    image = file_to_image(raw_file)
    npArray = pil_to_numpy(image)


    #retrieves the stored images from the database
    stored_faces = list(collection.find({}, {"data": 1, "_id": 0}))
    if not stored_faces:
        raise HTTPException(status_code=404, detail="No stored faces found in the database")
    
    for stored_face in stored_faces:

        try:
            base64_string  = stored_face["data"]

            base64_data = base64_string.split(",")[1] if "," in base64_string else base64_string
            image_data = base64.b64decode(base64_data)
            imageFromMongo = Image.open(BytesIO(image_data))
            image_np = np.array(imageFromMongo)

            try:
                if DeepFace.verify(npArray, image_np, model_name='Facenet', threshold=0.45)['verified']:
                    return {"message":"This dude is in the system"}
                else: 
                    return{"message":"This dude is not in the system"}
            
            except Exception as e:
                return{f"There was an error with the verify function: {str(e)}"}

        except Exception as e:
            return{f"There was an error:: {str(e)}"}

    return {"message": "No matching face found in the database"}

    
    



#gets all embeddings from mongo
@app.get("/getImages")
async def getImages():
    faces = []
    
    for x in collection.find():
        faces.append(x)
    return jsonable_encoder(str(faces))





def euclidean(embedding1, embedding2):
    if isinstance(embedding1, dict):
        embedding1 = embedding1['embedding']  
    if isinstance(embedding2, dict):
        embedding2 = embedding2['embedding']

    P1 = np.array(embedding1)
    P2 = np.array(embedding2)

    diff = {**P1 - P2}
    
    euclid_dist = np.sqrt(np.dot(diff.T, diff))
    
    return euclid_dist



def file_to_image(file):
  image = Image.open(BytesIO(file)).convert("RGB")
  return image



def pil_to_numpy(img: Image.Image) -> np.ndarray:
    return np.array(img)




def load_image_into_numpy_array(data):
    return np.array(Image.open(BytesIO(data)))



def extract_embedding(embeddings):
    # Extract the embedding from different possible formats
    if isinstance(embeddings, list) and len(embeddings) > 0:
        # Assuming the list contains dictionaries with an 'embedding' key
        embedding = embeddings[0]("embedding")
    elif isinstance(embeddings, dict):
        embedding = embeddings.get("embedding")
    else:
        raise ValueError(f"Unexpected embeddings format: {type(embeddings)}")

    if isinstance(embedding, np.ndarray):
        return embedding.tolist()
    elif isinstance(embedding, list):
        return embedding
    elif isinstance(embedding, float):
        return [embedding]
    else:
        raise ValueError(f"Unexpected embedding type: {type(embedding)}")
    

#converts the image binaries to a numpy array
def dict_to_numpy(image_dict):
    # Assuming 'image' key contains binary data
    image_data = image_dict.get('image')
    
    if image_data:
        # Convert binary data to numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        # Decode numpy array to image
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    else:
        raise ValueError("No image data found in dictionary")