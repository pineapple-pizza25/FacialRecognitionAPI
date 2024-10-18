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

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()

UPLOAD_DIR = "uploaded_images"
os.makedirs(UPLOAD_DIR, exist_ok=True)

origins = ["http://localhost:8000"]

client = MongoClient(settings.mongoUri)
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


@app.get("/")
def getImage():
    return {"Welcome"}

@app.post("/storeface")
async def storeFace(file:UploadFile = File(...)):
    
    raw_file = await file.read()
    image = file_to_image(raw_file)
    npArray = pil_to_numpy(image)

    try:
        embedding = DeepFace.represent(npArray, model_name="Facenet")

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
async def facialrecognition(file: UploadFile = File(...)):
    raw_file = await file.read()
    image = file_to_image(raw_file)
    npArray = pil_to_numpy(image)
    embedding = DeepFace.represent(npArray, model_name="Facenet")[0]['embedding']
    

    stored_faces = list(collection.find({}, {"embedding": 1, "_id": 0}))
    if not stored_faces:
        raise HTTPException(status_code=404, detail="No stored faces found in the database")
    
    for stored_face in stored_faces:

        try:
            
            stored_embedding = list(stored_face['embedding'])

            embedding_array = np.array(stored_embedding)
            
            distance = euclidean(embedding, embedding_array)
            threshold = 0.6
        
            try:
                if distance < threshold:
                    return {"message": "This is a match!"}
                else: 
                    continue
            
            except Exception as e:
                return{f"There was an error with the verify function: {str(e)}"}

        except Exception as e:
            return{f"There was an error:: {str(e)}"}
            continue


    return {"message": "No matching face found in the database"}

    




@app.post("/verifyface")
async def verifyface(file:UploadFile = File(...)):

    raw_file = await file.read()
    image = file_to_image(raw_file)
    npArray = pil_to_numpy(image)

    image1_path = "./face_image.jpg"


    try:
        try:
            face = DeepFace.detectFace(npArray, detector_backend='opencv')
        except Exception as e:
            return {"message": "No face detected"}
        

        if DeepFace.verify(npArray, image1_path,threshold=0.45)['verified']:
            return {"message":"This is the same dude"}
        else: 
            return{"message":"this is not the same dude"}
        

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Something went wrong: {str(e)}")


@app.post("/verifyfacelocal")
async def verifyfacelocal(data: NumpyArrayPayload):

    npArray = np.array(data.array)

    image1_path = "./face_image.jpg"


    try:
        try:
            face = DeepFace.detectFace(npArray, detector_backend='opencv')
        except Exception as e:
            return {"message": "No face detected"}
        

        if DeepFace.verify(npArray, image1_path,threshold=0.45)['verified']:
            return {"message":"This is the same dude"}
        else: 
            return{"message":"this is not the same dude"}
        

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Something went wrong: {str(e)}")


@app.get("/test")
async def test(testData: str):
    print("working")
    return {"received_data": testData}


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