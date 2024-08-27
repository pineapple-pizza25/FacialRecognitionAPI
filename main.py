from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from deepface import DeepFace
import os
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient
import bson.binary

app = FastAPI()

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
def storeImageEmbedding(img_path: str = "face_image.jpg"):
    #img = functions.preprocess_face(img_path, target_size=(224, 224))

    if not os.path.exists(img_path):
        raise HTTPException(status_code=404, detail="Image file not found")
    

    try:
        embedding = DeepFace.represent(img_path, model_name = "Facenet")

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

