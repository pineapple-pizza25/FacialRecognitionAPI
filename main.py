from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from deepface import DeepFace
import os

app = FastAPI()

origins = ["https://localhost:3000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def getImage():
    img_path = "face_image.jpg"

    #img = functions.preprocess_face(img_path, target_size=(224, 224))

    flag = False

    if os.path.exists(img_path):
        flag = True
    else:
        flag = False

    if flag==True:
        try:
            embedding = DeepFace.represent(img_path, model_name = "Facenet")
            return {"embedding": embedding}
        except Exception as e:
            raise HTTPException(status_code=500, detail="There was an error:"+str(e))
    else:
        return {"Image not valid"}