from fastapi import FastAPI, HTTPException, File, Form, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
from deepface import DeepFace
import os
from pymongo import MongoClient
from PIL import Image
import numpy as np
import logging
from io import BytesIO
import cv2
from fastapi.encoders import jsonable_encoder
import base64
from .settings import mongoUri, port
from datetime import datetime, timezone, timedelta
import pytz


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()

UPLOAD_DIR = "uploaded_images"
os.makedirs(UPLOAD_DIR, exist_ok=True)

origins = ["http://localhost:8000"]

client = MongoClient(mongoUri)
db = client['facial_recognition']
collection = db['Students']
attendanceCollection = db['Attendance']

class NumpyArrayPayload(BaseModel):
    array: list

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def current_time_to_nanoseconds():
    # Get current time
    now = datetime.now()
    # Calculate seconds since midnight
    seconds_since_midnight = now.hour * 3600 + now.minute * 60 + now.second
    # Convert to nanoseconds
    return seconds_since_midnight * 10_000_000

def getFormattedDate():
    """
    Returns the current date as a datetime object set to midnight UTC
    """
    current_date = datetime.now(pytz.utc) 
    midnight_date = current_date.replace(
        hour=0,
        minute=0,
        second=0,
        microsecond=0
    )
    return midnight_date

def getLesson():
    collection = db['Lessons']

    specified_date =  getFormattedDate()
    specified_time = current_time_to_nanoseconds()

    print(specified_date)
    print(specified_time)

    start_of_day = specified_date
    end_of_day = specified_date + timedelta(days=1)

    projection = {"_id": 1,}

    query = {
        "lessonDate": specified_date,
        "$and": [
            {"startTime": {"$lte": specified_time}},
            {"endTime": {"$gte": specified_time}}
        ]
    }

    test_query = {"lessonDate": specified_date}

    result = collection.find_one(query, projection)

    print("Query:", query)

    if result:
        print(result)
        return result.get('_id')
    else:
        print("No matching document found.")
        return None


def getStudentsInLesson(lesson_id):
    collection = db['Lessons']
    
    query = {"_id": lesson_id}
    
    projection = {"_id": 0, "Students": 1} 
    
    result = collection.find_one(query, projection)
    
    if result:
        return result.get('Students')  
    else:
        print(f"No lesson found with ID: {lesson_id}")
        return None


@app.get("/getStudents")
async def getStudents():
    lessonId = getLesson()

    if lessonId:
        students = getStudentsInLesson(lessonId)
        if students:
         print(students)
        else: print("no students in lesson")
    else: print("no lesson id")


@app.post("/storeface")
async def storeFace(file:UploadFile = File(...)):

    getLesson()
    
    raw_file = await file.read()
    image = file_to_image(raw_file)

    try:
        base64_image = base64.b64encode(raw_file).decode("utf-8")
        image_document = {
            "name": "image1",
            "image": base64_image
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
@app.post("/facialrecognition")
async def facialrecognition(file: UploadFile = File(...)):
    lessonId = getLesson()
    if not lessonId:
        return {"message": "No active lesson found"}

    lessonStudents = getStudentsInLesson(lessonId)
    if not lessonStudents:
        return {"message": "No students found in current lesson"}

    #converts image to a numpy array
    raw_file = await file.read()
    image = file_to_image(raw_file)
    npArray = pil_to_numpy(image)


    #retrieves the stored images from the database
    stored_faces = list(collection.find({}, {"image": 1, "_id": 1}))
    if not stored_faces:
        raise HTTPException(status_code=404, detail="No stored faces found in the database")
    
    face_identified = False
    student_id = None
    
    for lessonStudent in lessonStudents:

        try:
            student = collection.find_one({"_id": lessonStudent})
            if not student or "image" not in student:
                print("Skipping stored_face due to missing 'image' key or None value.")
                continue
    
            base64_string  = student["image"]

            if base64_string is None:
                print("Skipping stored_face due to none value in image.")
                continue

            base64_data = base64_string.split(",")[1] if "," in base64_string else base64_string
            image_data = base64.b64decode(base64_data)
            imageFromMongo = Image.open(BytesIO(image_data))
            image_np = np.array(imageFromMongo)

            try:
                if DeepFace.verify(npArray, image_np, model_name='Facenet', threshold=0.45)['verified']:
                    face_identified = True
                    student_id = student["_id"]
                    break
            
            except Exception as e:
                return{f"There was an error with the verify function: {str(e)}"}

        except Exception as e:
            return{f"There was an error:: {str(e)}"}
        
    if face_identified == True:
        attendance_record = {
            "lessonId": lessonId,
            "timestamp": datetime.now(),
            "status": "present"
        }       

        collection.update_one(
            {"_id": student_id},
            {
                "$push": {
                    "attendance": attendance_record
                }
            },
            upsert=True  # Creates the attendance array if it doesn't exist
        )
        
        return("This dude is in the system")
    else:
        return("this dude is not in the system")

    
    



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
    image_data = image_dict.get('image')
    
    if image_data:
        # Convert binary data to numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        # Decode numpy array to image
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    else:
        raise ValueError("No image data found in dictionary")