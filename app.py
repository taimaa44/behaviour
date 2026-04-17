from fastapi import FastAPI, File, UploadFile
import cv2
import shutil
import os
import uuid
from ultralytics import YOLO

app = FastAPI()

# تحميل الموديل
model = YOLO("best.pt")

classes = {
    0: "improper posture",
    1: "safe driving",
    2: "drowsy",
    3: "sleeping",
    4: "slight distraction",
    5: "phone usage",
    6: "talking on phone",
    7: "radio distraction",
    8: "drinking",
    9: "reaching side object",
    10: "mirror distraction",
    11: "talking to passenger",
    12: "multiple distractions"
}

@app.get("/")
def root():
    return {"message": "API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    filename = f"{uuid.uuid4()}.jpg"

    try:
        with open(filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        img = cv2.imread(filename)

        results = model(img, verbose=False)
        result = results[0]

        behavior = "No detection"
        cls_id = None

        if result.boxes is not None and len(result.boxes) > 0:
            cls_id = int(result.boxes.cls[0].item())
            behavior = classes.get(cls_id, f"class_{cls_id}")

        return {
            "behavior": behavior,
            "class_id": cls_id
        }

    finally:
        if os.path.exists(filename):
            os.remove(filename)
