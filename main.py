from fastapi import FastAPI, File, UploadFile
import cv2
import shutil
from model_utils import predict_behavior

app = FastAPI()


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # حفظ الصورة مؤقتًا
    file_path = "input.jpg"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # قراءة الصورة
    img = cv2.imread(file_path)

    # prediction
    result = predict_behavior(img)

    return result