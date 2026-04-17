
from fastapi import FastAPI, File, UploadFile
import cv2
import shutil
from ultralytics import YOLO

app = FastAPI()

# -------------------
# تحميل موديل YOLO
# -------------------
model = YOLO("best.pt")

# -------------------
# 🔥 حطي أسماء الكلاسات حسب تدريبك
# (عدّليهم إذا عندك أسماء مختلفة)
# -------------------
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

# -------------------
# API
# -------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # حفظ الصورة
    with open("input.jpg", "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # قراءة الصورة
    img = cv2.imread("input.jpg")

    # تشغيل YOLO
    results = model(img)
    result = results[0]

    behavior = "No detection"
    cls_id = None

    # إذا في detection
    if result.boxes is not None and len(result.boxes) > 0:
        cls_id = int(result.boxes.cls[0])

        # تحويل الرقم لاسم
        behavior = classes.get(cls_id, f"class_{cls_id}")

    return {
        "behavior": behavior,
        "class_id": cls_id
    }
