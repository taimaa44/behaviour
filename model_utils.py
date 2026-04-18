import os
from ultralytics import YOLO
from classes import CLASSES

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = YOLO(os.path.join(BASE_DIR, "models", "best.pt"))


def predict_behavior(img):

    results = model(img)
    result = results[0]

    if result.boxes is None or len(result.boxes) == 0:
        return {
            "class_id": None,
            "behavior": "No detection",
            "confidence": None
        }

    # 🔥 أهم تعديل هنا: خذ أعلى confidence مش أول box
    best_idx = result.boxes.conf.argmax()

    cls_id = int(result.boxes.cls[best_idx])
    conf = float(result.boxes.conf[best_idx])

    behavior = CLASSES.get(cls_id, f"unknown_{cls_id}")

    return {
        "class_id": cls_id,
        "behavior": behavior,
        "confidence": conf
    }