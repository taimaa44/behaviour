from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image
import io

app = FastAPI(title="Driver Behavior API")

# تحميل الموديل مرة وحدة عند بدء السيرفر
model = YOLO("best.pt")

class_map = {
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

label_to_ar = {
    "safe driving": "قيادة آمنة",
    "improper posture": "وضعية غير صحيحة",
    "drowsy": "نعاس",
    "sleeping": "نائم",
    "slight distraction": "تشتت بسيط",
    "phone usage": "استخدام الهاتف",
    "talking on phone": "التحدث على الهاتف",
    "radio distraction": "تشتت بسبب الراديو",
    "drinking": "شرب مشروب",
    "reaching side object": "الانشغال بشيء جانبي",
    "mirror distraction": "الانشغال بالمرآة",
    "talking to passenger": "التحدث مع الراكب",
    "multiple distractions": "تشتت لعدة أسباب"
}

def analyze(image: Image.Image, conf: float = 0.5):
    result = model.predict(source=image, conf=conf, verbose=False)[0]
    detections = []
    for box in result.boxes:
        cls_id = int(box.cls[0].item())
        score = float(box.conf[0].item())
        detections.append({
            "class_id": cls_id,
            "label": class_map.get(cls_id, f"unknown_{cls_id}"),
            "confidence": round(score, 4)
        })
    return sorted(detections, key=lambda x: x["confidence"], reverse=True)

def build_report(detections):
    if not detections:
        return {
            "primary_action": "لم يتم اكتشاف سلوك واضح",
            "secondary_actions": [],
            "safety_status": "غير واضح",
            "description_ar": "لم يتمكن النموذج من تحديد سلوك واضح للسائق.",
            "violations": []
        }

    primary = detections[0]
    label = primary["label"]
    conf = primary["confidence"]

    # حالة خاصة: تشتت بسيط بثقة منخفضة → آمن غالباً
    if label in ["slight distraction", "multiple distractions"] and conf < 0.8:
        return {
            "primary_action": f"safe driving (low confidence) ({conf:.2f})",
            "secondary_actions": [],
            "safety_status": "ملتزم غالبًا",
            "description_ar": "يبدو أن السائق يقود بشكل طبيعي، مع احتمال تشتت بسيط غير مؤكد.",
            "violations": []
        }

    descriptions = {
        "safe driving": ("يبدو أن السائق يقود بشكل آمن وطبيعي.", "ملتزم", []),
        "improper posture": ("يبدو أن السائق يجلس بوضعية غير صحيحة.", "غير ملتزم", ["وضعية قيادة غير صحيحة"]),
        "drowsy": ("توجد مؤشرات على نعاس لدى السائق.", "غير ملتزم", ["نعاس أو خمول"]),
        "sleeping": ("يبدو أن السائق نائم أثناء القيادة.", "غير ملتزم", ["النوم أثناء القيادة"]),
        "slight distraction": ("يبدو أن السائق في حالة تشتت بسيط.", "غير ملتزم", ["تشتت بسيط"]),
        "phone usage": ("يبدو أن السائق يستخدم الهاتف أثناء القيادة.", "غير ملتزم", ["استخدام الهاتف"]),
        "talking on phone": ("يبدو أن السائق يتحدث على الهاتف.", "غير ملتزم", ["التحدث على الهاتف"]),
        "radio distraction": ("يبدو أن السائق مشتت بسبب الراديو.", "غير ملتزم", ["تشتت بسبب الراديو"]),
        "drinking": ("يبدو أن السائق يشرب مشروبًا أثناء القيادة.", "غير ملتزم", ["الشرب أثناء القيادة"]),
        "reaching side object": ("يبدو أن السائق يمد يده لشيء جانبي.", "غير ملتزم", ["الانشغال بشيء جانبي"]),
        "mirror distraction": ("يبدو أن السائق منشغل بالمرآة.", "غير ملتزم", ["الانشغال بالمرآة"]),
        "talking to passenger": ("يبدو أن السائق يتحدث مع الراكب.", "غير ملتزم", ["التحدث مع الراكب"]),
        "multiple distractions": ("توجد مؤشرات على تشتت متعدد.", "غير ملتزم", ["تشتت لعدة أسباب"]),
    }

    desc, status, violations = descriptions.get(label, (f"تم اكتشاف: {label}", "غير واضح", []))
    secondary = [
        f"{label_to_ar.get(d['label'], d['label'])} ({d['confidence']:.2f})"
        for d in detections[1:] if d["confidence"] >= 0.65
    ]

    return {
        "primary_action": f"{label_to_ar.get(label, label)} ({conf:.2f})",
        "secondary_actions": secondary,
        "safety_status": status,
        "description_ar": desc,
        "violations": violations or []
    }

@app.get("/")
def root():
    return {"message": "Driver Behavior API is running"}

@app.post("/analyze")
async def analyze_driver(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    detections = analyze(image)
    report = build_report(detections)
    return JSONResponse(content={
        "detections": detections,
        "report": report
    })
