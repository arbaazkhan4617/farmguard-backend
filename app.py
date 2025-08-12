# app.py
import os
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from model_utils import load_model, predict
from db import SessionLocal, init_db, Detection
import shutil
from datetime import datetime
from PIL import Image
import io
import uuid

app = FastAPI(title="FarmGuard API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for image access
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# Config
MODEL_PATH = os.getenv("MODEL_PATH", "./model/saved_model")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Load model (at startup)
model = load_model(MODEL_PATH)
init_db()

def save_image(bytes_, filename):
    # compress & save as JPEG
    img = Image.open(io.BytesIO(bytes_)).convert("RGB")
    path = os.path.join(UPLOAD_DIR, filename)
    img.save(path, format="JPEG", quality=70, optimize=True)
    return path

def save_detection_to_db(image_path, label, confidence, advice, source="web"):
    db = SessionLocal()
    det = Detection(image_path=image_path, label=label, confidence=confidence, advice=advice, source=source)
    db.add(det)
    db.commit()
    db.refresh(det)
    db.close()
    return det.id

@app.get("/health")
def health():
    return {"status":"ok", "time": datetime.utcnow().isoformat()}

@app.post("/detect")
async def detect(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    # basic checks
    if file.content_type not in ("image/jpeg","image/png"):
        raise HTTPException(status_code=400, detail="Only jpeg/png allowed")
    contents = await file.read()
    if len(contents) > int(os.getenv("MAX_UPLOAD_SIZE", 3145728)):
        raise HTTPException(status_code=413, detail="File too large")
    # predict
    result = predict(model, contents)
    # save image in background
    filename = f"{uuid.uuid4().hex}.jpg"
    image_path = save_image(contents, filename)
    background_tasks.add_task(save_detection_to_db, image_path, result["label"], result["confidence"], result["advice"])
    return JSONResponse({
        "status":"ok",
        "result": result,
        "image_url": image_path,
        "timestamp": datetime.utcnow().isoformat()
    })

@app.get("/history")
def history(limit: int = 20):
    db = SessionLocal()
    items = db.query(Detection).order_by(Detection.timestamp.desc()).limit(limit).all()
    out = []
    for it in items:
        out.append({
            "id": it.id, "timestamp": it.timestamp.isoformat(), "label": it.label,
            "confidence": it.confidence, "advice": it.advice, "image_path": it.image_path
        })
    db.close()
    return {"items": out}

