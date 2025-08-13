# app.py
import os
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from model_utils import load_model, predict
from db import SessionLocal, init_db, Detection
import shutil
from datetime import datetime
from PIL import Image
import io
import uuid
import pickle
import numpy as np

# Pydantic models for new features
class CropRecommendation(BaseModel):
    N: int
    P: int
    K: int
    temperature: float
    humidity: float
    ph: float
    rainfall: float

class FertilizerRecommendation(BaseModel):
    cropType: str
    soilType: str
    nitrogen: int
    phosphorus: int
    potassium: int
    ph: float

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

# Crop recommendation data (simplified version - you can enhance this)
crop_recommendations = {
    "rice": {"min_temp": 20, "max_temp": 35, "min_rainfall": 100, "optimal_ph": 6.5},
    "maize": {"min_temp": 18, "max_temp": 32, "min_rainfall": 80, "optimal_ph": 6.0},
    "wheat": {"min_temp": 15, "max_temp": 25, "min_rainfall": 60, "optimal_ph": 6.5},
    "cotton": {"min_temp": 25, "max_temp": 40, "min_rainfall": 70, "optimal_ph": 6.0},
    "tomato": {"min_temp": 20, "max_temp": 30, "min_rainfall": 50, "optimal_ph": 6.5},
    "potato": {"min_temp": 15, "max_temp": 25, "min_rainfall": 60, "optimal_ph": 5.5},
    "sugarcane": {"min_temp": 25, "max_temp": 38, "min_rainfall": 120, "optimal_ph": 6.5},
    "coffee": {"min_temp": 18, "max_temp": 28, "min_rainfall": 100, "optimal_ph": 6.0}
}

# Fertilizer recommendations (simplified version - you can enhance this)
fertilizer_recommendations = {
    "rice": {"npk": "20-20-20", "deficiency": "Balanced NPK", "application": "Apply 250kg/ha at planting"},
    "maize": {"npk": "15-15-15", "deficiency": "Balanced NPK", "application": "Apply 200kg/ha at planting"},
    "wheat": {"npk": "18-18-18", "deficiency": "Balanced NPK", "application": "Apply 180kg/ha at planting"},
    "cotton": {"npk": "25-15-15", "deficiency": "High Nitrogen", "application": "Apply 300kg/ha at planting"},
    "tomato": {"npk": "20-20-20", "deficiency": "Balanced NPK", "application": "Apply 150kg/ha at planting"},
    "potato": {"npk": "15-15-15", "deficiency": "Balanced NPK", "application": "Apply 200kg/ha at planting"},
    "sugarcane": {"npk": "25-15-15", "deficiency": "High Nitrogen", "application": "Apply 400kg/ha at planting"},
    "coffee": {"npk": "20-20-20", "deficiency": "Balanced NPK", "application": "Apply 250kg/ha at planting"}
}

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

def get_crop_recommendation(data: CropRecommendation):
    """Simple crop recommendation logic - you can enhance this with ML models"""
    best_crops = []
    
    for crop, requirements in crop_recommendations.items():
        score = 0
        
        # Temperature check
        if requirements["min_temp"] <= data.temperature <= requirements["max_temp"]:
            score += 3
        elif abs(data.temperature - (requirements["min_temp"] + requirements["max_temp"])/2) <= 5:
            score += 2
        else:
            score += 1
            
        # Rainfall check
        if data.rainfall >= requirements["min_rainfall"]:
            score += 2
            
        # pH check
        if abs(data.ph - requirements["optimal_ph"]) <= 1:
            score += 2
        elif abs(data.ph - requirements["optimal_ph"]) <= 2:
            score += 1
            
        # N-P-K balance check
        if data.N >= 50 and data.P >= 30 and data.K >= 30:
            score += 2
            
        best_crops.append((crop, score))
    
    # Sort by score and return top recommendation
    best_crops.sort(key=lambda x: x[1], reverse=True)
    top_crop = best_crops[0][0].title()
    
    return f"{top_crop} is the best crop to be cultivated with your current soil and climate conditions."

def get_fertilizer_recommendation(data: FertilizerRecommendation):
    """Simple fertilizer recommendation logic - you can enhance this with ML models"""
    crop_lower = data.cropType.lower()
    
    if crop_lower in fertilizer_recommendations:
        rec = fertilizer_recommendations[crop_lower]
        return {
            "fertilizer": rec["npk"],
            "deficiency": rec["deficiency"],
            "application": rec["application"]
        }
    else:
        return {
            "fertilizer": "20-20-20 (Balanced NPK)",
            "deficiency": "General purpose fertilizer",
            "application": "Apply 200kg/ha at planting"
        }

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

@app.post("/crop-recommend")
async def crop_recommend(data: CropRecommendation):
    """Get crop recommendation based on soil and climate data"""
    try:
        recommendation = get_crop_recommendation(data)
        return JSONResponse({
            "status": "ok",
            "recommendation": recommendation,
            "timestamp": datetime.utcnow().isoformat()
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting crop recommendation: {str(e)}")

@app.post("/fertilizer-recommend")
async def fertilizer_recommend(data: FertilizerRecommendation):
    """Get fertilizer recommendation based on crop and soil data"""
    try:
        recommendation = get_fertilizer_recommendation(data)
        return JSONResponse({
            "status": "ok",
            "fertilizer": recommendation["fertilizer"],
            "deficiency": recommendation["deficiency"],
            "application": recommendation["application"],
            "timestamp": datetime.utcnow().isoformat()
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting fertilizer recommendation: {str(e)}")

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

