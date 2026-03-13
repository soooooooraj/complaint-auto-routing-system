import os
import time
import json
import tempfile
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import List
import uvicorn
import joblib

from api.schemas import ComplaintTextInput, ComplaintResponse, OfficerResult, SimilarComplaint
from pipeline.ingest import process_input
from pipeline.features import get_embedding
from models.priority_classifier import predict as predict_priority, MODEL_PATH as PRIO_PATH
from models.eta_regressor import predict as predict_eta, MODEL_PATH as ETA_PATH, ENCODER_PATH
from models.officer_router import route_complaint, EMBEDDINGS_PATH, MAPPING_PATH as OFFICER_MAP_PATH
from models.similarity_search import find_similar, load_index

app = FastAPI(title="Complaint Auto-Routing System")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
os.makedirs("api/static", exist_ok=True)

# Global variables to verify models are ready
models_ready = False
priority_model = None
eta_model = None
category_encoder = None
faiss_index = None
faiss_mapping = None
officer_embeddings = None
officer_mapping = None

@app.on_event("startup")
async def startup_event():
    global models_ready, priority_model, eta_model, category_encoder, faiss_index, faiss_mapping, officer_embeddings, officer_mapping
    print("Preloading all models into memory...")
    try:
        # Load Models
        faiss_index, faiss_mapping = load_index()
        priority_model = joblib.load(PRIO_PATH)
        eta_model = joblib.load(ETA_PATH)
        category_encoder = joblib.load(ENCODER_PATH)
        officer_embeddings = joblib.load(EMBEDDINGS_PATH)
        with open(OFFICER_MAP_PATH, 'r', encoding='utf-8') as f:
            officer_mapping = json.load(f)
            
        # Preload embeddings model
        get_embedding("preload")

        models_ready = True
        print("All models loaded and ready")
    except Exception as e:
        print(f"Error loading models: {e}")
        models_ready = False

@app.get("/health")
async def health_check():
    return {"status": "healthy", "models_loaded": models_ready}

def _run_inference_pipeline(clean_text, detected_lang, start_time):
    # 2. Priority Classification
    prio_result = predict_priority(clean_text, clf=priority_model)
    priority = prio_result["priority"]
    confidence = prio_result["confidence"]
    
    # 3. Similarity Search & Category Inference
    similar = find_similar(clean_text, top_k=5, index=faiss_index, mapping=faiss_mapping)
    if not similar:
        raise HTTPException(status_code=500, detail="Similarity search failed")
    
    inferred_category = similar[0]["category"]
    
    # 4. ETA Regression
    eta_days = predict_eta(clean_text, priority, inferred_category, reg=eta_model, cat_encoder=category_encoder)
    
    # 5. Officer Routing
    officers_list = route_complaint(clean_text, priority, category=inferred_category, top_k=3, embeddings=officer_embeddings, mapping=officer_mapping)
    
    processing_time = (time.time() - start_time) * 1000
    
    return ComplaintResponse(
        complaint_text=clean_text,
        detected_language=detected_lang,
        priority=priority,
        priority_confidence=confidence,
        eta_days=eta_days,
        assigned_officers=[OfficerResult(**o) for o in officers_list],
        similar_complaints=[SimilarComplaint(**s) for s in similar],
        processing_time_ms=round(processing_time, 2)
    )

@app.post("/complaint/text", response_model=ComplaintResponse)
def process_text_complaint(input_data: ComplaintTextInput):
    start_time = time.time()
    
    # 1. Pipeline Ingestion (Clean and Translate)
    ingest_result = process_input("text", input_data.text)
    
    return _run_inference_pipeline(ingest_result["clean_text"], ingest_result["detected_language"], start_time)

@app.post("/complaint/audio", response_model=ComplaintResponse)
def process_audio_complaint(file: UploadFile = File(...)):
    start_time = time.time()
    
    # Create temp file
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
        
    try:
        # 1. Pipeline Ingestion (Transcribe + Translate)
        ingest_result = process_input("audio", tmp_path)
        return _run_inference_pipeline(ingest_result["clean_text"], ingest_result["detected_language"], start_time)
    finally:
        os.remove(tmp_path)

@app.post("/complaint/video", response_model=ComplaintResponse)
def process_video_complaint(file: UploadFile = File(...)):
    start_time = time.time()
    
    # Create temp file
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
        
    try:
        # 1. Pipeline Ingestion (Transcribe + Translate)
        ingest_result = process_input("video", tmp_path)
        return _run_inference_pipeline(ingest_result["clean_text"], ingest_result["detected_language"], start_time)
    finally:
        os.remove(tmp_path)

@app.get("/officers")
async def get_officers():
    with open("data/officers.json", "r", encoding="utf-8") as f:
        officers = json.load(f)
    return officers

@app.get("/stats")
async def get_stats():
    with open("data/complaints.json", "r", encoding="utf-8") as f:
        complaints = json.load(f)
        
    total = len(complaints)
    cats = {}
    prios = {}
    langs = {}
    
    for c in complaints:
        cats[c["category"]] = cats.get(c["category"], 0) + 1
        prios[c["priority"]] = prios.get(c["priority"], 0) + 1
        langs[c["language"]] = langs.get(c["language"], 0) + 1
        
    return {
        "total_complaints": total,
        "complaints_per_category": cats,
        "complaints_per_priority": prios,
        "complaints_per_language": langs
    }

app.mount("/", StaticFiles(directory="api/static", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=True)
