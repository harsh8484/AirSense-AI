import os
import sys
import json
import sqlite3
from fastapi import FastAPI, BackgroundTasks, HTTPException, Query, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Add src to python path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from database import init_db, DATABASE_PATH
from generator import run_all as generate_mock_data
from ingestion import run_ingestion_pipeline
from collocation import run_collocation
from eda import generate_eda_metrics
from modeling import run_model_training
from forecasting import run_forecasting_training, get_forecast_for_station
from rag_assistant import LocalRAGAssistant
from mapping import generate_spatial_overlay

app = FastAPI(title="AirSense AI API", description="Satellite-Based Air Pollution Monitoring System")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup directories
os.makedirs("web/static", exist_ok=True)
os.makedirs("web/static/cache", exist_ok=True)
os.makedirs("web/templates", exist_ok=True)

# Mount static and templates
app.mount("/static", StaticFiles(directory="web/static"), name="static")
app.mount("/models", StaticFiles(directory="models"), name="models")
templates = Jinja2Templates(directory="web/templates")

# Initialize RAG assistant
rag_assistant = LocalRAGAssistant()

@app.on_event("startup")
def startup_event():
    init_db()
    # Attempt to load RAG documents index if documents already exist
    if os.path.exists("data/raw/documents"):
        rag_assistant.load_and_index_documents()

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Renders the main dashboard page."""
    return templates.TemplateResponse(request, "index.html")

@app.get("/api/status")
async def get_status():
    """Returns database size, training states, and available models."""
    status = {
        "has_data": False,
        "has_model": False,
        "has_forecast": False,
        "cpcb_records": 0,
        "aod_records": 0,
        "weather_records": 0,
        "best_model": "None",
        "models_available": []
    }
    
    if os.path.exists(DATABASE_PATH):
        try:
            conn = sqlite3.connect(DATABASE_PATH)
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM ground_measurements")
            status["cpcb_records"] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM satellite_aod")
            status["aod_records"] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM weather_data")
            status["weather_records"] = cursor.fetchone()[0]
            
            conn.close()
            
            if status["cpcb_records"] > 0:
                status["has_data"] = True
        except Exception as e:
            print(f"[ERROR] Reading status from DB: {e}")
            
    summary_path = "models/training_summary.json"
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r") as f:
                summary = json.load(f)
                status["has_model"] = True
                status["best_model"] = summary.get("best_model", "None")
                status["models_available"] = list(summary.get("metrics", {}).keys())
        except Exception as e:
            print(f"[ERROR] Reading training summary: {e}")
            
    forecast_path = "models/forecaster_summary.json"
    if os.path.exists(forecast_path):
        status["has_forecast"] = True
        
    return status

# Background Tasks
def process_data_task():
    try:
        run_ingestion_pipeline()
        run_collocation()
        print("[BACKGROUND] Ingestion and Collocation complete!")
    except Exception as e:
        print(f"[BACKGROUND ERROR] Ingestion/Collocation failed: {e}")

def train_models_task():
    try:
        run_model_training()
        run_forecasting_training()
        print("[BACKGROUND] Regression and Forecasting model training complete!")
    except Exception as e:
        print(f"[BACKGROUND ERROR] Model training failed: {e}")

@app.post("/api/generate-mock")
async def generate_mock(background_tasks: BackgroundTasks):
    """Triggers mock data generation."""
    try:
        generate_mock_data(days=30)
        # Re-index RAG documents since they are generated
        rag_assistant.load_and_index_documents()
        return {"status": "success", "message": "Synthetic dataset generated successfully!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/process")
async def process_data(background_tasks: BackgroundTasks):
    """Runs data ingestion and collocation pipelines."""
    background_tasks.add_task(process_data_task)
    return {"status": "processing", "message": "Data ingestion and collocation started in background..."}

@app.post("/api/train")
async def train_models(background_tasks: BackgroundTasks):
    """Trains regression models and sequence forecaster."""
    background_tasks.add_task(train_models_task)
    return {"status": "training", "message": "Model training pipeline started in background..."}

@app.get("/api/eda")
async def get_eda():
    """Returns exploratory data analysis metrics."""
    metrics = generate_eda_metrics()
    return metrics

@app.get("/api/ground-stations")
async def get_ground_stations():
    """Returns all ground stations with their latest PM2.5 measurements."""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get stations
        cursor.execute("SELECT * FROM stations")
        stations = [dict(r) for r in cursor.fetchall()]
        
        # For each station, get the latest measurement
        res = []
        for s in stations:
            cursor.execute("""
                SELECT pm25, pm10, timestamp FROM ground_measurements
                WHERE station_id = ?
                ORDER BY timestamp DESC LIMIT 1
            """, (s["id"],))
            m_row = cursor.fetchone()
            if m_row:
                s_dict = dict(s)
                s_dict["latest_pm25"] = m_row["pm25"]
                s_dict["latest_pm10"] = m_row["pm10"]
                s_dict["last_updated"] = m_row["timestamp"]
                res.append(s_dict)
            else:
                s_dict = dict(s)
                s_dict["latest_pm25"] = None
                s_dict["latest_pm10"] = None
                s_dict["last_updated"] = None
                res.append(s_dict)
                
        conn.close()
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/forecast/{station_id}")
async def get_forecast(station_id: str):
    """Gets forecasting predictions (1h, 24h, 72h) for a ground station."""
    forecast = get_forecast_for_station(station_id)
    if not forecast:
        raise HTTPException(status_code=404, detail="Forecasting not available. Train forecaster first.")
    return forecast

@app.get("/api/map-layer")
async def get_map_layer(date: str = Query("2026-07-11"), model: str = Query("Random Forest")):
    """Generates and returns grid AOD and predicted PM2.5 overlay details."""
    layer = generate_spatial_overlay(date, model)
    if not layer:
        raise HTTPException(status_code=404, detail="Data not available or map generation failed.")
    return layer

@app.post("/api/chat")
async def chat_assistant(payload: dict):
    """RAG AI Chatbot Endpoint."""
    query_text = payload.get("query", "")
    if not query_text:
        return {"answer": "Please ask a valid question about air quality."}
    res = rag_assistant.query(query_text)
    return res

@app.get("/api/predict-point")
async def predict_point(
    lat: float, lon: float, aod: float, temp: float, rh: float, ws: float, pblh: float,
    model_name: str = Query("Random Forest")
):
    """Real-time interactive prediction for a single coordinate point."""
    import joblib
    import pandas as pd
    import numpy as np
    
    model_filename = f"{model_name.lower().replace(' ', '_')}.joblib"
    model_path = os.path.join("models/regression", model_filename)
    
    if not os.path.exists(model_path):
        # Physical model simulation fallback
        pred = aod * 180.0 * (1000.0 / pblh) * (rh / 50.0)
        pred = max(10.0, min(350.0, float(pred)))
        return {"predicted_pm25": round(pred, 2), "model_used": "Physical Equation Proxy (Fallback)"}
        
    try:
        model = joblib.load(model_path)
        df_feat = pd.DataFrame({
            "aod": [aod],
            "temp": [temp],
            "rh": [rh],
            "wind_speed": [ws],
            "pblh": [pblh],
            "latitude": [lat],
            "longitude": [lon]
        })
        pred = model.predict(df_feat)[0]
        return {"predicted_pm25": round(float(pred), 2), "model_used": model_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

if __name__ == "__main__":
    import uvicorn
    # Initialize DB on start
    init_db()
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
