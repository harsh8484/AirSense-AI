# AirSense AI - Satellite-Based Air Pollution Monitoring System

AirSense AI is a platform designed to estimate surface-level Particulate Matter (PM2.5) concentrations by combining satellite Aerosol Optical Depth (AOD) measurements, ground-based CPCB observations, and meteorological reanalysis parameters using machine learning.

## Features
1. **Multi-Model Regression Suite**: Compares Linear Regression, Random Forest, and LightGBM models to predict PM2.5.
2. **LSTM Time-Series Forecasting**: Multi-step sequence model forecasting 1h, 24h, and 72h future air quality trends.
3. **Exploratory Data Analysis (EDA)**: Dashboard visualizations for AOD and PM2.5 distributions along with cross-variable correlation heatmaps.
4. **AI Assistant chatbot**: Offline local retrieval RAG system supporting policy queries mapped to WHO and CPCB guidelines.
5. **Interactive Mapping GIS**: Beautiful Leaflet map rendering ground stations and predicting high-resolution PM2.5 grid overlays across India.

## Setup and Running

1. **Prerequisites**:
   Ensure you have Python 3.10+ installed.

2. **Booting the System**:
   Double click the `run.bat` file in the root folder. This Windows batch script will automatically:
   - Create a Python virtual environment (`venv`).
   - Install required dependencies.
   - Run the FastAPI server.

3. **Accessing the Dashboard**:
   Open your browser and navigate to:
   [http://127.0.0.1:8000](http://127.0.0.1:8000)

## Directory Structure
- `data/`: Ingested raw datasets (satellite H5, ground CPCB CSVs, weather NetCDFs, and reference RAG documents).
- `src/`: Ingestion pipelines, spatial-temporal collocation matching, regression modeling, sequence forecasting, vector similarity matching, and overlay mapping.
- `web/`: Templates and static assets (Tailwind CSS styling, Leaflet, and Chart.js controllers).
- `main.py`: FastAPI server configuration.
- `run.bat`: Startup automation script.
