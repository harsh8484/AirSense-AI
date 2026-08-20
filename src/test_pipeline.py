import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__)))

from database import init_db, DATABASE_PATH
from generator import run_all as generate_mock_data
from ingestion import run_ingestion_pipeline
from collocation import run_collocation
from modeling import run_model_training
from forecasting import run_forecasting_training
from rag_assistant import LocalRAGAssistant

def verify_pipeline():
    print("======================================================")
    print("          AirSense AI - Pipeline Verification         ")
    print("======================================================")
    
    # Step 1: Initialize DB
    print("\n[STEP 1/6] Initializing Database...")
    init_db()
    
    # Step 2: Generate Mock Datasets
    print("\n[STEP 2/6] Generating Mock Datasets...")
    generate_mock_data(days=30)
    
    # Step 3: Run Ingestion
    print("\n[STEP 3/6] Running Data Ingestion...")
    g_count, s_count, w_count = run_ingestion_pipeline()
    
    # Step 4: Run Collocation
    print("\n[STEP 4/6] Running Data Collocation...")
    df_collocated = run_collocation()
    
    # Step 5: Train ML Models
    print("\n[STEP 5/6] Training Regression Models...")
    report = run_model_training()
    
    # Step 6: Train LSTM forecasting
    print("\n[STEP 6/6] Training Time Series Forecasting...")
    forecast_report = run_forecasting_training()
    
    # Final checks
    print("\n======================================================")
    print("                Pipeline Integrity Check              ")
    print("======================================================")
    
    checks = {
        "Database Created": os.path.exists(DATABASE_PATH),
        "Ground Records Ingested": g_count > 0,
        "Satellite Records Ingested": s_count > 0,
        "Weather Records Ingested": w_count > 0,
        "Collocated CSV Generated": os.path.exists("data/processed/collocated.csv"),
        "Models Directory Exists": os.path.exists("models/regression"),
        "Summary Report Written": os.path.exists("models/training_summary.json"),
        "Forecaster Trained": os.path.exists("models/forecaster/lstm_forecaster.joblib")
    }
    
    failures = 0
    for name, success in checks.items():
        status = "PASSED" if success else "FAILED"
        print(f"[{status}] {name}")
        if not success:
            failures += 1
            
    # Index RAG assistant to make sure it loads
    print("\n[INFO] Loading RAG Assistant Index...")
    rag = LocalRAGAssistant()
    rag_indexed = rag.load_and_index_documents()
    print(f"[{'PASSED' if rag_indexed else 'FAILED'}] RAG Documents Indexed")
    if not rag_indexed:
        failures += 1
        
    if failures == 0:
        print("\n[SUCCESS] Entire data science pipeline executed with 100% integrity!")
        print("You can now safely run: python main.py")
        sys.exit(0)
    else:
        print(f"\n[ERROR] Pipeline verification failed with {failures} issues.")
        sys.exit(1)

if __name__ == "__main__":
    verify_pipeline()
