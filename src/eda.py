import os
import pandas as pd
import numpy as np

def generate_eda_metrics(collocated_csv="data/processed/collocated.csv"):
    """
    Reads collocated dataset and calculates histograms and correlations for web charting.
    """
    if not os.path.exists(collocated_csv):
        print(f"[WARNING] Collocated data file not found at {collocated_csv}")
        return {}
        
    df = pd.read_csv(collocated_csv)
    
    # 1. Basic Stats
    summary_stats = {}
    cols_to_stats = ["pm25", "pm10", "aod", "temp", "rh", "wind_speed", "pblh"]
    for col in cols_to_stats:
        if col in df.columns:
            summary_stats[col] = {
                "mean": float(df[col].mean()),
                "median": float(df[col].median()),
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "std": float(df[col].std())
            }
            
    # 2. Histograms (for PM2.5 and AOD)
    histograms = {}
    
    if "pm25" in df.columns:
        counts, bins = np.histogram(df["pm25"].dropna(), bins=15)
        histograms["pm25"] = {
            "counts": [int(c) for c in counts],
            "bins": [round(float(b), 1) for b in bins]
        }
        
    if "aod" in df.columns:
        counts, bins = np.histogram(df["aod"].dropna(), bins=15)
        histograms["aod"] = {
            "counts": [int(c) for c in counts],
            "bins": [round(float(b), 2) for b in bins]
        }
        
    # 3. Correlation Heatmap matrix
    correlation_data = {}
    if all(c in df.columns for c in cols_to_stats):
        corr_matrix = df[cols_to_stats].corr()
        correlation_data = {
            "columns": cols_to_stats,
            "values": corr_matrix.values.tolist()
        }
        
    return {
        "stats": summary_stats,
        "histograms": histograms,
        "correlation": correlation_data,
        "total_records": len(df)
    }

if __name__ == "__main__":
    metrics = generate_eda_metrics()
    print("EDA total collocated rows:", metrics.get("total_records", 0))
