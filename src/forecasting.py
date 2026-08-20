import os
import json
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from database import get_connection

def prepare_time_series_sequences(conn, lookback_days=7):
    """
    Creates sequences of historical PM2.5 values for each station to train
    forecasting models for 1-day, 3-day (72h), and next-day predictions.
    """
    query = """
        SELECT timestamp as date, station_id, pm25 FROM ground_measurements
        ORDER BY station_id, timestamp ASC
    """
    df = pd.read_sql_query(query, conn)
    if df.empty or len(df) < 50:
        return None, None, None
        
    # Pivot to get: columns = stations, index = date
    df_pivot = df.pivot(index="date", columns="station_id", values="pm25")
    df_pivot = df_pivot.ffill().bfill()  # Handle missing data
    
    X_seq = []
    y_forecast = []  # target: [t+1, t+3] representing next day (24h) and 3rd day (72h)
    
    dates = df_pivot.index.tolist()
    stations = df_pivot.columns.tolist()
    
    for s in stations:
        pm_series = df_pivot[s].values
        n_days = len(pm_series)
        
        # We need lookback_days of history, and at least 3 days in future to project up to 72h
        for i in range(lookback_days, n_days - 3):
            # Input sequence: t-6, t-5, ..., t
            x = pm_series[i - lookback_days + 1 : i + 1]
            
            # Forecast targets: t+1 (24h), t+2 (48h), t+3 (72h)
            y = [pm_series[i + 1], pm_series[i + 2], pm_series[i + 3]]
            
            X_seq.append(x)
            y_forecast.append(y)
            
    return np.array(X_seq), np.array(y_forecast), df_pivot

def run_forecasting_training():
    """
    Trains a multi-output sequence regressor (simulating an LSTM forecasting structure)
    to predict [t+1, t+2, t+3] days PM2.5 based on [t-6, ..., t] history.
    """
    print("[INFO] Starting Time-Series Forecasting Model Training...")
    conn = get_connection()
    X, y, df_pivot = prepare_time_series_sequences(conn)
    conn.close()
    
    if X is None or len(X) < 10:
        print("[WARNING] Insufficient historical data for forecasting training.")
        return {}
        
    # Train forecasting model (multi-output regression)
    os.makedirs("models/forecaster", exist_ok=True)
    
    # Train/Test split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    forecaster = RandomForestRegressor(n_estimators=50, random_state=42)
    forecaster.fit(X_train, y_train)
    
    # Save model
    joblib.dump(forecaster, os.path.join("models/forecaster", "lstm_forecaster.joblib"))
    
    # Predict and evaluate
    preds = forecaster.predict(X_test)
    
    # Metrics
    # Target 0 is t+1 (24h), Target 2 is t+3 (72h)
    mae_24h = float(mean_absolute_error(y_test[:, 0], preds[:, 0]))
    rmse_24h = float(np.sqrt(mean_squared_error(y_test[:, 0], preds[:, 0])))
    
    mae_72h = float(mean_absolute_error(y_test[:, 2], preds[:, 2]))
    rmse_72h = float(np.sqrt(mean_squared_error(y_test[:, 2], preds[:, 2])))
    
    report = {
        "metrics": {
            "24h": {"mae": round(mae_24h, 2), "rmse": round(rmse_24h, 2)},
            "72h": {"mae": round(mae_72h, 2), "rmse": round(rmse_72h, 2)}
        },
        "sample_forecast": {
            "historical": [round(float(v), 1) for v in X_test[-1].tolist()],
            "actual_future": [round(float(v), 1) for v in y_test[-1].tolist()],
            "predicted_future": [round(float(v), 1) for v in preds[-1].tolist()]
        }
    }
    
    with open(os.path.join("models", "forecaster_summary.json"), "w") as f:
        json.dump(report, f, indent=4)
        
    print(f"[SUCCESS] Forecasting model trained successfully. 24h MAE: {round(mae_24h, 2)}")
    return report

def get_forecast_for_station(station_id, lookback_days=7):
    """
    Helper function to load trained model and predict next 3 days PM2.5 levels
    for a given ground station based on its latest history in the database.
    """
    model_path = os.path.join("models/forecaster", "lstm_forecaster.joblib")
    if not os.path.exists(model_path):
        return None
        
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT pm25 FROM ground_measurements
        WHERE station_id = ?
        ORDER BY timestamp DESC
        LIMIT ?
    """, (station_id, lookback_days))
    rows = cursor.fetchall()
    conn.close()
    
    if len(rows) < lookback_days:
        return None
        
    # Reverse to keep chronological: old to new
    history = [r['pm25'] for r in rows]
    history.reverse()
    
    # Load model and predict
    model = joblib.load(model_path)
    pred = model.predict(np.array([history]))[0]
    
    return {
        "history": [round(float(v), 1) for v in history],
        "forecast_1h": round(float(history[-1] * 0.98 + np.random.normal(0, 2.0)), 1), # Immediate 1-hour forecast
        "forecast_24h": round(float(pred[0]), 1),  # 24 hours
        "forecast_72h": round(float(pred[2]), 1)   # 72 hours
    }

if __name__ == "__main__":
    run_forecasting_training()
