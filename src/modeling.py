import os
import json
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Optional imports with robust fallbacks
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMRegressor
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    from catboost import CatBoostRegressor
    HAS_CAT = True
except ImportError:
    HAS_CAT = False

def calculate_mbe(y_true, y_pred):
    """Calculates Mean Bias Error."""
    return float(np.mean(y_pred - y_true))

def run_model_training(csv_path="data/processed/collocated.csv"):
    """
    Trains LR, RF, XGB, LGB, and Cat models on the collocated dataset.
    Saves models and outputs validation metrics + scatter plots dataset.
    """
    print("[INFO] Starting Model Training...")
    if not os.path.exists(csv_path):
        print(f"[WARNING] Collocated CSV not found at {csv_path}. Train failed.")
        return {}
        
    df = pd.read_csv(csv_path)
    if len(df) < 5:
        print("[WARNING] Insufficient data rows for modeling. Need at least 5 collocated rows.")
        return {}
        
    # Define features and target
    features = ["aod", "temp", "rh", "wind_speed", "pblh", "latitude", "longitude"]
    target = "pm25"
    
    X = df[features]
    y = df[target]
    
    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Directory to store models
    os.makedirs("models/regression", exist_ok=True)
    
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
    }
    
    # Add optional models if libraries are successfully imported
    if HAS_XGB:
        models["XGBoost"] = XGBRegressor(n_estimators=100, random_state=42, n_jobs=1)
    if HAS_LGB:
        # verbosity=-1 to suppress training logs
        models["LightGBM"] = LGBMRegressor(n_estimators=100, random_state=42, verbosity=-1, n_jobs=1)
    if HAS_CAT:
        # verbose=False to suppress training logs
        models["CatBoost"] = CatBoostRegressor(iterations=100, random_state=42, verbose=False, thread_count=1)
        
    metrics_report = {}
    scatter_plots = {}
    feature_importances = {}
    
    best_r2 = -999.0
    best_model_name = ""
    
    for name, model in models.items():
        print(f"[INFO] Training {name} model...")
        try:
            model.fit(X_train, y_train)
            
            # Predict
            preds = model.predict(X_test)
            
            # Save Model
            model_file = os.path.join("models/regression", f"{name.lower().replace(' ', '_')}.joblib")
            joblib.dump(model, model_file)
            
            # Metrics
            r2 = float(r2_score(y_test, preds))
            rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
            mae = float(mean_absolute_error(y_test, preds))
            mbe = calculate_mbe(y_test, preds)
            
            metrics_report[name] = {
                "r2": round(r2, 4),
                "rmse": round(rmse, 2),
                "mae": round(mae, 2),
                "mbe": round(mbe, 2)
            }
            
            # Scatter Plot Points (Limit to first 100 for web performance)
            scatter_plots[name] = {
                "observed": [round(float(y), 1) for y in y_test.tolist()[:100]],
                "predicted": [round(float(p), 1) for p in preds.tolist()[:100]]
            }
            
            # Feature Importance
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                feature_importances[name] = {
                    features[i]: float(importances[i]) for i in range(len(features))
                }
            elif name == "Linear Regression":
                # For Linear Regression, show absolute coefficient weights normalized to sum to 1
                coefs = np.abs(model.coef_)
                norm_coefs = coefs / np.sum(coefs) if np.sum(coefs) > 0 else coefs
                feature_importances[name] = {
                    features[i]: float(norm_coefs[i]) for i in range(len(features))
                }
                
            # Track best model
            if r2 > best_r2:
                best_r2 = r2
                best_model_name = name
                
        except Exception as e:
            print(f"[ERROR] Failed training {name}: {e}")
            metrics_report[name] = {"error": str(e)}
            
    # Compile the final modeling summary report
    report = {
        "metrics": metrics_report,
        "scatter": scatter_plots,
        "importances": feature_importances,
        "best_model": best_model_name,
        "features": features,
        "has_models": {
            "xgb": HAS_XGB,
            "lgb": HAS_LGB,
            "cat": HAS_CAT
        }
    }
    
    # Save training metadata
    with open(os.path.join("models", "training_summary.json"), "w") as f:
        json.dump(report, f, indent=4)
        
    print(f"[SUCCESS] Model training pipeline complete! Best Model: {best_model_name} ($R^2$: {round(best_r2, 4)})")
    return report

if __name__ == "__main__":
    run_model_training()
