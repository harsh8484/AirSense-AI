import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from database import get_connection

# Spatial extent of India
GRID_LAT_MIN, GRID_LAT_MAX = 6.0, 37.0
GRID_LON_MIN, GRID_LON_MAX = 68.0, 97.0
RESOLUTION = 0.5  # Grid spacing for prediction map

GRID_LATS = np.arange(GRID_LAT_MIN, GRID_LAT_MAX + RESOLUTION, RESOLUTION)
GRID_LONS = np.arange(GRID_LON_MIN, GRID_LON_MAX + RESOLUTION, RESOLUTION)
GRID_LON_2D, GRID_LAT_2D = np.meshgrid(GRID_LONS, GRID_LATS)

def generate_spatial_overlay(date_str, model_name="Random Forest"):
    """
    Interpolates satellite AOD and reanalysis weather parameters over a grid covering India,
    runs the selected regression model, and creates a transparent color-coded PNG overlay.
    """
    model_filename = f"{model_name.lower().replace(' ', '_')}.joblib"
    model_path = os.path.join("models/regression", model_filename)
    
    if not os.path.exists(model_path):
        print(f"[WARNING] Model file {model_path} not found. Using default formula overlay.")
        model = None
    else:
        model = joblib.load(model_path)
        
    conn = get_connection()
    cursor = conn.cursor()
    
    # 1. Fetch satellite AOD coordinates and values for the date
    cursor.execute("SELECT lat, lon, aod FROM satellite_aod WHERE timestamp = ?", (date_str,))
    aod_rows = cursor.fetchall()
    
    # 2. Fetch reanalysis weather variables for the date
    cursor.execute("SELECT lat, lon, temp, rh, wind_speed, pblh FROM weather_data WHERE timestamp = ?", (date_str,))
    weather_rows = cursor.fetchall()
    conn.close()
    
    if not aod_rows or not weather_rows:
        print(f"[WARNING] No AOD or weather data found in DB for date {date_str}")
        return None
        
    # Prepare interpolation inputs
    aod_coords = np.array([[r['lat'], r['lon']] for r in aod_rows])
    aod_vals = np.array([r['aod'] for r in aod_rows])
    
    w_coords = np.array([[r['lat'], r['lon']] for r in weather_rows])
    w_temp = np.array([r['temp'] for r in weather_rows])
    w_rh = np.array([r['rh'] for r in weather_rows])
    w_ws = np.array([r['wind_speed'] for r in weather_rows])
    w_pbl = np.array([r['pblh'] for r in weather_rows])
    
    # Interpolate variables over the 2D grid covering India
    grid_points = np.column_stack((GRID_LAT_2D.ravel(), GRID_LON_2D.ravel()))
    
    try:
        # Interpolate variables
        grid_aod = griddata(aod_coords, aod_vals, grid_points, method='linear', fill_value=np.nanmean(aod_vals))
        grid_temp = griddata(w_coords, w_temp, grid_points, method='linear', fill_value=np.nanmean(w_temp))
        grid_rh = griddata(w_coords, w_rh, grid_points, method='linear', fill_value=np.nanmean(w_rh))
        grid_ws = griddata(w_coords, w_ws, grid_points, method='linear', fill_value=np.nanmean(w_ws))
        grid_pbl = griddata(w_coords, w_pbl, grid_points, method='linear', fill_value=np.nanmean(w_pbl))
        
        # Build features dataframe
        df_grid = pd.DataFrame({
            "aod": grid_aod,
            "temp": grid_temp,
            "rh": grid_rh,
            "wind_speed": grid_ws,
            "pblh": grid_pbl,
            "latitude": grid_points[:, 0],
            "longitude": grid_points[:, 1]
        })
        
        # Predict PM2.5
        if model:
            predictions = model.predict(df_grid)
        else:
            # Fallback to physical equation proxy if model isn't trained yet
            predictions = df_grid["aod"] * 180.0 * (1000.0 / df_grid["pblh"]) * (df_grid["rh"] / 50.0)
            predictions = np.clip(predictions, 10.0, 350.0)
            
        # Reshape predictions back to 2D grid
        pm25_grid = predictions.reshape(GRID_LAT_2D.shape)
        
        # Apply mask outside India land mass (simple polygon proxy - coordinates cut off)
        # Lat/Lon bounding mask to clip the corners and keep it clean
        for i in range(len(GRID_LATS)):
            lat = GRID_LATS[i]
            for j in range(len(GRID_LONS)):
                lon = GRID_LONS[j]
                # Simulating a mask: cut extreme sea areas (e.g. bottom left/right corners)
                if (lat < 8.0) or (lat < 12.0 and lon < 74.0) or (lat < 12.0 and lon > 84.0) or (lat > 35.0 and lon < 72.0) or (lat > 35.0 and lon > 80.0):
                    pm25_grid[i, j] = np.nan
                    
        # Generate transparent PNG using Matplotlib
        cache_dir = "web/static/cache"
        os.makedirs(cache_dir, exist_ok=True)
        img_name = f"overlay_{date_str}_{model_name.lower().replace(' ', '_')}.png"
        img_path = os.path.join(cache_dir, img_name)
        
        # Custom Colormap for AQI (Green to Red to Maroon)
        # Normalize between 0 and 200 for good coloring
        norm = matplotlib.colors.Normalize(vmin=10.0, vmax=220.0)
        cmap = plt.cm.jet
        
        fig = plt.figure(figsize=(8, 8), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1], frameon=False, xticks=[], yticks=[])
        
        # Plot the 2D grid
        ax.imshow(
            pm25_grid, 
            cmap=cmap, 
            norm=norm, 
            extent=[GRID_LON_MIN, GRID_LON_MAX, GRID_LAT_MIN, GRID_LAT_MAX], 
            origin='lower', 
            alpha=0.6,
            interpolation='bicubic'
        )
        
        plt.savefig(img_path, transparent=True, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        
        # Return coordinates and metadata for Leaflet ImageOverlay
        # Leaflet expects bounding box: [[south, west], [north, east]]
        return {
            "image_url": f"/static/cache/{img_name}",
            "bounds": [[GRID_LAT_MIN, GRID_LON_MIN], [GRID_LAT_MAX, GRID_LON_MAX]],
            "grid_data": {
                "lats": GRID_LATS.tolist(),
                "lons": GRID_LONS.tolist(),
                "pm25": np.where(np.isnan(pm25_grid), -999.0, np.round(pm25_grid, 1)).tolist()
            }
        }
        
    except Exception as e:
        print(f"[ERROR] Failed to generate spatial map overlay: {e}")
        return None

if __name__ == "__main__":
    # Test execution
    res = generate_spatial_overlay("2026-07-11")
    if res:
        print("Overlay generated successfully at bounds:", res["bounds"])
