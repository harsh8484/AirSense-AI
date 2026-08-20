import os
import pandas as pd
import numpy as np
from database import get_connection

def find_nearest_satellite_aod(conn, date_str, station_lat, station_lon, max_dist=1.5):
    """
    Finds the nearest satellite AOD value on a specific date for a given station coordinate.
    If the closest point is further than max_dist, returns None.
    """
    cursor = conn.cursor()
    cursor.execute("""
        SELECT lat, lon, aod FROM satellite_aod
        WHERE timestamp = ? AND aod IS NOT NULL
    """, (date_str,))
    rows = cursor.fetchall()
    
    if not rows:
        return None
        
    min_dist = float('inf')
    best_aod = None
    
    for r in rows:
        lat, lon, aod = r['lat'], r['lon'], r['aod']
        dist = np.sqrt((lat - station_lat)**2 + (lon - station_lon)**2)
        if dist < min_dist:
            min_dist = dist
            best_aod = aod
            
    if min_dist <= max_dist:
        return best_aod
    return None

def find_nearest_weather(conn, date_str, station_lat, station_lon, max_dist=1.5):
    """
    Finds the nearest weather metrics on a specific date for a given station coordinate.
    """
    cursor = conn.cursor()
    cursor.execute("""
        SELECT lat, lon, temp, rh, wind_speed, pblh FROM weather_data
        WHERE timestamp = ?
    """, (date_str,))
    rows = cursor.fetchall()
    
    if not rows:
        return None
        
    min_dist = float('inf')
    best_weather = None
    
    for r in rows:
        lat, lon, temp, rh, ws, pb = r['lat'], r['lon'], r['temp'], r['rh'], r['wind_speed'], r['pblh']
        dist = np.sqrt((lat - station_lat)**2 + (lon - station_lon)**2)
        if dist < min_dist:
            min_dist = dist
            best_weather = {
                "temp": temp,
                "rh": rh,
                "wind_speed": ws,
                "pblh": pb
            }
            
    if min_dist <= max_dist:
        return best_weather
    return None

def run_collocation():
    """
    Collocates ground station measurements with corresponding nearest satellite AOD and
    meteorological weather parameters, saving the result to CSV.
    """
    print("[INFO] Starting Spatio-Temporal Collocation...")
    conn = get_connection()
    
    # Read ground measurements linked with station coordinates
    query = """
        SELECT gm.timestamp as date, gm.station_id, s.name as station_name,
               s.latitude, s.longitude, gm.pm25, gm.pm10
        FROM ground_measurements gm
        JOIN stations s ON gm.station_id = s.id
    """
    df_ground = pd.read_sql_query(query, conn)
    
    if df_ground.empty:
        print("[WARNING] No ground measurements found. Run ingestion first.")
        conn.close()
        return None
        
    collocated_data = []
    
    for _, row in df_ground.iterrows():
        date_str = row["date"]
        lat = row["latitude"]
        lon = row["longitude"]
        
        # 1. Match with nearest Satellite AOD
        aod = find_nearest_satellite_aod(conn, date_str, lat, lon)
        
        # 2. Match with nearest Weather parameter
        weather = find_nearest_weather(conn, date_str, lat, lon)
        
        if aod is not None and weather is not None:
            collocated_data.append({
                "date": date_str,
                "station_id": row["station_id"],
                "station_name": row["station_name"],
                "latitude": lat,
                "longitude": lon,
                "pm25": row["pm25"],
                "pm10": row["pm10"],
                "aod": aod,
                "temp": weather["temp"],
                "rh": weather["rh"],
                "wind_speed": weather["wind_speed"],
                "pblh": weather["pblh"]
            })
            
    conn.close()
    
    if not collocated_data:
        print("[WARNING] Collocation yielded 0 matched rows. Spatial distance threshold might be too tight.")
        return None
        
    df_collocated = pd.DataFrame(collocated_data)
    
    # Save processed collocated CSV
    os.makedirs("data/processed", exist_ok=True)
    out_path = "data/processed/collocated.csv"
    df_collocated.to_csv(out_path, index=False)
    
    print(f"[SUCCESS] Collocated {len(df_collocated)} data rows saved at {out_path}")
    return df_collocated

if __name__ == "__main__":
    run_collocation()
