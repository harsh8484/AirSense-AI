import os
import sqlite3

DATABASE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "database")
DATABASE_PATH = os.path.join(DATABASE_DIR, "airsense.db")

def get_connection():
    """Returns a connection to the SQLite database, ensuring parent directory exists."""
    os.makedirs(DATABASE_DIR, exist_ok=True)
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes the database schema by creating all required tables."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # 1. Ground Stations
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS stations (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            state TEXT,
            city TEXT
        )
    """)
    
    # 2. CPCB Ground Measurements
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ground_measurements (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            station_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            pm25 REAL,
            pm10 REAL,
            FOREIGN KEY (station_id) REFERENCES stations (id),
            UNIQUE(station_id, timestamp)
        )
    """)
    
    # 3. Satellite AOD Observations
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS satellite_aod (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            lat REAL NOT NULL,
            lon REAL NOT NULL,
            aod REAL,
            UNIQUE(timestamp, lat, lon)
        )
    """)
    
    # 4. Atmospheric Reanalysis Weather Parameters
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS weather_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            lat REAL NOT NULL,
            lon REAL NOT NULL,
            temp REAL,
            rh REAL,
            wind_speed REAL,
            pblh REAL,
            UNIQUE(timestamp, lat, lon)
        )
    """)
    
    # 5. Model Predictions (Spatial/Temporal)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            lat REAL NOT NULL,
            lon REAL NOT NULL,
            pm25 REAL NOT NULL,
            model_name TEXT NOT NULL,
            UNIQUE(timestamp, lat, lon, model_name)
        )
    """)
    
    conn.commit()
    conn.close()
    print(f"[INFO] Database initialized at {DATABASE_PATH}")

if __name__ == "__main__":
    init_db()
