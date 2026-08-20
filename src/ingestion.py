import os
import glob
import re
import datetime
import numpy as np
import pandas as pd
import h5py
from netCDF4 import Dataset
from database import get_connection

def parse_date_from_satellite_filename(filename):
    """
    Parses date from INSAT-3D filename format: SSNNN_DDMMMYYYY_HHmm_LOP_XXX.h5
    Example: 3DIMG_11JUL2026_0600_L2B_AOD.h5
    """
    basename = os.path.basename(filename)
    match = re.search(r'3DIMG_(\d{2})([A-Z]{3})(\d{4})_', basename)
    if match:
        day, month_str, year = match.groups()
        try:
            date_obj = datetime.datetime.strptime(f"{day}{month_str}{year}", "%d%b%Y")
            return date_obj.strftime("%Y-%m-%d")
        except ValueError:
            pass
    return None

def parse_date_from_reanalysis_filename(filename):
    """
    Parses date from MERRA-2 filename format: MERRA2_400.tavg1_2d_slv_Nx.YYYYMMDD.nc
    """
    basename = os.path.basename(filename)
    match = re.search(r'\.(\d{8})\.nc', basename)
    if match:
        date_str = match.group(1)
        try:
            date_obj = datetime.datetime.strptime(date_str, "%Y%m%d")
            return date_obj.strftime("%Y-%m-%d")
        except ValueError:
            pass
    return None

def ingest_ground_data(csv_path="data/raw/ground/cpcb_ground_data.csv"):
    """
    Reads ground CPCB measurements from CSV, performs QA/QC, and loads into SQLite.
    """
    if not os.path.exists(csv_path):
        print(f"[WARNING] CPCB Ground CSV not found at {csv_path}")
        return 0
        
    df = pd.read_csv(csv_path)
    
    # QA/QC steps:
    # 1. Drop rows with invalid coordinates or dates
    df = df.dropna(subset=["date", "station_id", "latitude", "longitude"])
    # 2. Filter out negative PM values (impossible physics)
    df = df[(df["PM2.5"] >= 0) & (df["PM10"] >= 0)]
    
    conn = get_connection()
    cursor = conn.cursor()
    
    # Ingest stations first
    stations = df[["station_id", "station_name", "latitude", "longitude", "state", "city"]].drop_duplicates()
    for _, s in stations.iterrows():
        cursor.execute("""
            INSERT OR REPLACE INTO stations (id, name, latitude, longitude, state, city)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (s["station_id"], s["station_name"], s["latitude"], s["longitude"], s["state"], s["city"]))
        
    # Ingest measurements
    count = 0
    for _, row in df.iterrows():
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO ground_measurements (station_id, timestamp, pm25, pm10)
                VALUES (?, ?, ?, ?)
            """, (row["station_id"], row["date"], row["PM2.5"], row["PM10"]))
            count += 1
        except Exception as e:
            print(f"[ERROR] Failed to ingest ground measurement: {e}")
            
    conn.commit()
    conn.close()
    print(f"[SUCCESS] Ingested {count} CPCB ground measurements.")
    return count

def ingest_satellite_aod(satellite_dir="data/raw/satellite"):
    """
    Reads geostationary satellite AOD HDF5 files, cleans data, and loads gridded data.
    """
    h5_files = glob.glob(os.path.join(satellite_dir, "*.h5"))
    if not h5_files:
        print(f"[WARNING] No satellite H5 files found in {satellite_dir}")
        return 0
        
    conn = get_connection()
    cursor = conn.cursor()
    
    total_records = 0
    
    for h5_file in h5_files:
        date_str = parse_date_from_satellite_filename(h5_file)
        if not date_str:
            print(f"[WARNING] Could not parse date from satellite file {h5_file}")
            continue
            
        with h5py.File(h5_file, 'r') as f:
            try:
                # Find latitude, longitude, and AOD datasets
                # Standard geostationary grids could be inside groups:
                lat_ds = f["Geolocation_Information/Latitude"][:]
                lon_ds = f["Geolocation_Information/Longitude"][:]
                aod_ds = f["Geophysical_Parameters/AOD"]
                
                aod_data = aod_ds[:]
                fill_value = aod_ds.attrs.get("_FillValue", -999.0)
                scale_factor = aod_ds.attrs.get("scale_factor", 1.0)
                
                # Check for flattened/gridded dimensions consistency
                assert lat_ds.shape == lon_ds.shape == aod_data.shape, "Array dimensions mismatch"
                
                # Rescale and mask fill values
                aod_data = np.where(aod_data == fill_value, np.nan, aod_data)
                aod_data = aod_data * scale_factor
                
                # Filter out values outside physical limits (AOD typically is between 0.0 and 2.0)
                aod_data = np.where((aod_data >= 0.0) & (aod_data <= 2.5), aod_data, np.nan)
                
                # Extract valid cells to insert
                rows, cols = aod_data.shape
                records = []
                for r in range(rows):
                    for c in range(cols):
                        val = aod_data[r, c]
                        if not np.isnan(val):
                            lat = float(lat_ds[r, c])
                            lon = float(lon_ds[r, c])
                            records.append((date_str, round(lat, 2), round(lon, 2), float(val)))
                            
                # Insert in batch
                cursor.executemany("""
                    INSERT OR REPLACE INTO satellite_aod (timestamp, lat, lon, aod)
                    VALUES (?, ?, ?, ?)
                """, records)
                total_records += len(records)
                
            except Exception as e:
                print(f"[ERROR] Failed to parse H5 file {h5_file}: {e}")
                
    conn.commit()
    conn.close()
    print(f"[SUCCESS] Ingested {total_records} satellite AOD points.")
    return total_records

def ingest_reanalysis_weather(reanalysis_dir="data/raw/reanalysis"):
    """
    Reads MERRA-2 meteorological NetCDF files, cleans, calculates wind speed, and loads to SQLite.
    """
    nc_files = glob.glob(os.path.join(reanalysis_dir, "*.nc"))
    if not nc_files:
        print(f"[WARNING] No reanalysis NetCDF files found in {reanalysis_dir}")
        return 0
        
    conn = get_connection()
    cursor = conn.cursor()
    
    total_records = 0
    
    for nc_file in nc_files:
        date_str = parse_date_from_reanalysis_filename(nc_file)
        if not date_str:
            print(f"[WARNING] Could not parse date from reanalysis file {nc_file}")
            continue
            
        try:
            with Dataset(nc_file, 'r') as rootgrp:
                lat_arr = rootgrp.variables['lat'][:]
                lon_arr = rootgrp.variables['lon'][:]
                
                # Read 2m temperature (T2M), RH, wind components U10, V10, and PBLH
                t2m = rootgrp.variables['T2M'][0, :, :]
                rh = rootgrp.variables['RH'][0, :, :]
                u10 = rootgrp.variables['U10'][0, :, :]
                v10 = rootgrp.variables['V10'][0, :, :]
                pblh = rootgrp.variables['PBLH'][0, :, :]
                
                # Calculate wind speed from components: WS = sqrt(U10^2 + V10^2)
                wind_speed = np.sqrt(u10**2 + v10**2)
                
                # Convert T2M from Kelvin to Celsius: C = K - 273.15
                temp_c = t2m - 273.15
                
                records = []
                for i in range(len(lat_arr)):
                    lat = lat_arr[i]
                    for j in range(len(lon_arr)):
                        lon = lon_arr[j]
                        
                        # Validate meteorological variables
                        tc = float(temp_c[i, j])
                        r = float(rh[i, j])
                        ws = float(wind_speed[i, j])
                        pb = float(pblh[i, j])
                        
                        # Check fill limits
                        if -900.0 not in [tc, r, ws, pb]:
                            records.append((
                                date_str, 
                                round(float(lat), 2), 
                                round(float(lon), 2), 
                                round(tc, 2), 
                                round(r, 2), 
                                round(ws, 2), 
                                round(pb, 1)
                            ))
                            
                # Batch Insert
                cursor.executemany("""
                    INSERT OR REPLACE INTO weather_data (timestamp, lat, lon, temp, rh, wind_speed, pblh)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, records)
                total_records += len(records)
                
        except Exception as e:
            print(f"[ERROR] Failed to parse NetCDF file {nc_file}: {e}")
            
    conn.commit()
    conn.close()
    print(f"[SUCCESS] Ingested {total_records} reanalysis weather grids.")
    return total_records

def run_ingestion_pipeline():
    """Run all ingestion pipelines in sequence."""
    print("[INFO] Starting Data Ingestion Pipeline...")
    g_count = ingest_ground_data()
    s_count = ingest_satellite_aod()
    w_count = ingest_reanalysis_weather()
    print("[SUCCESS] Data Ingestion Pipeline Complete!")
    return g_count, s_count, w_count

if __name__ == "__main__":
    run_ingestion_pipeline()
