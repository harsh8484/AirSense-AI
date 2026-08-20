import os
import datetime
import numpy as np
import pandas as pd
import h5py
from netCDF4 import Dataset
from database import init_db

# Constants for spatial domain (covering India)
MIN_LAT, MAX_LAT = 5.0, 38.0
MIN_LON, MAX_LON = 68.0, 98.0
GRID_RES = 1.0  # 1 degree grid for simple simulation

LATS = np.arange(MIN_LAT, MAX_LAT + GRID_RES, GRID_RES)
LONS = np.arange(MIN_LON, MAX_LON + GRID_RES, GRID_RES)
N_LAT = len(LATS)
N_LON = len(LONS)

STATIONS_METADATA = [
    {"id": "DL001", "name": "Delhi_RK_Puram", "lat": 28.56, "lon": 77.18, "state": "Delhi", "city": "Delhi", "pm_base": 180.0, "pm_var": 50.0},
    {"id": "MH001", "name": "Mumbai_Bandra", "lat": 19.05, "lon": 72.84, "state": "Maharashtra", "city": "Mumbai", "pm_base": 65.0, "pm_var": 20.0},
    {"id": "WB001", "name": "Kolkata_Victoria", "lat": 22.54, "lon": 88.34, "state": "West Bengal", "city": "Kolkata", "pm_base": 90.0, "pm_var": 30.0},
    {"id": "TN001", "name": "Chennai_Alandur", "lat": 13.00, "lon": 80.20, "state": "Tamil Nadu", "city": "Chennai", "pm_base": 45.0, "pm_var": 15.0},
    {"id": "KA001", "name": "Bengaluru_Silk_Board", "lat": 12.92, "lon": 77.62, "state": "Karnataka", "city": "Bengaluru", "pm_base": 35.0, "pm_var": 10.0},
    {"id": "TG001", "name": "Hyderabad_Sanathnagar", "lat": 17.45, "lon": 78.43, "state": "Telangana", "city": "Hyderabad", "pm_base": 55.0, "pm_var": 18.0},
    {"id": "BR001", "name": "Patna_Sanjay_Gandhi", "lat": 25.60, "lon": 85.10, "state": "Bihar", "city": "Patna", "pm_base": 140.0, "pm_var": 40.0},
    {"id": "UP001", "name": "Lucknow_Lalbagh", "lat": 26.85, "lon": 80.94, "state": "Uttar Pradesh", "city": "Lucknow", "pm_base": 150.0, "pm_var": 45.0},
    {"id": "RJ001", "name": "Jaipur_Adarsh_Nagar", "lat": 26.90, "lon": 75.82, "state": "Rajasthan", "city": "Jaipur", "pm_base": 85.0, "pm_var": 25.0},
    {"id": "GJ001", "name": "Ahmedabad_Maninagar", "lat": 23.00, "lon": 72.61, "state": "Gujarat", "city": "Ahmedabad", "pm_base": 95.0, "pm_var": 30.0}
]

def make_dirs():
    """Create data directories."""
    paths = [
        "data/raw/satellite",
        "data/raw/ground",
        "data/raw/reanalysis",
        "data/raw/documents",
        "database"
    ]
    for p in paths:
        os.makedirs(p, exist_ok=True)
    print("[INFO] Directories created successfully.")

def generate_rag_documents():
    """Generates text reference documents in data/raw/documents/"""
    docs = {
        "cpcb_guidelines.txt": (
            "Central Pollution Control Board (CPCB) - National Ambient Air Quality Standards (NAAQS) for India.\n"
            "The standard 24-hour limit for PM2.5 is 60 ug/m3, and the annual limit is 40 ug/m3.\n"
            "For PM10, the 24-hour limit is 100 ug/m3 and the annual limit is 60 ug/m3.\n"
            "AQI Categories in India:\n"
            "- Good (0-50): Minimal impact.\n"
            "- Satisfactory (51-100): Minor breathing discomfort to sensitive people.\n"
            "- Moderately Polluted (101-200): Breathing discomfort to people with lungs, asthma, and heart diseases.\n"
            "- Poor (201-300): Breathing discomfort to most people on prolonged exposure.\n"
            "- Very Poor (301-400): Respiratory illness on prolonged exposure.\n"
            "- Severe (401-500): Affects healthy people and seriously impacts those with existing diseases."
        ),
        "who_air_quality_guidelines.txt": (
            "World Health Organization (WHO) Global Air Quality Guidelines (2021 update).\n"
            "WHO recommends an annual mean limit of 5 ug/m3 for PM2.5 (down from 10 ug/m3 in previous guidelines).\n"
            "The 24-hour exposure limit should not exceed 15 ug/m3 for PM2.5.\n"
            "For PM10, WHO recommends an annual limit of 15 ug/m3 and a 24-hour limit of 45 ug/m3.\n"
            "WHO estimates that exposure to ambient air pollution causes millions of premature deaths globally each year, "
            "primarily from stroke, heart disease, lung cancer, and acute respiratory infections."
        ),
        "pm25_research_paper.txt": (
            "Research Review: Mapping Surface PM2.5 Concentrations Using Geostationary Satellite AOD.\n"
            "Aerosol Optical Depth (AOD) measures the total column integration of aerosol light extinction. "
            "Surface PM2.5 estimation from AOD is governed by the physical relationship:\n"
            "PM2.5 = AOD * (1 / PBLH) * f(RH) * C\n"
            "Where PBLH represents the planetary boundary layer height. A lower PBLH compresses particulates near the surface, "
            "increasing PM2.5 for a given column AOD. Relative humidity (RH) causes water absorption (hygroscopic growth) "
            "in aerosols, changing their scattering efficiency. Surface wind speed affects dispersion, carrying particulates away "
            "under high winds or stagnating pollution under low wind speeds. Random Forest, XGBoost, and deep learning architectures "
            "like LSTMs have proven highly effective in modeling these complex non-linear relations."
        ),
        "policy_options.txt": (
            "Policy Interventions for Air Quality Management in India.\n"
            "Key strategies to control PM2.5 pollution:\n"
            "1. Stubble Burning Mitigation: Promotion of Happy Seeder machines, bio-decomposers (like Pusa Decomposer), "
            "and biomass pellet fuel plants to reuse crop residues.\n"
            "2. Electric Vehicles (EV): Rapid scaling of charging infrastructure, public bus transit electrification, and subsidies.\n"
            "3. Stricter Industrial Standards: Implementation of Flue Gas Desulfurization (FGD) in coal plants and adoption of Zig-Zag technology in brick kilns.\n"
            "4. Dust Management: Mechanical sweepers, water sprinklers, and green belt planting around highways."
        )
    }
    
    for filename, content in docs.items():
        filepath = os.path.join("data", "raw", "documents", filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
    print("[INFO] Generated RAG text documents.")

def generate_ground_cpcb(days=30):
    """Generates ground CSV readings for CPCB stations for the past N days."""
    dates = [datetime.date.today() - datetime.timedelta(days=i) for i in range(days)]
    dates.reverse()
    
    records = []
    for d in dates:
        date_str = d.strftime("%Y-%m-%d")
        for s in STATIONS_METADATA:
            # Add some weather-dependent noise to simulate pollution fluctuations
            # E.g. assume weekends or specific seasons are slightly different
            noise = np.random.normal(0, 10.0)
            
            # Simple simulation: let's couple it with latitude (higher in North, i.e., Delhi)
            # Add random walk factor
            pm25 = max(5.0, s["pm_base"] + noise + np.random.uniform(-15.0, 15.0))
            pm10 = pm25 * np.random.uniform(1.4, 2.0)  # PM10 is always higher
            
            records.append({
                "date": date_str,
                "station_id": s["id"],
                "station_name": s["name"],
                "latitude": s["lat"],
                "longitude": s["lon"],
                "state": s["state"],
                "city": s["city"],
                "PM2.5": round(pm25, 2),
                "PM10": round(pm10, 2)
            })
            
    df = pd.DataFrame(records)
    csv_path = os.path.join("data", "raw", "ground", "cpcb_ground_data.csv")
    df.to_csv(csv_path, index=False)
    print(f"[INFO] Generated CPCB Ground Station CSV with {len(df)} records at {csv_path}")

def generate_satellite_aod(days=30):
    """Generates mock HDF5 satellite AOD files for the past N days."""
    dates = [datetime.date.today() - datetime.timedelta(days=i) for i in range(days)]
    
    # 2D Grid coordinates
    lat_grid, lon_grid = np.meshgrid(LATS, LONS, indexing='ij')
    
    for d in dates:
        date_str = d.strftime("%Y%m%d")
        # Format matching INSAT-3D naming: 3DIMG_DDMMMYYYY_HHmm_L2B_AOD.h5
        # Example: 3DIMG_11JUL2026_0600_L2B_AOD.h5
        month_abbr = d.strftime("%b").upper()
        day_str = d.strftime("%d")
        year_str = d.strftime("%Y")
        h5_name = f"3DIMG_{day_str}{month_abbr}{year_str}_0600_L2B_AOD.h5"
        h5_path = os.path.join("data", "raw", "satellite", h5_name)
        
        # Simulate AOD grid
        # AOD is high in North (Indo-Gangetic Plain) near Lat 25-30, Lon 75-85
        # AOD increases near major sources
        aod = np.zeros((N_LAT, N_LON))
        for i in range(N_LAT):
            for j in range(N_LON):
                lat = LATS[i]
                lon = LONS[j]
                
                # Distance to Indo-Gangetic Plain center (Delhi area)
                dist_igp = np.sqrt((lat - 27.0)**2 + (lon - 80.0)**2)
                base_aod = 0.6 * np.exp(-dist_igp / 12.0) + 0.15
                
                # Dynamic daily variation + random noise
                daily_fluct = 0.1 * np.sin(d.day / 5.0)
                noise = np.random.normal(0, 0.05)
                val = base_aod + daily_fluct + noise
                aod[i, j] = max(0.05, min(1.5, val))
                
        # Simulate cloud mask: set ~15% of cells to NaN (simulating clouds)
        cloud_mask = np.random.choice([False, True], size=(N_LAT, N_LON), p=[0.85, 0.15])
        aod[cloud_mask] = np.nan
        
        with h5py.File(h5_path, 'w') as f:
            # Create group structure matching INSAT-3D
            geo_grp = f.create_group("Geolocation_Information")
            geo_grp.create_dataset("Latitude", data=lat_grid)
            geo_grp.create_dataset("Longitude", data=lon_grid)
            
            geophys_grp = f.create_group("Geophysical_Parameters")
            # In HDF5, NaN can be stored as float nan, or we can use fill value -999.0
            # Let's replace NaN with -999.0 for standard satellite representation
            aod_clean = np.where(np.isnan(aod), -999.0, aod)
            ds = geophys_grp.create_dataset("AOD", data=aod_clean)
            ds.attrs["_FillValue"] = -999.0
            ds.attrs["scale_factor"] = 1.0
            
    print(f"[INFO] Generated {days} INSAT-3D Satellite AOD H5 files.")

def generate_reanalysis_met(days=30):
    """Generates mock MERRA-2 meteorological NetCDF files for the past N days."""
    dates = [datetime.date.today() - datetime.timedelta(days=i) for i in range(days)]
    
    for d in dates:
        date_str = d.strftime("%Y%m%d")
        # Format: MERRA2_100.inst1_2d_asm_Nx.YYYYMMDD.nc
        nc_name = f"MERRA2_400.tavg1_2d_slv_Nx.{date_str}.nc"
        nc_path = os.path.join("data", "raw", "reanalysis", nc_name)
        
        # Create dataset
        with Dataset(nc_path, 'w', format='NETCDF4') as rootgrp:
            # Create dimensions
            rootgrp.createDimension('time', 1)
            rootgrp.createDimension('lat', N_LAT)
            rootgrp.createDimension('lon', N_LON)
            
            # Create variables
            time_var = rootgrp.createVariable('time', 'f4', ('time',))
            lat_var = rootgrp.createVariable('lat', 'f4', ('lat',))
            lon_var = rootgrp.createVariable('lon', 'f4', ('lon',))
            
            # Create meteorological variables
            # 2m Temperature (T2M) - Kelvin
            t2m_var = rootgrp.createVariable('T2M', 'f4', ('time', 'lat', 'lon'), fill_value=-999.0)
            # Relative Humidity (RH) - Percentage
            rh_var = rootgrp.createVariable('RH', 'f4', ('time', 'lat', 'lon'), fill_value=-999.0)
            # Wind speed components U10 and V10
            u10_var = rootgrp.createVariable('U10', 'f4', ('time', 'lat', 'lon'), fill_value=-999.0)
            v10_var = rootgrp.createVariable('V10', 'f4', ('time', 'lat', 'lon'), fill_value=-999.0)
            # Planetary Boundary Layer Height (PBLH) - meters
            pblh_var = rootgrp.createVariable('PBLH', 'f4', ('time', 'lat', 'lon'), fill_value=-999.0)
            
            # Write coordinate data
            time_var[:] = [0.0]
            lat_var[:] = LATS
            lon_var[:] = LONS
            
            # Simulate meteorological variables
            # Temperature is higher in South (lower lat), lower in North/Himalayas (high lat)
            # Relative humidity is higher in coastal South (low lat, near oceans)
            # PBLH is higher in dry regions, lower in morning/winter/coastal
            temp_map = np.zeros((1, N_LAT, N_LON))
            rh_map = np.zeros((1, N_LAT, N_LON))
            u_map = np.zeros((1, N_LAT, N_LON))
            v_map = np.zeros((1, N_LAT, N_LON))
            pbl_map = np.zeros((1, N_LAT, N_LON))
            
            for i in range(N_LAT):
                lat = LATS[i]
                for j in range(N_LON):
                    lon = LONS[j]
                    
                    # Temp: base is 303 K (30C) in south, dropping to 285 K (12C) in extreme north
                    temp_map[0, i, j] = 305.0 - 0.45 * (lat - 5.0) + np.random.normal(0, 1.5)
                    
                    # RH: high near coast (Lat 5-15, Lon 70-80), dry in Rajasthan (Lat 26, Lon 72)
                    dist_to_rajasthan = np.sqrt((lat - 26.0)**2 + (lon - 72.0)**2)
                    rh_map[0, i, j] = max(10.0, min(95.0, 75.0 - 1.2 * lat + 1.5 * dist_to_rajasthan + np.random.normal(0, 4.0)))
                    
                    # Wind: general trade winds
                    u_map[0, i, j] = np.random.uniform(-4.0, 4.0)
                    v_map[0, i, j] = np.random.uniform(-3.0, 3.0)
                    
                    # PBLH: high in dry areas (Rajasthan) up to 2000m, low in coastal/mountains (300m)
                    pbl_map[0, i, j] = max(200.0, 1000.0 + 30.0 * dist_to_rajasthan - 10.0 * lat + np.random.normal(0, 100.0))
                    
            t2m_var[:] = temp_map
            rh_var[:] = rh_map
            u10_var[:] = u_map
            v10_var[:] = v_map
            pblh_var[:] = pbl_map
            
    print(f"[INFO] Generated {days} MERRA-2 NetCDF weather files.")

def run_all(days=30):
    """Initializes everything and generates full synthetic dataset."""
    print("======================================================")
    print("        AirSense AI Synthetic Data Generator          ")
    print("======================================================")
    
    # 1. Initialize folders
    make_dirs()
    
    # 2. Initialize Database
    init_db()
    
    # 3. Generate documents
    generate_rag_documents()
    
    # 4. Generate datasets
    generate_ground_cpcb(days)
    generate_satellite_aod(days)
    generate_reanalysis_met(days)
    
    print("\n[SUCCESS] Mock data generated successfully! Ready for ingestion.")
    print("======================================================")

if __name__ == "__main__":
    run_all()
