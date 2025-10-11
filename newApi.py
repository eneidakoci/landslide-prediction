import os, math, requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import sqlite3
from typing import Optional
from pyproj import Transformer
from pathlib import Path
import hashlib
from datetime import datetime, timedelta

MODEL_PATH = "xgboost_model.pkl"
MODEL = joblib.load(MODEL_PATH)

OPENMETEO_BASE = "https://api.open-meteo.com/v1"
USGS_EQ_BASE = "https://earthquake.usgs.gov/fdsnws/event/1/query"

session = requests.Session()
session.headers.update({'User-Agent': 'LandslideRiskApp/1.0'})

app = FastAPI(title="Landslide Risk Albania API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

transformer = Transformer.from_crs("EPSG:32634", "EPSG:4326", always_xy=True)

API_CACHE = {}
CACHE_DURATION = 86400
REGIONAL_CACHE = {}


def get_cache_key(func_name, *args):
    key_str = f"{func_name}_{args}"
    return hashlib.md5(key_str.encode()).hexdigest()


def get_regional_key(lat, lon, precision=0.1):
    return f"{round(lat / precision) * precision}_{round(lon / precision) * precision}"


def get_cached_or_fetch(func_name, func, *args, cache_duration=CACHE_DURATION, use_regional=False):
    cache_key = get_cache_key(func_name, *args)

    if cache_key in API_CACHE:
        cached_data, cached_time = API_CACHE[cache_key]
        if time.time() - cached_time < cache_duration:
            print(f"  ⚡ {func_name} - using exact cache")
            return cached_data

    if use_regional and len(args) >= 2:
        regional_key = get_regional_key(args[0], args[1])
        if regional_key in REGIONAL_CACHE:
            cached_data, cached_time = REGIONAL_CACHE[regional_key]
            if time.time() - cached_time < cache_duration:
                print(f"  ⚡ {func_name} - using regional cache")
                return cached_data

    try:
        result = func(*args)
        API_CACHE[cache_key] = (result, time.time())
        if use_regional and len(args) >= 2:
            regional_key = get_regional_key(args[0], args[1])
            REGIONAL_CACHE[regional_key] = (result, time.time())
        return result
    except Exception as e:
        print(f"   {func_name} failed: {e}")
        return get_smart_fallback(func_name, *args)


def get_smart_fallback(func_name, *args):
    if "precip" in func_name:
        lat = args[0] if args else 41.0
        if lat > 42.0:
            return 1800.0
        elif lat > 41.5:
            return 1500.0
        elif lat > 40.5:
            return 1200.0
        else:
            return 1000.0

    elif "moisture" in func_name:
        return 0.25

    elif "elev" in func_name:
        lat = args[0] if args else 41.0
        if lat > 42.0:
            return 800.0
        elif lat > 41.5:
            return 400.0
        else:
            return 200.0

    elif "eq" in func_name:
        return None

    return None


def get_precipitation_yearly(lat, lon):
    try:
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=30)
        url = f"{OPENMETEO_BASE}/archive"
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "daily": "precipitation_sum",
            "timezone": "UTC"
        }
        r = session.get(url, params=params, timeout=2)
        r.raise_for_status()
        j = r.json()
        daily_precip = j.get("daily", {}).get("precipitation_sum", [])
        if daily_precip:
            monthly_total = sum(p for p in daily_precip if p is not None)
            yearly_estimate = monthly_total * 12
            print(f"   Precipitation (monthly x12): {yearly_estimate:.1f} mm")
            return float(yearly_estimate)
    except:
        pass

    fallback = 1800.0 if lat > 42.0 else 1500.0 if lat > 41.5 else 1200.0 if lat > 40.5 else 1000.0
    print(f"   Using geographic fallback: {fallback} mm")
    return fallback


def get_soil_moisture(lat, lon):
    try:
        url = f"{OPENMETEO_BASE}/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": "soil_moisture_0_to_1cm",
            "timezone": "UTC",
            "forecast_days": 1
        }
        r = session.get(url, params=params, timeout=1.5)
        r.raise_for_status()
        j = r.json()
        arr = j.get("hourly", {}).get("soil_moisture_0_to_1cm", [])
        return float(arr[-1]) if arr else 0.25
    except:
        return 0.25


def get_elevation(lat, lon):
    try:
        url = "https://api.open-meteo.com/v1/elevation"
        params = {"latitude": lat, "longitude": lon}
        r = session.get(url, params=params, timeout=1.5)
        r.raise_for_status()
        j = r.json()
        elev = j.get("elevation", [0.0])
        return float(elev[0]) if isinstance(elev, list) else float(elev)
    except:
        if lat > 42.0:
            return 800.0
        elif lat > 41.5:
            return 400.0
        else:
            return 200.0


def estimate_slope_from_elev_samples(lat, lon, meters=100):
    try:
        deglat = meters / 111320.0
        deglon = meters / (111320.0 * math.cos(math.radians(lat)) + 1e-9)
        pts = [
            (lat, lon),
            (lat + deglat, lon),
            (lat - deglat, lon),
            (lat, lon + deglon),
            (lat, lon - deglon)
        ]

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(get_cached_or_fetch, f"elev_{i}", get_elevation, la, lo, use_regional=True): i
                       for i, (la, lo) in enumerate(pts)}
            results = [None] * 5
            for future in as_completed(futures, timeout=2):
                idx = futures[future]
                try:
                    results[idx] = future.result(timeout=0.5)
                except:
                    results[idx] = 200.0
            elevations = results

        if None in elevations or not elevations:
            return 15.0

        dz_ns = elevations[1] - elevations[2]
        dz_ew = elevations[3] - elevations[4]
        slope_rad = math.atan(math.sqrt((dz_ns / (2 * meters)) ** 2 + (dz_ew / (2 * meters)) ** 2))
        return float(math.degrees(slope_rad))
    except:
        return 15.0


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def get_recent_earthquake_info(lat, lon, days=30):
    try:
        end = datetime.utcnow().date()
        start = end - timedelta(days=days)

        params = {
            "format": "geojson",
            "starttime": str(start),
            "endtime": str(end),
            "minlatitude": lat - 1.5,
            "maxlatitude": lat + 1.5,
            "minlongitude": lon - 1.5,
            "maxlongitude": lon + 1.5,
            "minmagnitude": 2.5,
            "orderby": "time-asc"
        }

        r = session.get(USGS_EQ_BASE, params=params, timeout=2)
        r.raise_for_status()
        j = r.json()
        feats = j.get("features", [])

        if not feats:
            return None

        best = None
        best_dist = 1e9
        for f in feats:
            mag = f["properties"].get("mag")
            coords = f["geometry"]["coordinates"]
            elon, elat = coords[0], coords[1]
            d = haversine_km(lat, lon, elat, elon)
            if d < best_dist:
                best_dist = d
                best = {"mag": mag, "distance_km": d}

        return best
    except:
        return None


def estimate_pga_from_mag_dist(mag, dist_km):
    if mag is None or dist_km is None:
        return 0.3
    a, b, c = -3.0, 0.8, 1.2
    log10_pga = a + b * mag - c * math.log10(dist_km + 1.0)
    return max(0.0, min(1.0, 10 ** log10_pga))


def calculate_rusle_erosion(precip_mm: float, slope_deg: float, geology_code: int, land_code: int,
                            moisture: float = 0.25):
    """
    erosion_class where:
        0 = Very High Erosion (BAD - high landslide risk)
        1 = Medium Erosion
        2 = Unknown
        3 = Low Erosion (GOOD - low landslide risk)
    """
    # R factor (Rainfall erosivity)
    if precip_mm < 400:
        R = 50.0
    elif precip_mm > 2500:
        R = 800.0
    else:
        R = 0.0483 * (precip_mm ** 1.61)
        R = min(max(R, 50), 800)

    # K factor (Soil erodibility)
    k_values = {
        1: 0.45, 2: 0.38, 3: 0.18, 4: 0.15, 5: 0.42,
        7: 0.40, 8: 0.35, 10: 0.22, 12: 0.38, 13: 0.30, 17: 0.25
    }
    K = k_values.get(geology_code, 0.30)

    # LS factor (Slope length and steepness)
    if slope_deg < 0.5:
        LS = 0.1
    else:
        if slope_deg < 9:
            S = 10.8 * math.sin(math.radians(slope_deg)) + 0.03
        else:
            S = 16.8 * math.sin(math.radians(slope_deg)) - 0.50
        m = 0.5 if slope_deg >= 5 else 0.4 if slope_deg >= 3 else 0.3
        L = (100 / 22.13) ** m
        LS = min(L * S, 20.0)

    # C factor (Cover management)
    c_values = {
        0: 0.35, 1: 0.55, 2: 0.001, 3: 0.40, 4: 0.15,
        5: 0.002, 6: 0.70, 7: 0.20, 8: 0.30, 9: 0.40
    }
    C = c_values.get(land_code, 0.40)

    # P factor (Support practice)
    if land_code == 0:
        P = 0.5 if slope_deg > 20 else 0.6 if slope_deg > 10 else 0.8
    elif land_code in [2, 5]:
        P = 0.4
    elif land_code in [4, 7, 8]:
        P = 0.7
    elif land_code == 1:
        P = 0.9
    else:
        P = 0.85

    # M factor (Moisture adjustment)
    if moisture < 0.15:
        M = 0.8
    elif moisture < 0.30:
        M = 1.0
    elif moisture < 0.40:
        M = 1.15
    else:
        M = 1.3

    # Calculate soil loss (tons/hectare/year)
    A = R * K * LS * C * P * M

    # Convert to erosion index (0-5)
    if A <= 10:
        erosion_index = (A / 10.0) * 1.0
    elif A <= 25:
        erosion_index = 1.0 + ((A - 10) / 15.0) * 1.0
    elif A <= 50:
        erosion_index = 2.0 + ((A - 25) / 25.0) * 1.0
    elif A <= 75:
        erosion_index = 3.0 + ((A - 50) / 25.0) * 1.0
    else:
        erosion_index = 4.0 + min((A - 75) / 50.0, 1.0)

    erosion_index = max(0.0, min(erosion_index, 5.0))


    if A is None or A == 0:
        erosion_class = 2
    elif erosion_index >= 3.5:
        erosion_class = 0
    elif erosion_index >= 2.0:
        erosion_class = 1
    elif erosion_index >= 0.5:
        erosion_class = 3
    else:
        erosion_class = 3

    print(f"   Erosion: A={A:.1f}, index={erosion_index:.2f}, class={erosion_class}")
    return erosion_class


class InputFeatures(BaseModel):
    Elevation: float
    Slope_inclination_degrees: float
    Geology_of_the_mass: float
    Seismicity_PGA: float
    Land_classification: float
    Erosion_rate: float
    Precipitation: float
    Moisture: float


DB_PATH = "landslides.db"

def get_db_connection():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db():
    if not Path(DB_PATH).exists():
        print(" Database not found. Please run the CSV converter script first.")
        return False

    try:
        with get_db_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM landslides")
            count = cursor.fetchone()[0]
            print(f" Database loaded: {count} landslides")
            return True
    except Exception as e:
        print(f" Database error: {e}")
        return False

db_ready = init_db()

@app.get("/")
def root():
    return {
        "status": "OK",
        "db_ready": db_ready,
        "cache_entries": len(API_CACHE),
        "regional_cache": len(REGIONAL_CACHE)
    }

@app.get("/fetch_features/")
def fetch_features(lat: float, lon: float):
    print(f"\n{'=' * 60}")
    print(f" Fetching features for: {lat:.4f}, {lon:.4f}")
    start_time = time.time()

    if not (39.5 <= lat <= 42.7 and 19.0 <= lon <= 21.5):
        raise HTTPException(status_code=400, detail="Koordinatat duhet të jenë brenda Shqipërisë")

    lat_rounded = round(lat, 2)
    lon_rounded = round(lon, 2)

    with ThreadPoolExecutor(max_workers=4) as executor:
        future_precip = executor.submit(get_cached_or_fetch, "precip", get_precipitation_yearly,
                                        lat_rounded, lon_rounded, use_regional=True)
        future_moisture = executor.submit(get_cached_or_fetch, "moisture", get_soil_moisture,
                                          lat_rounded, lon_rounded, use_regional=True)
        future_elev = executor.submit(get_cached_or_fetch, "elev", get_elevation, lat, lon, use_regional=True)
        future_eq = executor.submit(get_cached_or_fetch, "eq", get_recent_earthquake_info,
                                    lat_rounded, lon_rounded, use_regional=True)

        try:
            precipitation = future_precip.result(timeout=3)
        except:
            precipitation = get_smart_fallback("precip", lat_rounded, lon_rounded)

        try:
            moisture = future_moisture.result(timeout=2)
        except:
            moisture = 0.25

        try:
            elevation = future_elev.result(timeout=2)
        except:
            elevation = get_smart_fallback("elev", lat, lon)

        try:
            eq = future_eq.result(timeout=3)
        except:
            eq = None

    slope = estimate_slope_from_elev_samples(lat, lon)
    geology_val = 3
    land_code = 0

    pga = estimate_pga_from_mag_dist(
        eq.get("mag") if eq else None,
        eq.get("distance_km") if eq else None
    )

    erosion = calculate_rusle_erosion(
        precip_mm=precipitation,
        slope_deg=slope,
        geology_code=geology_val,
        land_code=land_code,
        moisture=moisture
    )

    elapsed = time.time() - start_time
    print(f" TOTAL: {elapsed:.2f}s | Cache: {len(API_CACHE)} | Regional: {len(REGIONAL_CACHE)}")
    print(f"{'=' * 60}\n")

    return {
        "lat": lat,
        "lon": lon,
        "Elevation": elevation,
        "Slope_inclination_degrees": slope,
        "Geology_of_the_mass": geology_val,
        "Seismicity_PGA": pga,
        "Land_classification": land_code,
        "Erosion_rate": erosion,
        "Precipitation": precipitation,
        "Moisture": moisture,
        "nearest_eq": eq,
        "fetch_time_seconds": round(elapsed, 2)
    }


@app.post("/predict/")
def predict_risk(data: InputFeatures):
    try:
        arr = np.array([[
            data.Elevation,
            data.Slope_inclination_degrees,
            data.Geology_of_the_mass,
            data.Seismicity_PGA,
            data.Land_classification,
            data.Erosion_rate,
            data.Precipitation,
            data.Moisture
        ]])

        prob = MODEL.predict_proba(arr)[0][1] if hasattr(MODEL, "predict_proba") else MODEL.predict(arr)[0]
        prob_pct = round(float(prob) * 100, 2)

        risk = (
            "Shumë e Ulët" if prob_pct < 20 else
            "E Ulët" if prob_pct < 40 else
            "Mesatare" if prob_pct < 60 else
            "E Lartë" if prob_pct < 80 else
            "Shumë e Lartë"
        )

        return {
            "probability": prob_pct,
            "risk_level": risk
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/landslides/")
def get_landslides(year: Optional[int] = None):
    if not db_ready:
        raise HTTPException(status_code=500, detail="Database not initialized")

    try:
        with get_db_connection() as conn:
            conn.row_factory = sqlite3.Row

            if year:
                query = """
                SELECT * FROM landslides 
                WHERE (
                    CAST(substr("Data e ndodhjes ", -4) AS INTEGER) = ?
                    OR strftime('%Y', "Data e ndodhjes ") = ?
                )
                AND Latitude IS NOT NULL 
                AND Longitude IS NOT NULL
                """
                cursor = conn.execute(query, (year, str(year)))
            else:
                query = """
                SELECT * FROM landslides 
                WHERE Latitude IS NOT NULL 
                AND Longitude IS NOT NULL
                """
                cursor = conn.execute(query)

            rows = cursor.fetchall()

            landslide_list = []
            for row in rows:
                item = dict(row)
                landslide_list.append(item)

            return {
                "year": year,
                "count": len(landslide_list),
                "landslides": landslide_list
            }

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8503)