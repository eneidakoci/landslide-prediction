import os, math, requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

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


# TODO: take yearly precipitation instead
def get_precipitation_24h(lat, lon):
    """Merr reshjet e 24 orëve të fundit"""
    try:
        url = f"{OPENMETEO_BASE}/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": "precipitation_sum",
            "timezone": "UTC",
            "forecast_days": 1
        }
        r = session.get(url, params=params, timeout=5)
        r.raise_for_status()
        j = r.json()
        val = j.get("daily", {}).get("precipitation_sum", [0.0])
        return float(val[0]) if val else 0.0
    except Exception as e:
        print(f"Precipitation error: {e}")
        return 0.0


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
        r = session.get(url, params=params, timeout=5)
        r.raise_for_status()
        j = r.json()
        arr = j.get("hourly", {}).get("soil_moisture_0_to_1cm", [])
        return float(arr[-1]) if arr else 0.25
    except Exception as e:
        print(f"Soil moisture error: {e}")
        return 0.25  # default


def get_elevation(lat, lon):
    try:
        url = "https://api.open-meteo.com/v1/elevation"
        params = {"latitude": lat, "longitude": lon}
        r = session.get(url, params=params, timeout=3)
        r.raise_for_status()
        j = r.json()
        elev = j.get("elevation", [0.0])
        return float(elev[0]) if isinstance(elev, list) else float(elev)
    except Exception as e:
        print(f"Elevation error: {e}")
        return 0.0


def estimate_slope_from_elev_samples(lat, lon, meters=100):
    try:
        deglat = meters / 111320.0
        deglon = meters / (111320.0 * math.cos(math.radians(lat)) + 1e-9)

        # Merr 5 pika rreth lokacionit
        pts = [
            (lat, lon),
            (lat + deglat, lon),
            (lat - deglat, lon),
            (lat, lon + deglon),
            (lat, lon - deglon)
        ]

        # Use ThreadPoolExecutor to retrieve elevation data at the same time/in parallel
        elevations = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(get_elevation, la, lo): i for i, (la, lo) in enumerate(pts)}
            results = [None] * 5
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result(timeout=2)
                except:
                    results[idx] = 0.0
            elevations = results

        if None in elevations or not elevations:
            return 15.0  # default slope in Albania (kodra)

        dz_ns = elevations[1] - elevations[2]
        dz_ew = elevations[3] - elevations[4]
        slope_rad = math.atan(math.sqrt((dz_ns / (2 * meters)) ** 2 + (dz_ew / (2 * meters)) ** 2))
        return float(math.degrees(slope_rad))
    except Exception as e:
        print(f"Slope error: {e}")
        return 15.0


def haversine_km(lat1, lon1, lat2, lon2):
    """Kalkulon distancën në km"""
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def get_recent_earthquake_info(lat, lon, days=30):
    try:
        import datetime
        end = datetime.datetime.utcnow().date()
        start = end - datetime.timedelta(days=days)

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

        r = session.get(USGS_EQ_BASE, params=params, timeout=5)
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
    except Exception as e:
        print(f"Earthquake error: {e}")
        return None


def estimate_pga_from_mag_dist(mag, dist_km):
    """Kalkulon PGA nga magnituda dhe distanca"""
    if mag is None or dist_km is None:
        return 0.3  # default in Albania (zone sizmike)
    a, b, c = -3.0, 0.8, 1.2
    log10_pga = a + b * mag - c * math.log10(dist_km + 1.0)
    return max(0.0, min(1.0, 10 ** log10_pga))


def estimate_erosion(precip_mm, slope_deg):
    """Vlereson shkallen e erozionit"""
    score = 0.0
    score += precip_mm / 100.0 if precip_mm else 0.0
    score += slope_deg / 10.0 if slope_deg else 0.0
    return min(score, 10.0)

class InputFeatures(BaseModel):
    Elevation: float
    Slope_inclination_degrees: float
    Geology_of_the_mass: float
    Seismicity_PGA: float
    Land_classification: float
    Erosion_rate: float
    Precipitation: float
    Moisture: float

# Endpoints
@app.get("/")
def root():
    return {"status": "Landslide API is running", "endpoints": ["/fetch_features/", "/predict/"]}

@app.get("/fetch_features/")
def fetch_features(lat: float, lon: float):
    print(f"Fetching features for: {lat}, {lon}")
    start_time = time.time()

    # Validate coordinates
    if not (39.5 <= lat <= 42.7 and 19.0 <= lon <= 21.5):
        raise HTTPException(status_code=400, detail="Koordinatat duhet të jenë brenda Shqipërisë")

    results = {}

    # Take all the data at the same time with threadpoolexecutor
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_precip = executor.submit(get_precipitation_24h, lat, lon)
        future_moisture = executor.submit(get_soil_moisture, lat, lon)
        future_elev = executor.submit(get_elevation, lat, lon)
        future_eq = executor.submit(get_recent_earthquake_info, lat, lon)

        # get results
        try:
            precipitation = future_precip.result(timeout=6)
            moisture = future_moisture.result(timeout=6)
            elevation = future_elev.result(timeout=4)
            eq = future_eq.result(timeout=6)
        except Exception as e:
            print(f"Thread error: {e}")
            precipitation = 0.0
            moisture = 0.25
            elevation = 0.0
            eq = None

    #slope estimation
    slope = estimate_slope_from_elev_samples(lat, lon)

    # Placeholders per geology dhe land classification
    #TODO: REPLACE GEOLOGY OF THE MASS AND LAND CLASSIFICATION - to be retrieved from SHGJSH
    geology_val = 3  # Carbonate sedimentary (tipik)
    land_code = 0  # Agricultural land

    # Kalkulo PGA
    pga = estimate_pga_from_mag_dist(
        eq.get("mag") if eq else None,
        eq.get("distance_km") if eq else None
    )

    # Kalkulo eroizon
    erosion = estimate_erosion(precipitation, slope)

    elapsed = time.time() - start_time
    print(f"Features fetched in {elapsed:.2f}s")

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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8502)