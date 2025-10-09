#backend code
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
from pydantic import BaseModel
from typing import Literal, Optional, Dict, Any
import math
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("xgboost_model.pkl")

# Match field names in the training CSV
class InputFeatures(BaseModel):
    Elevation: float
    Slope_inclination_degrees: float
    Geology_of_the_mass: int
    Seismicity_PGA: float
    Land_classification: int
    Erosion_rate: float
    Precipitation: float
    Moisture: float

@app.post("/predict/")
def predict_risk(data: InputFeatures):
    input_array = np.array([[data.Elevation,
                             data.Slope_inclination_degrees,
                             data.Geology_of_the_mass,
                             data.Seismicity_PGA,
                             data.Land_classification,
                             data.Erosion_rate,
                             data.Precipitation,
                             data.Moisture]])

    probability = model.predict_proba(input_array)[0][1]
    prob_rounded = round(float(probability) * 100, 2)

    if prob_rounded < 20:
        risk = "Very Low"
    elif prob_rounded < 40:
        risk = "Low"
    elif prob_rounded < 60:
        risk = "Moderate"
    elif prob_rounded < 80:
        risk = "High"
    else:
        risk = "Very High"

    return {
        "probability": prob_rounded,
        "risk_level": risk
    }


# --- Feature aggregation (lightweight stub) ---

class LatLng(BaseModel):
    lat: float
    lng: float

class AggregatedFeatures(BaseModel):
    Elevation: float
    Slope_inclination_degrees: float
    Geology_of_the_mass: int
    Seismicity_PGA: float
    Land_classification: int
    Erosion_rate: float
    Precipitation: float
    Moisture: int

class FeatureSources(BaseModel):
    geology: Literal["real", "synthetic"]
    erosion: Literal["real", "synthetic"]

def _bounded(val: float, low: float, high: float) -> float:
    return max(low, min(high, val))

"""
External vendor-specific queries removed to keep the backend fully open-source.
Geology and erosion are synthesized deterministically from lat/lng until
open data sources are integrated.
"""


def aggregate_features(lat: float, lng: float) -> tuple[AggregatedFeatures, FeatureSources, dict[str, Any]]:
    # NOTE: Stubbed feature generator to introduce realistic variation without external deps
    # Elevation: vary 50-1500m over Albania bounds
    elevation = _bounded(800 + 600 * math.sin(lat) + 300 * math.cos(lng), 50, 1500)

    # Slope: 0-45 degrees based on lat/lng mix
    slope = _bounded(abs(30 * math.sin(lat * 2) + 20 * math.cos(lng * 2)), 0, 45)

    # Geology: synthetic mapping (no external dependency)
    geology: int
    geology_mapping = {
        "Deluvium": 1,
        "Breccia & Deluvium": 2,
        "Carbonate sedimentary": 3,
        "Carbonates": 4,
        "Claystone": 5,
        "Deluvium & Eluvium": 7,
        "Sandstone & Claystone & Deluvium & Eluvium": 8,
        "Breccia": 10,
        "Shale & Siltstone": 12,
        "Unknown": 13,
        "Sandstone": 17,
    }
    geology_source: Literal["real", "synthetic"] = "synthetic"
    geology_options = [1, 2, 3, 4, 5, 7, 8, 10, 12, 13, 17]
    geology_idx = int(abs(lat * 10 + lng * 7)) % len(geology_options)
    geology = geology_options[geology_idx]

    # Seismicity PGA: 0.05 - 0.35
    pga = _bounded(0.2 + 0.15 * math.sin(lat + lng), 0.05, 0.35)

    # Land classification
    land_options = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    land = land_options[int(abs(lat * 5 - lng * 3)) % len(land_options)]

    # Erosion rate: synthetic 0.0-5.0 (no external dependency)
    erosion_source: Literal["real", "synthetic"] = "synthetic"
    erosion = _bounded(1.0 + 2.0 * abs(math.sin(lat) * math.cos(lng)), 0.0, 5.0)

    # Precipitation: 600 - 2000 mm
    precipitation = _bounded(1200 + 500 * math.sin(lng) - 200 * math.cos(lat * 1.5), 600, 2000)

    # Moisture categorical 1/2/3
    moisture = 1 if math.sin(lat + lng) < -0.2 else (3 if math.sin(lat + lng) > 0.2 else 2)

    feats = AggregatedFeatures(
        Elevation=float(round(elevation, 2)),
        Slope_inclination_degrees=float(round(slope, 2)),
        Geology_of_the_mass=int(geology),
        Seismicity_PGA=float(round(pga, 3)),
        Land_classification=int(land),
        Erosion_rate=float(round(erosion, 2)),
        Precipitation=float(round(precipitation, 1)),
        Moisture=int(moisture),
    )
    sources = FeatureSources(geology=geology_source, erosion=erosion_source)
    extras: dict[str, Any] = {}
    return feats, sources, extras


@app.get("/features")
def get_features(lat: float, lng: float):
    try:
        feats, sources, extras = aggregate_features(lat, lng)
        return {"features": feats.model_dump(), "sources": sources.model_dump(), **extras}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feature aggregation failed: {e}")


@app.get("/predict_at")
def predict_at(lat: float, lng: float):
    try:
        feats, sources, extras = aggregate_features(lat, lng)
        input_array = np.array([[
            feats.Elevation,
            feats.Slope_inclination_degrees,
            feats.Geology_of_the_mass,
            feats.Seismicity_PGA,
            feats.Land_classification,
            feats.Erosion_rate,
            feats.Precipitation,
            feats.Moisture
        ]])
        probability = model.predict_proba(input_array)[0][1]
        prob_rounded = round(float(probability) * 100, 2)

        if prob_rounded < 20:
            risk = "Very Low"
        elif prob_rounded < 40:
            risk = "Low"
        elif prob_rounded < 60:
            risk = "Moderate"
        elif prob_rounded < 80:
            risk = "High"
        else:
            risk = "Very High"

        return {
            "probability": prob_rounded,
            "risk_level": risk,
            "features": feats.model_dump(),
            "sources": sources.model_dump(),
            **extras,
            "lat": lat,
            "lng": lng,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
