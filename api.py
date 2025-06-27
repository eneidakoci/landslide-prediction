#backend code
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
from pydantic import BaseModel

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
