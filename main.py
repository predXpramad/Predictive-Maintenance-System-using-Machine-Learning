"""
main.py - FastAPI Backend for Predictive Maintenance
======================================================
Exposes two endpoints:
  GET  /          → Health check
  POST /predict   → Accepts sensor readings, returns failure prediction
""" 

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
import joblib
import numpy as np
import uvicorn
from pathlib import Path

# ─────────────────────────────────────────
# Load model artefact at startup
# ─────────────────────────────────────────
MODEL_PATH = Path("model.pkl")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "model.pkl not found. Run `python train.py` first."
    )

artifact = joblib.load(MODEL_PATH)
MODEL         = artifact["model"]
FEATURE_NAMES = artifact["feature_names"]
MODEL_NAME    = artifact["model_name"]

print(f"✅ Loaded model: {MODEL_NAME}")
print(f"   Expects features: {FEATURE_NAMES}")

# ─────────────────────────────────────────
# Pydantic input schema
# ─────────────────────────────────────────
class SensorInput(BaseModel):
    """
    Input features for a single machine reading.
    Feature order must match the training pipeline exactly.
    """
    Type: int = Field(
        ...,
        ge=0, le=2,
        description="Machine type encoded: L=0, M=1, H=2"
    )
    air_temperature_K: float = Field(
        ..., alias="Air temperature [K]",
        ge=290, le=310,
        description="Air temperature in Kelvin"
    )
    process_temperature_K: float = Field(
        ..., alias="Process temperature [K]",
        ge=300, le=320,
        description="Process temperature in Kelvin"
    )
    rotational_speed_rpm: int = Field(
        ..., alias="Rotational speed [rpm]",
        ge=1000, le=3000,
        description="Rotational speed in RPM"
    )
    torque_Nm: float = Field(
        ..., alias="Torque [Nm]",
        ge=0, le=100,
        description="Torque in Newton-meters"
    )
    tool_wear_min: int = Field(
        ..., alias="Tool wear [min]",
        ge=0, le=300,
        description="Tool wear in minutes"
    )

    class Config:
        populate_by_name = True   # allow both alias and field name


class PredictionResponse(BaseModel):
    prediction: int           # 0 = No Failure, 1 = Failure
    label: str                # human-readable label
    probability_failure: float  # P(failure)
    model_used: str


# ─────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────
app = FastAPI(
    title="Predictive Maintenance API",
    description=(
        "Binary classifier that predicts machine failure "
        "from real-time sensor readings."
    ),
    version="1.0.0",
)


@app.get("/", tags=["Health"])
def health_check():
    """Simple health-check endpoint."""
    return {
        "status": "ok",
        "model":  MODEL_NAME,
        "features": FEATURE_NAMES,
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(data: SensorInput):
    """
    Predict machine failure from sensor input.

    Returns:
    - **prediction**: 0 (No Failure) or 1 (Failure)
    - **label**: Human-readable result
    - **probability_failure**: Confidence score for failure class
    - **model_used**: Name of the model that made the prediction
    """
    # Build feature array in the exact order used during training
    # Note: column names are sanitised (brackets replaced) to match training
    feature_map = {
        "Type":                       data.Type,
        "Air temperature (K)":        data.air_temperature_K,
        "Process temperature (K)":    data.process_temperature_K,
        "Rotational speed (rpm)":     data.rotational_speed_rpm,
        "Torque (Nm)":                data.torque_Nm,
        "Tool wear (min)":            data.tool_wear_min,
    }

    try:
        features = np.array([[feature_map[f] for f in FEATURE_NAMES]])
    except KeyError as e:
        raise HTTPException(
            status_code=422,
            detail=f"Missing expected feature: {e}"
        )

    prediction  = int(MODEL.predict(features)[0])
    proba       = float(MODEL.predict_proba(features)[0][1])
    label       = "⚠️ Machine Failure Detected" if prediction == 1 else "✅ No Failure"

    return PredictionResponse(
        prediction=prediction,
        label=label,
        probability_failure=round(proba, 4),
        model_used=MODEL_NAME,
    )


# ─────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
