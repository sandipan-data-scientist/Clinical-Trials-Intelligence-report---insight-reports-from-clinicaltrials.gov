"""
FastAPI inference endpoint.
Run:  uvicorn api.app:app --host 0.0.0.0 --port 8000
Docker: see api/Dockerfile
"""

from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import numpy as np
import os

BASE = os.path.join(os.path.dirname(__file__), "..", "artifacts")

model          = joblib.load(os.path.join(BASE, "model_gb.pkl"))
le_intervention = joblib.load(os.path.join(BASE, "le_intervention.pkl"))
le_domain       = joblib.load(os.path.join(BASE, "le_domain.pkl"))
feature_cols    = joblib.load(os.path.join(BASE, "feature_cols.pkl"))

app = FastAPI(
    title="Clinical Trial Failure Predictor",
    description="Predicts trial termination probability using a GradientBoostingClassifier.",
    version="1.0.0"
)


class TrialInput(BaseModel):
    phase_rank:              float = Field(..., ge=0, le=4)
    log_enrollment:          float = Field(..., ge=0)
    duration_months:         float = Field(..., gt=0)
    sponsor_size:            int   = Field(..., ge=1)
    sponsor_hist_fail_rate:  float = Field(..., ge=0.0, le=1.0)
    funder_rank:             int   = Field(..., ge=0, le=3)
    masking_rank:            int   = Field(..., ge=0, le=4)
    is_randomized:           int   = Field(..., ge=0, le=1)
    intervention_type:       str
    medical_domain:          str
    years_since_2005:        int   = Field(..., ge=0)


class PredictionOutput(BaseModel):
    failure_probability: float
    risk_level:          str
    recommendation:      str


def _safe_encode(encoder, value):
    try:
        return encoder.transform([value])[0]
    except ValueError:
        return 0


@app.get("/")
def root():
    return {"message": "Clinical Trial Failure Predictor API", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok", "model": "GradientBoostingClassifier"}


@app.post("/predict", response_model=PredictionOutput)
def predict(trial: TrialInput):
    features = np.array([[
        trial.phase_rank,
        trial.log_enrollment,
        trial.duration_months,
        trial.sponsor_size,
        trial.sponsor_hist_fail_rate,
        trial.funder_rank,
        trial.masking_rank,
        trial.is_randomized,
        _safe_encode(le_intervention, trial.intervention_type),
        _safe_encode(le_domain, trial.medical_domain),
        trial.years_since_2005,
    ]])
    prob = float(model.predict_proba(features)[0][1])

    if prob < 0.20:
        risk = "LOW"
        rec  = "Trial profile matches historically successful trials. Standard monitoring."
    elif prob < 0.40:
        risk = "MODERATE"
        rec  = "Some risk factors present. Review protocol design and enrollment targets."
    else:
        risk = "HIGH"
        rec  = "High termination risk. Recommend protocol review and contingency planning."

    return PredictionOutput(failure_probability=round(prob, 4), risk_level=risk, recommendation=rec)