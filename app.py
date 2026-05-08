# FastAPI application for clinical trial failure prediction

from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import numpy as np
import uvicorn

# load saved artifacts at startup
model = joblib.load("model_gb.pkl")
feature_cols = joblib.load("feature_cols.pkl")
le_intervention = joblib.load("le_intervention.pkl")
le_domain = joblib.load("le_domain.pkl")

app = FastAPI(
    title="Clinical Trial Failure Predictor",
    description=(
        "Predicts the probability that a clinical trial will be terminated "
        "before completion, given trial characteristics. Built on ClinicalTrials.gov "
        "data from 2005-2025."
    ),
    version="1.0.0"
)


class TrialInput(BaseModel):
    phase_rank: float = Field(
        ..., ge=0, le=4,
        description="Phase numeric rank: Phase1=1, Phase2=2, Phase3=3, Phase4=4"
    )
    log_enrollment: float = Field(
        ..., ge=0,
        description="log(1 + enrollment_count). Use numpy.log1p(enrollment)."
    )
    duration_months: float = Field(
        ..., gt=0,
        description="Expected or actual trial duration in months."
    )
    sponsor_size: int = Field(
        ..., ge=1,
        description="Number of trials the sponsor has run historically."
    )
    sponsor_hist_fail_rate: float = Field(
        ..., ge=0.0, le=1.0,
        description="Sponsor's historical failure rate as a fraction (0.0 to 1.0)."
    )
    funder_rank: int = Field(
        ..., ge=0, le=3,
        description="Funder type rank: INDUSTRY=3, NIH/FED=2, OTHER=1, UNKNOWN=0"
    )
    masking_rank: int = Field(
        ..., ge=0, le=4,
        description="Masking type rank: NONE=0, SINGLE=1, DOUBLE=2, TRIPLE=3, QUADRUPLE=4"
    )
    is_randomized: int = Field(
        ..., ge=0, le=1,
        description="1 if trial is randomized, 0 otherwise."
    )
    intervention_type: str = Field(
        ...,
        description="Intervention type string: DRUG, BIOLOGICAL, DEVICE, PROCEDURE, BEHAVIORAL, OTHER"
    )
    medical_domain: str = Field(
        ...,
        description="Medical domain: Oncology, Infectious, Cardiovascular, Neurology, Immunology, etc."
    )
    years_since_2005: int = Field(
        ..., ge=0,
        description="Trial start year minus 2005. E.g., for a 2023 trial: 2023-2005=18"
    )


class PredictionOutput(BaseModel):
    failure_probability: float
    risk_level: str
    recommendation: str


@app.get("/")
def root():
    return {
        "message": "Clinical Trial Failure Predictor API is running.",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health")
def health():
    return {"status": "ok", "model": "GradientBoostingClassifier"}


@app.post("/predict", response_model=PredictionOutput)
def predict_failure(trial: TrialInput):
    # encode categorical fields using saved label encoders
    try:
        intervention_code = le_intervention.transform([trial.intervention_type])[0]
    except ValueError:
        intervention_code = 0  # default for unseen categories

    try:
        domain_code = le_domain.transform([trial.medical_domain])[0]
    except ValueError:
        domain_code = 0

    # assemble feature vector in the same order as training
    features = np.array([[
        trial.phase_rank,
        trial.log_enrollment,
        trial.duration_months,
        trial.sponsor_size,
        trial.sponsor_hist_fail_rate,
        trial.funder_rank,
        trial.masking_rank,
        trial.is_randomized,
        intervention_code,
        domain_code,
        trial.years_since_2005
    ]])

    prob = model.predict_proba(features)[0][1]

    if prob < 0.20:
        risk = "LOW"
        rec = "Trial profile is similar to historically successful trials. Standard monitoring recommended."
    elif prob < 0.40:
        risk = "MODERATE"
        rec = "Some risk factors detected. Consider reviewing protocol design and enrollment projections."
    else:
        risk = "HIGH"
        rec = "High failure risk detected. Recommend detailed protocol review and contingency planning."

    return PredictionOutput(
        failure_probability=round(float(prob), 4),
        risk_level=risk,
        recommendation=rec
    )


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)