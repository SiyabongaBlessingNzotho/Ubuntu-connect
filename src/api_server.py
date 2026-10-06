"""Risk API - Person 7 (/predict endpoint).

Run from the project root with the virtual environment active:
    python -m uvicorn src.api_server:app --reload
Then open http://127.0.0.1:8000/docs
"""
from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# SETTINGS - the model file and its matching data file must be a pair.
#   no-NLP model: rf_smote_no_nlp.joblib   + student_features.csv
#   NLP model:    rf_smote_with_nlp.joblib + student_features_nlp.csv
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "rf_smote_no_nlp.joblib"
DATA_PATH = BASE_DIR / "data" / "student_features.csv"
AT_RISK_CLASS = 1    # value of label_at_risk that means "At-Risk"
MEDIUM_MIN = 0.33    # probability below this -> Low
HIGH_MIN = 0.66      # probability at/above this -> High; in between -> Medium

# ---------------------------------------------------------------------------
# Load the model once, when the server starts
# ---------------------------------------------------------------------------
model = joblib.load(MODEL_PATH)
if not hasattr(model, "feature_names_in_"):
    raise RuntimeError(
        "This model file has no feature names (it was not trained on a pandas "
        "DataFrame, or it is a pipeline/dict). Ask the model owner which file to use."
    )
FEATURES = [str(c) for c in model.feature_names_in_]
CLASSES = list(model.classes_)
if AT_RISK_CLASS not in CLASSES:
    raise RuntimeError(
        f"Model classes are {CLASSES}; expected {AT_RISK_CLASS} to mean At-Risk."
    )
AT_RISK_INDEX = CLASSES.index(AT_RISK_CLASS)

# Training-data averages, used only to explain a Medium/High result
try:
    _train = pd.read_csv(DATA_PATH)[FEATURES]
    TRAIN_MEAN = _train.mean()
    TRAIN_STD = _train.std().replace(0, 1).fillna(1)
except Exception as exc:  # the API still works without reasons
    print(f"WARNING: could not load {DATA_PATH} for reasons: {exc}")
    TRAIN_MEAN = None
    TRAIN_STD = None

IMPORTANCE = dict(zip(FEATURES, model.feature_importances_))

# Friendlier names for known columns; any other column is shown by its own name
NICE_NAMES = {
    "absences": "absences",
    "failures": "prior failures",
    "g1_to_g2_change": "change in grade from G1 to G2",
    "avg_g1_g2": "average of G1 and G2 grades",
    "sentiment_polarity": "teacher-note sentiment",
}

app = FastAPI(title="Ubuntu Connect Risk API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    student_id: str
    features: Dict[str, float]


class PredictResponse(BaseModel):
    student_id: str
    at_risk: bool
    risk_probability: float
    risk_level: str
    reason: Optional[str] = None


def risk_level(prob: float) -> str:
    if prob >= HIGH_MIN:
        return "High"
    if prob >= MEDIUM_MIN:
        return "Medium"
    return "Low"


def make_reason(row: pd.DataFrame) -> str:
    """Name the features that most separate this student from the training average."""
    if TRAIN_MEAN is None:
        return "Overall pattern resembles students who struggled."
    scored = []
    for name in FEATURES:
        value = float(row[name].iloc[0])
        z = (value - TRAIN_MEAN[name]) / TRAIN_STD[name]
        if abs(z) >= 0.5:
            scored.append((IMPORTANCE[name] * abs(z), name, value, z))
    scored.sort(reverse=True)
    if not scored:
        return "No single feature stands out; overall pattern resembles students who struggled."
    parts = []
    for _, name, value, z in scored[:3]:
        direction = "above" if z > 0 else "below"
        parts.append(f"{NICE_NAMES.get(name, name)} = {value:.4g} ({direction} average)")
    return "Main factors: " + "; ".join(parts)


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_PATH.name, "n_features": len(FEATURES)}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    missing = [c for c in FEATURES if c not in req.features]
    unknown = [c for c in req.features if c not in FEATURES]
    if missing or unknown:
        raise HTTPException(
            status_code=422,
            detail={"missing_features": missing, "unknown_features": unknown},
        )
    row = pd.DataFrame([[req.features[c] for c in FEATURES]], columns=FEATURES)
    prob = float(model.predict_proba(row)[0][AT_RISK_INDEX])
    level = risk_level(prob)
    return PredictResponse(
        student_id=req.student_id,
        at_risk=level != "Low",
        risk_probability=round(prob, 4),
        risk_level=level,
        reason=make_reason(row) if level != "Low" else None,
    )
