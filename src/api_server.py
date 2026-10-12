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

# ===========================================================================
# Person 8 - /chat endpoint (rule-based intervention tips for staff)
# Add this to the END of src/api_server.py. Uses Person 7's model, FEATURES,
# risk_level(), MEDIUM_MIN, HIGH_MIN, etc. -- those must already be defined
# above this block by Person 7's /predict code. Do not paste this before
# Person 7's code, and do not remove anything above it.
# ===========================================================================

# Defensive re-imports: harmless if Person 7's code already imported these
# under the same names. Having them here means this block still works even
# if Person 7's import style changes slightly.
from typing import List, Optional, Dict
from pydantic import BaseModel
from fastapi import HTTPException
import pandas as pd

# Which direction makes each feature a RISK factor ("above" = higher than
# average is worse). Only features listed here produce tips. Age and other
# personal attributes are deliberately NOT listed, so the chatbot never
# gives advice based on them (separate from the open question of whether
# age should be a model *input* at all -- raise that with the team lead).
RISK_DIRECTION = {
    "absences": "above",
    "failures": "above",
    "avg_g1_g2": "below",
    "g1_to_g2_change": "below",
    "studytime": "below",
}

FACTOR_TIPS = {
    "absences": "Meet the student to find out why they are missing classes and agree a simple "
                "attendance plan. Check in weekly and acknowledge improvement.",
    "failures": "Review the earlier failed subject(s) with the student, find the topics that caused "
                "the problem, and offer catch-up tutoring or a peer study partner.",
    "avg_g1_g2": "Arrange extra support for the weaker topics (tutoring, past papers, office hours) "
                 "and set one small target for the next assessment.",
    "g1_to_g2_change": "Grades are falling between periods: ask what changed and review the most "
                        "recent assessments to see where marks dropped.",
    "studytime": "Help the student build a weekly study timetable and consider a supervised "
                 "study session.",
}

NEXT_STEP = {
    "Medium": "Suggested next step: check in with the student within two weeks "
              "and watch the next assessment.",
    "High": "Suggested next step: contact the student this week, and record the "
            "intervention so its effect can be measured.",
}

# Topics the chatbot can answer from the staff member's message alone,
# even with no student features supplied.
TOPIC_TIPS = {
    "attendance": (
        ["attendance", "absent", "absence", "skipping", "truant"],
        ["Talk to the student privately about why they miss classes (transport, work, family, motivation).",
         "Agree a small attendance target and review it weekly."],
    ),
    "grades": (
        ["grade", "grades", "marks", "scores", "failing", "falling behind"],
        ["Identify the topics where marks dropped and offer tutoring or office-hour time for them.",
         "Set one small, specific target for the next assessment."],
    ),
    "previous failures": (
        ["failure", "failures", "repeat", "failed"],
        ["Review the failed subject(s) with the student and find the topics that caused the problem.",
         "Offer catch-up tutoring or a peer study partner."],
    ),
    "study habits": (
        ["study", "studying", "homework", "revision", "timetable"],
        ["Help the student plan a weekly study timetable.",
         "Offer a supervised study session or a study group."],
    ),
}

HELP_TEXT = ("I can give intervention tips on attendance, grades, previous failures and study habits, "
             "or explain the risk levels. Include the student's features to get tips for that student.")

DISCLAIMER = ("These are suggestions for staff. Confirm with the student before acting: the risk score "
              "is a prompt to talk, not a verdict.")


class ChatRequest(BaseModel):
    message: str
    student_id: Optional[str] = None
    features: Optional[Dict[str, float]] = None


class ChatResponse(BaseModel):
    reply: str
    risk_level: Optional[str] = None
    risk_probability: Optional[float] = None
    tips: List[str] = []


def top_risk_factors(row: pd.DataFrame, n: int = 3) -> List[str]:
    """Feature names (from RISK_DIRECTION) where this student is clearly on the risky side."""
    if TRAIN_MEAN is None:
        return []
    scored = []
    for name in FEATURES:
        direction = RISK_DIRECTION.get(name)
        if direction is None:
            continue
        z = (float(row[name].iloc[0]) - TRAIN_MEAN[name]) / TRAIN_STD[name]
        risky = z >= 0.5 if direction == "above" else z <= -0.5
        if risky:
            scored.append((IMPORTANCE[name] * abs(z), name))
    scored.sort(reverse=True)
    return [name for _, name in scored[:n]]


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    msg = req.message.lower().strip()
    if not msg:
        raise HTTPException(status_code=422, detail="message must not be empty")

    lines: List[str] = []
    tips: List[str] = []
    level = None
    prob = None

    if req.features is not None:
        missing = [c for c in FEATURES if c not in req.features]
        unknown = [c for c in req.features if c not in FEATURES]
        if missing or unknown:
            raise HTTPException(
                status_code=422,
                detail={"missing_features": missing, "unknown_features": unknown},
            )
        row = pd.DataFrame([[req.features[c] for c in FEATURES]], columns=FEATURES)
        prob = round(float(model.predict_proba(row)[0][AT_RISK_INDEX]), 4)
        level = risk_level(prob)

        who = f"Student {req.student_id}" if req.student_id else "This student"
        lines.append(f"{who}: {level} risk (probability {prob:.2f}).")

        if level == "Low":
            lines.append("No intervention is needed now; keep monitoring.")
        else:
            factors = top_risk_factors(row)
            for name in factors:
                tips.append(FACTOR_TIPS[name])
            if not factors:
                lines.append("No single measured factor stands out, so start with a general "
                              "check-in conversation with the student.")
            lines.append(NEXT_STEP[level])

    if "risk level" in msg:
        lines.append(f"Risk levels: Low is below {MEDIUM_MIN}, Medium is {MEDIUM_MIN} up to but not "
                      f"including {HIGH_MIN}, and High is {HIGH_MIN} and above.")

    for topic, (keywords, topic_tips) in TOPIC_TIPS.items():
        if any(k in msg for k in keywords):
            for tip in topic_tips:
                if tip not in tips:
                    tips.append(tip)

    if not lines and not tips:
        lines.append(HELP_TEXT)

    parts = list(lines)
    if tips:
        parts.append("Tips:\n" + "\n".join(f"{i}. {t}" for i, t in enumerate(tips, 1)))
    parts.append(DISCLAIMER)
    reply = "\n".join(parts)

    return ChatResponse(reply=reply, risk_level=level, risk_probability=prob, tips=tips)
