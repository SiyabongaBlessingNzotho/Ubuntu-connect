import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import joblib

np.random.seed(42)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def load_xy(csv_path):
    df = pd.read_csv(csv_path)
    X = df.drop(columns=["student_id", "label_at_risk"])
    y = df["label_at_risk"]
    return X, y


def run_variant(csv_path, label):
    X, y = load_xy(csv_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # SMOTE fit on TRAINING data only. Never touch X_test/y_test.
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"\n=== {label} ===")
    print(f"Training set (after SMOTE): {len(X_train)} students")
    print(f"Test set: {len(X_test)} students")
    print(classification_report(y_test, y_pred, target_names=["Not At-Risk", "At-Risk"]))

    return model


if __name__ == "__main__":
    model_a = run_variant(DATA_DIR / "student_features.csv", "Random Forest + SMOTE (no NLP)")
    joblib.dump(model_a, MODELS_DIR / "rf_smote_no_nlp.joblib")

    model_b = run_variant(DATA_DIR / "student_features_nlp.csv", "Random Forest + SMOTE (with NLP)")
    joblib.dump(model_b, MODELS_DIR / "rf_smote_with_nlp.joblib")

    print("\nModels saved to models/rf_smote_no_nlp.joblib and models/rf_smote_with_nlp.joblib")
    print("Compare At-Risk recall across all four: RF baseline, Keras, RF+SMOTE (no NLP), RF+SMOTE (with NLP).")
