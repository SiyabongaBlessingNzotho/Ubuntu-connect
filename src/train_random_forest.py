
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_DIR / "student_features.csv")

X = df.drop(columns=['student_id', 'label_at_risk'])
y = df['label_at_risk']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training set: {len(X_train)} students")
print(f"Test set: {len(X_test)} students")

model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("\n=== RANDOM FOREST BASELINE RESULTS ===")
print(classification_report(y_test, y_pred, target_names=["Not At-Risk", "At-Risk"]))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

joblib.dump(model, MODELS_DIR / "rf_baseline.pkl")
joblib.dump(X.columns.tolist(), MODELS_DIR / "feature_columns.pkl")
print("\nModel saved to models/rf_baseline.pkl")