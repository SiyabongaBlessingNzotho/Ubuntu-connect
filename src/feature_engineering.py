import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

df = pd.read_csv(DATA_DIR / "student_data_combined.csv")
print(f"Loaded {len(df)} students")

# --- 1. Time-series features from G1 -> G2 (both known BEFORE the final grade) ---
# We deliberately do NOT create a G2->G3 feature. G3 is the final grade
# used to define the label, so any feature derived from it would leak the
# answer into the model and defeat the whole point of an "early warning" system.
df['g1_to_g2_change'] = df['G2'] - df['G1']
df['avg_g1_g2'] = (df['G1'] + df['G2']) / 2

# --- 2. Encode categorical columns (separate encoder per column, kept for reference) ---
categoricals = ['sex', 'address', 'famsize', 'Pstatus', 'schoolsup', 'famsup']
encoders = {}
for col in categoricals:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le  # keep if you need df[col] = le.inverse_transform(...) later

# --- 3. Target: At Risk if final grade < 10 ---
df['label_at_risk'] = (df['G3'] < 10).astype(int)
print(f"At-Risk students: {df['label_at_risk'].sum()} out of {len(df)}")

# --- 4. Final feature set for Machine Learning (no G3-derived features) ---
feature_columns = [
    'student_id',
    'age', 'studytime', 'failures', 'absences',
    'g1_to_g2_change', 'avg_g1_g2',
    'sex', 'address', 'famsize', 'Pstatus', 'schoolsup', 'famsup',
    'label_at_risk'
]

df_features = df[feature_columns]
df_features.to_csv(DATA_DIR / "student_features.csv", index=False)

print(f"Features saved! Shape: {df_features.shape}")
print(f"Feature columns: {df_features.columns.tolist()}")