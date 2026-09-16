import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from tensorflow import keras
import tensorflow as tf
# Seed everything so results (and your RF-vs-DL comparison) are reproducible
np.random.seed(42)
tf.random.set_seed(42)
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
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(f"Training set: {len(X_train)} students")
print(f"Test set: {len(X_test)} students")
# Modern Keras pattern: explicit Input layer instead of input_shape= on Dense
model = keras.Sequential([
keras.Input(shape=(X_train.shape[1],)),
keras.layers.Dense(32, activation='relu'),
keras.layers.Dropout(0.3),
keras.layers.Dense(16, activation='relu'),
keras.layers.Dropout(0.2),
keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(
    X_train_scaled, y_train,
    epochs=50, batch_size=16,
    validation_split=0.2, verbose=0
)
y_pred = (model.predict(X_test_scaled) > 0.5).astype(int)
print("\n=== DEEP LEARNING (KERAS) RESULTS ===")
print(classification_report(y_test, y_pred, target_names=["Not At-Risk", "At-Risk"]))
model.save(MODELS_DIR / "deep_model.keras")
print("\nModel saved to models/deep_model.keras")
print("Compare this to Person 3's Random Forest results for your report.")
