import pandas as pd
from pathlib import Path

# Resolve paths relative to this file, not the current working directory,
# so this script works whether you run it from the repo root or from src/.
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Load both datasets (note: they use semicolon separators)
math = pd.read_csv(DATA_DIR / "student-mat.csv", sep=';')
por = pd.read_csv(DATA_DIR / "student-por.csv", sep=';')

# Add a unique student ID to each row
math['student_id'] = range(1, len(math) + 1)
por['student_id'] = range(1000, 1000 + len(por))

# Combine them
combined = pd.concat([math, por], ignore_index=True)

# Save the combined file
combined.to_csv(DATA_DIR / "student_data_combined.csv", index=False)

print("Combined dataset saved!")
print(f"Total students: {len(combined)} (Math: {len(math)}, Portuguese: {len(por)})")
print(f"Columns: {combined.columns.tolist()}")