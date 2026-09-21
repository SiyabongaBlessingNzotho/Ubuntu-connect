import random
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

random.seed(42)

POSITIVE = [
    "Consistently well prepared and contributes confidently in class.",
    "Hands work in on time and asks thoughtful questions.",
    "A pleasure to teach, shows real interest in the subject.",
    "Strong effort this term, clearly enjoys the work.",
    "Excellent attendance and a positive attitude throughout.",
]

NEUTRAL = [
    "Attends regularly and completes the required work.",
    "Quiet in class but keeps up with the material.",
    "Performance is steady, no particular concerns raised.",
    "Does what is asked, could participate more in discussion.",
    "Average progress so far this term.",
]

NEGATIVE = [
    "Frequently absent and falling behind on assignments.",
    "Seems disengaged and rarely completes homework.",
    "Struggling badly with the basics, needs urgent support.",
    "Poor attendance is hurting an already weak performance.",
    "Distracted in class and repeatedly misses deadlines.",
]

df = pd.read_csv(DATA_DIR / "student_features.csv")
print(f"Loaded {len(df)} students")

def pick_note(row):
    concerns = 0
    if row["absences"] >= 10:      concerns += 1
    if row["failures"] >= 1:       concerns += 1
    if row["avg_g1_g2"] < 10:      concerns += 1
    if row["g1_to_g2_change"] < 0: concerns += 1
    if concerns >= 2:   pool = NEGATIVE
    elif concerns == 1: pool = NEUTRAL
    else: pool = POSITIVE if row["avg_g1_g2"] >= 12 else NEUTRAL
    return random.choice(pool)

notes = pd.DataFrame({
    "student_id":   df["student_id"],
    "teacher_note": df.apply(pick_note, axis=1),
})

notes.to_csv(DATA_DIR / "teacher_notes.csv", index=False)
print(f"Saved {len(notes)} notes to data/teacher_notes.csv")
print(notes.head())
