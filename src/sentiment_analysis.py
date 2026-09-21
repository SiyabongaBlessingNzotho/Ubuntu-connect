import pandas as pd
from pathlib import Path
from textblob import TextBlob

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

features = pd.read_csv(DATA_DIR / "student_features.csv")
notes    = pd.read_csv(DATA_DIR / "teacher_notes.csv")
print(f"Loaded {len(features)} students and {len(notes)} teacher notes")

def score(text):
    blob = TextBlob(str(text))
    return pd.Series({
        "sentiment_polarity":     blob.sentiment.polarity,
        "sentiment_subjectivity": blob.sentiment.subjectivity,
    })

notes = notes.join(notes["teacher_note"].apply(score))

def bucket(p):
    if p > 0.1:  return "Positive"
    if p < -0.1: return "Negative"
    return "Neutral"

notes["sentiment_label"]         = notes["sentiment_polarity"].apply(bucket)
notes["sentiment_negative_flag"] = (notes["sentiment_label"] == "Negative").astype(int)

print("\n=== SENTIMENT DISTRIBUTION ===")
print(notes["sentiment_label"].value_counts())

nlp_cols = ["student_id", "sentiment_polarity",
            "sentiment_subjectivity", "sentiment_negative_flag"]
df = features.merge(notes[nlp_cols], on="student_id", how="left")
cols = ["sentiment_polarity", "sentiment_subjectivity", "sentiment_negative_flag"]
df[cols] = df[cols].fillna(0)

print("\n=== DOES SENTIMENT RELATE TO RISK? ===")
print(df.groupby("label_at_risk")["sentiment_polarity"].mean().rename(
    index={0: "Not At-Risk", 1: "At-Risk"}))
print("\nSentiment label vs actual outcome:")
print(pd.crosstab(notes["sentiment_label"], df["label_at_risk"]))

df.to_csv(DATA_DIR / "student_features_nlp.csv", index=False)
print(f"\nSaved! Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
