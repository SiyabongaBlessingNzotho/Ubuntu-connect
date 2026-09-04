
# Ubuntu-Connect: AI-Powered Student At-Risk Detection

An early-warning system that scores every student's academic risk and explains *why*, so staff can intervene before a student fails — not after.

## The Problem

Academic staff typically discover a struggling student only after formal results confirm failure, usually at the end of a term, by which point little time remains for meaningful intervention. Attendance registers, continuous assessment marks, and assignment records are already collected by most institutions, but are rarely analysed together in real time — so early warning signs (declining attendance, falling scores, missed submissions) go unnoticed until academic damage is already done.

Ubuntu-Connect applies predictive machine learning to convert data institutions already hold into a proactive, explainable early-warning system — the same approach already common in banking (credit risk scoring) and healthcare (patient risk scoring), applied to education.

## Business Objectives

1. Identify students at academic risk earlier than the current end-of-term review process allows.
2. Give academic staff a clear, explainable reason behind every risk flag — not just a raw score.
3. Support a measurable, longer-term reduction in avoidable student failure and dropout through earlier intervention.

## Success Criteria

- The risk model achieves at least 70% precision on the "High Risk" class.
- 100% of Medium/High risk flags are accompanied by at least one machine-generated reason.
- A measurable reduction in the fail rate of students who received intervention after being flagged.

## How It Works

```
[Weekly CSV Data] → [Feature Engineering: G1/G2 time-series] → [ML/DL Models: Random Forest + Keras] → [Risk API: FastAPI] → [Dashboard + Chatbot]
```

- **Time-series features**: the dataset doesn't contain weekly observations, so the three sequential grading periods (G1, G2, G3) are treated as discrete time steps. Features like `g1_to_g2_change` and `avg_g1_g2` capture temporal trends in student performance, fulfilling the time-based feature extraction requirement.
- **At-risk threshold**: a student is classified "At Risk" if their final grade (G3) is below 10/20 — the passing grade in the Portuguese education system that the dataset is sourced from. This creates a binary pass/fail classification problem predicted from earlier academic and behavioural indicators.
- **Class imbalance**: handled with SMOTE (`imbalanced-learn`) to improve recall on the At-Risk class.
- **Explainability**: every Medium/High risk flag is paired with a machine-generated reason (e.g. "High absences (12 days), grades declining between periods").

## Tech Stack

Python 3 · Pandas & NumPy · Scikit-learn · TensorFlow/Keras · TextBlob (sentiment analysis on teacher notes) · imbalanced-learn/SMOTE · FastAPI

## Project Structure

```
Ubuntu-connect/
├── data/                     # Raw dataset (student-mat.csv, student-por.csv)
├── models/                   # Saved trained models (output of main.py)
├── docs/                     # Report, poster outline, project docs
├── src/
│   ├── load_data.py          # Load and merge raw CSVs
│   ├── feature_engineering.py# Time-series & derived features (G1→G2→G3)
│   ├── train_random_forest.py
│   ├── train_with_smote.py   # Random Forest + SMOTE for class imbalance
│   ├── train_deep_learning.py# Keras neural network
│   ├── sentiment_analysis.py # TextBlob sentiment on teacher notes
│   └── api_server.py         # FastAPI server — serves predictions from a trained model
├── main.py                   # Entry point: runs the full training pipeline
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone https://github.com/SiyabongaBlessingNzotho/Ubuntu-connect.git
cd Ubuntu-connect
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

**1. Train the models** (loads data → engineers features → trains → evaluates → saves to `models/`):

```bash
python main.py
```

**2. Serve predictions via API** (after a model exists in `models/`):

```bash
uvicorn src.api_server:app --reload
```

## Data Source

Cortez, P., & Silva, A. (2008). *Using Data Mining to Predict Secondary School Student Performance.* UCI Machine Learning Repository — Student Performance Dataset. 33 attributes covering demographics, family background, study habits, and grades (G1, G2, G3) for 395 students in Mathematics and 395 in Portuguese.

## Constraints & Risks

- No real consented student data is used — the dataset is sourced from Kaggle/UCI.
- 9-week development timeframe, no budget for paid services (all open-source).
- Model bias mitigated by never using protected attributes as features.
- Automation bias mitigated by always presenting supporting reasons alongside every risk score.

## Results

_To be added once `train_random_forest.py`, `train_with_smote.py`, and `train_deep_learning.py` are implemented and evaluated._

| Model | Precision | Recall | F1 |
|---|---|---|---|
| Random Forest | — | — | — |
| Random Forest + SMOTE | — | — | — |
| Deep Learning (Keras) | — | — | — |

## Team

9-person team project. Team Leader :Siyabonga Nzotho 

1-Person =
2-Person
3-Person
4-Person
5-Person
6-Person
7-Person
8-Person
9-Person
