# Ubuntu-Connect: AI-Powered Student At-Risk Detection

An early-warning system that scores every student's academic risk and explains why, so staff can intervene before a student fails, not after.

## The Problem

Academic staff typically discover a struggling student only after formal results confirm failure, usually at the end of a term, by which point little time remains for meaningful intervention. Attendance registers, continuous assessment marks, and assignment records are already collected by most institutions, but are rarely analysed together in real time, so early warning signs (declining attendance, falling scores, missed submissions) go unnoticed until academic damage is already done.

Ubuntu-Connect applies predictive machine learning to convert data institutions already hold into a proactive, explainable early-warning system: the same approach already common in banking (credit risk scoring) and healthcare (patient risk scoring), applied to education.

## Business Objectives

- Identify students at academic risk earlier than the current end-of-term review process allows.
- Give academic staff a clear, explainable reason behind every risk flag, not just a raw score.
- Support a measurable, longer-term reduction in avoidable student failure and dropout through earlier intervention.

## Success Criteria

- The risk model achieves at least 70% precision on the "High Risk" class.
- 100% of Medium/High risk flags are accompanied by at least one machine-generated reason.
- A measurable reduction in the fail rate of students who received intervention after being flagged.

## How It Works

```
[Weekly CSV Data] -> [Feature Engineering: G1/G2 time-series] -> [ML/DL Models: Random Forest + Keras] -> [Risk API: FastAPI] -> [Dashboard + Chatbot]
```

- **Time-series features:** the dataset doesn't contain weekly observations, so the three sequential grading periods (G1, G2, G3) are treated as discrete time steps. Features like `g1_to_g2_change` and `avg_g1_g2` capture temporal trends in student performance, fulfilling the time-based feature extraction requirement.
- **At-risk threshold:** a student is classified "At Risk" if their final grade (G3) is below 10/20, the passing grade in the Portuguese education system that the dataset is sourced from. This creates a binary pass/fail classification problem predicted from earlier academic and behavioural indicators.
- **Class imbalance:** the Random Forest baseline uses `class_weight='balanced'`; SMOTE (imbalanced-learn) is compared against it to improve recall on the At-Risk class.
- **Text features (NLP):** sentiment scores from teacher notes (TextBlob) are added as extra features. See [NLP Sentiment Analysis](#nlp-sentiment-analysis-teacher-notes).
- **Explainability:** every Medium/High risk flag is paired with a machine-generated reason (e.g. "High absences (12 days), grades declining between periods").

## Tech Stack

Python 3 · Pandas & NumPy · Scikit-learn · TensorFlow/Keras · TextBlob (sentiment analysis on teacher notes) · imbalanced-learn/SMOTE · FastAPI

## Project Structure

```
Ubuntu-connect/
├── data/                        # Raw and generated datasets
│   ├── student-mat.csv          # Raw: Mathematics
│   ├── student-por.csv          # Raw: Portuguese
│   ├── student_data_combined.csv
│   ├── student_features.csv     # Engineered features + label_at_risk
│   ├── teacher_notes.csv        # Synthetic teacher notes (NLP)
│   └── student_features_nlp.csv # Features + sentiment columns
├── models/                      # Saved trained models
├── docs/                        # Report, poster outline, project docs
├── src/
│   ├── load_data.py             # Load and merge raw CSVs
│   ├── feature_engineering.py   # Time-series & derived features (G1 -> G2 -> G3)
│   ├── generate_teacher_notes.py# Synthetic teacher notes (no access to the label)
│   ├── sentiment_analysis.py    # TextBlob sentiment on teacher notes
│   ├── train_random_forest.py   # Random Forest baseline
│   ├── train_with_smote.py      # Random Forest + SMOTE for class imbalance
│   ├── train_deep_learning.py   # Keras neural network
│   └── api_server.py            # FastAPI server: serves predictions from a trained model
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

There is no single entry point. Run the pipeline steps in order from the repo root (on Windows use `python src\...`):

1. **Prepare the data**

   ```bash
   python src/load_data.py
   python src/feature_engineering.py        # -> data/student_features.csv
   ```

2. **Add the NLP features**

   ```bash
   python src/generate_teacher_notes.py     # -> data/teacher_notes.csv
   python src/sentiment_analysis.py         # -> data/student_features_nlp.csv
   ```

3. **Train and evaluate the models** (each saves its model to `models/`)

   ```bash
   python src/train_random_forest.py        # -> models/rf_baseline.pkl
   python src/train_with_smote.py
   python src/train_deep_learning.py        # -> models/deep_model.keras
   ```

4. **Serve predictions via API** (after a model exists in `models/`)

   ```bash
   uvicorn src.api_server:app --reload
   ```

## NLP Sentiment Analysis (Teacher Notes)

The UCI student performance dataset contains no free-text field, so teacher comments were synthesised from each student's absences, prior failures and grade trajectory (G1/G2) in order to demonstrate the NLP component. The generation process does not use G3 or `label_at_risk`. In a live deployment these comments would be drawn from real learning-management-system entries.

- `generate_teacher_notes.py` writes `data/teacher_notes.csv`.
- `sentiment_analysis.py` scores each note with TextBlob and writes `data/student_features_nlp.csv`.
- New columns: `sentiment_polarity`, `sentiment_subjectivity`, `sentiment_negative_flag`.
- Requires `pip install textblob`.

Because the notes are built from features already in `student_features.csv`, any recall gain from the NLP features should be read with that in mind. The final comparison trains the model on both `student_features.csv` and `student_features_nlp.csv`.

## Deep Learning Model (Keras)

`src/train_deep_learning.py` trains a neural network on `data/student_features.csv` to predict `label_at_risk`, for comparison with the Random Forest.

- **Split:** 80/20 train/test, stratified, `random_state=42` (same split as the Random Forest baseline)
- **Preprocessing:** `StandardScaler` fitted on the training set only
- **Architecture:** Dense(32, ReLU) -> Dropout(0.3) -> Dense(16, ReLU) -> Dropout(0.2) -> Dense(1, sigmoid)
- **Training:** Adam, binary cross-entropy, 50 epochs, batch size 16, 20% validation split
- **Reproducibility:** NumPy and TensorFlow seeds set to 42
- **Output:** `models/deep_model.keras`
- **Requires:** `tensorflow`

## Data Source

Cortez, P., & Silva, A. (2008). *Using Data Mining to Predict Secondary School Student Performance.* UCI Machine Learning Repository, Student Performance Dataset. 33 attributes covering demographics, family background, study habits, and grades (G1, G2, G3) for 395 students in Mathematics and 649 in Portuguese (1,044 records combined).

## Constraints & Risks

- No real consented student data is used; the dataset is sourced from Kaggle/UCI.
- Teacher notes are synthetic (see the NLP section above).
- 9-week development timeframe, no budget for paid services (all open-source).
- Model bias mitigated by never using protected attributes as features.
- Automation bias mitigated by always presenting supporting reasons alongside every risk score.

## Results

All figures are for the **At-Risk** class on the 209-student test set (80/20 stratified split, `random_state=42`).

| Model | Precision | Recall | F1 |
|---|---|---|---|
| Random Forest (baseline, `class_weight='balanced'`) | 0.64 | 0.89 | 0.75 |
| Random Forest + SMOTE | pending | pending | pending |
| Deep Learning (Keras) | 0.78 | 0.76 | 0.77 |

**Random Forest baseline:** 100 trees, trained on 835 students. Overall accuracy 0.87. Confusion matrix on the test set:

| | Predicted Not At-Risk | Predicted At-Risk |
|---|---|---|
| **Actual Not At-Risk** | 140 | 23 |
| **Actual At-Risk** | 5 | 41 |

The model catches 41 of the 46 at-risk students and misses 5. It also raises 23 false alarms, which is why precision on the At-Risk class (0.64) is lower than recall (0.89). For an early-warning system this trade-off is generally acceptable, since missing an at-risk student costs more than a false alarm. Precision on the At-Risk class is below the 70% target in the success criteria.

**Deep Learning (Keras):** trained on the same 835 students and evaluated on the same 209-student test set. Overall accuracy 0.90. Precision on the At-Risk class is 0.78, which meets the 70% target, but recall is 0.76, lower than the Random Forest's 0.89.

**Comparison:** the Random Forest catches more at-risk students, while the Keras model raises fewer false alarms. Because missing an at-risk student costs more than a false alarm, the team treats recall as the priority metric, which favours the Random Forest so far. The two models are not fully like-for-like: the Random Forest uses `class_weight='balanced'` while the Keras model uses no class weighting, and the test set contains only 46 at-risk students, so small differences between the models should be treated with caution.

## Team

9-person team project. Team Leader: Spha (load data)

<!-- TODO: complete the team table with each person's role. -->
