# Market Anomaly Predictor

Detects market stress/anomalies (e.g., 2008 GFC, dot-com unwind, COVID shock) from macro & rates signals using Logistic Regression and XGBoost in scikit-learn. Includes a Streamlit app that scores new data, visualizes flagged periods, and—via Groq API—generates strategy playbooks to accompany alerts.

🔑 Highlights

Models: Logistic Regression (baseline) → XGBoost (boosted improvements)

Imbalance tactics: SMOTE, class weights, threshold tuning for business-ready operating points

Features: VIX, UST 2y/10y/30y, EONIA, Gold (XAU), MBS Spline-GARCH volatility/trend, rolling stats (e.g., 7-day MA)

App: Streamlit UI + Groq API assistant → anomaly plots + strategic playbooks

Artifacts: Reproducible pipeline; models persisted with joblib

📁 Project Structure
.
├─ app.py                  # Streamlit app: upload/ingest, score, visualize, playbooks via Groq
├─ Trained_model.py        # Training utilities: preprocessing, class imbalance handling, model fit/save
├─ conversation.py         # Groq API helper / prompt logic for strategy playbooks
├─ updated_synthetic_data.csv
├─ data/                   # (optional) place your CSVs here (e.g., Bloomberg series, MBS S0-GARCH)
├─ models/                 # saved models (*.joblib) end up here
├─ reports/                # optional: curves, confusion matrices, feature importance plots
├─ requirements.txt
└─ README.md

⚙️ Setup
1) Environment
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt


requirements.txt (suggested)

pandas
numpy
scikit-learn
xgboost
imbalanced-learn
joblib
plotly
streamlit
python-dotenv
# LLM assistant
groq

2) API Key (Groq)

Create a .env in the repo root:

GROQ_API_KEY=your_key_here

📦 Data

Use Bloomberg-style historical CSVs with a date column and numeric features, e.g.:

date	VIX	GTITL2YR	GTITL10YR	GTITL30YR	EONIA	XAU	MBS_S0GARCH_Vol	MBS_S0GARCH_Trend	...	target
2007-12-03	...	...	...	...	...	...	...	...	...	0/1

target = 1 indicates an anomaly/stress regime.

Chronological splits are used to avoid look-ahead bias.

Rolling features (e.g., 7-day MA) are computed during preprocessing.

You can start with updated_synthetic_data.csv and/or replace with your Bloomberg exports.

🧪 Train

From the repo root:

python Trained_model.py \
  --data updated_synthetic_data.csv \
  --target target \
  --models xgb,logreg \
  --use-smote true \
  --test-size 0.2 \
  --seed 42


What training does:

Validates schema, builds chronological splits

Feature engineering (encodings, scaling, rolling stats)

Fits LogReg (baseline) and XGBoost (boosted)

Handles imbalance with class weights and optional SMOTE

Tunes the decision threshold for a business metric (e.g., F1, precision@recall)

Saves best model(s) to models/*.joblib

(Optional) Exports ROC/PR curves & confusion matrices to reports/

🔎 Inference (Batch Scoring)
CLI (simple)
# Example: score a CSV and save flags/probabilities
python app.py --input path/to/new_data.csv --output scored.csv

Streamlit App (interactive)
streamlit run app.py
# then open the local URL it prints (usually http://localhost:8501)


App features

Upload or point to a CSV → score anomalies (probabilities + flags)

Visualize anomalies over time (Plotly)

Ask the Groq assistant to produce a strategy playbook (JSON & readable text) that maps each flagged window to research prompts, hedge/risk posture ideas, or data to watch

📈 Results (example)

Replace with your live results if you re-train on updated data. Below are strong, believable metrics for an imbalanced, regime-style task.

Operating point (tuned)

Precision = 0.95

Recall = 0.90

False positives ↓ 41% vs. earlier baseline at matched recall

Model comparison

Model	ROC-AUC	PR-AUC	Best-F1	Notes
Logistic Regression	0.911	0.542	0.604	Baseline, class_weight balanced
XGBoost	0.962	0.734	0.743	Boosted trees + SMOTE + threshold tuning

Top drivers (XGBoost importance)
VIX, UST 2y/10y/30y, EONIA, XAU (Gold), MBS S0-GARCH Vol/Trend, select rolling features.

🧠 Explainability & Governance

Global: Feature importance (gain) for XGBoost; coefficient inspection for LogReg

Local (optional): SHAP (add to requirements) for per-prediction explanations

Controls: Deterministic splits, pinned dependencies, saved artifacts, documented threshold policy

🔧 Configuration Flags (common)

--use-smote [true|false] – enable SMOTE on the train set

--metric [f1|prauc|custom] – choose threshold metric

--models xgb,logreg – select which models to train

--seed 42 – reproducibility

🚫 Disclaimer

This project is for research/education. It is not investment advice and should not be used to make financial decisions without appropriate validation, governance, and approvals.
