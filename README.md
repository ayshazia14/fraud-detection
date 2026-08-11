# Credit Card Fraud Detection

A comparative study of fraud detection approaches, XGBoost vs. a PyTorch neural network, each evaluated under two class-imbalance strategies (class weighting vs. SMOTE) and deployed as an interactive Streamlit dashboard.

## Overview

Credit card fraud is a textbook extreme class-imbalance problem: in this dataset, fraudulent transactions make up just **0.17%** of all transactions. A model that predicts "not fraud" every time would be 99.8% accurate and completely useless. This project explores how model choice and resampling strategy interact to solve that problem, and packages the results into a dashboard that lets you simulate transactions and inspect model behaviour in real time.

**Live dashboard:** *Run locally with `streamlit run dashboard/app.py` (see Usage below).*

## Key Findings

| Model | Strategy | PR-AUC | Fraud Precision | Fraud Recall |
|---|---|---|---|---|
| XGBoost | Class Weights | 0.8791 | 87% | 83% |
| XGBoost | SMOTE | 0.8775 | 77% | 86% |
| PyTorch MLP | Class Weights | 0.7044 | 7.8% | — |
| PyTorch MLP | SMOTE | 0.8245 | 59.4% | — |

**The headline result:** resampling strategy barely moves XGBoost's performance as it handles imbalance reasonably well on its own. It however makes a drastic difference for the neural network: SMOTE lifts the PyTorch model's fraud precision from **7.8% to 59.4%**, and PR-AUC from 0.70 to 0.82. Tree-based models and neural nets respond very differently to the same imbalance-handling techniques, which is a genuinely useful, non-obvious takeaway for choosing a resampling strategy in practice.

The confusion matrices make the practical impact concrete: the class-weighted MLP raised **1,044 false alarms** on the test set; the SMOTE-trained MLP raised just **56**, at the cost of a few additional missed fraud cases (16 vs. 10 false negatives). That's the precision/recall tradeoff in action, and it's exactly what the dashboard's threshold slider lets you explore interactively.

## Approach

**1. Exploratory Data Analysis** — quantified the class imbalance, examined transaction `Amount` and `Time` distributions by class, and ranked PCA-derived features (`V1`–`V28`) by correlation with fraud.

**2. Resampling strategies** — split data before any resampling to avoid leakage, then built two parallel training sets:
   - **Class weights**: original imbalanced data, with `scale_pos_weight` (XGBoost) / `pos_weight` (PyTorch) penalising missed fraud ~577x more than false alarms
   - **SMOTE**: synthetic oversampling of the minority class to a balanced 50/50 training set

**3. Modelling** — trained two model families under both strategies:
   - **XGBoost** (baseline): fast, strong out-of-the-box performance on tabular data
   - **PyTorch MLP** (main model): a 3-layer feedforward network (30 → 64 → 32 → 1) with dropout, trained with a manual training loop and `BCEWithLogitsLoss`

**4. Evaluation** — PR-AUC (Average Precision) over ROC-AUC or accuracy, since both are misleading under this level of imbalance. Precision/recall/F1 reported specifically for the fraud class.

**5. Dashboard** — a Streamlit app for interactive inference (sample real test-set transactions and see live predictions) and a performance artifacts tab (PR curves, confusion matrices, threshold trade-off analysis).

## Dataset

[Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud) (Kaggle / ULB Machine Learning Group) — 284,807 transactions over two days, 492 labelled as fraud. Features `V1`–`V28` are PCA-anonymised for confidentiality; `Time` and `Amount` are the only original features retained.

The raw CSV is not included in this repo (144MB, over GitHub's file size limit). To reproduce training:
1. Download `creditcard.csv` from the [Kaggle link above](https://www.kaggle.com/mlg-ulb/creditcardfraud)
2. Place it at `data/creditcard.csv`

## Project Structure

```
├── notebooks/
│   └── eda.ipynb              # Exploratory data analysis
├── src/
│   ├── train.py                # Model training (XGBoost + PyTorch)
│   ├── evaluate.py             # Evaluation metrics and plots
│   └── predict.py              # FraudPredictor inference class
├── saved_models/
│   ├── pytorch_smote_model.pth
│   ├── pytorch_weighted_model.pth
│   ├── scaler.joblib
│   ├── test_data.joblib
│   └── *.png                   # Evaluation artifacts
├── dashboard/
│   └── app.py                  # Streamlit dashboard
└── requirements.txt
```

## Setup

```bash
git clone https://github.com/ayshazia14/fraud-detection.git
cd fraud-detection
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Download the dataset (see [Dataset](#dataset) above) and place it at `data/creditcard.csv`.

## Usage

**Train models from scratch:**
```bash
python3 src/train.py
```

**Run the dashboard** (uses the pre-trained models already committed to `saved_models/`):
```bash
streamlit run dashboard/app.py
```

## Tech Stack

- **Modelling:** XGBoost, PyTorch, scikit-learn, imbalanced-learn (SMOTE)
- **Data:** pandas, NumPy
- **Visualisation:** matplotlib, seaborn
- **Dashboard:** Streamlit

## Future Work

- Containerise with Docker and deploy to AWS (EC2/ECS)
- CI/CD pipeline via GitHub Actions (lint → test → build → deploy)
- SHAP-based feature importance in the dashboard
- Experiment tracking (MLflow / Weights & Biases) for model comparisons

## Author

**Aysha Ziaulhaque**
