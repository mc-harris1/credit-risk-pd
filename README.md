# Credit Risk Scoring — Probability of Default (PD) Prediction

This project builds an **end-to-end credit risk scoring system** that predicts the **probability that a loan will default**, using real-world consumer lending datasets.  
It includes a full ML pipeline (data → features → modeling → evaluation → serving), a **FastAPI scoring service**, and an optional **Streamlit dashboard** for model exploration and applicant-level explanations.

The goal is to demonstrate production-quality ML engineering, risk modeling, explainability, and deployment patterns commonly used in **FinTech** environments.

---

## **📌 Features**

- **Binary classification model** predicting the probability of default (PD) on consumer loans  
- **Time-based train/validation/test split** to mimic real-world lending workflows  
- **Advanced modeling** with LightGBM / XGBoost (plus logistic regression baseline)  
- **Explainability**:
  - SHAP global feature importance
  - Local explanations for individual applicants  
- **FastAPI REST service** for real-time scoring  
- **Dockerized deployment**  
- **CI/CD ready** (linting, tests, reproducibility)  
- **Optional Streamlit UI** for demoing risk predictions and exploring model outputs  

---

## **📁 Repository Structure**

```
credit-risk-pd/
├── README.md
├── requirements.txt or pyproject.toml
├── Makefile
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_prototyping.ipynb
├── src/
│   ├── config.py
│   ├── data/
│   │   ├── load_data.py
│   │   └── preprocess.py
│   ├── features/
│   │   └── build_features.py
│   ├── models/
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   ├── predict.py
│   │   └── metrics.py
│   ├── serving/
│   │   └── api.py
│   └── utils/
│       └── io.py
├── models/
│   ├── artifacts/
│   └── metadata/
├── streamlit_app/
│   └── app.py
├── tests/
│   ├── test_data.py
│   ├── test_features.py
│   ├── test_models.py
│   └── test_api.py
└── docker/
    └── Dockerfile
```

---

## **📊 Data**

### **Primary Dataset (Recommended)**  
This project uses **Lending Club / LendingClub-style consumer loan data**, available publicly through several curated sources.

Datasets generally include:  
- Loan amount, term, interest rate  
- Borrower attributes (income, employment length, home ownership)  
- Debt-to-income (DTI), revolving utilization  
- Credit grade & subgrade  
- Loan status (fully paid, charged off, default, late, etc.)

### **Target Definition**

We model a binary classification target:

```
default = 1   → Loan is charged off, defaulted, or severely delinquent  
default = 0   → Loan is fully paid
```

Loans with ambiguous statuses (“Current”, “In Grace Period”, etc.) are removed for label clarity.

---

## **🛠️ ML Pipeline Overview**

### **1. Preprocessing**
- Clean categorical values  
- Handle missing data  
- Filter to relevant loan statuses  
- Remove data leakage sources (e.g., post-origination fields)

### **2. Feature Engineering**
Examples:
- Loan-to-income ratio  
- Interest rate buckets  
- DTI normalization  
- Employment length encoding  
- Grade/subgrade ordering  
- One-hot / target encoding for categoricals  

### **3. Modeling**
Baseline:  
- Logistic Regression (interpretable benchmark)

Advanced:  
- **LightGBM** or **XGBoost** gradient boosting  
- Optional: Probability calibration (Isotonic / Platt scaling)

Metrics captured:  
- ROC-AUC  
- PR-AUC (important for imbalance)  
- Brier Score (probability quality)  
- Confusion matrix & cost curves  
- Calibration curves  

### **4. Explainability**
Using SHAP:  
- **Global importance** → which features matter most for predicting default  
- **Local explanations** → why a specific applicant received their risk score  

---

## **🚀 Model Serving (FastAPI)**

A production-ready scoring service is included.

### **Endpoints**

#### `GET /health`
Health check.

#### `POST /score`

Example response:

```json
{
  "prob_default": 0.182,
  "risk_band": "Medium",
  "top_factors": [
    {"feature": "interest_rate", "impact": 0.07, "direction": "up"},
    {"feature": "dti", "impact": 0.05, "direction": "up"}
  ]
}
```

Run locally:

```bash
uvicorn src.serving.api:app --reload --port 8000
```

Docs at:  
`http://localhost:8000/docs`

---

## **🐳 Docker**

```bash
docker build -t credit-risk-api -f docker/Dockerfile .
docker run -p 8000:8000 credit-risk-api
```

---

## **📊 Optional: Streamlit Demo App**

```
streamlit run streamlit_app/app.py
```

---

## **🧪 Testing**

```
pytest -q
```

---

## **🧭 How to Run the Full Pipeline**

```bash
pip install -r requirements.txt
python -m src.data.preprocess
python -m src.features.build_features
python -m src.models.train
uvicorn src.serving.api:app --reload
```

---

## **📦 Future Improvements**

- Add LGD/EAD for full expected loss modeling  
- Survival analysis for time-to-default  
- Reject inference to correct approval bias  
- Drift monitoring (PSI, feature drift)  
- Cloud deployment (AWS Lambda, ECS, GCP Cloud Run)

---

## **📄 License**
No license.
