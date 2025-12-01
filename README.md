Credit Risk Scoring — Probability of Default (PD) Prediction

This project builds an end-to-end credit risk scoring system that predicts the probability that a loan will default, using real-world consumer lending datasets.

It includes a full ML pipeline (data → features → modeling → evaluation → serving), a FastAPI scoring service, and an optional Streamlit dashboard for model exploration and applicant-level explanations.

The goal is to demonstrate production-quality ML engineering, risk modeling, explainability, and deployment patterns commonly used in FinTech environments.

# 📌 Features:  

* Binary classification model predicting the probability of default (PD) on consumer loans  
* Time-based train/validation/test split to mimic real-world lending workflows.
* Advanced modeling with LightGBM / XGBoost (plus logistic regression baseline).
* Explainability:
  * SHAP global feature importance
  * Local explanations for individual applicants
* FastAPI REST service for real-time scoring
* Dockerized deployment
* CI/CD ready (linting, tests, reproducibility)
* Optional Streamlit UI for demoing risk predictions and exploring model outputs
