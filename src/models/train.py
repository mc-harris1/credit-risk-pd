# Copyright (c) 2025 Mark Harris
# Licensed under the MIT License. See LICENSE file in the project root.

import os
from datetime import datetime

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src.config import METADATA_DIR, MODELS_DIR, PROCESSED_DATA_DIR, RANDOM_STATE
from src.features.transforms import add_domain_features

FEATURES_FILE = "loans_features.parquet"
MODEL_FILE = "pd_model_xgb.pkl"


def load_feature_data() -> tuple[pd.DataFrame, pd.Series]:
    input_path = os.path.join(PROCESSED_DATA_DIR, FEATURES_FILE)
    df = pd.read_parquet(input_path)
    df.drop(columns=["loan_status"], inplace=True)
    y = df["default"]
    X = df.drop(columns=["default"])
    return X, y


def train_model() -> None:
    X, y = load_feature_data()
    X = add_domain_features(X)

    print(X.head())

    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    clf = XGBClassifier(
        objective="binary:logistic",
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", clf),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    pipeline.fit(X_train, y_train)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    print(f"Validation ROC-AUC: {auc:.4f}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, MODEL_FILE)
    joblib.dump(pipeline, model_path)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    METADATA_FILE = f"pd_model_xgb_{timestamp}.meta"
    os.makedirs(METADATA_DIR, exist_ok=True)
    metadata_path = os.path.join(METADATA_DIR, METADATA_FILE)

    # if files in metadata_path do not equal METADATA_FILE pattern, delete
    for f in os.listdir(METADATA_DIR):
        if f != METADATA_FILE:
            os.remove(os.path.join(METADATA_DIR, f))

    joblib.dump(
        {
            "model_file": MODEL_FILE,
            "trained_at": timestamp,
            "roc_auc": auc,
            "model_type": "XGBClassifier",
            "features": X.columns.tolist(),
        },
        metadata_path,
    )
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    train_model()
