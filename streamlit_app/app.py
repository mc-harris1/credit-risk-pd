# streamlit_app/app.py
from __future__ import annotations

import os
from typing import Any, Dict, List

import altair as alt
import pandas as pd
import requests
import streamlit as st
from src.serving.schemas import LoanApplication

API_URL = os.getenv("PD_API_URL", "http://127.0.0.1:8000/predict")


def _call_api(app_data: LoanApplication) -> Dict[str, Any]:
    """Call the FastAPI prediction endpoint and return the JSON payload."""
    payload = app_data.model_dump()
    try:
        resp = requests.post(API_URL, json=payload, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        st.error(f"Error calling API: {exc}")
        return {}


def _render_top_factors(top_factors: List[Dict[str, Any]]) -> None:
    """Render SHAP top factors as table, chart, and natural language."""
    if not top_factors:
        st.info("No SHAP explanations returned for this prediction.")
        return

    st.markdown("---")
    st.header("Model Explainability (SHAP)")

    df = pd.DataFrame(top_factors)
    if "shap_value" not in df.columns or "feature" not in df.columns:
        st.warning("Unexpected SHAP format from API.")
        st.write(df)
        return

    df["abs_value"] = df["shap_value"].abs()
    df_sorted = df.sort_values("abs_value", ascending=False)

    # Table
    st.subheader("Top Contributing Features")
    st.dataframe(df_sorted[["feature", "shap_value"]], use_container_width=True)

    # Bar chart
    st.subheader("Feature Contribution Chart")
    chart_df = df_sorted.copy()
    chart_df = chart_df.sort_values("abs_value", ascending=True)

    chart = (
        alt.Chart(chart_df)
        .mark_bar()
        .encode(
            x=alt.X("shap_value:Q", title="SHAP Contribution"),
            y=alt.Y("feature:N", sort=chart_df["feature"].tolist()),
            color=alt.condition(
                alt.datum.shap_value > 0,
                alt.value("#2ECC71"),  # positive → green
                alt.value("#E74C3C"),  # negative → red
            ),
            tooltip=["feature", "shap_value"],
        )
        .properties(height=300)
    )
    st.altair_chart(chart, use_container_width=True)

    # Natural language explanation
    st.subheader("Why this prediction?")
    for row in df_sorted.to_dict("records"):
        direction = "increased" if row["shap_value"] > 0 else "decreased"
        st.write(
            f"- **{row['feature']}** {direction} the predicted default probability "
            f"by **{abs(row['shap_value']):.4f}**"
        )


def main() -> None:
    st.set_page_config(page_title="Credit Risk PD Scoring", layout="wide")
    st.title("Credit Risk – Probability of Default (PD) Scoring")

    st.markdown(
        "This app sends loan applications to a FastAPI service, which returns "
        "a predicted probability of default and SHAP-based feature explanations."
    )

    st.sidebar.header("Loan Application Inputs")

    # Basic numeric inputs
    loan_amnt = st.sidebar.number_input(
        "Loan Amount", min_value=1000.0, max_value=100000.0, value=10000.0, step=500.0
    )
    annual_inc = st.sidebar.number_input(
        "Annual Income", min_value=10000.0, max_value=500000.0, value=85000.0, step=5000.0
    )
    dti = st.sidebar.number_input(
        "Debt-to-Income (DTI)", min_value=0.0, max_value=60.0, value=18.0, step=0.5
    )

    # Categorical inputs (tune options to match your training data)
    term = st.sidebar.selectbox("Term", ["36 months", "60 months"])
    home_ownership = st.sidebar.selectbox("Home Ownership", ["RENT", "MORTGAGE", "OWN", "OTHER"])
    grade = st.sidebar.selectbox("Grade", ["A", "B", "C", "D", "E", "F", "G"])
    # For sub_grade, you could constrain based on grade, but keep it simple for now
    sub_grade = st.sidebar.text_input("Subgrade (e.g. B3)", value="B3")

    st.sidebar.markdown("---")
    st.sidebar.write(f"API URL: `{API_URL}`")

    if st.button("Score Application"):
        # Build Pydantic model from inputs
        app_data = LoanApplication(
            loan_amnt=loan_amnt,
            annual_inc=annual_inc,
            dti=dti,
            term=term,
            home_ownership=home_ownership,
            grade=grade,
            sub_grade=sub_grade,
        )

        with st.spinner("Calling PD scoring API..."):
            result = _call_api(app_data)

        if not result:
            return

        prob_default = float(result.get("prob_default", 0.0))
        risk_band = result.get("risk_band", "Unknown")

        st.subheader("Prediction")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Probability of Default", f"{prob_default:.2%}")
        with col2:
            # Color-code risk band
            if risk_band == "Low":
                st.success(f"Risk Band: **{risk_band}**")
            elif risk_band == "Medium":
                st.warning(f"Risk Band: **{risk_band}**")
            else:
                st.error(f"Risk Band: **{risk_band}**")

        # SHAP explanations
        top_factors = result.get("top_factors") or []
        _render_top_factors(top_factors)


if __name__ == "__main__":
    main()
