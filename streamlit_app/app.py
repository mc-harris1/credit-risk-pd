from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import altair as alt
import pandas as pd
import requests
import streamlit as st
from src.serving.schemas import LoanApplication

DEFAULT_API_URL = "http://127.0.0.1:8000/predict"
API_URL = os.getenv("PD_API_URL", DEFAULT_API_URL)


# -----------------------------
# Models / helpers
# -----------------------------
@dataclass(frozen=True)
class PredictionResult:
    prob_default: float
    risk_band: str
    top_factors: List[Dict[str, Any]]
    raw: Dict[str, Any]


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _normalize_prediction_payload(payload: Dict[str, Any]) -> PredictionResult:
    """
    Tolerant to upstream response changes.
    PD may be returned as: prob_default | pd | probability_default
    Risk band may be: risk_band | band | risk
    SHAP list may be: top_factors | attributions | shap_top_factors | explanations
    """
    prob_default = _safe_float(
        payload.get("prob_default", payload.get("pd", payload.get("probability_default", 0.0)))
    )
    risk_band = str(payload.get("risk_band", payload.get("band", payload.get("risk", "Unknown"))))

    # SHAP factors: try multiple keys
    top_factors = payload.get("top_factors")
    if top_factors is None:
        top_factors = payload.get("attributions")
    if top_factors is None:
        top_factors = payload.get("shap_top_factors")
    if top_factors is None:
        top_factors = payload.get("explanations")

    if not isinstance(top_factors, list):
        top_factors = []

    return PredictionResult(
        prob_default=prob_default,
        risk_band=risk_band,
        top_factors=top_factors,
        raw=payload,
    )


# -----------------------------
# API layer
# -----------------------------
@st.cache_data(show_spinner=False, ttl=60)
def call_prediction_api(
    payload: Dict[str, Any], api_url: str
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        resp = requests.post(api_url, json=payload, timeout=15)
        resp.raise_for_status()
        return resp.json(), None
    except requests.HTTPError as exc:
        # include response body if available
        body = ""
        try:
            body = resp.text  # type: ignore[name-defined]
        except Exception:
            body = ""
        msg = f"HTTP error calling API: {exc}"
        if body:
            msg = f"{msg}\n\nResponse body:\n{body}"
        return None, msg
    except Exception as exc:
        return None, f"Error calling API: {exc}"


# -----------------------------
# UI builders
# -----------------------------
def _subgrades_for_grade(grade: str) -> List[str]:
    """
    Typical LendingClub-style subgrades: A1..A5, B1..B5, etc.
    If your training data differs, tweak counts here.
    """
    SUBGRADE_COUNTS = {"A": 5, "B": 5, "C": 5, "D": 5, "E": 5, "F": 5, "G": 5}
    n = SUBGRADE_COUNTS.get(grade, 5)
    return [f"{grade}{i}" for i in range(1, n + 1)]


def build_application_from_ui() -> LoanApplication:
    st.sidebar.header("Loan Application Inputs")

    loan_amnt = st.sidebar.number_input(
        "Loan Amount", min_value=1000.0, max_value=100000.0, value=10000.0, step=500.0
    )
    annual_inc = st.sidebar.number_input(
        "Annual Income", min_value=10000.0, max_value=500000.0, value=85000.0, step=5000.0
    )
    dti = st.sidebar.number_input(
        "Debt-to-Income (DTI)", min_value=0.0, max_value=60.0, value=18.0, step=0.5
    )

    term = st.sidebar.selectbox("Term", ["36 months", "60 months"], index=0)
    home_ownership = st.sidebar.selectbox(
        "Home Ownership", ["RENT", "MORTGAGE", "OWN", "OTHER"], index=1
    )

    # Grade -> Subgrade filtering
    grade = st.sidebar.selectbox("Grade", ["A", "B", "C", "D", "E", "F", "G"], index=1)
    subgrade_options = _subgrades_for_grade(grade)

    prev_sub = st.session_state.get("sub_grade", subgrade_options[0])
    if prev_sub not in subgrade_options:
        prev_sub = subgrade_options[0]

    sub_grade = st.sidebar.selectbox(
        "Subgrade",
        subgrade_options,
        index=subgrade_options.index(prev_sub),
        key="sub_grade",
    )

    st.sidebar.markdown("---")
    st.sidebar.write(f"API URL: `{API_URL}`")

    return LoanApplication(
        loan_amnt=loan_amnt,
        annual_inc=annual_inc,
        dti=dti,
        term=term,
        home_ownership=home_ownership,
        grade=grade,
        sub_grade=sub_grade,
    )


# -----------------------------
# Rendering
# -----------------------------
def render_prediction(result: PredictionResult) -> None:
    st.subheader("Prediction")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Probability of Default", f"{result.prob_default:.2%}")

    with col2:
        band = (result.risk_band or "Unknown").strip()
        band_norm = band.lower()
        if band_norm == "low":
            st.success(f"Risk Band: **{band}**")
        elif band_norm == "medium":
            st.warning(f"Risk Band: **{band}**")
        elif band_norm in {"high", "very high"}:
            st.error(f"Risk Band: **{band}**")
        else:
            st.info(f"Risk Band: **{band}**")


def render_shap_explanation(top_factors: List[Dict[str, Any]]) -> None:
    st.markdown("---")
    st.header("Model Explainability (SHAP)")

    if not top_factors:
        st.info(
            "No SHAP explanations were returned for this prediction (top_factors/attributions missing)."
        )
        return

    df = pd.DataFrame(top_factors)

    # Handle slight naming differences
    if "feature" not in df.columns and "name" in df.columns:
        df = df.rename(columns={"name": "feature"})
    if "shap_value" not in df.columns:
        for alt_name in ("value", "contribution", "shap"):
            if alt_name in df.columns:
                df = df.rename(columns={alt_name: "shap_value"})
                break

    if "feature" not in df.columns or "shap_value" not in df.columns:
        st.warning("Unexpected SHAP format from API. Showing raw payload.")
        st.dataframe(df, use_container_width=True)
        return

    df = df.copy()
    df["shap_value"] = pd.to_numeric(df["shap_value"], errors="coerce").fillna(0.0)
    df["abs_value"] = df["shap_value"].abs()
    df_sorted = df.sort_values("abs_value", ascending=False)

    st.subheader("Top Contributing Features")
    st.dataframe(df_sorted[["feature", "shap_value"]], use_container_width=True)

    st.subheader("Feature Contribution Chart")
    chart_df = df_sorted.sort_values("abs_value", ascending=True)

    chart = (
        alt.Chart(chart_df)
        .mark_bar()
        .encode(
            x=alt.X("shap_value:Q", title="SHAP Contribution"),
            y=alt.Y("feature:N", sort=chart_df["feature"].tolist(), title=None),
            color=alt.condition(
                alt.datum.shap_value > 0,
                alt.value("#2ECC71"),
                alt.value("#E74C3C"),
            ),
            tooltip=[alt.Tooltip("feature:N"), alt.Tooltip("shap_value:Q", format=".4f")],
        )
        .properties(height=min(420, 24 * max(6, len(chart_df))))
    )
    st.altair_chart(chart, use_container_width=True)

    st.subheader("Why this prediction?")
    for row in df_sorted.to_dict("records"):
        shap_val = float(row.get("shap_value", 0.0))
        direction = "increased" if shap_val > 0 else "decreased"
        st.write(
            f"- **{row['feature']}** {direction} the predicted default probability "
            f"by **{abs(shap_val):.4f}**"
        )


# -----------------------------
# App
# -----------------------------
def main() -> None:
    st.set_page_config(page_title="Credit Risk PD Scoring", layout="wide")
    st.title("Credit Risk – Probability of Default (PD) Scoring")

    st.markdown(
        "This app sends loan applications to a FastAPI service, which returns "
        "a predicted probability of default, a risk band, and SHAP-based explanations."
    )

    app_data = build_application_from_ui()

    with st.form("score_form", clear_on_submit=False):
        submitted = st.form_submit_button("Score Application")

    if not submitted:
        st.info("Adjust inputs in the sidebar, then click **Score Application**.")
        return

    payload = app_data.model_dump()

    with st.spinner("Calling PD scoring API..."):
        json_resp, err = call_prediction_api(payload, API_URL)

    if err:
        st.error(err)
        with st.expander("Request payload"):
            st.json(payload)
        return

    assert json_resp is not None
    result = _normalize_prediction_payload(json_resp)

    render_prediction(result)
    render_shap_explanation(result.top_factors)

    with st.expander("Raw API response"):
        st.json(result.raw)


if __name__ == "__main__":
    main()
