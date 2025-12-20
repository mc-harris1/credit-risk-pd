from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlsplit, urlunsplit

import altair as alt
import pandas as pd
import requests
import streamlit as st
from src.serving.schemas import LoanApplication

# -----------------------------
# Config
# -----------------------------
DEFAULT_API_URL = "http://127.0.0.1:8000/predict"
API_URL = os.getenv("PD_API_URL", DEFAULT_API_URL).strip()

# Toggle debug UI via env var (recommended off in prod)
DEBUG_UI = os.getenv("DEBUG_UI", "0").strip().lower() in {"1", "true", "yes", "on"}

# Brand links (override if you want)
PORTFOLIO_URL = os.getenv("PORTFOLIO_URL", "https://mc-harris.dev").strip()
GITHUB_URL = os.getenv("GITHUB_URL", "https://github.com/mc-harris1").strip()
API_DOCS_URL = os.getenv("API_DOCS_URL", "https://api.mc-harris.dev/docs").strip()


# -----------------------------
# Models / helpers
# -----------------------------
@dataclass(frozen=True)
class PredictionResult:
    prob_default: float
    risk_band: str
    top_factors: List[Dict[str, Any]]
    raw: Dict[str, Any]


def _health_url_from_predict_url(predict_url: str) -> str:
    parts = urlsplit(predict_url)
    # replace path with /health
    return urlunsplit((parts.scheme, parts.netloc, "/health", "", ""))


@st.cache_data(show_spinner=False, ttl=30)
def check_api_health(api_url: str) -> bool:
    try:
        health_url = _health_url_from_predict_url(api_url)
        r = requests.get(health_url, timeout=3)
        return r.status_code == 200
    except Exception:
        return False


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


def inject_brand_css() -> None:
    css_path = Path(__file__).parent / "assets" / "brand.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)


def render_top_nav() -> None:
    st.markdown(
        f"""
        <div id="mh-topnav" style="
          position: fixed; top: 0; left: 0; right: 0;
          z-index: 2147483647; /* max z-index to beat Streamlit layers */
          background: rgba(5,8,22,0.92);
          backdrop-filter: blur(12px);
          -webkit-backdrop-filter: blur(12px);
          border-bottom: 1px solid rgba(31,41,55,0.6);
          padding: 12px 0;
          pointer-events: none;
        ">
          <div style="
            max-width: 980px; margin: 0 auto; padding: 0 16px;
            display: flex; align-items: center; justify-content: space-between;
          ">
            <div style="
              font-weight: 600; letter-spacing: .08em; font-size: .85rem;
              text-transform: uppercase; color: #9ca3af;
            ">
              MARK HARRIS
            </div>
            <div style="display:flex; gap: 16px; font-size: .85rem;">
              <a href="{PORTFOLIO_URL}" style="color:#9ca3af; pointer-events:auto;">Home</a>
              <a href="{API_DOCS_URL}" style="color:#9ca3af; pointer-events:auto;">API Docs</a>
              <a href="{GITHUB_URL}" style="color:#9ca3af; pointer-events:auto;">GitHub</a>
            </div>
          </div>
        </div>

        <!-- spacer so page content isn't hidden under fixed nav -->
        <div style="height: 56px;"></div>
        """,
        unsafe_allow_html=True,
    )

    api_ok = check_api_health(API_URL)

    st.markdown(
        f"""
        <div style="margin-bottom:0.75rem;">
        <div style="display:inline-flex; align-items:center; gap:0.5rem; padding:0.25rem 0.75rem;
                    border-radius:999px; background:rgba(15,23,42,0.9);
                    border:1px solid rgba(148,163,184,0.35); color:#9ca3af; font-size:0.75rem;">
            <span style="width:8px; height:8px; border-radius:999px; background:{"#22c55e" if api_ok else "#ef4444"};
                        box-shadow:0 0 0 4px {"rgba(34,197,94,0.35)" if api_ok else "rgba(239,68,68,0.35)"};
                        display:inline-block;"></span>
            <span>{"API Connected" if api_ok else "API Unreachable"}</span>
        </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------
# API layer
# -----------------------------
@st.cache_data(show_spinner=False, ttl=60)
def call_prediction_api(
    payload: Dict[str, Any], api_url: str
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    resp = None
    try:
        resp = requests.post(api_url, json=payload, timeout=15)
        resp.raise_for_status()
        return resp.json(), None
    except requests.HTTPError as exc:
        body = ""
        try:
            body = resp.text if resp is not None else ""
        except Exception:
            body = ""
        msg = f"API request failed: {exc}"
        if body and DEBUG_UI:
            msg = f"{msg}\n\nResponse body:\n{body}"
        return None, msg
    except Exception as exc:
        return None, f"Unable to reach the scoring API: {exc}"


# -----------------------------
# UI builders
# -----------------------------
def _subgrades_for_grade(grade: str) -> List[str]:
    SUBGRADE_COUNTS = {"A": 5, "B": 5, "C": 5, "D": 5, "E": 5, "F": 5, "G": 5}
    n = SUBGRADE_COUNTS.get(grade, 5)
    return [f"{grade}{i}" for i in range(1, n + 1)]


def build_application_from_ui() -> LoanApplication:
    st.sidebar.markdown("### Application Inputs")
    st.sidebar.caption("Adjust values and run a score. Defaults are realistic but illustrative.")

    loan_amnt = st.sidebar.number_input(
        "Loan amount ($)",
        min_value=1_000.0,
        max_value=100_000.0,
        value=10_000.0,
        step=500.0,
    )
    annual_inc = st.sidebar.number_input(
        "Annual income ($)",
        min_value=10_000.0,
        max_value=500_000.0,
        value=85_000.0,
        step=5_000.0,
    )
    dti = st.sidebar.number_input(
        "Debt-to-income (DTI %)",
        min_value=0.0,
        max_value=60.0,
        value=18.0,
        step=0.5,
    )

    st.sidebar.markdown("---")
    term = st.sidebar.selectbox("Term", ["36 months", "60 months"], index=0)
    home_ownership = st.sidebar.selectbox(
        "Home ownership",
        ["RENT", "MORTGAGE", "OWN", "OTHER"],
        index=1,
    )

    st.sidebar.markdown("---")
    grade = st.sidebar.selectbox("Credit grade", ["A", "B", "C", "D", "E", "F", "G"], index=1)
    subgrade_options = _subgrades_for_grade(grade)

    prev_sub = st.session_state.get("sub_grade", subgrade_options[0])
    if prev_sub not in subgrade_options:
        prev_sub = subgrade_options[0]

    sub_grade = st.sidebar.selectbox(
        "Credit sub-grade",
        subgrade_options,
        index=subgrade_options.index(prev_sub),
        key="sub_grade",
    )

    if DEBUG_UI:
        st.sidebar.markdown("---")
        st.sidebar.caption("Debug")
        st.sidebar.code(f"PD_API_URL={API_URL}", language="text")

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
    st.markdown("## Score")

    col1, col2, col3 = st.columns([1.2, 1.0, 1.2])

    with col1:
        st.metric("Probability of Default", f"{result.prob_default:.2%}")

    with col2:
        st.metric("Risk Band", (result.risk_band or "Unknown").strip())

    with col3:
        band = (result.risk_band or "Unknown").strip().lower()
        if band == "low":
            st.success("Low risk")
        elif band == "medium":
            st.warning("Medium risk")
        elif band in {"high", "very high"}:
            st.error("High risk")
        else:
            st.info("Uncategorized")


def render_shap_explanation(top_factors: List[Dict[str, Any]]) -> None:
    st.markdown("---")
    st.markdown("## Explainability")

    if not top_factors:
        st.info("Explainability data was not returned for this prediction.")
        return

    df = pd.DataFrame(top_factors)

    if "feature" not in df.columns and "name" in df.columns:
        df = df.rename(columns={"name": "feature"})
    if "shap_value" not in df.columns:
        for alt_name in ("value", "contribution", "shap"):
            if alt_name in df.columns:
                df = df.rename(columns={alt_name: "shap_value"})
                break

    if "feature" not in df.columns or "shap_value" not in df.columns:
        st.warning("Unexpected explainability format. Showing returned fields.")
        st.dataframe(df, use_container_width=True)
        return

    df = df.copy()
    df["shap_value"] = pd.to_numeric(df["shap_value"], errors="coerce").fillna(0.0)
    df["abs_value"] = df["shap_value"].abs()
    df_sorted = df.sort_values("abs_value", ascending=False)
    df_sorted = df_sorted.head(12)

    left, right = st.columns([1.0, 1.2], gap="large")

    with left:
        st.markdown("### Top drivers")
        st.dataframe(df_sorted[["feature", "shap_value"]], use_container_width=True, height=360)

    with right:
        st.markdown("### Contribution chart")
        chart_df = df_sorted.sort_values("abs_value", ascending=True)
        chart = (
            alt.Chart(chart_df)
            .mark_bar()
            .encode(
                x=alt.X("shap_value:Q", title="Impact on PD"),
                y=alt.Y("feature:N", sort=chart_df["feature"].tolist(), title=None),
                color=alt.condition(
                    alt.datum.shap_value > 0,
                    alt.value("#ef4444"),  # PD increases (riskier)
                    alt.value("#22c55e"),  # PD decreases (safer)
                ),
                tooltip=[alt.Tooltip("feature:N"), alt.Tooltip("shap_value:Q", format=".4f")],
            )
            .properties(height=min(420, 24 * max(6, len(chart_df))))
        )
        st.altair_chart(chart, use_container_width=True)

    st.markdown("### Plain-English summary")
    for row in df_sorted.to_dict("records"):
        shap_val = float(row.get("shap_value", 0.0))
        direction = "increased" if shap_val > 0 else "decreased"
        st.write(f"- **{row['feature']}** {direction} PD by **{abs(shap_val):.4f}**")


def render_footer() -> None:
    st.markdown("---")
    st.markdown(
        f"""
        <div class="muted" style="display:flex; justify-content:space-between; gap:1rem; flex-wrap:wrap;">
          <span>© Mark Harris · Credit Risk PD Demo</span>
          <span>
            <a href="{PORTFOLIO_URL}">mc-harris.dev</a> ·
            <a href="{GITHUB_URL}">GitHub</a> ·
            <a href="{API_DOCS_URL}">API Docs</a>
          </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------
# App
# -----------------------------
def main() -> None:
    st.set_page_config(
        page_title="Mark Harris | Credit Risk PD Demo",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    inject_brand_css()
    render_top_nav()

    st.markdown(
        """
        <div style="margin-bottom:0.75rem;">
          <div style="display:inline-flex; align-items:center; gap:0.5rem; padding:0.25rem 0.75rem;
                      border-radius:999px; background:rgba(15,23,42,0.9);
                      border:1px solid rgba(148,163,184,0.35); color:#9ca3af; font-size:0.75rem;">
            <span style="width:8px; height:8px; border-radius:999px; background:#22c55e;
                         box-shadow:0 0 0 4px rgba(34,197,94,0.35); display:inline-block;"></span>
            <span>Credit Risk · Probability of Default Scoring</span>
          </div>
        </div>
        <h1 style="margin:0.2rem 0 0.35rem 0; letter-spacing:-0.02em;">
          Credit Risk — <span style="color:#3b82f6;">PD Scoring</span>
        </h1>
        <div class="muted" style="max-width:55rem;">
          Submit a loan application and receive a calibrated probability of default, a risk band,
          and an explanation of the top drivers (SHAP-style contributions).
        </div>
        """,
        unsafe_allow_html=True,
    )

    app_data = build_application_from_ui()

    st.markdown("---")
    with st.form("score_form", clear_on_submit=False):
        c1, c2 = st.columns([1.0, 2.0])
        with c1:
            submitted = st.form_submit_button("Score application", type="primary")
        with c2:
            st.caption("Tip: Adjust inputs in the sidebar. Results refresh per submission.")

    if not submitted:
        # render_footer()
        return

    payload = app_data.model_dump()

    with st.spinner("Scoring application..."):
        json_resp, err = call_prediction_api(payload, API_URL)

    if err:
        st.error(err)
        if DEBUG_UI:
            with st.expander("Debug: request payload"):
                st.json(payload)
        # render_footer()
        return

    assert json_resp is not None
    result = _normalize_prediction_payload(json_resp)

    render_prediction(result)
    render_shap_explanation(result.top_factors)

    if DEBUG_UI:
        with st.expander("Debug: raw API response"):
            st.json(result.raw)

    # render_footer()


if __name__ == "__main__":
    main()
