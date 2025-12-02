import requests
import streamlit as st

API_URL = "http://localhost:8000/score"


def main() -> None:
    st.title("Credit Risk Scoring Demo (PD)")

    st.sidebar.header("Loan Application")

    loan_amnt = st.sidebar.number_input("Loan amount", value=10000.0, min_value=0.0)
    annual_inc = st.sidebar.number_input("Annual income", value=60000.0, min_value=0.0)
    dti = st.sidebar.number_input("DTI", value=18.0, min_value=0.0)
    term = st.sidebar.selectbox("Term", ["36 months", "60 months"])
    home_ownership = st.sidebar.selectbox("Home ownership", ["RENT", "MORTGAGE", "OWN", "OTHER"])
    grade = st.sidebar.selectbox("Grade", ["A", "B", "C", "D", "E", "F", "G"])

    if st.button("Score application"):
        payload = {
            "loan_amnt": loan_amnt,
            "annual_inc": annual_inc,
            "dti": dti,
            "term": term,
            "home_ownership": home_ownership,
            "grade": grade,
        }
        try:
            resp = requests.post(API_URL, json=payload, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            st.subheader("Result")
            st.metric("Probability of Default", f"{data['prob_default']:.2%}")
            st.write(f"Risk band: **{data['risk_band']}**")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Error calling API: {exc}")


if __name__ == "__main__":
    main()
