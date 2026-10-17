import requests
import streamlit as st

st.set_page_config(page_title="AI Loan Underwriting", page_icon="🏦")

st.title("AI Loan Underwriting System")
st.write("Predict loan approval probability and explain the main factors behind the prediction.")

st.subheader("Loan Application")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=75, value=35)
    credit_amount = st.number_input("Credit Amount", min_value=100, max_value=100000, value=5000)
    duration = st.number_input("Loan Duration (months)", min_value=1, max_value=72, value=24)
    credit_history = st.selectbox(
        "Credit History",
        ["existing paid", "all paid", "no credits/all paid", "delayed previously", "critical/other existing credit"],
    )
    savings_status = st.selectbox(
        "Savings",
        ["<100", "100<=X<500", "500<=X<1000", ">=1000", "no known savings"],
    )
    employment = st.selectbox(
        "Employment",
        ["unemployed", "<1", "1<=X<4", "4<=X<7", ">=7"],
    )
    checking_status = st.selectbox(
        "Checking Account",
        ["<0", "0<=X<200", ">=200", "no checking"],
    )
    purpose = st.selectbox(
        "Purpose",
        ["radio/tv", "new car", "used car", "education", "business", "furniture/equipment", "repairs", "domestic appliance", "vacation/others"],
    )
    installment_commitment = st.slider("Installment Commitment", 1, 4, 2)
    residence_since = st.slider("Years at Current Residence", 1, 4, 2)

with col2:
    personal_status = st.selectbox("Personal Status", ["male single", "male mar/wid", "male div/sep", "female div/dep/mar"])
    other_parties = st.selectbox("Other Parties", ["none", "guarantor", "co applicant"])
    property_magnitude = st.selectbox("Property", ["real estate", "car", "life insurance", "no known property"])
    other_payment_plans = st.selectbox("Other Payment Plans", ["none", "bank", "stores"])
    housing = st.selectbox("Housing", ["own", "rent", "for free"])
    existing_credits = st.slider("Existing Credits", 1, 4, 1)
    job = st.selectbox("Job", ["skilled", "unskilled resident", "high qualif/self emp/mgmt", "unemp/non res"])
    num_dependents = st.slider("Dependents", 1, 2, 1)
    own_telephone = st.selectbox("Telephone", ["yes", "none"])
    foreign_worker = st.selectbox("Foreign Worker", ["yes", "no"])

if st.button("Run Underwriting", type="primary"):
    payload = {
        "checking_status": checking_status,
        "duration": duration,
        "credit_history": credit_history,
        "purpose": purpose,
        "credit_amount": credit_amount,
        "savings_status": savings_status,
        "employment": employment,
        "installment_commitment": installment_commitment,
        "personal_status": personal_status,
        "other_parties": other_parties,
        "residence_since": residence_since,
        "property_magnitude": property_magnitude,
        "age": age,
        "other_payment_plans": other_payment_plans,
        "housing": housing,
        "existing_credits": existing_credits,
        "job": job,
        "num_dependents": num_dependents,
        "own_telephone": own_telephone,
        "foreign_worker": foreign_worker,
    }

    try:
        response = requests.post("http://127.0.0.1:8000/predict", json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()

        st.divider()
        st.subheader("Prediction")
        st.metric("Approval Probability", f"{result['approval_probability']}%")
        st.write(f"**Decision:** {result['decision']}")

        st.subheader("Top SHAP Factors")
        for factor in result["top_factors"]:
            direction = "positive" if factor["impact"] >= 0 else "negative"
            st.write(f"**{factor['feature']}** — {factor['impact']:+.4f} ({direction})")

        st.subheader("Underwriting Analysis")
        report = result["underwriting_report"]
        st.write(f"**Credit Analysis:** {report['credit_analysis']}")
        st.write(f"**Risk Analysis:** {report['risk_analysis']}")
        st.write(f"**Recommendation:** {report['recommendation']}")

    except requests.RequestException as exc:
        st.error(f"Could not reach the FastAPI backend: {exc}")
