from pathlib import Path

import joblib
import pandas as pd
import shap
from fastapi import FastAPI
from pydantic import BaseModel

from src.graph import run_underwriting

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "loan_model.pkl"
PREPROCESSOR_PATH = ROOT / "models" / "preprocessor.pkl"
THRESHOLD_PATH = ROOT / "models" / "threshold.pkl"

app = FastAPI(title="Loan Underwriting API")

model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)
threshold = float(joblib.load(THRESHOLD_PATH))
explainer = shap.TreeExplainer(model)


class LoanApplication(BaseModel):
    checking_status: str
    duration: int
    credit_history: str
    purpose: str
    credit_amount: int
    savings_status: str
    employment: str
    installment_commitment: int
    personal_status: str
    other_parties: str
    residence_since: int
    property_magnitude: str
    age: int
    other_payment_plans: str
    housing: str
    existing_credits: int
    job: str
    num_dependents: int
    own_telephone: str
    foreign_worker: str


def get_shap_values(processed):
    values = explainer.shap_values(processed)
    if isinstance(values, list):
        return values[1][0]
    return values[0]


def clean_feature_name(name: str) -> str:
    name = name.replace("num__", "").replace("cat__", "")
    return name.replace("_", " ").title()


@app.get("/")
def home():
    return {"message": "Loan Underwriting API is running"}


@app.post("/predict")
def predict(application: LoanApplication):
    data = application.model_dump()
    input_df = pd.DataFrame([data])
    processed = preprocessor.transform(input_df)

    probability = float(model.predict_proba(processed)[0][1])
    prediction = int(probability >= threshold)
    decision = "Approved" if prediction else "Rejected"

    shap_values = get_shap_values(processed)
    feature_names = preprocessor.get_feature_names_out()

    factors = [
        {
            "feature": clean_feature_name(name),
            "impact": round(float(impact), 4),
        }
        for name, impact in zip(feature_names, shap_values)
    ]
    factors.sort(key=lambda item: abs(item["impact"]), reverse=True)
    top_factors = factors[:5]

    report = run_underwriting(
        applicant=data,
        probability=probability,
        decision=decision,
        top_factors=top_factors,
    )

    return {
        "approval_probability": round(probability * 100, 2),
        "decision": decision,
        "threshold": round(threshold, 2),
        "top_factors": top_factors,
        "underwriting_report": report,
    }
