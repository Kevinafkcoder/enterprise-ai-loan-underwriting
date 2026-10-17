# AI Loan Underwriting System

A simple end-to-end loan underwriting application using XGBoost, SHAP, LangGraph, FastAPI and Streamlit.

## Flow

Streamlit → FastAPI → preprocessing → XGBoost → SHAP → LangGraph → Streamlit

## Components

- **XGBoost:** predicts loan approval probability.
- **SHAP:** explains the main feature contributions behind the prediction.
- **LangGraph:** organizes credit analysis, risk analysis and recommendation into a workflow.
- **FastAPI:** exposes the prediction pipeline through a `/predict` endpoint.
- **Streamlit:** provides the user interface.

## Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Start FastAPI:

```bash
python -m uvicorn src.api:app --reload
```

In another terminal, start Streamlit:

```bash
python -m streamlit run app.py
```

Open the Streamlit URL shown in the terminal.

The `models/` folder contains the trained model, preprocessing pipeline and decision threshold used by the API.

