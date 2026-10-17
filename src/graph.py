from typing import Any, Dict, List, TypedDict

from langgraph.graph import END, StateGraph


class LoanState(TypedDict, total=False):
    applicant: Dict[str, Any]
    probability: float
    decision: str
    top_factors: List[Dict[str, Any]]
    credit_analysis: str
    risk_analysis: str
    recommendation: str


def credit_analysis(state: LoanState):
    applicant = state["applicant"]
    history = applicant["credit_history"]
    employment = applicant["employment"]

    if history in {"existing paid", "all paid", "no credits/all paid"}:
        credit = "The applicant has a relatively positive credit history."
    else:
        credit = "The applicant's credit history shows some previous credit concerns."

    if employment in {"4<=X<7", ">=7"}:
        credit += " Employment history also indicates reasonable stability."
    else:
        credit += " Employment stability should be considered during review."

    return {"credit_analysis": credit}


def risk_analysis(state: LoanState):
    applicant = state["applicant"]
    risks = []

    if applicant["savings_status"] in {"<100", "no known savings"}:
        risks.append("limited savings")
    if applicant["credit_amount"] >= 10000:
        risks.append("higher loan amount")
    if applicant["employment"] == "unemployed":
        risks.append("unemployment")
    if applicant["credit_history"] in {"delayed previously", "critical/other existing credit"}:
        risks.append("credit history concerns")

    if risks:
        risk = "Key risk factors: " + ", ".join(risks) + "."
    else:
        risk = "No major risk factor was identified from the supplied application fields."

    return {"risk_analysis": risk}


def recommendation(state: LoanState):
    probability = state["probability"]
    decision = state["decision"]

    if 0.45 <= probability < 0.65:
        recommendation_text = "Further manual review is recommended because the model confidence is moderate."
    elif decision == "Approved":
        recommendation_text = "The model supports approval, subject to normal underwriting checks."
    else:
        recommendation_text = "The model does not support approval; manual review can be considered if additional information is available."

    return {"recommendation": recommendation_text}


def build_report(state: LoanState):
    return {
        "credit_analysis": state["credit_analysis"],
        "risk_analysis": state["risk_analysis"],
        "recommendation": state["recommendation"],
    }


workflow = StateGraph(LoanState)
workflow.add_node("credit_analysis", credit_analysis)
workflow.add_node("risk_analysis", risk_analysis)
workflow.add_node("recommendation", recommendation)
workflow.set_entry_point("credit_analysis")
workflow.add_edge("credit_analysis", "risk_analysis")
workflow.add_edge("risk_analysis", "recommendation")
workflow.add_edge("recommendation", END)

graph = workflow.compile()


def run_underwriting(applicant, probability, decision, top_factors):
    state = graph.invoke(
        {
            "applicant": applicant,
            "probability": probability,
            "decision": decision,
            "top_factors": top_factors,
        }
    )
    return build_report(state)
