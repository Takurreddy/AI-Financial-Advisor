"""pytest suite for the recommendation engine.  Run:  pytest -q test_engine.py"""
import copy, math
from pathlib import Path
import numpy as np, pandas as pd, pytest
from recommendation_engine import FinancialAdvisor, UserFinancialInput, inr
from demo import PERSONAS

import os
# Datasets are looked up in: $FINANCE_DATA_DIR, ./data, then the project's parent folder.
_CANDIDATES = [Path(os.environ["FINANCE_DATA_DIR"])] if os.environ.get("FINANCE_DATA_DIR") else []
_CANDIDATES += [Path(__file__).parent / "data", Path(__file__).parent.parent]
DATA = next((d for d in _CANDIDATES if (d / "expense_prediction_dataset.csv").exists()), None)
adv = FinancialAdvisor()
P = list(PERSONAS.values())


def test_inr_formatting():
    assert inr(40000) == "₹40,000" and inr(1234567) == "₹12,34,567" and inr(-5000) == "-₹5,000"


def test_cluster_labels_are_distinct_and_semantic():
    # saved pickle labelled two clusters 'Debt-Heavy Spender'; engine re-derives them
    assert sorted(adv.profile_labels.values()) == ["Debt-Heavy Spender", "High Spender", "Saver"]


def test_report_structure_and_money_invariants():
    for u in P:
        r = adv.analyze(u)
        assert r.predictions["credit_risk"] in {"Low", "Medium", "High"}
        assert abs(sum(r.predictions["credit_risk_probs"].values()) - 1) < 1e-6
        b = r.budget
        assert b["needs_budget"] + b["wants_budget"] + b["savings_and_debt_budget"] <= u.monthly_income + 1
        assert sum(r.savings["recommended_allocation"].values()) <= b["savings_and_debt_budget"] + 1
        assert 1 <= len(r.priority_actions) <= 3
        assert r.to_json()                      # serialisable


def test_saver_gets_growth_and_no_alerts_stressed_gets_critical():
    saver, stressed = adv.analyze(P[1]), adv.analyze(P[2])
    assert saver.investment["category"] == "Growth" and not saver.alerts
    assert stressed.investment["category"] == "Capital Preservation"
    assert any(a["severity"] == "critical" for a in stressed.alerts)
    assert stressed.predictions["credit_risk"] == "High"


def test_expense_prediction_close_to_recent_spending():
    for u in P:
        r = adv.analyze(u)
        assert 0.75 * min(u.previous_month_expense, u.avg_3_month_expense) <= r.predictions["predicted_expense"]
        assert abs(r.metrics["expense_change_pct"]) < 20


def test_inconsistent_savings_does_not_break_expense_model():
    """Regression test: raw model gave -32k..+307k when savings was inconsistent with income."""
    base = P[0]
    preds = []
    for s in (0, 3000, 20000, 38000):
        u = copy.deepcopy(base); u.savings_amount = s
        r = adv.analyze(u); preds.append(r.predictions["predicted_expense"])
        assert not r.predictions["expense_clamped"]
    assert max(preds) - min(preds) < 1.0       # reconciled -> identical expense prediction


def test_guardrail_clamps_out_of_range():
    u = copy.deepcopy(P[0]); u.monthly_income = 5_000_000   # far outside training range
    r = adv.analyze(u)
    lo, hi = 0.75 * 28000, 1.35 * 29000
    assert lo <= r.predictions["predicted_expense"] <= hi


@pytest.mark.parametrize("bad", [dict(age=10), dict(monthly_income=0), dict(total_debt=-1),
                                 dict(credit_utilization=3.0), dict(food_expense=-5)])
def test_validation_rejects_bad_input(bad):
    u = copy.deepcopy(P[0])
    for k, v in bad.items(): setattr(u, k, v)
    with pytest.raises(ValueError):
        adv.analyze(u)


def test_minimal_input_runs_and_reports_assumptions():
    r = adv.analyze(P[3])
    assert r.assumptions and any("credit risk" in w for w in r.warnings)


def test_deficit_user_triggers_deficit_alert():
    u = UserFinancialInput(age=30, monthly_income=20000, previous_month_expense=26000,
                           avg_3_month_expense=25000, savings_amount=0)
    r = adv.analyze(u)
    assert any(a["code"] == "DEFICIT" for a in r.alerts) and "Close the monthly gap" in r.priority_actions[0]


@pytest.mark.skipif(DATA is None, reason="CSV datasets not found - put them in ./data or set FINANCE_DATA_DIR")
def test_stress_on_3000_stitched_users_from_real_datasets():
    """Stitch rows from the three real datasets into unified users; engine must never crash / emit NaN."""
    ed = pd.read_csv(DATA / "expense_prediction_dataset.csv").sample(3000, random_state=1).reset_index(drop=True)
    cd = pd.read_csv(DATA / "credit_risk_dataset.csv").sample(3000, random_state=2).reset_index(drop=True)
    pdf = pd.read_csv(DATA / "financial_profile_dataset.csv").sample(3000, random_state=3).reset_index(drop=True)
    clamped = 0
    for i in range(3000):
        e, c, p = ed.iloc[i], cd.iloc[i], pdf.iloc[i]
        u = UserFinancialInput(
            age=int(e.age), monthly_income=float(e.monthly_income), previous_month_expense=float(e.previous_month_expense),
            avg_3_month_expense=float(e.avg_3_month_expense), savings_amount=float(e.savings_amount),
            food_expense=float(e.food_expense), transport_expense=float(e.transport_expense),
            shopping_expense=float(e.shopping_expense), entertainment_expense=float(e.entertainment_expense),
            medical_expense=float(e.medical_expense), education_expense=float(e.education_expense),
            utilities_expense=float(e.utilities_expense), other_expense=float(e.other_expense),
            transaction_count=int(e.transaction_count), expense_growth_rate=float(e.expense_growth_rate),
            total_debt=float(p.debt_amount), monthly_emi=float(c.monthly_emi), credit_utilization=float(c.credit_utilization),
            late_payment_count=int(c.late_payment_count), missed_payment_count=int(c.missed_payment_count),
            repayment_ratio=float(c.repayment_ratio), emergency_fund=float(p.emergency_fund),
            investment_amount=float(p.investment_amount))
        r = adv.analyze(u)
        assert math.isfinite(r.predictions["predicted_expense"]) and r.predictions["predicted_expense"] > 0
        assert r.budget["savings_and_debt_budget"] >= 0
        clamped += r.predictions["expense_clamped"]
    assert clamped / 3000 < 0.02, f"guardrail fired on {clamped/3000:.1%} of in-distribution users"


def test_negative_savings_is_accepted_and_flagged():
    u = UserFinancialInput(age=30, monthly_income=25000, previous_month_expense=26000,
                           avg_3_month_expense=25500, savings_amount=-1000)
    r = adv.analyze(u)
    assert any(a["code"] == "DEFICIT" for a in r.alerts)


def test_chart_data_is_consistent():
    for u in P:
        r = adv.analyze(u); cd = r.chart_data
        assert abs(sum(cd["category_current"].values()) - r.predictions["predicted_expense"]) < 1
        assert sum(cd["category_recommended"].values()) <= sum(cd["category_current"].values()) + 1
        assert sum(cd["income_plan"].values()) <= u.monthly_income + 1
        pc = cd["profile_compare"]
        assert len(pc["user"]) == len(pc["features"]) and len(pc["clusters"]) == 3