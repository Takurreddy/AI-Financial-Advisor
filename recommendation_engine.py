"""
AI Personal Finance Advisor - Recommendation Engine (Module 4)
==============================================================

Pipeline:
    UserFinancialInput
        -> validate & feature-build (per model)
        -> [1] Expense model   : predicted next-month expense
        -> [2] Credit model    : Low / Medium / High risk (+ probabilities)
        -> [3] Profile model   : Saver / High Spender / Debt-Heavy Spender
        -> rule-based aggregation
        -> AdvisorReport (budget, savings, investment category, alerts, actions)

The engine does NOT merge the three datasets; it combines the *outputs* of the
three trained models, exactly as described in the project report.

Academic decision-support tool - not professional financial advice.
"""
from __future__ import annotations

import json
import re
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Configuration (all thresholds in one place so the team can tune / justify)
# --------------------------------------------------------------------------
RISK_LABELS = {0: "Low", 1: "Medium", 2: "High"}   # verified against dataset

ESSENTIAL_CATS = ["food", "transport", "medical", "education", "utilities"]
DISCRETIONARY_CATS = ["shopping", "entertainment", "other"]
ALL_CATS = ESSENTIAL_CATS + DISCRETIONARY_CATS

# Average category shares in the expense dataset (used only if user gives no split)
DEFAULT_CAT_SHARE = {
    "food": 0.2796, "transport": 0.1196, "shopping": 0.1197,
    "entertainment": 0.0801, "medical": 0.0806, "education": 0.0699,
    "utilities": 0.1504, "other": 0.1001,
}

# Median values from the credit dataset - used for optional fields the user omits
CREDIT_MEDIANS = {
    "employment_years": 20, "credit_utilization": 0.5486, "previous_loans": 1,
    "late_payment_count": 3, "missed_payment_count": 1, "repayment_ratio": 0.7283,
    "credit_limit": 133268.5, "bank_balance": 7175.5,
    "credit_history_years": 20, "number_of_credit_accounts": 4, "loan_term": 60,
}

# Expense-model guardrail: in the training data next-month expense stayed within
# 0.76x-1.33x of recent spending. Predictions outside this band are clamped.
PRED_BAND_LOW, PRED_BAND_HIGH = 0.75, 1.35

# Budget: max share of income for discretionary ("wants") spending, by profile
WANTS_CAP = {"Saver": 0.30, "High Spender": 0.20, "Debt-Heavy Spender": 0.15}
WANTS_FLOOR = 0.10                    # never recommend squeezing wants below this
HIGH_RISK_WANTS_REDUCTION = 0.05      # extra squeeze for High credit risk
MIN_SAVINGS_RATE = {"Saver": 0.20, "High Spender": 0.20, "Debt-Heavy Spender": 0.15}

# Emergency fund target (months of expenses) by credit risk
EMERGENCY_MONTHS = {"Low": 3, "Medium": 6, "High": 6}

# Alert thresholds
SPEND_RATIO_WARN, SPEND_RATIO_HIGH, SPEND_RATIO_CRIT = 0.75, 0.90, 1.00
GROWTH_WARN, GROWTH_HIGH = 0.05, 0.10
DTI_WARN, DTI_CRIT = 0.30, 0.40
EMI_CRIT = 0.40
UTIL_WARN, UTIL_CRIT = 0.50, 0.75
EF_CRITICAL_MONTHS = 1          # below this -> Capital Preservation
LOW_SAVINGS_RATE = 0.10
DISCRETIONARY_WARN = 0.45

SEVERITY_ORDER = {"critical": 0, "high": 1, "warning": 2, "info": 3}


def inr(x: float) -> str:
    """Format a number as Indian-grouped rupees, e.g. 1234567 -> ₹12,34,567."""
    x = int(round(x))
    sign = "-" if x < 0 else ""
    s = str(abs(x))
    if len(s) > 3:
        head, tail = s[:-3], s[-3:]
        head = re.sub(r"(\d)(?=(\d\d)+$)", r"\1,", head)
        s = f"{head},{tail}"
    return f"{sign}₹{s}"


# --------------------------------------------------------------------------
# Input / output containers
# --------------------------------------------------------------------------
@dataclass
class UserFinancialInput:
    """One unified user record. Only the first five fields are mandatory."""
    # --- required ---
    age: int
    monthly_income: float
    previous_month_expense: float
    avg_3_month_expense: float
    savings_amount: float                       # amount saved per month
    # --- optional: spending detail ---
    food_expense: Optional[float] = None
    transport_expense: Optional[float] = None
    shopping_expense: Optional[float] = None
    entertainment_expense: Optional[float] = None
    medical_expense: Optional[float] = None
    education_expense: Optional[float] = None
    utilities_expense: Optional[float] = None
    other_expense: Optional[float] = None
    transaction_count: Optional[int] = None
    expense_growth_rate: Optional[float] = None
    # --- optional: debt / credit behaviour ---
    total_debt: float = 0.0
    existing_loan_amount: float = 0.0
    monthly_emi: float = 0.0
    loan_amount: float = 0.0
    loan_term: Optional[int] = None
    credit_limit: Optional[float] = None
    credit_utilization: Optional[float] = None
    late_payment_count: Optional[int] = None
    missed_payment_count: Optional[int] = None
    repayment_ratio: Optional[float] = None
    previous_loans: Optional[int] = None
    credit_history_years: Optional[int] = None
    number_of_credit_accounts: Optional[int] = None
    employment_years: Optional[int] = None
    bank_balance: Optional[float] = None
    # --- optional: assets ---
    investment_amount: float = 0.0              # invested per month
    emergency_fund: float = 0.0                 # current emergency corpus


@dataclass
class AdvisorReport:
    inputs_summary: Dict
    predictions: Dict
    metrics: Dict
    budget: Dict
    savings: Dict
    investment: Dict
    alerts: List[Dict]
    priority_actions: List[str]
    rules_triggered: List[str]
    assumptions: List[str]
    warnings: List[str]
    disclaimer: str = ("Academic ML decision-support output - not professional "
                       "financial advice.")
    chart_data: Dict = field(default_factory=dict)     # chart-ready numbers for the UI

    def to_dict(self) -> Dict:
        return asdict(self)

    def to_json(self, **kw) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False, default=float, **kw)

    # ------------------------------------------------------------------
    def summary_text(self) -> str:
        p, m = self.predictions, self.metrics
        L = []
        L.append("=" * 68)
        L.append("AI PERSONAL FINANCE ADVISOR - PERSONALISED REPORT")
        L.append("=" * 68)
        L.append(f"Monthly income        : {inr(self.inputs_summary['monthly_income'])}")
        L.append(f"Avg monthly expense   : {inr(self.inputs_summary['avg_3_month_expense'])}")
        L.append(f"Predicted expense     : {inr(p['predicted_expense'])}  "
                 f"({m['expense_change_pct']:+.1f}% vs 3-month avg)")
        L.append(f"Credit risk           : {p['credit_risk']}  "
                 f"(confidence {p['credit_risk_probs'][p['credit_risk']]*100:.0f}%)")
        L.append(f"Financial profile     : {p['financial_profile']}  "
                 f"(confidence {p['profile_confidence']*100:.0f}%)")
        L.append("")
        L.append("[BUDGET]     " + self.budget["message"])
        L.append("[SAVINGS]    " + self.savings["message"])
        L.append("[INVESTMENT] " + self.investment["message"])
        L.append("")
        L.append("ALERTS")
        if not self.alerts:
            L.append("  (none)")
        for a in self.alerts:
            L.append(f"  [{a['severity'].upper():8}] {a['message']}")
        L.append("")
        L.append("TOP ACTIONS")
        for i, a in enumerate(self.priority_actions, 1):
            L.append(f"  {i}. {a}")
        if self.assumptions:
            L.append("")
            L.append("DEFAULTS USED (not supplied by user): " + "; ".join(self.assumptions))
        if self.warnings:
            L.append("DATA WARNINGS: " + "; ".join(self.warnings))
        L.append("")
        L.append(self.disclaimer)
        return "\n".join(L)


# --------------------------------------------------------------------------
# The engine
# --------------------------------------------------------------------------
class FinancialAdvisor:
    def __init__(self, models_dir: Optional[str] = None):
        d = Path(models_dir) if models_dir else Path(__file__).parent / "models"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")           # sklearn version-skew noise
            load = lambda n: joblib.load(d / f"{n}.pkl")
            self.expense_model, self.expense_scaler = load("best_expense_model"), load("expense_scaler")
            self.expense_features = load("expense_features")
            self.credit_model, self.credit_scaler = load("best_credit_model"), load("credit_scaler")
            self.credit_features = load("credit_features")
            self.kmeans, self.svm = load("kmeans_financial_profile_model"), load("svm_financial_profile_model")
            self.profile_scaler = load("financial_profile_scaler")
            self.profile_features = load("financial_profile_features")
            self.saved_cluster_labels = load("financial_profile_cluster_labels")
        self.profile_labels = self._derive_profile_labels()

    # ---- cluster naming ---------------------------------------------------
    def _derive_profile_labels(self) -> Dict[int, str]:
        """
        The saved label file calls two clusters 'Debt-Heavy Spender'
        ({0: Debt-Heavy, 1: Saver, 2: Debt-Heavy}). Cluster 2 is actually the
        high-discretionary / high-transaction group, so labels are re-derived
        from the cluster centroids (robust if the model is retrained).
        """
        centers = pd.DataFrame(self.profile_scaler.inverse_transform(self.kmeans.cluster_centers_),
                               columns=self.profile_features)
        if len(centers) != 3:
            return dict(self.saved_cluster_labels)
        saver = int(centers["savings_rate"].idxmax())
        debt = int(centers.drop(index=saver)["debt_to_income_ratio"].idxmax())
        spender = [i for i in range(3) if i not in (saver, debt)][0]
        return {saver: "Saver", debt: "Debt-Heavy Spender", spender: "High Spender"}

    # ---- validation -------------------------------------------------------
    @staticmethod
    def _validate(u: UserFinancialInput) -> List[str]:
        errs, warns = [], []
        if not 18 <= u.age <= 100: errs.append("age must be between 18 and 100")
        if u.monthly_income <= 0: errs.append("monthly_income must be > 0")
        # savings_amount may be negative (overspending month) - that is a valid, important case
        if u.savings_amount < -u.monthly_income: errs.append("savings_amount cannot be below minus one month of income")
        for f in ("previous_month_expense", "avg_3_month_expense", "total_debt",
                  "monthly_emi", "investment_amount", "emergency_fund"):
            if getattr(u, f) < 0: errs.append(f"{f} cannot be negative")
        for c in ALL_CATS:
            v = getattr(u, f"{c}_expense")
            if v is not None and v < 0: errs.append(f"{c}_expense cannot be negative")
        for f in ("credit_utilization", "repayment_ratio"):
            v = getattr(u, f)
            if v is not None and not 0 <= v <= 1.5: errs.append(f"{f} should be a ratio between 0 and 1")
        if errs:
            raise ValueError("Invalid input: " + "; ".join(errs))
        if u.avg_3_month_expense <= 0: raise ValueError("avg_3_month_expense must be > 0")
        given = [getattr(u, f"{c}_expense") for c in ALL_CATS]
        if all(v is not None for v in given):
            tot = sum(given)
            if abs(tot - u.previous_month_expense) / max(u.previous_month_expense, 1) > 0.15:
                warns.append("category expenses differ from previous_month_expense by >15%")
        if not 0.4 <= u.previous_month_expense / u.avg_3_month_expense <= 2.5:
            warns.append("previous month expense is very different from the 3-month average")
        return warns

    # ---- feature builders -------------------------------------------------
    def _category_amounts(self, u: UserFinancialInput, assumptions: List[str]) -> Dict[str, float]:
        given = {c: getattr(u, f"{c}_expense") for c in ALL_CATS}
        if all(v is None for v in given.values()):
            assumptions.append("category split (dataset-average shares)")
            return {c: DEFAULT_CAT_SHARE[c] * u.previous_month_expense for c in ALL_CATS}
        missing = [c for c, v in given.items() if v is None]
        if missing:
            assumptions.append(f"missing categories set to 0: {', '.join(missing)}")
        return {c: (v or 0.0) for c, v in given.items()}

    def _expense_frame(self, u, cats, tx, growth, notes: List[str]) -> pd.DataFrame:
        """
        Build the 17 expense-model features.

        The saved LinearRegression was trained on data where
        savings_amount ~= income - previous_month_expense (corr 0.97) and the
        categories sum to previous_month_expense. That near-collinearity gives the
        model huge opposing weights, so inconsistent inputs make it explode
        (e.g. +-Rs 3 lakh swings). Features are therefore reconciled to that
        manifold before prediction; the user's real figures are still used by
        the credit/profile models and by the recommendation rules.
        """
        prev = u.previous_month_expense
        implied_savings = u.monthly_income - prev
        if abs(u.savings_amount - implied_savings) > 0.10 * u.monthly_income:
            notes.append(f"savings_amount ({inr(u.savings_amount)}) differs from income minus last month's "
                         f"expense ({inr(implied_savings)}); expense model was fed the reconciled value")
        cat_sum = sum(cats.values())
        scale = prev / cat_sum if cat_sum > 0 else 1.0
        row = {"age": u.age, "monthly_income": u.monthly_income,
               "previous_month_expense": prev,
               "avg_3_month_expense": u.avg_3_month_expense,
               **{f"{c}_expense": cats[c] * scale for c in ALL_CATS},
               "transaction_count": tx,
               "savings_amount": implied_savings,
               "savings_rate": implied_savings / u.monthly_income,
               "expense_growth_rate": growth,
               # engineered feature (verified against saved scaler: mean~10, std~1220)
               "expense_deviation": prev - u.avg_3_month_expense}
        return pd.DataFrame([row])[self.expense_features]

    def _credit_frame(self, u, assumptions) -> pd.DataFrame:
        def opt(name, default):
            v = getattr(u, name)
            if v is None:
                assumptions.append(f"{name}={default:g}")
                return default
            return v
        hist_cap = max(u.age - 18, 0)
        row = {"age": u.age, "monthly_income": u.monthly_income,
               "employment_years": min(opt("employment_years", CREDIT_MEDIANS["employment_years"]), hist_cap),
               "existing_loan_amount": u.existing_loan_amount, "monthly_emi": u.monthly_emi,
               "total_debt": u.total_debt,
               "credit_limit": opt("credit_limit", CREDIT_MEDIANS["credit_limit"]),
               "credit_utilization": opt("credit_utilization", CREDIT_MEDIANS["credit_utilization"]),
               "previous_loans": opt("previous_loans", CREDIT_MEDIANS["previous_loans"]),
               "late_payment_count": opt("late_payment_count", CREDIT_MEDIANS["late_payment_count"]),
               "missed_payment_count": opt("missed_payment_count", CREDIT_MEDIANS["missed_payment_count"]),
               "repayment_ratio": opt("repayment_ratio", CREDIT_MEDIANS["repayment_ratio"]),
               "debt_to_income_ratio": u.total_debt / (u.monthly_income * 12),
               "savings_amount": u.savings_amount,
               "bank_balance": opt("bank_balance", CREDIT_MEDIANS["bank_balance"]),
               "credit_history_years": min(opt("credit_history_years", CREDIT_MEDIANS["credit_history_years"]), hist_cap),
               "number_of_credit_accounts": opt("number_of_credit_accounts", CREDIT_MEDIANS["number_of_credit_accounts"]),
               "loan_amount": u.loan_amount,
               "loan_term": u.loan_term if u.loan_term is not None else (CREDIT_MEDIANS["loan_term"] if u.loan_amount > 0 else 0)}
        return pd.DataFrame([row])[self.credit_features]

    def _profile_frame(self, u, cats, tx) -> pd.DataFrame:
        total = max(sum(cats.values()), 1.0)
        ess = sum(cats[c] for c in ESSENTIAL_CATS) / total
        monthly_expense = u.avg_3_month_expense
        row = {"monthly_income": u.monthly_income, "monthly_expense": monthly_expense,
               "savings_amount": u.savings_amount, "savings_rate": u.savings_amount / u.monthly_income,
               "essential_expense_ratio": ess, "discretionary_expense_ratio": 1 - ess,
               "food_ratio": cats["food"] / total, "shopping_ratio": cats["shopping"] / total,
               "entertainment_ratio": cats["entertainment"] / total,
               "transaction_frequency": tx, "debt_amount": u.total_debt,
               "debt_to_income_ratio": u.total_debt / (u.monthly_income * 12),
               "investment_amount": u.investment_amount, "emergency_fund": u.emergency_fund}
        return pd.DataFrame([row])[self.profile_features]

    # ---- main entry point -------------------------------------------------
    def analyze(self, user: UserFinancialInput) -> AdvisorReport:
        data_warnings = self._validate(user)
        assumptions: List[str] = []
        cats = self._category_amounts(user, assumptions)
        tx = user.transaction_count
        if tx is None:
            tx = 30; assumptions.append("transaction_count=30")
        growth = user.expense_growth_rate
        if growth is None:
            growth = 0.0; assumptions.append("expense_growth_rate=0")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # [1] expense
            Xe = self.expense_scaler.transform(self._expense_frame(user, cats, tx, growth, data_warnings))
            raw_pred = float(self.expense_model.predict(Xe)[0])
            lo = PRED_BAND_LOW * min(user.previous_month_expense, user.avg_3_month_expense)
            hi = PRED_BAND_HIGH * max(user.previous_month_expense, user.avg_3_month_expense)
            predicted = float(min(max(raw_pred, lo), hi))
            pred_clamped = abs(predicted - raw_pred) > 1e-6
            if pred_clamped:
                data_warnings.append(f"expense model output {inr(raw_pred)} was outside the plausible band "
                                     f"({inr(lo)}-{inr(hi)}) and was clamped - input may be out of training range")
            # [2] credit risk
            Xc = self.credit_scaler.transform(self._credit_frame(user, assumptions))
            probs = self.credit_model.predict_proba(Xc)[0]
            cls = self.credit_model.classes_
            risk_probs = {RISK_LABELS[int(c)]: float(p) for c, p in zip(cls, probs)}
            risk = RISK_LABELS[int(cls[int(np.argmax(probs))])]
            # [3] profile (KMeans = primary, SVM = confidence / cross-check)
            Xp = self.profile_scaler.transform(self._profile_frame(user, cats, tx))
            k = int(self.kmeans.predict(Xp)[0])
            svm_probs = self.svm.predict_proba(Xp)[0]
            svm_cls = list(self.svm.classes_)
            profile = self.profile_labels[k]
            profile_conf = float(svm_probs[svm_cls.index(k)])
            svm_agrees = int(self.svm.predict(Xp)[0]) == k
        n_defaulted = sum(a.split("=")[0] in ("credit_utilization", "late_payment_count",
                                              "missed_payment_count", "repayment_ratio") for a in assumptions)
        if n_defaulted >= 3:
            data_warnings.append("credit risk is based mostly on dataset-median defaults (payment history not "
                                 "provided) - provide utilisation, late/missed payments and repayment ratio for a reliable estimate")
        if not svm_agrees:
            data_warnings.append("KMeans and SVM disagree on the profile - treat profile as low confidence")

        preds = {"predicted_expense": predicted, "expense_model_raw": raw_pred, "expense_clamped": pred_clamped,
                  "credit_risk": risk, "credit_risk_probs": risk_probs,
                 "financial_profile": profile, "profile_confidence": profile_conf,
                 "profile_cluster_id": k, "profile_models_agree": svm_agrees}
        report = self._recommend(user, cats, preds, assumptions, data_warnings)
        report.chart_data["profile_compare"] = self._profile_compare(self._profile_frame(user, cats, tx))
        return report

    PROFILE_COMPARE_FEATURES = ["savings_rate", "debt_to_income_ratio", "discretionary_expense_ratio",
                                "shopping_ratio", "entertainment_ratio"]

    def _profile_compare(self, frame: pd.DataFrame) -> Dict:
        """User's profile ratios next to each cluster's average (for a grouped bar chart)."""
        centers = pd.DataFrame(self.profile_scaler.inverse_transform(self.kmeans.cluster_centers_),
                               columns=self.profile_features)
        f = self.PROFILE_COMPARE_FEATURES
        return {"features": f, "user": [float(frame.iloc[0][c]) for c in f],
                "clusters": {self.profile_labels[i]: [float(centers.loc[i, c]) for c in f] for i in range(len(centers))}}

    # ----------------------------------------------------------------------
    # Rule-based recommendation layer
    # ----------------------------------------------------------------------
    def _recommend(self, u, cats, preds, assumptions, data_warnings) -> AdvisorReport:
        I, E = u.monthly_income, preds["predicted_expense"]
        risk, profile = preds["credit_risk"], preds["financial_profile"]
        rules: List[str] = []
        user_gave = lambda f: getattr(u, f) is not None

        # ---- derived metrics ----
        cat_total = max(sum(cats.values()), 1.0)
        needs_share = sum(cats[c] for c in ESSENTIAL_CATS) / cat_total
        needs_pred = E * needs_share
        wants_pred = E - needs_pred
        spend_ratio = E / I
        change = E / u.avg_3_month_expense - 1
        dti = u.total_debt / (I * 12)
        emi_ratio = u.monthly_emi / I
        proj_savings = I - E
        proj_savings_rate = proj_savings / I
        ef_months = u.emergency_fund / E if E > 0 else 0.0
        target_months = EMERGENCY_MONTHS[risk]
        if profile == "Debt-Heavy Spender":
            target_months = max(target_months, 6)
        metrics = {"spend_to_income_ratio": spend_ratio, "expense_change_pct": change * 100,
                   "debt_to_income_ratio": dti, "emi_to_income_ratio": emi_ratio,
                   "projected_monthly_surplus": proj_savings, "projected_savings_rate": proj_savings_rate,
                   "emergency_fund_months": ef_months, "emergency_target_months": target_months,
                   "essential_share": needs_share, "discretionary_share": 1 - needs_share}

        # ---------------- BUDGET ----------------
        wants_cap_pct = WANTS_CAP[profile]
        rules.append(f"profile={profile} -> wants cap {wants_cap_pct:.0%} of income")
        if risk == "High":
            wants_cap_pct = max(WANTS_FLOOR, wants_cap_pct - HIGH_RISK_WANTS_REDUCTION)
            rules.append(f"credit risk=High -> wants cap tightened to {wants_cap_pct:.0%}")
        wants_cap = wants_cap_pct * I
        wants_cut = max(0.0, wants_pred - wants_cap)
        wants_cats = {c: cats[c] / cat_total * E for c in DISCRETIONARY_CATS}
        wants_sum = max(sum(wants_cats.values()), 1.0)
        category_cuts = {c: round(wants_cut * v / wants_sum, 0) for c, v in wants_cats.items()} if wants_cut > 0 else {}
        pool = max(0.0, I - needs_pred - min(wants_pred, wants_cap))   # savings + debt pool
        budget = {"needs_budget": needs_pred, "wants_budget": min(wants_pred, wants_cap),
                  "savings_and_debt_budget": pool, "wants_cap": wants_cap,
                  "discretionary_reduction_needed": wants_cut, "category_reductions": category_cuts}
        if needs_pred >= I:
            rules.append("essential spending >= income")
            budget["message"] = (f"Essential spending alone ({inr(needs_pred)}) consumes the full income. "
                                 "Review fixed costs (rent, utilities, EMIs) or raise income before any discretionary spend.")
        elif wants_cut > 0:
            rules.append("discretionary spending above cap -> reduce")
            top = max(category_cuts, key=category_cuts.get)
            budget["message"] = (f"Reduce discretionary spending by about {inr(wants_cut)}/month "
                                 f"(projected {inr(wants_pred)} vs cap {inr(wants_cap)}); biggest saving: {top} "
                                 f"({inr(category_cuts[top])}). Projected expenses "
                                 f"{'are increasing' if change >= GROWTH_WARN else 'look stable'}.")
        else:
            rules.append("discretionary spending within cap")
            budget["message"] = (f"Discretionary spending ({inr(wants_pred)}) is within the {inr(wants_cap)} cap - "
                                 "maintain the current budget and avoid lifestyle inflation.")

        # ---------------- SAVINGS ----------------
        ef_target = target_months * E
        ef_gap = max(0.0, ef_target - u.emergency_fund)
        min_rate = MIN_SAVINGS_RATE[profile]
        target_savings = max(min_rate * I, 0.0)
        debt_priority = risk == "High" or profile == "Debt-Heavy Spender" or dti > DTI_WARN
        if debt_priority:
            rules.append("debt priority (High risk / Debt-Heavy / DTI>30%)")
            alloc_debt, rest = 0.5 * pool, 0.5 * pool
            if ef_gap <= 0: alloc_debt, rest = 0.7 * pool, 0.3 * pool
            alloc = {"debt_repayment": alloc_debt, "emergency_fund": rest if ef_gap > 0 else 0.0,
                     "investments": rest if ef_gap <= 0 else 0.0}
        elif ef_gap > 0:
            rules.append("emergency fund below target -> build fund first")
            alloc = {"debt_repayment": 0.0, "emergency_fund": 0.8 * pool, "investments": 0.2 * pool}
        else:
            rules.append("emergency fund met -> growth allocation")
            alloc = {"debt_repayment": 0.0, "emergency_fund": 0.0, "investments": pool}
        months_to_goal = (ef_gap / alloc["emergency_fund"]) if alloc["emergency_fund"] > 0 and ef_gap > 0 else None
        savings = {"target_savings_rate": min_rate, "target_monthly_savings": target_savings,
                   "recommended_allocation": alloc, "emergency_fund_target": ef_target,
                   "emergency_fund_gap": ef_gap, "months_to_emergency_goal": months_to_goal}
        if ef_gap > 0:
            when = f"~{months_to_goal:.0f} months at the suggested pace" if months_to_goal else "once surplus is available"
            savings["message"] = (f"Build an emergency fund of {inr(ef_target)} ({target_months} months of expenses); "
                                  f"current gap {inr(ef_gap)}, reachable {when}. "
                                  f"Increase emergency savings before increasing discretionary spending.")
        else:
            savings["message"] = (f"Emergency fund target of {inr(ef_target)} is met. Keep saving at least "
                                  f"{min_rate:.0%} of income ({inr(target_savings)}/month).")
        if debt_priority and alloc["debt_repayment"] > 0:
            savings["message"] += f" Direct about {inr(alloc['debt_repayment'])}/month to debt repayment."

        # ---------------- INVESTMENT CATEGORY ----------------
        if risk == "High" or ef_months < EF_CRITICAL_MONTHS or dti > DTI_CRIT:
            cat_name = "Capital Preservation"
            mix = {"Emergency fund (savings account / liquid fund / FD)": 70, "High-interest debt prepayment": 30}
            reasons = ([f"credit risk is High"] if risk == "High" else []) + \
                      ([f"emergency fund covers only {ef_months:.1f} months"] if ef_months < EF_CRITICAL_MONTHS else []) + \
                      ([f"debt is {dti:.0%} of annual income"] if dti > DTI_CRIT else [])
            why = "; ".join(reasons)
        elif risk == "Medium" or ef_months < target_months or profile == "Debt-Heavy Spender":
            cat_name = "Conservative"
            mix = {"Debt funds / FD / PPF": 70, "Large-cap index fund SIP": 20, "Gold": 10}
            reasons = ([f"credit risk is Medium"] if risk == "Medium" else []) + \
                      ([f"emergency fund ({ef_months:.1f} months) is below the {target_months}-month target"] if ef_months < target_months else []) + \
                      (["profile is Debt-Heavy Spender"] if profile == "Debt-Heavy Spender" else [])
            why = "; ".join(reasons)
        elif profile == "High Spender" or proj_savings_rate < 0.20:
            cat_name = "Balanced"
            mix = {"Debt funds / PPF": 45, "Index fund SIP (auto-debit)": 40, "Gold": 15}
            why = "credit risk is Low and emergency fund is funded, but spending discipline is still being built - automate SIPs"
        else:
            cat_name = "Growth"
            mix = {"Diversified equity / index funds": 60, "Debt funds / PPF": 30, "Gold": 10}
            why = "credit risk is Low, savings rate is healthy and the emergency reserve is funded"
        rules.append(f"investment category -> {cat_name}")
        investment = {"category": cat_name, "indicative_mix_pct": mix, "rationale": why,
                      "monthly_investable": alloc["investments"],
                      "message": (f"{cat_name} - {why}. Prioritise "
                                  f"{'lower-risk options until financial stability improves' if cat_name in ('Capital Preservation', 'Conservative') else 'regular SIP investing'}.")}

        # ---------------- ALERTS ----------------
        alerts: List[Dict] = []
        add = lambda sev, code, msg: alerts.append({"severity": sev, "code": code, "message": msg})
        if spend_ratio >= SPEND_RATIO_CRIT:
            add("critical", "DEFICIT", f"Projected spending {inr(E)} exceeds income {inr(I)} - a monthly deficit of {inr(E - I)}.")
        elif spend_ratio >= SPEND_RATIO_HIGH:
            add("high", "SPEND_NEAR_INCOME", f"Projected spending is {spend_ratio:.0%} of income - almost no buffer left.")
        elif spend_ratio >= SPEND_RATIO_WARN:
            add("warning", "SPEND_APPROACHING_LIMIT", f"Projected spending ({inr(E)}) is approaching the income limit ({spend_ratio:.0%} of income).")
        if change >= GROWTH_HIGH:
            add("high", "EXPENSE_SURGE", f"Expenses are projected to jump {change:.0%} above the 3-month average.")
        elif change >= GROWTH_WARN:
            add("warning", "EXPENSE_RISING", f"Expenses are trending up (+{change:.0%} vs 3-month average).")
        if dti > DTI_CRIT:
            add("critical", "DTI_HIGH", f"Debt is {dti:.0%} of annual income (>{DTI_CRIT:.0%}).")
        elif dti > DTI_WARN:
            add("warning", "DTI_ELEVATED", f"Debt is {dti:.0%} of annual income (>{DTI_WARN:.0%}).")
        if emi_ratio > EMI_CRIT:
            add("critical", "EMI_BURDEN", f"EMIs take {emi_ratio:.0%} of monthly income.")
        if user_gave("credit_utilization"):
            if u.credit_utilization > UTIL_CRIT:
                add("critical", "UTILISATION_HIGH", f"Credit utilisation is {u.credit_utilization:.0%} - pay cards down below 30%.")
            elif u.credit_utilization > UTIL_WARN:
                add("warning", "UTILISATION_ELEVATED", f"Credit utilisation is {u.credit_utilization:.0%}; aim below 30%.")
        if user_gave("missed_payment_count") and u.missed_payment_count > 0:
            add("high", "MISSED_PAYMENTS", f"{u.missed_payment_count} missed payment(s) on record - set up auto-pay immediately.")
        elif user_gave("late_payment_count") and u.late_payment_count >= 3:
            add("warning", "LATE_PAYMENTS", f"{u.late_payment_count} late payments on record.")
        if ef_months < 3:
            add("warning", "LOW_EMERGENCY_FUND", f"Emergency fund covers only {ef_months:.1f} months of expenses (minimum 3).")
        if proj_savings_rate < LOW_SAVINGS_RATE:
            add("warning", "LOW_SAVINGS", f"Projected savings rate is {proj_savings_rate:.0%} (<{LOW_SAVINGS_RATE:.0%}).")
        if (1 - needs_share) > DISCRETIONARY_WARN:
            add("info", "DISCRETIONARY_HEAVY", f"Discretionary spending is {1 - needs_share:.0%} of total expenses.")
        if risk == "High":
            add("high", "CREDIT_RISK_HIGH", "Credit-risk model rates this profile High risk.")
        alerts.sort(key=lambda a: SEVERITY_ORDER[a["severity"]])
        if alerts: rules.append("alerts fired: " + ", ".join(a["code"] for a in alerts))

        # ---------------- PRIORITY ACTIONS (max 3, ordered) ----------------
        codes = {a["code"] for a in alerts}
        actions: List[str] = []
        if "DEFICIT" in codes or needs_pred >= I:
            actions.append(f"Close the monthly gap: cut {inr(max(E - I, wants_cut))} of spending (start with discretionary items).")
        if "MISSED_PAYMENTS" in codes:
            actions.append("Clear overdue payments and enable auto-pay to protect your credit score.")
        if debt_priority and alloc["debt_repayment"] > 0:
            actions.append(f"Put {inr(alloc['debt_repayment'])}/month towards the highest-interest debt first.")
        if wants_cut > 0 and not any("Close the monthly gap" in a for a in actions):
            actions.append(f"Trim discretionary spending by {inr(wants_cut)}/month ({', '.join(f'{c} -{inr(v)}' for c, v in sorted(category_cuts.items(), key=lambda x: -x[1]) if v > 0)}).")
        if ef_gap > 0 and alloc["emergency_fund"] > 0:
            actions.append(f"Save {inr(alloc['emergency_fund'])}/month into a liquid emergency fund until it reaches {inr(ef_target)}.")
        if not actions or (ef_gap <= 0 and alloc["investments"] > 0):
            actions.append(f"Invest {inr(alloc['investments'])}/month via a {cat_name.lower()} mix.")
        actions = actions[:3]

        # ---------------- CHART DATA (for Streamlit / any UI) ----------------
        cat_pred = {c: cats[c] / cat_total * E for c in ALL_CATS}
        cat_rec = {c: cat_pred[c] - category_cuts.get(c, 0.0) for c in ALL_CATS}
        chart_data = {
            "expense_trend": {"previous_month": u.previous_month_expense, "avg_3_month": u.avg_3_month_expense,
                              "predicted_next_month": E, "income": I},
            "spending_now": {"Needs": needs_pred, "Wants": wants_pred, "Left over": max(I - E, 0.0),
                             "Overspend": max(E - I, 0.0)},
            "income_plan": {"Needs": budget["needs_budget"], "Wants": budget["wants_budget"],
                            "Debt repayment": alloc["debt_repayment"], "Emergency fund": alloc["emergency_fund"],
                            "Investments": alloc["investments"]},
            "category_current": cat_pred, "category_recommended": cat_rec,
            "emergency_fund": {"current": u.emergency_fund, "target": ef_target,
                               "monthly_contribution": alloc["emergency_fund"]},
        }
        return AdvisorReport(
            inputs_summary={"monthly_income": I, "avg_3_month_expense": u.avg_3_month_expense,
                            "previous_month_expense": u.previous_month_expense, "age": u.age},
            predictions=preds, metrics=metrics, budget=budget, savings=savings,
            investment=investment, alerts=alerts, priority_actions=actions,
            rules_triggered=rules, assumptions=assumptions, warnings=data_warnings,
            chart_data=chart_data)


# --------------------------------------------------------------------------
def advise_from_dict(d: Dict, models_dir: Optional[str] = None) -> AdvisorReport:
    """Convenience wrapper: dict (e.g. parsed JSON / Streamlit form) -> report."""
    return FinancialAdvisor(models_dir).analyze(UserFinancialInput(**d))


if __name__ == "__main__":
    import argparse, sys
    for _s in (sys.stdout, sys.stderr):      # Windows consoles default to cp1252 and cannot print the rupee sign
        try: _s.reconfigure(encoding="utf-8")
        except Exception: pass
    ap = argparse.ArgumentParser(description="AI Personal Finance Advisor - recommendation engine")
    ap.add_argument("profile_json", help="path to a JSON file with the user's financial data")
    ap.add_argument("--json", action="store_true", help="print full JSON instead of text report")
    a = ap.parse_args()
    rep = advise_from_dict(json.loads(Path(a.profile_json).read_text()))
    print(rep.to_json() if a.json else rep.summary_text())