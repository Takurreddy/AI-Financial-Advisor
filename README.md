# Recommendation Engine (Member 4) - AI Personal Finance Advisor

Combines the **outputs** of the three trained models (datasets are never merged) with a transparent rule layer.

```
UserFinancialInput ─► validate ─► feature builders ─┬─► Expense model (LinearRegression)  → predicted expense
                                                    ├─► Credit model  (LogisticRegression) → Low/Medium/High (+probs)
                                                    └─► KMeans (+SVM confidence)           → Saver / High Spender / Debt-Heavy Spender
                                                                       │
                                  rule-based aggregation ◄─────────────┘
                                                                       ▼
              AdvisorReport: budget · savings · investment category · alerts · top-3 actions · rules triggered · assumptions
```

## Run
```bash
pip install -r requirements.txt
python demo.py                                   # 4 example personas (incl. slide-7 example)
python recommendation_engine.py sample_profile.json [--json]
streamlit run app.py                             # UI
pytest -q test_engine.py                         # 16 tests (set FINANCE_DATA_DIR to the CSV folder)
```
```python
from recommendation_engine import FinancialAdvisor, UserFinancialInput
report = FinancialAdvisor().analyze(UserFinancialInput(age=32, monthly_income=40000,
         previous_month_expense=29000, avg_3_month_expense=28000, savings_amount=3000))
print(report.summary_text());  report.to_dict()
```
Only 5 fields are required; omitted ones fall back to dataset medians and are listed under `assumptions`.

## Rule summary (all thresholds are constants at the top of `recommendation_engine.py`)
| Output | Logic |
|---|---|
| **Budget** | Needs = essential share of predicted expense. Wants cap = 30% (Saver) / 20% (High Spender) / 15% (Debt-Heavy) of income, −5 pts if credit risk High (floor 10%). Excess is split across shopping/entertainment/other. |
| **Savings** | Emergency target = 3 months (Low risk) or 6 months (Medium/High or Debt-Heavy). Surplus pool split: debt-priority (High risk / Debt-Heavy / DTI>30%) → 50-70% to debt; else emergency fund first, then investing. |
| **Investment category** | Capital Preservation (High risk, EF<1 month, or DTI>40%) → Conservative (Medium risk, EF below target, or Debt-Heavy) → Balanced (High Spender or savings<20%) → Growth. |
| **Alerts** | Deficit, spend ≥75/90/100% of income, expense growth ≥5/10%, DTI >30/40%, EMI >40%, utilisation >50/75%, missed/late payments, EF <3 months, savings <10%, discretionary >45%, High credit risk. |

## Issues found in the upstream models (please share with Members 1-3)
1. **Expense model is fragile (Member 1).** In the data `savings_amount ≈ income − previous_month_expense` (corr 0.97), and categories sum to previous expense. The LinearRegression therefore has huge opposing coefficients; changing only `savings_amount` on one row swings the prediction from −₹32,889 to +₹307,515. Two features (previous + 3-month avg) already give R²≈0.975 vs 0.984 for all 17. *Engine workaround:* reconcile features to the training manifold + clamp to 0.75–1.35× recent spend. *Recommended fix:* retrain with Ridge/XGBoost or drop `savings_amount`, `savings_rate`, and the category columns.
2. **`expense_deviation` was not saved/documented (Member 1).** It is not in the CSV; verified from the scaler's stored mean/std that it equals `previous_month_expense − avg_3_month_expense`.
3. **Cluster labels bug (Member 3).** `financial_profile_cluster_labels.pkl` = {0: Debt-Heavy Spender, 1: Saver, 2: Debt-Heavy Spender}. Cluster 2 is a high-discretionary / high-transaction group (shopping+entertainment ≈ 43% of spend, ~94 transactions/month, DTI only 0.14) → "High Spender". The engine re-derives labels from centroids; please fix the pickle.
4. **Credit model probabilities are over-confident (Member 2)** (often 100%). Accuracy ≈ 90% on a 2,000-row sample; class mapping 0/1/2 → Low/Medium/High verified via late-payment/repayment averages. Consider calibration (`CalibratedClassifierCV`).

## Limitations
Rule thresholds are reasonable defaults, not tuned against outcomes. Stress-test users are stitched from independent datasets, so output mix there validates robustness, not realism. Investment mixes are generic category guidance (Indian instruments), not product advice.
