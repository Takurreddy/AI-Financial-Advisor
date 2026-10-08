# Recommendation Engine — Run & Test Report

**Date:** 2025-07-14  
**Environment:** Windows, Python 3.14, scikit-learn 1.8.0

---

## 1. Setup

```
pip install -r requirements.txt   ✅  (all packages installed cleanly)
```

---

## 2. Demo Run (`python demo.py`)

Cluster labels re-derived from centroids (engine bug-fix for upstream pickle):
```
{1: 'Saver', 0: 'Debt-Heavy Spender', 2: 'High Spender'}
```

### Persona A — Slide-7 example (income ₹40k)
| Field | Value |
|---|---|
| Predicted expense | ₹28,894 (+3.2% vs 3-month avg) |
| Credit risk | Low (100% confidence) |
| Financial profile | Debt-Heavy Spender (81% confidence) |
| Investment category | Conservative |
| Alerts | Credit utilisation 55% (warn), Emergency fund 1.4 months (warn) |
| Top action | ₹7,535/month to debt; trim discretionary by ₹3,964 |

### Persona B — Disciplined Saver (income ₹90k)
| Field | Value |
|---|---|
| Predicted expense | ₹53,045 (+4.0% vs 3-month avg) |
| Credit risk | Low (100% confidence) |
| Financial profile | Saver (100% confidence) |
| Investment category | **Growth** |
| Alerts | None |
| Top action | Invest ₹36,955/month via a growth mix |

### Persona C — Debt-stressed, missed payments (income ₹35k)
| Field | Value |
|---|---|
| Predicted expense | ₹34,830 (+2.4% vs 3-month avg) |
| Credit risk | **High** (100% confidence) |
| Financial profile | Debt-Heavy Spender (100% confidence) |
| Investment category | **Capital Preservation** |
| Alerts | 2× CRITICAL (debt 90% of annual income, utilisation 88%), 3× HIGH (spend at 100% of income, 3 missed payments, High credit risk), 2× WARNING |
| Top action | Clear overdue payments + enable auto-pay |

### Persona D — Minimal input (5 fields only)
| Field | Value |
|---|---|
| Predicted expense | ₹22,011 (+4.8% vs 3-month avg) |
| Credit risk | Low (100% confidence) |
| Financial profile | Saver (100% confidence) |
| Investment category | Capital Preservation (no emergency fund) |
| Alerts | Emergency fund 0.0 months (warn) |
| Assumptions used | Category split, transaction count, all credit fields — all from dataset medians |

---

## 3. Sample Profile Run (`python recommendation_engine.py sample_profile.json`)

Output matches Persona A exactly (same input values). ✅

---

## 4. Test Suite (`pytest -q test_engine.py`)

| Test | Result |
|---|---|
| `test_inr_formatting` | ✅ PASSED |
| `test_cluster_labels_are_distinct_and_semantic` | ✅ PASSED |
| `test_report_structure_and_money_invariants` | ✅ PASSED |
| `test_saver_gets_growth_and_no_alerts_stressed_gets_critical` | ✅ PASSED |
| `test_expense_prediction_close_to_recent_spending` | ✅ PASSED |
| `test_inconsistent_savings_does_not_break_expense_model` | ✅ PASSED |
| `test_guardrail_clamps_out_of_range` | ✅ PASSED |
| `test_validation_rejects_bad_input[bad0]` (age=10) | ✅ PASSED |
| `test_validation_rejects_bad_input[bad1]` (income=0) | ✅ PASSED |
| `test_validation_rejects_bad_input[bad2]` (debt=-1) | ✅ PASSED |
| `test_validation_rejects_bad_input[bad3]` (utilisation=3.0) | ✅ PASSED |
| `test_validation_rejects_bad_input[bad4]` (food=-5) | ✅ PASSED |
| `test_minimal_input_runs_and_reports_assumptions` | ✅ PASSED |
| `test_deficit_user_triggers_deficit_alert` | ✅ PASSED |
| `test_negative_savings_is_accepted_and_flagged` | ✅ PASSED |
| `test_stress_on_3000_stitched_users_from_real_datasets` | ⏭ SKIPPED (CSV datasets not present) |

**Result: 14 passed, 1 skipped (dataset-dependent stress test), 1 deprecation warning (pytest-asyncio / Python 3.16 future removal — harmless)**

---

## 5. Observations & Known Issues

### Works correctly
- Engine re-derives cluster labels from centroids, fixing the upstream pickle bug (cluster 2 correctly identified as "High Spender" not "Debt-Heavy Spender").
- Expense model reconciliation prevents the ±₹300k swing from inconsistent `savings_amount` values.
- Guardrail clamps out-of-distribution predictions to 0.75–1.35× recent spend.
- Budget, savings, and investment logic all produce internally consistent numbers (budget components sum ≤ income; allocation ≤ savings budget).
- Minimal-input mode fills missing fields from dataset medians and lists every assumption transparently.

### Known issues (upstream models — flagged in README)
1. **Expense model fragility (Member 1):** Raw model has huge opposing coefficients due to near-perfect collinearity (`savings_amount ≈ income − expense`). Engine workaround (reconcile + clamp) is effective but a retrain with Ridge/XGBoost is recommended.
2. **`expense_deviation` not in CSV (Member 1):** Engine re-derives it as `previous_month_expense − avg_3_month_expense`. Member 1 should document/save this feature.
3. **Cluster label pickle bug (Member 3):** `financial_profile_cluster_labels.pkl` maps cluster 2 → "Debt-Heavy Spender" (wrong). Engine overrides this. Member 3 should fix the pickle.
4. **Over-confident credit probabilities (Member 2):** Model outputs 100% confidence on almost every prediction. `CalibratedClassifierCV` is recommended.

### Windows-specific note
Running `python demo.py` directly on Windows fails with `UnicodeEncodeError` on the ₹ symbol. Fix: set `PYTHONIOENCODING=utf-8` before running, or add `sys.stdout.reconfigure(encoding='utf-8')` at the top of `demo.py`.

---

## 6. Summary

The recommendation engine is **fully functional**. All 14 runnable tests pass. The four demo personas produce sensible, internally consistent financial advice. The one skipped test (`test_stress_on_3000_stitched_users_from_real_datasets`) only requires the raw CSV files to be placed in a `data/` folder or pointed to via `FINANCE_DATA_DIR`.
