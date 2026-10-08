"""Run the recommendation engine on four example personas (incl. the slide-7 example)."""
import sys
for _s in (sys.stdout, sys.stderr):
    try: _s.reconfigure(encoding="utf-8")
    except Exception: pass
from recommendation_engine import FinancialAdvisor, UserFinancialInput

PERSONAS = {
 "A. Slide-7 example (income 40k, avg expense 28k)": UserFinancialInput(
    age=32, monthly_income=40000, previous_month_expense=29000, avg_3_month_expense=28000,
    savings_amount=3000, food_expense=8000, transport_expense=3500, shopping_expense=5000,
    entertainment_expense=3000, medical_expense=2000, education_expense=1500, utilities_expense=4000,
    other_expense=2000, transaction_count=45, total_debt=120000, monthly_emi=3500, credit_utilization=0.55,
    late_payment_count=2, missed_payment_count=0, repayment_ratio=0.85, emergency_fund=40000, investment_amount=1000),
 "B. Disciplined saver": UserFinancialInput(
    age=38, monthly_income=90000, previous_month_expense=52000, avg_3_month_expense=51000,
    savings_amount=26000, food_expense=14000, transport_expense=6000, shopping_expense=5000,
    entertainment_expense=3000, medical_expense=5000, education_expense=8000, utilities_expense=9000,
    other_expense=2000, transaction_count=28, total_debt=40000, monthly_emi=0, credit_utilization=0.12,
    late_payment_count=0, missed_payment_count=0, repayment_ratio=1.0, emergency_fund=450000, investment_amount=15000),
 "C. Debt-stressed, missed payments": UserFinancialInput(
    age=45, monthly_income=35000, previous_month_expense=36500, avg_3_month_expense=34000,
    savings_amount=500, food_expense=10000, transport_expense=3000, shopping_expense=4500,
    entertainment_expense=2500, medical_expense=3500, education_expense=3000, utilities_expense=5000,
    other_expense=5000, transaction_count=40, total_debt=380000, monthly_emi=14000, credit_utilization=0.88,
    late_payment_count=7, missed_payment_count=3, repayment_ratio=0.45, emergency_fund=5000),
 "D. Minimal input (only 5 required fields)": UserFinancialInput(
    age=26, monthly_income=30000, previous_month_expense=22000, avg_3_month_expense=21000, savings_amount=4000),
}

if __name__ == "__main__":
    adv = FinancialAdvisor()
    print("Cluster labels used:", adv.profile_labels, "\n")
    for name, user in PERSONAS.items():
        print("#" * 68, "\n#", name)
        print(adv.analyze(user).summary_text(), "\n")