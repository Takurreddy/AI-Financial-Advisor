AI PERSONAL FINANCE ADVISOR
Recommendation Engine

OVERVIEW
This project combines the outputs of three trained models with a transparent rule-based layer. The datasets remain separate.

The engine uses:
- A Linear Regression expense model to predict expenses.
- A Logistic Regression credit model to estimate Low, Medium, or High credit risk.
- KMeans and an SVM confidence model to identify a Saver, High Spender, or Debt-Heavy Spender profile.
- Rule-based logic to produce a budget, savings guidance, an investment category, alerts, and recommended actions.

INSTALLATION
Install the required Python packages:

    pip install -r requirements.txt

RUN THE PROJECT
Run the example personas:

    python demo.py

Analyze the sample profile:

    python recommendation_engine.py sample_profile.json

To print the report as JSON, add the --json option:

    python recommendation_engine.py sample_profile.json --json

Start the Streamlit application:

    streamlit run app.py

EXAMPLE
The following Python example analyzes a financial profile:

    from recommendation_engine import FinancialAdvisor, UserFinancialInput

    profile = UserFinancialInput(
        age=32,
        monthly_income=40000,
        previous_month_expense=29000,
        avg_3_month_expense=28000,
        savings_amount=3000
    )

    report = FinancialAdvisor().analyze(profile)
    print(report.summary_text())
    print(report.to_dict())

Only five fields are required. Missing values use dataset medians and are listed in the report assumptions.

RULE SUMMARY
Budget: Essential expenses are based on predicted expenses. The wants limit is 30 percent of income for Savers, 20 percent for High Spenders, and 15 percent for Debt-Heavy Spenders. High credit risk lowers the limit by five percentage points, with a minimum of 10 percent. Any excess is divided among shopping, entertainment, and other spending.

Savings: The emergency fund target is three months of expenses for Low credit risk and six months for Medium or High credit risk, or for a Debt-Heavy Spender. When debt priority applies, 50 to 70 percent of the surplus is directed to debt. Otherwise, the emergency fund is prioritized before investing.

Investment category: The engine selects Capital Preservation, Conservative, Balanced, or Growth based on credit risk, emergency savings, debt-to-income ratio, spending profile, and savings level.

Alerts: The engine can flag a budget deficit, high spending compared with income, expense growth, high debt-to-income ratio, high EMI or credit utilization, missed or late payments, a low emergency fund, low savings, high discretionary spending, or high credit risk.

MODEL NOTES
Expense model: Savings amount is strongly related to income and previous expenses in the training data. This can make the Linear Regression model unstable when inputs do not resemble its training data. The engine reconciles related inputs and limits its prediction to a range around recent spending. Retraining with Ridge or XGBoost, or removing savings and category features, may improve the model.

Expense deviation: This feature is not present in the source CSV. The saved scaler indicates that it was calculated as previous month expenses minus the three-month average expenses.

Cluster labels: The saved cluster labels identify cluster 2 as Debt-Heavy Spender, although its spending pattern is closer to High Spender. The engine derives labels from cluster centroids to handle this inconsistency.

Credit model: The credit model can return overconfident probabilities. Calibration may improve probability estimates. The class mapping of 0, 1, and 2 to Low, Medium, and High was checked against late-payment and repayment averages.

LIMITATIONS
Rule thresholds are defaults and have not been tuned against real outcomes. Test profiles combine information from independent datasets, so their output distribution may not represent real users. Investment categories provide general guidance for Indian instruments and are not recommendations for specific financial products.
