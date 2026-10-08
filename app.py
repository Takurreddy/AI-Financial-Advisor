"""
AI Personal Finance Advisor - Streamlit dashboard (v2 UI)
Run:  streamlit run app.py
"""
import html

import plotly.graph_objects as go
import streamlit as st

from demo import PERSONAS
from recommendation_engine import ALL_CATS, FinancialAdvisor, UserFinancialInput, inr

st.set_page_config(page_title="AI Personal Finance Advisor", page_icon="💰", layout="wide")

# ------------------------------------------------------------------ styling
TEAL, NAVY = "#0F766E", "#0B3C5D"
SEV = {"critical": ("#dc2626", "🚨"), "high": ("#ea580c", "⚠️"), "warning": ("#d97706", "⚠️"), "info": ("#2563eb", "ℹ️")}
RISK_COLOR = {"Low": "#16a34a", "Medium": "#d97706", "High": "#dc2626"}
PROFILE_ICON = {"Saver": "🏦", "High Spender": "🛍️", "Debt-Heavy Spender": "💳"}
SERIES = ["#0F766E", "#F59E0B", "#3B82F6", "#EF4444", "#8B5CF6", "#14B8A6", "#EC4899", "#64748B"]

st.markdown(f"""
<style>
.block-container {{padding-top: 1.6rem; max-width: 1250px;}}
.hero {{background: linear-gradient(120deg, {NAVY} 0%, {TEAL} 100%); color: white; border-radius: 18px;
        padding: 22px 28px; margin-bottom: 18px;}}
.hero h1 {{margin: 0; font-size: 1.9rem; color: white;}}
.hero p {{margin: 4px 0 0 0; opacity: .9; font-size: .95rem;}}
.card {{border: 1px solid rgba(128,128,128,.28); border-radius: 14px; padding: 14px 18px;
        background: rgba(128,128,128,.07); height: 100%;}}
.kpi-label {{font-size: .78rem; text-transform: uppercase; letter-spacing: .05em; opacity: .7; margin-bottom: 4px;}}
.kpi-value {{font-size: 1.75rem; font-weight: 700; line-height: 1.15;}}
.kpi-sub {{font-size: .82rem; opacity: .75; margin-top: 3px;}}
.pill {{display: inline-block; padding: 2px 12px; border-radius: 999px; color: white; font-weight: 600; font-size: .95rem;}}
.alert {{border-left: 5px solid; border-radius: 8px; padding: 9px 14px; margin: 7px 0; background: rgba(128,128,128,.07);}}
.action {{display: flex; gap: 12px; align-items: flex-start; padding: 10px 14px; margin: 8px 0;
          border-radius: 12px; background: rgba(15,118,110,.10); border: 1px solid rgba(15,118,110,.30);}}
.action .n {{background: {TEAL}; color: white; border-radius: 50%; min-width: 28px; height: 28px;
             display: flex; align-items: center; justify-content: center; font-weight: 700;}}
.section {{font-size: 1.05rem; font-weight: 700; margin: 6px 0 4px 0;}}
div[data-testid="stTabs"] button {{font-size: 1rem; font-weight: 600;}}
footer {{visibility: hidden;}}
</style>
""", unsafe_allow_html=True)


def show(container, fig):
    """plotly_chart that works on both old and new Streamlit versions."""
    try:
        container.plotly_chart(fig, width="stretch")
    except TypeError:
        container.plotly_chart(fig, use_container_width=True)


def style(fig, title, height=340, legend=True):
    fig.update_layout(title=dict(text=title, x=0, font=dict(size=16)), height=height,
                      margin=dict(l=10, r=10, t=50, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      showlegend=legend, legend=dict(orientation="h", y=-0.15), font=dict(size=13))
    return fig


def kpi(col, label, value, sub="", value_html=None):
    col.markdown(f"<div class='card'><div class='kpi-label'>{label}</div>"
                 f"<div class='kpi-value'>{value_html or html.escape(value)}</div>"
                 f"<div class='kpi-sub'>{html.escape(sub)}</div></div>", unsafe_allow_html=True)


def health_score(r):
    """Transparent 0-100 summary built only from engine metrics (a heuristic, not a model output)."""
    m, risk = r.metrics, r.predictions["credit_risk"]
    parts = {
        "Savings rate": 25 * min(max(m["projected_savings_rate"], 0) / 0.25, 1),
        "Spending vs income": 20 * min(max((1.0 - m["spend_to_income_ratio"]) / 0.4, 0), 1),
        "Debt load": 20 * min(max((0.5 - m["debt_to_income_ratio"]) / 0.35, 0), 1),
        "Emergency fund": 20 * min(m["emergency_fund_months"] / max(m["emergency_target_months"], 1), 1),
        "Credit risk": {"Low": 15, "Medium": 8, "High": 0}[risk],
    }
    maxes = {"Savings rate": 25, "Spending vs income": 20, "Debt load": 20, "Emergency fund": 20, "Credit risk": 15}
    return sum(parts.values()), parts, maxes


@st.cache_resource
def get_advisor():
    return FinancialAdvisor()


# ------------------------------------------------------------------ hero
st.markdown("<div class='hero'><h1>💰 AI Personal Finance Advisor</h1>"
            "<p>Predict next-month spending · assess credit risk · profile your behaviour · get a personalised plan</p></div>",
            unsafe_allow_html=True)

# ------------------------------------------------------------------ sidebar
preset_names = ["Custom (blank form)"] + list(PERSONAS)
preset = st.sidebar.selectbox("📂 Load an example profile", preset_names, index=1)
base = PERSONAS.get(preset, PERSONAS[list(PERSONAS)[0]])
k = preset[:2]


def g(field, default):
    v = getattr(base, field, None)
    return default if v is None else v


with st.sidebar:
    with st.expander("👤 Basics (required)", expanded=True):
        age = st.number_input("Age", 18, 100, int(base.age), key=f"{k}age")
        income = st.number_input("Monthly income (₹)", 1000, 5_000_000, int(base.monthly_income), 1000, key=f"{k}inc")
        prev = st.number_input("Last month's expense (₹)", 0, 5_000_000, int(base.previous_month_expense), 500, key=f"{k}prev")
        avg3 = st.number_input("3-month avg expense (₹)", 1, 5_000_000, int(base.avg_3_month_expense), 500, key=f"{k}avg")
        sav = st.number_input("Monthly savings (₹)", -1_000_000, 5_000_000, int(base.savings_amount), 500, key=f"{k}sav")
    with st.expander("🛒 Spending breakdown"):
        use_cats = st.checkbox("I know my category split", value=base.food_expense is not None, key=f"{k}uc")
        cats = {}
        if use_cats:
            for c in ALL_CATS:
                cats[c] = st.number_input(f"{c.title()} (₹)", 0, 5_000_000, int(g(f"{c}_expense", 0)), 500, key=f"{k}c{c}")
            tx = st.number_input("Transactions / month", 0, 500, int(g("transaction_count", 30)), key=f"{k}tx")
    with st.expander("💳 Credit behaviour"):
        use_credit = st.checkbox("I know my credit details", value=base.credit_utilization is not None, key=f"{k}ucr")
        credit = {}
        if use_credit:
            credit["credit_utilization"] = st.slider("Credit utilisation", 0.0, 1.0, float(g("credit_utilization", 0.4)), key=f"{k}ut")
            credit["late_payment_count"] = st.number_input("Late payments", 0, 50, int(g("late_payment_count", 0)), key=f"{k}lt")
            credit["missed_payment_count"] = st.number_input("Missed payments", 0, 50, int(g("missed_payment_count", 0)), key=f"{k}ms")
            credit["repayment_ratio"] = st.slider("Repayment ratio", 0.0, 1.0, float(g("repayment_ratio", 0.9)), key=f"{k}rp")
    with st.expander("🏦 Debt & assets"):
        debt = st.number_input("Total debt (₹)", 0, 100_000_000, int(base.total_debt), 5000, key=f"{k}debt")
        emi = st.number_input("Monthly EMI (₹)", 0, 5_000_000, int(base.monthly_emi), 500, key=f"{k}emi")
        ef = st.number_input("Emergency fund (₹)", 0, 100_000_000, int(base.emergency_fund), 5000, key=f"{k}ef")
        inv = st.number_input("Monthly investment (₹)", 0, 5_000_000, int(base.investment_amount), 500, key=f"{k}inv")
    st.caption("Results update live as you change any value.")

kw = dict(age=age, monthly_income=income, previous_month_expense=prev, avg_3_month_expense=avg3, savings_amount=sav,
          emergency_fund=ef, investment_amount=inv, total_debt=debt, monthly_emi=emi, **credit)
if use_cats:
    kw.update({f"{c}_expense": v for c, v in cats.items()}); kw["transaction_count"] = tx
try:
    r = get_advisor().analyze(UserFinancialInput(**kw))
except ValueError as e:
    st.error(str(e)); st.stop()

p, m, cd = r.predictions, r.metrics, r.chart_data
score, parts, maxes = health_score(r)
band, band_col = (("Excellent", "#16a34a") if score >= 80 else ("Good", "#65a30d") if score >= 60
                  else ("Needs attention", "#d97706") if score >= 40 else ("At risk", "#dc2626"))

# ------------------------------------------------------------------ KPI cards
c1, c2, c3, c4, c5 = st.columns(5)
kpi(c1, "Predicted expense", inr(p["predicted_expense"]), f"{m['expense_change_pct']:+.1f}% vs 3-month avg")
kpi(c2, "Credit risk", "", f"{p['credit_risk_probs'][p['credit_risk']]:.0%} model confidence",
    value_html=f"<span class='pill' style='background:{RISK_COLOR[p['credit_risk']]}'>{p['credit_risk']}</span>")
kpi(c3, "Financial profile", f"{PROFILE_ICON.get(p['financial_profile'], '')} {p['financial_profile']}",
    f"{p['profile_confidence']:.0%} confidence")
kpi(c4, "Monthly surplus", inr(m["projected_monthly_surplus"]), f"{m['projected_savings_rate']:.0%} of income")
kpi(c5, "Health score", "", band, value_html=f"<span style='color:{band_col}'>{score:.0f}<small>/100</small></span>")
st.write("")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "🧾 Spending & budget", "🏦 Savings & investing",
                                        "🎯 Risk & profile", "🔍 Why this advice?"])

# ------------------------------------------------------------------ TAB 1
with tab1:
    left, right = st.columns([3, 2])
    with left:
        st.markdown("<div class='section'>🎯 Top actions for you</div>", unsafe_allow_html=True)
        for i, act in enumerate(r.priority_actions, 1):
            st.markdown(f"<div class='action'><div class='n'>{i}</div><div>{html.escape(act)}</div></div>", unsafe_allow_html=True)
        st.markdown("<div class='section'>🔔 Alerts</div>", unsafe_allow_html=True)
        if not r.alerts:
            st.success("No alerts - your finances look healthy.")
        for a in r.alerts:
            col, icon = SEV[a["severity"]]
            st.markdown(f"<div class='alert' style='border-color:{col}'><b style='color:{col}'>{icon} {a['severity'].upper()}</b>"
                        f" &nbsp;{html.escape(a['message'])}</div>", unsafe_allow_html=True)
    with right:
        gauge = go.Figure(go.Indicator(mode="gauge+number", value=score, number={"suffix": "/100"},
                                       title={"text": f"Financial health: <b>{band}</b>"},
                                       gauge={"axis": {"range": [0, 100]}, "bar": {"color": band_col},
                                              "steps": [{"range": [0, 40], "color": "rgba(220,38,38,.25)"},
                                                        {"range": [40, 60], "color": "rgba(217,119,6,.25)"},
                                                        {"range": [60, 80], "color": "rgba(101,163,13,.25)"},
                                                        {"range": [80, 100], "color": "rgba(22,163,74,.25)"}]}))
        show(right, style(gauge, "", 270, False))
        fig = go.Figure(go.Bar(y=list(parts), x=[parts[x] for x in parts], orientation="h",
                               marker_color=TEAL, text=[f"{parts[x]:.0f}/{maxes[x]}" for x in parts], textposition="auto"))
        fig.update_yaxes(autorange="reversed"); fig.update_xaxes(range=[0, 25], visible=False)
        show(right, style(fig, "What makes up the score", 230, False))
        right.caption("Heuristic summary built from the engine's metrics (savings rate, spend ratio, debt, emergency fund, credit risk) - not a model output.")

    et = cd["expense_trend"]
    fig = go.Figure(go.Bar(x=["Last month", "3-month avg", "Predicted next month"],
                           y=[et["previous_month"], et["avg_3_month"], et["predicted_next_month"]],
                           marker_color=["#93c5fd", "#60a5fa", NAVY],
                           text=[inr(v) for v in (et["previous_month"], et["avg_3_month"], et["predicted_next_month"])],
                           textposition="outside"))
    fig.add_hline(y=et["income"], line_dash="dash", line_color="#16a34a", annotation_text=f"Income {inr(et['income'])}")
    fig.update_yaxes(range=[0, max(et["income"], et["predicted_next_month"]) * 1.2], title="₹")
    show(st, style(fig, "Expense trend vs income", 320, False))

# ------------------------------------------------------------------ TAB 2
with tab2:
    a, b = st.columns(2)
    now = {k_: v for k_, v in cd["spending_now"].items() if v > 0}
    d1 = go.Figure(go.Pie(labels=list(now), values=list(now.values()), hole=0.55, sort=False,
                          marker_colors=["#3B82F6", "#F59E0B", "#16a34a", "#dc2626"][:len(now)]))
    show(a, style(d1, "Where your income goes now"))
    plan = {k_: v for k_, v in cd["income_plan"].items() if v > 0}
    d2 = go.Figure(go.Pie(labels=list(plan), values=list(plan.values()), hole=0.55, sort=False,
                          marker_colors=["#3B82F6", "#F59E0B", "#dc2626", "#16a34a", "#14B8A6"]))
    show(b, style(d2, "Recommended plan"))

    cur, rec = cd["category_current"], cd["category_recommended"]
    fig = go.Figure()
    fig.add_bar(name="Predicted now", x=[c.title() for c in ALL_CATS], y=[cur[c] for c in ALL_CATS], marker_color="#93c5fd")
    fig.add_bar(name="Recommended", x=[c.title() for c in ALL_CATS], y=[rec[c] for c in ALL_CATS], marker_color=TEAL)
    fig.update_layout(barmode="group", yaxis_title="₹")
    show(st, style(fig, "Spending by category: predicted vs recommended"))
    cuts = {c: cur[c] - rec[c] for c in ALL_CATS if cur[c] - rec[c] > 1}
    if cuts:
        st.markdown("**Suggested cuts:** " + " · ".join(f"{c.title()} −{inr(v)}" for c, v in sorted(cuts.items(), key=lambda x: -x[1])))
    st.info(r.budget["message"])

# ------------------------------------------------------------------ TAB 3
with tab3:
    a, b = st.columns(2)
    efd = cd["emergency_fund"]
    pct = min(efd["current"] / efd["target"], 1.0) if efd["target"] else 1.0
    a.markdown("<div class='section'>🛟 Emergency fund</div>", unsafe_allow_html=True)
    a.progress(pct, text=f"{inr(efd['current'])} of {inr(efd['target'])}  ({m['emergency_target_months']}-month target)")
    contrib = a.slider("What if I save this much per month? (₹)", 0, max(int(income), 1000), int(efd["monthly_contribution"]), 500)
    gap = max(efd["target"] - efd["current"], 0)
    months = list(range(0, 25))
    series = [min(efd["current"] + contrib * mth, max(efd["target"], efd["current"])) for mth in months]
    fig = go.Figure(go.Scatter(x=months, y=series, mode="lines+markers", line=dict(color=TEAL, width=3)))
    fig.add_hline(y=efd["target"], line_dash="dash", line_color="#dc2626", annotation_text="Target")
    fig.update_layout(xaxis_title="Months from now", yaxis_title="₹")
    show(a, style(fig, "Emergency-fund projection", 300, False))
    if gap <= 0:
        a.success("Emergency fund target already met 🎉")
    elif contrib > 0:
        a.info(f"At {inr(contrib)}/month you reach the target in about **{gap / contrib:.0f} months**.")
    else:
        a.warning("Set a monthly amount to see the timeline.")

    b.markdown(f"<div class='section'>📈 Investment category: {r.investment['category']}</div>", unsafe_allow_html=True)
    mix = r.investment["indicative_mix_pct"]
    fig = go.Figure(go.Pie(labels=list(mix), values=list(mix.values()), hole=0.55, sort=False, marker_colors=SERIES))
    show(b, style(fig, "Indicative mix"))
    b.write(r.investment["message"])
    st.info(r.savings["message"])

# ------------------------------------------------------------------ TAB 4
with tab4:
    a, b = st.columns(2)
    probs = p["credit_risk_probs"]
    fig = go.Figure(go.Bar(x=list(probs), y=[v * 100 for v in probs.values()], marker_color=[RISK_COLOR[x] for x in probs],
                           text=[f"{v:.0%}" for v in probs.values()], textposition="outside"))
    fig.update_yaxes(title="%", range=[0, 115])
    show(a, style(fig, "Credit-risk model: class probabilities", 320, False))
    a.caption("The credit model is often very confident; treat probabilities as indicative.")

    pc = cd["profile_compare"]
    nice = {"savings_rate": "Savings rate", "debt_to_income_ratio": "Debt / annual income",
            "discretionary_expense_ratio": "Discretionary share", "shopping_ratio": "Shopping share",
            "entertainment_ratio": "Entertainment share"}
    x = [nice[f] for f in pc["features"]]
    fig = go.Figure()
    fig.add_bar(name="You", x=x, y=pc["user"], marker_color="#111827")
    for (lab, vals), col in zip(pc["clusters"].items(), ["#16a34a", "#F59E0B", "#3B82F6"]):
        fig.add_bar(name=f"{lab} avg", x=x, y=vals, marker_color=col, opacity=0.8)
    fig.update_layout(barmode="group", yaxis_tickformat=".0%")
    show(b, style(fig, f"You vs profile averages (matched: {p['financial_profile']})", 320))

    d1, d2, d3 = st.columns(3)
    def status(v, w, c): return "✅ OK" if v <= w else "⚠️ Elevated" if v <= c else "🚨 Critical"
    kpi(d1, "Debt / annual income", f"{m['debt_to_income_ratio']:.0%}", status(m["debt_to_income_ratio"], .30, .40))
    kpi(d2, "EMI / monthly income", f"{m['emi_to_income_ratio']:.0%}", status(m["emi_to_income_ratio"], .25, .40))
    kpi(d3, "Emergency fund cover", f"{m['emergency_fund_months']:.1f} months", f"target {m['emergency_target_months']} months")

# ------------------------------------------------------------------ TAB 5
with tab5:
    a, b = st.columns(2)
    a.markdown("<div class='section'>📜 Rules triggered</div>", unsafe_allow_html=True)
    for rule in r.rules_triggered:
        a.markdown(f"- {rule}")
    b.markdown("<div class='section'>🧩 Defaults assumed (not supplied)</div>", unsafe_allow_html=True)
    b.write(", ".join(r.assumptions) or "None - all values were provided.")
    if r.warnings:
        b.markdown("<div class='section'>⚠️ Data warnings</div>", unsafe_allow_html=True)
        for w in r.warnings:
            b.warning(w)
    d1, d2 = st.columns(2)
    d1.download_button("⬇️ Download report (JSON)", r.to_json(), "finance_report.json", "application/json")
    d2.download_button("⬇️ Download report (text)", r.summary_text(), "finance_report.txt", "text/plain")
    with st.expander("Full JSON report"):
        st.json(r.to_dict())

st.caption("⚖️ " + r.disclaimer)