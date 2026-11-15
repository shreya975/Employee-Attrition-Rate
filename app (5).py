"""
Attrition IQ - Employee Attrition Analytics Platform
----------------------------------------------------
Streamlit app for the IBM HR Analytics Employee Attrition dataset.

Model: the leak-free Scikit-learn pipeline from the project notebook
(ColumnTransformer with StandardScaler + OneHotEncoder, then Logistic
Regression, stratified 80:20 split, random_state=10). The target is never
scaled, and preprocessing is learned from the training data only.

Run locally:  streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ----------------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Attrition IQ",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Design tokens
PAPER = "#F3F6F5"
SURFACE = "#FFFFFF"
INK = "#0E1B1A"
MUTED = "#5A6B69"
LINE = "#D8E0DE"
TEAL = "#0B6E66"
TEAL_SOFT = "#DCEEEB"
RISK = "#C8372D"
AMBER = "#C98512"
OK = "#2F8F5B"

DISPLAY_FONT = "'Bricolage Grotesque', 'Segoe UI', sans-serif"
BODY_FONT = "'IBM Plex Sans', 'Segoe UI', sans-serif"

DATA_CANDIDATES = [
    Path(__file__).parent / "HR-Employee-Attrition.csv",
    Path(__file__).parent / "Datasets" / "HR-Employee-Attrition.csv",
    Path(__file__).parent.parent / "Datasets" / "HR-Employee-Attrition.csv",
]

DROP_COLUMNS = ["EmployeeNumber", "EmployeeCount", "Over18", "StandardHours"]
RANDOM_STATE = 10

FRIENDLY = {
    "Age": "Age",
    "BusinessTravel": "Business travel",
    "DailyRate": "Daily rate",
    "Department": "Department",
    "DistanceFromHome": "Distance from home",
    "Education": "Education level",
    "EducationField": "Education field",
    "EnvironmentSatisfaction": "Environment satisfaction",
    "Gender": "Gender",
    "HourlyRate": "Hourly rate",
    "JobInvolvement": "Job involvement",
    "JobLevel": "Job level",
    "JobRole": "Job role",
    "JobSatisfaction": "Job satisfaction",
    "MaritalStatus": "Marital status",
    "MonthlyIncome": "Monthly income",
    "MonthlyRate": "Monthly rate",
    "NumCompaniesWorked": "Companies worked for",
    "OverTime": "Overtime",
    "PercentSalaryHike": "Last salary hike (%)",
    "PerformanceRating": "Performance rating",
    "RelationshipSatisfaction": "Relationship satisfaction",
    "StockOptionLevel": "Stock option level",
    "TotalWorkingYears": "Total working years",
    "TrainingTimesLastYear": "Trainings last year",
    "WorkLifeBalance": "Work-life balance",
    "YearsAtCompany": "Years at company",
    "YearsInCurrentRole": "Years in current role",
    "YearsSinceLastPromotion": "Years since last promotion",
    "YearsWithCurrManager": "Years with current manager",
}

LEVEL4 = {1: "Low", 2: "Medium", 3: "High", 4: "Very high"}
WLB = {1: "Bad", 2: "Good", 3: "Better", 4: "Best"}
EDUCATION = {1: "Below college", 2: "College", 3: "Bachelor", 4: "Master", 5: "Doctor"}
PERFORMANCE = {3: "Excellent", 4: "Outstanding"}


# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

html, body, .stApp, [data-testid="stAppViewContainer"] {
  background: #F3F6F5;
  color: #0E1B1A;
  font-family: 'IBM Plex Sans', 'Segoe UI', sans-serif;
  font-variant-numeric: tabular-nums;
}
[data-testid="stHeader"], [data-testid="stToolbar"], footer, #MainMenu { display: none !important; }
.block-container { max-width: 1180px; padding: 1.6rem 1.5rem 4rem; }

h1, h2, h3, h4 { font-family: 'Bricolage Grotesque', 'Segoe UI', sans-serif; color: #0E1B1A; letter-spacing: -0.01em; }
p, li, label, span, div { font-family: inherit; }

/* Masthead */
.mast { display: flex; justify-content: space-between; align-items: center; gap: 1rem; flex-wrap: wrap;
  padding-bottom: 1rem; border-bottom: 1px solid #D8E0DE; }
.mark { display: flex; align-items: center; gap: .6rem; font-family: 'Bricolage Grotesque', sans-serif;
  font-weight: 700; font-size: 1.25rem; }
.mark svg { display: block; }
.mast-note { color: #5A6B69; font-size: .9rem; max-width: 34rem; text-align: right; }

/* Lede */
.lede { padding: 2.2rem 0 1.4rem; }
.lede h1 { font-size: clamp(1.9rem, 4.4vw, 3.1rem); line-height: 1.08; margin: 0 0 .8rem; max-width: 22ch; font-weight: 700; }
.lede p { color: #5A6B69; font-size: 1.05rem; line-height: 1.55; max-width: 62ch; margin: 0; }

/* Stat strip */
.stats { display: grid; grid-template-columns: repeat(4, 1fr); border-top: 1px solid #D8E0DE; border-bottom: 1px solid #D8E0DE;
  margin: 1.4rem 0 2rem; background: #FFFFFF; }
.stat { padding: 1.1rem 1.3rem; border-left: 1px solid #D8E0DE; }
.stat:first-child { border-left: none; }
.stat b { display: block; font-family: 'Bricolage Grotesque', sans-serif; font-size: 2rem; line-height: 1.1; font-weight: 700; }
.stat span { color: #5A6B69; font-size: .9rem; }
@media (max-width: 760px) { .stats { grid-template-columns: repeat(2, 1fr); } .stat:nth-child(3) { border-left: none; } .stat:nth-child(n+3) { border-top: 1px solid #D8E0DE; } }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 1.6rem; border-bottom: 1px solid #D8E0DE; }
.stTabs [data-baseweb="tab"] { padding: .7rem 0; background: transparent; font-family: 'Bricolage Grotesque', sans-serif;
  font-size: 1.05rem; font-weight: 500; color: #5A6B69; height: auto; }
.stTabs [aria-selected="true"] { color: #0B6E66 !important; font-weight: 700; }
.stTabs [data-baseweb="tab-highlight"] { background-color: #0B6E66 !important; height: 3px; }
.stTabs [data-baseweb="tab-border"] { display: none; }

/* Section headings */
.sec { margin: 2rem 0 .3rem; font-family: 'Bricolage Grotesque', sans-serif; font-size: 1.45rem; font-weight: 700; }
.sec-sub { color: #5A6B69; margin: 0 0 1rem; max-width: 64ch; line-height: 1.5; }

/* Panels */
.panel { background: #FFFFFF; border: 1px solid #D8E0DE; border-radius: 10px; padding: 1.2rem 1.3rem; }
.panel + .panel { margin-top: 1rem; }
.panel h4 { margin: 0 0 .6rem; font-size: 1.1rem; }

/* Highest-risk list */
.hr-list { list-style: none; margin: 0; padding: 0; }
.hr-list li { display: flex; justify-content: space-between; align-items: baseline; gap: 1rem; padding: .7rem 0; border-top: 1px solid #D8E0DE; }
.hr-list li:first-child { border-top: none; }
.hr-list .grp { font-weight: 500; }
.hr-list .grp small { display: block; color: #5A6B69; font-weight: 400; }
.hr-list .rate { font-family: 'Bricolage Grotesque', sans-serif; font-size: 1.35rem; font-weight: 700; color: #C8372D; }

/* Form */
div[data-testid="stForm"] { background: #FFFFFF; border: 1px solid #D8E0DE; border-radius: 10px; padding: 1.2rem 1.3rem 1.4rem; }
.form-h { font-family: 'Bricolage Grotesque', sans-serif; font-weight: 700; font-size: 1.05rem; margin: .9rem 0 .2rem; padding-top: .8rem; border-top: 1px solid #D8E0DE; }
.form-h.first { border-top: none; margin-top: 0; padding-top: 0; }
div[data-testid="stFormSubmitButton"] button, .stButton > button {
  background: #0B6E66; color: #fff; border: none; border-radius: 8px; padding: .65rem 1.4rem;
  font-family: 'Bricolage Grotesque', sans-serif; font-size: 1.05rem; font-weight: 700; width: 100%; }
div[data-testid="stFormSubmitButton"] button:hover, .stButton > button:hover { background: #09574F; color: #fff; }
button:focus-visible, input:focus-visible, [role="radio"]:focus-visible { outline: 3px solid #E0A31B !important; outline-offset: 2px; }

/* Sticky result column */
div[data-testid="stColumn"]:has(.result-anchor), div[data-testid="column"]:has(.result-anchor) { position: sticky; top: 1rem; align-self: flex-start; }

/* Result */
.reading { display: flex; align-items: flex-end; gap: 1rem; flex-wrap: wrap; }
.reading .num { font-family: 'Bricolage Grotesque', sans-serif; font-weight: 700; font-size: 4.2rem; line-height: .95; }
.reading .band { font-family: 'Bricolage Grotesque', sans-serif; font-weight: 700; font-size: 1.2rem; padding: .25rem .7rem; border-radius: 6px; margin-bottom: .4rem; }
.band.low { background: #DDF1E6; color: #1E6B42; }
.band.mid { background: #FAEBCB; color: #8A5A05; }
.band.high { background: #F8DAD7; color: #9C2119; }
.reading-sub { color: #5A6B69; margin: .6rem 0 1.1rem; line-height: 1.5; }

.strip { position: relative; display: flex; align-items: flex-end; gap: 2px; height: 74px; padding-bottom: 2px; border-bottom: 1px solid #0E1B1A; }
.strip i { flex: 1; background: #BFD3D0; border-radius: 2px 2px 0 0; min-height: 2px; }
.pin { position: absolute; bottom: 0; top: -8px; width: 3px; background: #C8372D; border-radius: 2px; left: var(--pos); animation: land 1.1s cubic-bezier(.2,.8,.2,1) both; }
.pin::after { content: ""; position: absolute; top: -2px; left: -4px; width: 11px; height: 11px; border-radius: 50%; background: #C8372D; }
@keyframes land { from { left: 0%; } to { left: var(--pos); } }
@media (prefers-reduced-motion: reduce) { .pin { animation: none; } }
.axis { display: flex; justify-content: space-between; color: #5A6B69; font-size: .8rem; margin-top: .3rem; }

.whatif { list-style: none; padding: 0; margin: 0; }
.whatif li { display: flex; justify-content: space-between; gap: 1rem; padding: .55rem 0; border-top: 1px solid #D8E0DE; }
.whatif li:first-child { border-top: none; }
.whatif .d { font-weight: 600; color: #2F8F5B; white-space: nowrap; }

.note { padding: .6rem .8rem; border-left: 4px solid #C98512; background: #FDF5E3; border-radius: 0 6px 6px 0; margin-bottom: .5rem; line-height: 1.45; }
.note.danger { border-color: #C8372D; background: #FBECEA; }
.note.good { border-color: #2F8F5B; background: #E8F5EE; }

.empty { color: #5A6B69; line-height: 1.6; }
.empty b { color: #0E1B1A; }
.fine { color: #5A6B69; font-size: .85rem; line-height: 1.5; }

/* Tables */
.stDataFrame, [data-testid="stTable"] { font-size: .95rem; }
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def style_fig(fig: go.Figure, height: int = 360) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=28, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=BODY_FONT, size=13, color=INK),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        hoverlabel=dict(font_family=BODY_FONT),
    )
    fig.update_xaxes(gridcolor=LINE, zerolinecolor=LINE, linecolor=LINE)
    fig.update_yaxes(gridcolor=LINE, zerolinecolor=LINE, linecolor=LINE)
    return fig


# ----------------------------------------------------------------------------
# Data and model
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    for path in DATA_CANDIDATES:
        if path.exists():
            return pd.read_csv(path, encoding="utf-8-sig")
    raise FileNotFoundError(
        "HR-Employee-Attrition.csv was not found next to app.py. "
        "Place the dataset in the same folder as this file."
    )


def _dense(matrix: Any) -> np.ndarray:
    return matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)


def _metrics(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    pred = model.predict(x_test)
    proba = model.predict_proba(x_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, proba)
    return {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, proba),
        "cm": confusion_matrix(y_test, pred, labels=[0, 1]),
        "fpr": fpr,
        "tpr": tpr,
    }


@st.cache_resource(show_spinner="Training the model...")
def build_model(raw: pd.DataFrame) -> Dict[str, Any]:
    df = raw.copy()
    df["Attrition"] = df["Attrition"].map({"No": 0, "Yes": 1})
    df = df.drop(columns=DROP_COLUMNS)

    x = df.drop(columns=["Attrition"])
    y = df["Attrition"]
    numeric = x.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical = [c for c in x.columns if c not in numeric]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    def make(class_weight: Optional[str]) -> Pipeline:
        pre = ColumnTransformer(
            [
                ("numerical", StandardScaler(), numeric),
                ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
            ]
        )
        clf = LogisticRegression(max_iter=1000, class_weight=class_weight, random_state=RANDOM_STATE)
        return Pipeline([("preprocessor", pre), ("classifier", clf)])

    standard, balanced = make(None), make("balanced")
    standard.fit(x_train, y_train)
    balanced.fit(x_train, y_train)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_standard = cross_val_score(make(None), x_train, y_train, cv=cv, scoring="f1")
    cv_balanced = cross_val_score(make("balanced"), x_train, y_train, cv=cv, scoring="f1")

    use_balanced = cv_balanced.mean() > cv_standard.mean()
    final = balanced if use_balanced else standard

    pre = final.named_steps["preprocessor"]
    names = pre.get_feature_names_out()
    coef = final.named_steps["classifier"].coef_[0]

    def origin(name: str) -> str:
        body = name.split("__", 1)[1]
        if name.startswith("numerical__"):
            return body
        return next(c for c in sorted(categorical, key=len, reverse=True) if body.startswith(c + "_"))

    owner = np.array([origin(n) for n in names])
    z_train = _dense(pre.transform(x_train))
    baseline = (z_train * coef).mean(axis=0)

    return {
        "final": final,
        "final_name": "Class-weighted Logistic Regression" if use_balanced else "Logistic Regression",
        "standard": _metrics(standard, x_test, y_test),
        "balanced": _metrics(balanced, x_test, y_test),
        "selected_metrics": _metrics(final, x_test, y_test),
        "cv": {"standard": cv_standard, "balanced": cv_balanced},
        "columns": x.columns.tolist(),
        "numeric": numeric,
        "categorical": categorical,
        "names": names,
        "coef": coef,
        "owner": owner,
        "baseline": baseline,
        "all_proba": final.predict_proba(x)[:, 1],
        "split": (len(x_train), len(x_test), int(y_test.sum()), int((1 - y_test).sum())),
        "defaults": {
            c: (x[c].median() if c in numeric else x[c].mode().iloc[0]) for c in x.columns
        },
    }


def predict(model_info: Dict[str, Any], row: Dict[str, Any]) -> float:
    frame = pd.DataFrame([row])[model_info["columns"]]
    return float(model_info["final"].predict_proba(frame)[0, 1])


def explain(model_info: Dict[str, Any], row: Dict[str, Any]) -> pd.DataFrame:
    """Contribution of each original feature versus the average training employee."""
    final = model_info["final"]
    frame = pd.DataFrame([row])[model_info["columns"]]
    z = _dense(final.named_steps["preprocessor"].transform(frame))[0]
    contrib = z * model_info["coef"] - model_info["baseline"]
    grouped = pd.Series(contrib).groupby(model_info["owner"]).sum()
    out = grouped.rename("effect").reset_index().rename(columns={"index": "feature"})
    out["label"] = out["feature"].map(lambda f: f"{FRIENDLY.get(f, f)}: {format_value(f, row[f])}")
    return out.reindex(out["effect"].abs().sort_values(ascending=False).index)


def format_value(feature: str, value: Any) -> str:
    if feature in ("JobSatisfaction", "EnvironmentSatisfaction", "RelationshipSatisfaction", "JobInvolvement"):
        return LEVEL4.get(int(value), str(value))
    if feature == "WorkLifeBalance":
        return WLB.get(int(value), str(value))
    if feature == "Education":
        return EDUCATION.get(int(value), str(value))
    if feature == "BusinessTravel":
        return str(value).replace("_", " ").replace("Travel ", "Travels ").lower().capitalize()
    if isinstance(value, (int, float, np.integer, np.floating)):
        return f"{value:,.0f}"
    return str(value)


# ----------------------------------------------------------------------------
# Workforce analytics helpers
# ----------------------------------------------------------------------------
def prepare_analysis(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df["Left"] = (df["Attrition"] == "Yes").astype(int)
    df["Tenure"] = pd.cut(
        df["YearsAtCompany"], [-1, 2, 5, 10, 60], labels=["0-2 years", "3-5 years", "6-10 years", "11+ years"]
    ).astype(str)
    df["Age band"] = pd.cut(df["Age"], [17, 25, 35, 45, 60], labels=["18-25", "26-35", "36-45", "46-60"]).astype(str)
    df["Job satisfaction"] = df["JobSatisfaction"].map(LEVEL4)
    df["Work-life balance"] = df["WorkLifeBalance"].map(WLB)
    df["Stock options"] = df["StockOptionLevel"].map(lambda v: f"Level {v}")
    df["Business travel"] = df["BusinessTravel"].map(
        {"Non-Travel": "No travel", "Travel_Rarely": "Rarely", "Travel_Frequently": "Frequently"}
    )
    df["Overtime"] = df["OverTime"].map({"Yes": "Works overtime", "No": "No overtime"})
    return df


FACTORS: Dict[str, Optional[List[str]]] = {
    "Overtime": None,
    "Job role": None,
    "Department": None,
    "Tenure": ["0-2 years", "3-5 years", "6-10 years", "11+ years"],
    "Age band": ["18-25", "26-35", "36-45", "46-60"],
    "Business travel": ["No travel", "Rarely", "Frequently"],
    "Job satisfaction": ["Low", "Medium", "High", "Very high"],
    "Work-life balance": ["Bad", "Good", "Better", "Best"],
    "Stock options": ["Level 0", "Level 1", "Level 2", "Level 3"],
    "Marital status": None,
}
COLUMN_OF = {"Job role": "JobRole", "Marital status": "MaritalStatus"}


def segment_table(df: pd.DataFrame, factor: str) -> pd.DataFrame:
    col = COLUMN_OF.get(factor, factor)
    g = df.groupby(col)["Left"].agg(rate="mean", n="size", left="sum").reset_index()
    g["rate"] *= 100
    g = g.rename(columns={col: "segment"})
    order = FACTORS[factor]
    if order:
        g["segment"] = pd.Categorical(g["segment"], order, ordered=True)
        g = g.sort_values("segment")
        g["segment"] = g["segment"].astype(str)
    else:
        g = g.sort_values("rate", ascending=False)
    return g.reset_index(drop=True)


def highest_risk_groups(df: pd.DataFrame, overall: float, top: int = 6) -> pd.DataFrame:
    rows = []
    for factor in FACTORS:
        for _, r in segment_table(df, factor).iterrows():
            if r["n"] >= 40:
                rows.append({"factor": factor, "segment": r["segment"], "rate": r["rate"], "n": int(r["n"])})
    out = pd.DataFrame(rows)
    out = out[out["rate"] > overall].sort_values("rate", ascending=False)
    return out.head(top)


# ----------------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------------
def masthead() -> None:
    st.markdown(
        """
        <div class="mast">
          <div class="mark">
            <svg width="30" height="30" viewBox="0 0 30 30" aria-hidden="true">
              <rect width="30" height="30" rx="7" fill="#0B6E66"/>
              <path d="M6 9h6v5h6v5h6" fill="none" stroke="#fff" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
            Attrition IQ
          </div>
          <div class="mast-note">Built on 1,470 employee records from the IBM HR Analytics dataset. For analysis and learning, not for employment decisions.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def workforce_tab(df: pd.DataFrame) -> None:
    n = len(df)
    left = int(df["Left"].sum())
    overall = df["Left"].mean() * 100
    ot_yes = df.loc[df["OverTime"] == "Yes", "Left"].mean() * 100
    ot_no = df.loc[df["OverTime"] == "No", "Left"].mean() * 100

    st.markdown(
        f"""
        <div class="lede">
          <h1>{overall:.1f}% of employees left. Overtime workers left at {ot_yes:.1f}%.</h1>
          <p>Employees who don't work overtime left at {ot_no:.1f}%. Use the explorer below to see where else attrition
          is concentrated, then check a specific employee in the Risk check tab.</p>
        </div>
        <div class="stats">
          <div class="stat"><b>{n:,}</b><span>Employees in the data</span></div>
          <div class="stat"><b>{left}</b><span>Left the company</span></div>
          <div class="stat"><b>{overall:.1f}%</b><span>Attrition rate</span></div>
          <div class="stat"><b>{df['YearsAtCompany'].median():.0f} years</b><span>Median tenure</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sec">Where attrition concentrates</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sec-sub">Pick a factor to compare attrition rates across its groups. '
        "Bars in red are above the company average.</p>",
        unsafe_allow_html=True,
    )

    chart_col, list_col = st.columns([1.55, 1], gap="large")
    with chart_col:
        factor = st.selectbox("Compare attrition by", list(FACTORS), index=0)
        seg = segment_table(df, factor)
        colors = [RISK if r > overall else TEAL for r in seg["rate"]]
        fig = go.Figure(
            go.Bar(
                x=seg["rate"],
                y=seg["segment"],
                orientation="h",
                marker_color=colors,
                text=[f"{r:.1f}%" for r in seg["rate"]],
                textposition="outside",
                customdata=np.stack([seg["n"], seg["left"]], axis=-1),
                hovertemplate="%{y}<br>Attrition: %{x:.1f}%<br>%{customdata[1]} of %{customdata[0]} employees left<extra></extra>",
            )
        )
        fig.add_vline(x=overall, line_dash="dash", line_color=INK, annotation_text=f"Company {overall:.1f}%", annotation_position="top")
        fig.update_yaxes(autorange="reversed", title=None)
        fig.update_xaxes(title="Attrition rate (%)", range=[0, max(seg["rate"].max() * 1.2, overall * 1.5)])
        st.plotly_chart(style_fig(fig, 80 + 48 * len(seg)), width="stretch", config={"displayModeBar": False})

    with list_col:
        top = highest_risk_groups(df, overall)
        items = "".join(
            f'<li><div class="grp">{r.segment}<small>{r.factor}, {r.n} employees</small></div><div class="rate">{r.rate:.1f}%</div></li>'
            for r in top.itertuples()
        )
        st.markdown(
            f'<div class="panel"><h4>Highest-risk groups</h4><ul class="hr-list">{items}</ul>'
            f'<p class="fine" style="margin-top:.8rem">Groups with at least 40 employees, ranked by attrition rate across all ten factors.</p></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="sec">Pay and tenure</div>', unsafe_allow_html=True)
    a, b = st.columns(2, gap="large")
    with a:
        fig = go.Figure()
        for status, color in (("No", TEAL), ("Yes", RISK)):
            sub = df[df["Attrition"] == status]
            fig.add_trace(
                go.Box(
                    y=sub["MonthlyIncome"],
                    name="Stayed" if status == "No" else "Left",
                    marker_color=color,
                    boxpoints=False,
                )
            )
        fig.update_yaxes(title="Monthly income")
        fig.update_layout(title=dict(text="Employees who left earned less", x=0, font=dict(size=16, family=DISPLAY_FONT)), showlegend=False)
        st.plotly_chart(style_fig(fig, 340), width="stretch", config={"displayModeBar": False})
    with b:
        capped = df.assign(Y=df["YearsAtCompany"].clip(upper=20))
        t = capped.groupby("Y")["Left"].agg(rate="mean", n="size").reset_index()
        t = t[t["n"] >= 8]
        fig = go.Figure(
            go.Scatter(
                x=t["Y"],
                y=t["rate"] * 100,
                mode="lines+markers",
                line=dict(color=RISK, width=3),
                marker=dict(size=7),
                customdata=t["n"],
                hovertemplate="%{x} years<br>%{y:.1f}% left (n=%{customdata})<extra></extra>",
            )
        )
        fig.add_hline(y=overall, line_dash="dash", line_color=INK)
        fig.update_xaxes(title="Years at company (20 = 20 or more)")
        fig.update_yaxes(title="Attrition rate (%)")
        fig.update_layout(title=dict(text="Risk is highest in the first years", x=0, font=dict(size=16, family=DISPLAY_FONT)))
        st.plotly_chart(style_fig(fig, 340), width="stretch", config={"displayModeBar": False})


def validate(row: Dict[str, Any]) -> Optional[str]:
    if row["YearsAtCompany"] > row["TotalWorkingYears"]:
        return "Years at company can't be more than total working years."
    if row["YearsInCurrentRole"] > row["YearsAtCompany"]:
        return "Years in current role can't be more than years at company."
    if row["YearsWithCurrManager"] > row["YearsAtCompany"]:
        return "Years with current manager can't be more than years at company."
    if row["YearsSinceLastPromotion"] > row["YearsAtCompany"]:
        return "Years since last promotion can't be more than years at company."
    if row["TotalWorkingYears"] > row["Age"] - 14:
        return "Total working years looks too high for this age."
    return None


def band_of(p: float) -> Dict[str, str]:
    if p < 0.15:
        return {"name": "Low", "css": "low"}
    if p < 0.40:
        return {"name": "Elevated", "css": "mid"}
    return {"name": "High", "css": "high"}


def what_if(model_info: Dict[str, Any], row: Dict[str, Any], base: float) -> List[Dict[str, Any]]:
    scenarios = []

    def add(label: str, changes: Dict[str, Any]) -> None:
        trial = {**row, **changes}
        scenarios.append({"label": label, "delta": (predict(model_info, trial) - base) * 100})

    if row["OverTime"] == "Yes":
        add("Moves off overtime", {"OverTime": "No"})
    if row["BusinessTravel"] == "Travel_Frequently":
        add("Travels only rarely", {"BusinessTravel": "Travel_Rarely"})
    if row["YearsSinceLastPromotion"] > 0:
        add("Gets promoted now", {"YearsSinceLastPromotion": 0})
    if row["JobSatisfaction"] < 4:
        add("Job satisfaction rises to very high", {"JobSatisfaction": 4})
    if row["WorkLifeBalance"] < 3:
        add("Work-life balance rises to better", {"WorkLifeBalance": 3})
    if row["StockOptionLevel"] == 0:
        add("Receives stock options (level 1)", {"StockOptionLevel": 1})
    add("Monthly income rises by 15%", {"MonthlyIncome": round(row["MonthlyIncome"] * 1.15)})
    return sorted([s for s in scenarios if s["delta"] < -0.4], key=lambda s: s["delta"])[:5]


def review_notes(df: pd.DataFrame, row: Dict[str, Any]) -> List[tuple]:
    dept_income = df.loc[df["Department"] == row["Department"], "MonthlyIncome"].mean()
    notes = []
    if row["OverTime"] == "Yes":
        notes.append(("danger", "Works overtime. In this data, overtime workers left at about three times the rate of others."))
    if row["MonthlyIncome"] < dept_income:
        notes.append(("", f"Monthly income of {row['MonthlyIncome']:,} is below the {row['Department']} average of {dept_income:,.0f}."))
    if row["YearsAtCompany"] <= 2:
        notes.append(("", "In the first two years at the company, when attrition is highest."))
    if row["JobSatisfaction"] <= 2:
        notes.append(("danger", "Reports low job satisfaction."))
    if row["WorkLifeBalance"] <= 1:
        notes.append(("danger", "Reports bad work-life balance."))
    if row["YearsSinceLastPromotion"] >= 5:
        notes.append(("", "No promotion in 5 or more years."))
    if row["NumCompaniesWorked"] >= 5:
        notes.append(("", "Has worked for 5 or more companies before."))
    if not notes:
        notes.append(("good", "No major risk signals from the factors checked here."))
    return notes


def result_panel(model_info: Dict[str, Any], df: pd.DataFrame, result: Dict[str, Any]) -> None:
    p = result["p"]
    row = result["row"]
    band = band_of(p)
    percentile = float((model_info["all_proba"] < p).mean() * 100)

    counts, _ = np.histogram(model_info["all_proba"], bins=40, range=(0, 1))
    heights = np.sqrt(counts / counts.max()) * 100
    bars = "".join(f'<i style="height:{h:.0f}%"></i>' for h in heights)

    st.markdown(
        f"""
        <div class="panel">
          <div class="reading">
            <div class="num">{p * 100:.0f}%</div>
            <div class="band {band['css']}">{band['name']} risk</div>
          </div>
          <p class="reading-sub">Estimated chance this employee leaves. That is higher than {percentile:.0f}% of the
          {len(model_info['all_proba']):,} employees in the dataset.</p>
          <div class="strip" role="img" aria-label="Distribution of predicted risk with this employee marked">
            {bars}<div class="pin" style="--pos:{p * 100:.1f}%"></div>
          </div>
          <div class="axis"><span>0%</span><span>Predicted risk across the dataset</span><span>100%</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    effects = explain(model_info, row).head(8).iloc[::-1]
    fig = go.Figure(
        go.Bar(
            x=effects["effect"],
            y=effects["label"],
            orientation="h",
            marker_color=[RISK if v > 0 else TEAL for v in effects["effect"]],
            hovertemplate="%{y}<br>Effect on risk score: %{x:+.2f}<extra></extra>",
        )
    )
    fig.update_xaxes(title="Pushes risk down  |  Pushes risk up", zeroline=True, zerolinecolor=INK)
    st.markdown('<div class="panel"><h4>What drives this score</h4><p class="fine">Compared with a typical employee in the training data. Red bars raise the risk, teal bars lower it.</p>', unsafe_allow_html=True)
    st.plotly_chart(style_fig(fig, 340), width="stretch", config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)

    scenarios = what_if(model_info, row, p)
    if scenarios:
        items = "".join(
            f'<li><span>{s["label"]}</span><span class="d">{s["delta"]:.0f} points</span></li>' for s in scenarios
        )
        st.markdown(
            f'<div class="panel"><h4>What could lower the risk</h4><ul class="whatif">{items}</ul>'
            f'<p class="fine" style="margin-top:.7rem">Model estimates of how the score would shift. They show associations in the data, not proof that the change would cause it.</p></div>',
            unsafe_allow_html=True,
        )

    notes = "".join(f'<div class="note {c}">{t}</div>' for c, t in review_notes(df, row))
    st.markdown(f'<div class="panel"><h4>Review points for HR</h4>{notes}</div>', unsafe_allow_html=True)


def risk_tab(model_info: Dict[str, Any], df: pd.DataFrame) -> None:
    d = model_info["defaults"]
    st.markdown('<div class="sec" style="margin-top:1.6rem">Check an employee</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sec-sub">Enter the employee profile and select Check attrition risk. '
        "Anything you don't change keeps the dataset's typical value.</p>",
        unsafe_allow_html=True,
    )

    form_col, result_col = st.columns([1.05, 1], gap="large")
    with form_col:
        with st.form("profile"):
            st.markdown('<div class="form-h first">About the employee</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            age = c1.number_input("Age", 18, 60, int(d["Age"]))
            gender = c2.radio("Gender", ["Male", "Female"], horizontal=True, index=0 if d["Gender"] == "Male" else 1)
            marital = c1.selectbox("Marital status", ["Single", "Married", "Divorced"], index=["Single", "Married", "Divorced"].index(d["MaritalStatus"]))
            education = c2.selectbox("Education level", list(EDUCATION), index=int(d["Education"]) - 1, format_func=EDUCATION.get)
            field = c1.selectbox("Education field", sorted(df["EducationField"].unique()), index=sorted(df["EducationField"].unique()).index(d["EducationField"]))
            distance = c2.slider("Distance from home (km)", 1, 29, int(d["DistanceFromHome"]))

            st.markdown('<div class="form-h">Role and pay</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            depts = sorted(df["Department"].unique())
            department = c1.selectbox("Department", depts, index=depts.index(d["Department"]))
            roles = sorted(df["JobRole"].unique())
            role = c2.selectbox("Job role", roles, index=roles.index(d["JobRole"]))
            level = c1.slider("Job level", 1, 5, int(d["JobLevel"]))
            income = c2.number_input("Monthly income", 1000, 25000, int(d["MonthlyIncome"]), step=100)
            stock = c1.select_slider("Stock option level", [0, 1, 2, 3], value=int(d["StockOptionLevel"]))
            hike = c2.slider("Last salary hike (%)", 11, 25, int(d["PercentSalaryHike"]))
            travel = c1.selectbox(
                "Business travel",
                ["Non-Travel", "Travel_Rarely", "Travel_Frequently"],
                index=["Non-Travel", "Travel_Rarely", "Travel_Frequently"].index(d["BusinessTravel"]),
                format_func=lambda v: {"Non-Travel": "No travel", "Travel_Rarely": "Rarely", "Travel_Frequently": "Frequently"}[v],
            )
            overtime = c2.radio("Works overtime", ["No", "Yes"], horizontal=True, index=0 if d["OverTime"] == "No" else 1)

            st.markdown('<div class="form-h">Experience</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            total_years = c1.number_input("Total working years", 0, 40, int(d["TotalWorkingYears"]))
            years_company = c2.number_input("Years at company", 0, 40, int(d["YearsAtCompany"]))
            years_role = c1.number_input("Years in current role", 0, 18, int(d["YearsInCurrentRole"]))
            years_promo = c2.number_input("Years since last promotion", 0, 15, int(d["YearsSinceLastPromotion"]))
            years_mgr = c1.number_input("Years with current manager", 0, 17, int(d["YearsWithCurrManager"]))
            companies = c2.number_input("Companies worked for", 0, 9, int(d["NumCompaniesWorked"]))
            training = c1.number_input("Trainings last year", 0, 6, int(d["TrainingTimesLastYear"]))
            performance = c2.selectbox("Performance rating", list(PERFORMANCE), index=list(PERFORMANCE).index(int(d["PerformanceRating"])), format_func=PERFORMANCE.get)

            st.markdown('<div class="form-h">Satisfaction</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            job_sat = c1.select_slider("Job satisfaction", [1, 2, 3, 4], value=int(d["JobSatisfaction"]), format_func=LEVEL4.get)
            env_sat = c2.select_slider("Environment satisfaction", [1, 2, 3, 4], value=int(d["EnvironmentSatisfaction"]), format_func=LEVEL4.get)
            rel_sat = c1.select_slider("Relationship satisfaction", [1, 2, 3, 4], value=int(d["RelationshipSatisfaction"]), format_func=LEVEL4.get)
            involvement = c2.select_slider("Job involvement", [1, 2, 3, 4], value=int(d["JobInvolvement"]), format_func=LEVEL4.get)
            wlb = c1.select_slider("Work-life balance", [1, 2, 3, 4], value=int(d["WorkLifeBalance"]), format_func=WLB.get)

            with st.expander("Pay rates (optional)"):
                e1, e2, e3 = st.columns(3)
                daily = e1.number_input("Daily rate", 100, 1500, int(d["DailyRate"]))
                hourly = e2.number_input("Hourly rate", 30, 100, int(d["HourlyRate"]))
                monthly_rate = e3.number_input("Monthly rate", 2000, 27000, int(d["MonthlyRate"]))

            submitted = st.form_submit_button("Check attrition risk")

    if submitted:
        row = {
            "Age": age, "BusinessTravel": travel, "DailyRate": daily, "Department": department,
            "DistanceFromHome": distance, "Education": education, "EducationField": field,
            "EnvironmentSatisfaction": env_sat, "Gender": gender, "HourlyRate": hourly,
            "JobInvolvement": involvement, "JobLevel": level, "JobRole": role, "JobSatisfaction": job_sat,
            "MaritalStatus": marital, "MonthlyIncome": income, "MonthlyRate": monthly_rate,
            "NumCompaniesWorked": companies, "OverTime": overtime, "PercentSalaryHike": hike,
            "PerformanceRating": performance, "RelationshipSatisfaction": rel_sat,
            "StockOptionLevel": stock, "TotalWorkingYears": total_years, "TrainingTimesLastYear": training,
            "WorkLifeBalance": wlb, "YearsAtCompany": years_company, "YearsInCurrentRole": years_role,
            "YearsSinceLastPromotion": years_promo, "YearsWithCurrManager": years_mgr,
        }
        problem = validate(row)
        if problem:
            st.session_state["problem"] = problem
            st.session_state["result"] = None
        else:
            st.session_state["problem"] = None
            st.session_state["result"] = {"row": row, "p": predict(model_info, row)}

    with result_col:
        st.markdown('<span class="result-anchor"></span>', unsafe_allow_html=True)
        problem = st.session_state.get("problem")
        result = st.session_state.get("result")
        if problem:
            st.error(f"{problem} Fix this and check again.")
        elif result:
            result_panel(model_info, df, result)
        else:
            st.markdown(
                '<div class="panel empty"><h4>Your result appears here</h4>'
                "<p>You'll see the employee's estimated attrition risk, where it sits among the "
                f"{len(df):,} employees in the dataset, the factors behind it, and changes that could lower it.</p>"
                "<p><b>Tip:</b> start with overtime, tenure and satisfaction. They move the score most.</p></div>",
                unsafe_allow_html=True,
            )


def label_for(name: str, categorical: List[str]) -> str:
    body = name.split("__", 1)[1]
    if name.startswith("numerical__"):
        return FRIENDLY.get(body, body)
    feat = next(c for c in sorted(categorical, key=len, reverse=True) if body.startswith(c + "_"))
    return f"{FRIENDLY.get(feat, feat)}: {body[len(feat) + 1:].replace('_', ' ')}"


def drivers_tab(model_info: Dict[str, Any]) -> None:
    st.markdown('<div class="sec" style="margin-top:1.6rem">What the model weighs most</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sec-sub">Each bar is a coefficient from the Logistic Regression. Numeric features are standardised, '
        "so their sizes can be compared. Red raises predicted attrition risk and teal lowers it.</p>",
        unsafe_allow_html=True,
    )
    coef = pd.DataFrame(
        {"label": [label_for(n, model_info["categorical"]) for n in model_info["names"]], "coef": model_info["coef"]}
    )
    top = coef.reindex(coef["coef"].abs().sort_values(ascending=False).index).head(15).iloc[::-1]
    fig = go.Figure(
        go.Bar(
            x=top["coef"],
            y=top["label"],
            orientation="h",
            marker_color=[RISK if v > 0 else TEAL for v in top["coef"]],
            hovertemplate="%{y}<br>Coefficient: %{x:+.2f}<extra></extra>",
        )
    )
    fig.update_xaxes(title="Coefficient", zeroline=True, zerolinecolor=INK)
    st.plotly_chart(style_fig(fig, 560), width="stretch", config={"displayModeBar": False})
    st.markdown(
        '<p class="fine">Coefficients show how the model uses each feature, not what causes people to leave. '
        "Some features overlap (for example, job role and department), so read single bars with care.</p>",
        unsafe_allow_html=True,
    )


def model_tab(model_info: Dict[str, Any]) -> None:
    std, bal = model_info["standard"], model_info["balanced"]
    cv = model_info["cv"]
    n_train, n_test, test_left, test_stayed = model_info["split"]

    st.markdown('<div class="sec" style="margin-top:1.6rem">How well it works</div>', unsafe_allow_html=True)
    st.markdown(
        f'<p class="sec-sub">Results on {n_test} employees held out from training ({test_left} who left and {test_stayed} who stayed). '
        f"The selected model is <b>{model_info['final_name']}</b>, chosen for its higher cross-validated F1 score.</p>",
        unsafe_allow_html=True,
    )

    table = pd.DataFrame(
        {
            "Metric": ["Accuracy", "Precision (left)", "Recall (left)", "F1 (left)", "ROC-AUC", "5-fold CV mean F1"],
            "Logistic Regression": [f"{std['accuracy']:.1%}", f"{std['precision']:.1%}", f"{std['recall']:.1%}", f"{std['f1']:.1%}", f"{std['roc_auc']:.3f}", f"{cv['standard'].mean():.3f}"],
            "Class-weighted": [f"{bal['accuracy']:.1%}", f"{bal['precision']:.1%}", f"{bal['recall']:.1%}", f"{bal['f1']:.1%}", f"{bal['roc_auc']:.3f}", f"{cv['balanced'].mean():.3f}"],
        }
    )
    left, right = st.columns([1, 1.1], gap="large")
    with left:
        st.dataframe(table, hide_index=True, width="stretch")
        st.markdown(
            f"""
            <div class="panel">
              <h4>Reading the numbers</h4>
              <p class="fine" style="font-size:.95rem">About 84% of employees stay, so accuracy alone flatters any model.
              The standard model is precise but finds only {std['recall']:.0%} of leavers. The class-weighted model finds
              {bal['recall']:.0%} of leavers but wrongly flags many who stay ({bal['precision']:.0%} precision).
              Choose by what costs more: missing a leaver or reviewing someone who would have stayed.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=std["fpr"], y=std["tpr"], name=f"Logistic Regression (AUC {std['roc_auc']:.2f})", line=dict(color=TEAL, width=3)))
        fig.add_trace(go.Scatter(x=bal["fpr"], y=bal["tpr"], name=f"Class-weighted (AUC {bal['roc_auc']:.2f})", line=dict(color=RISK, width=3)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Random guess", line=dict(color=MUTED, dash="dash")))
        fig.update_xaxes(title="False positive rate")
        fig.update_yaxes(title="True positive rate")
        st.plotly_chart(style_fig(fig, 340), width="stretch", config={"displayModeBar": False})

    cm = model_info["selected_metrics"]["cm"]
    fig = go.Figure(
        go.Heatmap(
            z=cm,
            x=["Predicted: stayed", "Predicted: left"],
            y=["Actually stayed", "Actually left"],
            colorscale=[[0, "#EAF2F1"], [1, TEAL]],
            showscale=False,
            text=cm,
            texttemplate="%{text}",
            textfont=dict(size=22, family=DISPLAY_FONT),
            hovertemplate="%{y}, %{x}: %{z}<extra></extra>",
        )
    )
    fig.update_yaxes(autorange="reversed")
    a, b = st.columns([1, 1.1], gap="large")
    with a:
        st.markdown('<div class="sec" style="font-size:1.15rem">Confusion matrix</div>', unsafe_allow_html=True)
        st.plotly_chart(style_fig(fig, 280), width="stretch", config={"displayModeBar": False})
    with b:
        st.markdown(
            f"""
            <div class="sec" style="font-size:1.15rem">How it was built</div>
            <div class="panel">
              <p class="fine" style="font-size:.95rem">
              Data: IBM HR Analytics, {n_train + n_test:,} employees, 30 input features after dropping 4 ID or constant columns.<br><br>
              Preprocessing: standardised numeric features and one-hot encoded categories, inside a Scikit-learn pipeline fitted on the
              training set only, so no information leaks from the test set.<br><br>
              Validation: stratified 80:20 split ({n_train:,} training, {n_test} testing) and 5-fold cross-validation.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        '<p class="fine" style="margin-top:1.4rem">This is a baseline on a public benchmark dataset. Predictions are analytical indicators, '
        "not judgments about individual people, and should never be the sole basis for an employment decision.</p>",
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------------
# App
# ----------------------------------------------------------------------------
def main() -> None:
    inject_css()
    try:
        raw = load_data()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    model_info = build_model(raw)
    df = prepare_analysis(raw)

    masthead()
    tab1, tab2, tab3, tab4 = st.tabs(["Workforce", "Risk check", "Drivers", "Model"])
    with tab1:
        workforce_tab(df)
    with tab2:
        risk_tab(model_info, df)
    with tab3:
        drivers_tab(model_info)
    with tab4:
        model_tab(model_info)


if __name__ == "__main__":
    main()
