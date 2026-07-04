"""
==================================================================================
AI EMPLOYEE ATTRITION INTELLIGENCE PLATFORM
==================================================================================
Single-file Streamlit application. The machine learning pipeline below is an
EXACT reproduction of `Employee_Attrition_Rate.ipynb` — including its specific
(and unusual) preprocessing order. Nothing about the model logic has been
changed; only the presentation layer is new.

REPRODUCED PIPELINE (verbatim order from the notebook):

    1. Load HR-Employee-Attrition.csv
    2. Attrition: "Yes" -> 1, "No" -> 0
    3. Identify ALL numeric columns (this includes Attrition itself, since it
    is now 0/1 integer) and fit a single StandardScaler across all of them
    on the FULL dataset (before any train/test split — matching the
    notebook exactly).
    4. Drop high-correlation columns: JobLevel, PerformanceRating,
    YearsWithCurrManager, YearsInCurrentRole, TotalWorkingYears
    5. x = all remaining columns except Attrition (29 features)
    y = the SCALED Attrition column
    6. train_test_split(test_size=0.2, random_state=10)
    7. y_train / y_test cast to int — because Attrition was scaled with the
    rest of the numeric columns, its two possible values (~-0.438 for
    "No" and ~2.281 for "Yes") truncate to integers 0 and 2 respectively.
    This means the trained classifier's classes_ are literally [0, 2],
    not [0, 1]. The app below reads classes_ dynamically rather than
    assuming label values, so predictions are correctly interpreted
    regardless of this quirk.
    8. Model: make_pipeline(category_encoders.OneHotEncoder(), LogisticRegression())

For live predictions, a new employee's raw inputs are passed through the SAME
fitted StandardScaler (numeric columns) before being handed to the pipeline,
exactly mirroring how x_train/x_test were derived from the notebook's df1.
==================================================================================
"""

from __future__ import annotations

import io
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from category_encoders import OneHotEncoder as CategoryOneHotEncoder

    _ENCODER_IMPORT_ERROR: Optional[str] = None
except Exception as _exc:  # pragma: no cover
    CategoryOneHotEncoder = None  # type: ignore
    _ENCODER_IMPORT_ERROR = str(_exc)


CSV_PATH = "HR-Employee-Attrition.csv"
DROP_HIGH_CORR_COLUMNS = [
    "JobLevel", "PerformanceRating", "YearsWithCurrManager",
    "YearsInCurrentRole", "TotalWorkingYears",
]
CATEGORICAL_FEATURES = [
    "BusinessTravel", "Department", "EducationField", "Gender",
    "JobRole", "MaritalStatus", "Over18", "OverTime",
]

st.set_page_config(
    page_title="Attrition IQ — AI Employee Attrition Intelligence Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==================================================================================
# THEME
# ==================================================================================
THEME = {
    "bg_a": "#04050d", "bg_b": "#0a0c1c", "bg_c": "#11132b",
    "text": "#eef1fb", "muted": "#99a2c4",
    "surface": "rgba(255,255,255,0.045)", "surface_border": "rgba(255,255,255,0.09)",
    "accent1": "#8b5cf6", "accent2": "#22d3ee", "accent3": "#f472b6",
    "danger": "#f87171", "warn": "#fbbf24", "success": "#34d399",
}


# ==================================================================================
# CSS INJECTION
# ==================================================================================
def inject_css() -> None:
    t = THEME
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@300;400;500;600;700;800&display=swap');

        :root {{
            --bg-a:{t['bg_a']}; --bg-b:{t['bg_b']}; --bg-c:{t['bg_c']};
            --text:{t['text']}; --muted:{t['muted']};
            --surface:{t['surface']}; --surface-border:{t['surface_border']};
            --accent1:{t['accent1']}; --accent2:{t['accent2']}; --accent3:{t['accent3']};
            --danger:{t['danger']}; --warn:{t['warn']}; --success:{t['success']};
        }}

        html, body, [class*="css"] {{ font-family:'Inter',sans-serif; }}
        h1,h2,h3,h4,.grotesk {{ font-family:'Space Grotesk',sans-serif; }}
        #MainMenu, footer, header {{ visibility:hidden; }}
        .block-container {{ padding-top:1.5rem; max-width:1220px; }}

        .stApp {{
            background:
                radial-gradient(circle at 12% -8%, color-mix(in srgb, var(--accent1) 24%, transparent) 0%, transparent 42%),
                radial-gradient(circle at 92% 6%, color-mix(in srgb, var(--accent2) 18%, transparent) 0%, transparent 38%),
                linear-gradient(180deg, var(--bg-a) 0%, var(--bg-b) 55%, var(--bg-c) 100%);
            color: var(--text);
            background-size: 200% 200%, 200% 200%, 100% 100%;
            animation: bgshift 18s ease-in-out infinite;
        }}
        @keyframes bgshift {{
            0%,100% {{ background-position: 0% 0%, 100% 0%, 0 0; }}
            50% {{ background-position: 30% 20%, 70% 30%, 0 0; }}
        }}
        p, span, label, li, div {{ color: var(--text); }}
        .muted {{ color: var(--muted) !important; }}

        /* NAV */
        .topnav {{
            display:flex; align-items:center; justify-content:space-between;
            padding:0.9rem 1.6rem; border-radius:18px; background:var(--surface);
            border:1px solid var(--surface-border); backdrop-filter:blur(18px);
            margin-bottom:1.5rem;
        }}
        .brand {{ font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:1.15rem;
            display:flex; align-items:center; gap:0.55rem; letter-spacing:-0.02em; }}
        .brand-dot {{ width:11px; height:11px; border-radius:4px;
            background:linear-gradient(135deg,var(--accent1),var(--accent2));
            box-shadow:0 0 18px color-mix(in srgb, var(--accent1) 70%, transparent);
            animation: glow 2.4s ease-in-out infinite; }}
        @keyframes glow {{ 0%,100%{{opacity:1;}} 50%{{opacity:0.55;}} }}
        .nav-pill {{ font-size:0.78rem; font-weight:700; padding:0.32rem 0.9rem; border-radius:999px;
            background:color-mix(in srgb, var(--success) 16%, transparent);
            border:1px solid color-mix(in srgb, var(--success) 40%, transparent); color:var(--success); }}

        /* HERO */
        .hero {{
            position:relative; overflow:hidden; border-radius:28px; padding:3.6rem 2.6rem;
            background:linear-gradient(135deg, color-mix(in srgb, var(--accent1) 85%, black 8%) 0%, color-mix(in srgb, var(--accent3) 55%, black 5%) 55%, color-mix(in srgb, var(--accent2) 65%, black 5%) 100%);
            background-size:220% 220%; animation: heroshift 12s ease-in-out infinite, fadeUp 0.7s ease-out;
            margin-bottom:2rem; box-shadow:0 30px 80px -20px color-mix(in srgb, var(--accent1) 55%, transparent);
        }}
        @keyframes heroshift {{ 0%,100%{{background-position:0% 50%;}} 50%{{background-position:100% 50%;}} }}
        .hero::after {{ content:""; position:absolute; inset:0;
            background-image: radial-gradient(circle, rgba(255,255,255,0.35) 1px, transparent 1px);
            background-size: 26px 26px; opacity:0.10; pointer-events:none; }}
        .hero-eyebrow {{ display:inline-flex; align-items:center; gap:0.4rem; font-size:0.78rem;
            font-weight:700; letter-spacing:0.08em; text-transform:uppercase;
            background:rgba(255,255,255,0.16); border:1px solid rgba(255,255,255,0.3);
            padding:0.4rem 0.9rem; border-radius:999px; color:#fff; backdrop-filter:blur(6px); }}
        .hero-title {{ font-family:'Space Grotesk',sans-serif; font-weight:700; letter-spacing:-0.03em;
            font-size:3.1rem; line-height:1.1; color:#fff; margin:1.1rem 0 0.8rem 0;
            text-shadow:0 8px 30px rgba(0,0,0,0.25); }}
        .hero-sub {{ font-size:1.1rem; color:rgba(255,255,255,0.92); max-width:660px; line-height:1.6; }}
        .hero-badges {{ display:flex; gap:0.6rem; flex-wrap:wrap; margin-top:1.4rem; }}
        .hero-badge {{ background:rgba(255,255,255,0.16); border:1px solid rgba(255,255,255,0.3);
            padding:0.35rem 0.9rem; border-radius:999px; font-size:0.8rem; font-weight:700; color:#fff;
            animation: floaty 4s ease-in-out infinite; backdrop-filter:blur(6px); }}
        .hero-badge:nth-child(2) {{ animation-delay:0.6s; }}
        .hero-badge:nth-child(3) {{ animation-delay:1.2s; }}
        .hero-badge:nth-child(4) {{ animation-delay:1.8s; }}
        @keyframes floaty {{ 0%,100%{{transform:translateY(0);}} 50%{{transform:translateY(-6px);}} }}
        @keyframes fadeUp {{ from{{opacity:0; transform:translateY(20px);}} to{{opacity:1; transform:translateY(0);}} }}
        @keyframes popIn {{ 0%{{opacity:0; transform:scale(0.9);}} 100%{{opacity:1; transform:scale(1);}} }}
        @keyframes pulse {{ 0%,100%{{opacity:1;}} 50%{{opacity:0.55;}} }}

        /* GLASS CARD */
        .glass {{ background:var(--surface); border:1px solid var(--surface-border); border-radius:20px;
            padding:1.5rem 1.6rem; backdrop-filter:blur(16px); transition:all 0.25s ease;
            box-shadow:0 10px 30px rgba(0,0,0,0.15); }}
        .glass:hover {{ transform:translateY(-3px);
            border-color:color-mix(in srgb, var(--accent1) 45%, var(--surface-border));
            box-shadow:0 18px 44px color-mix(in srgb, var(--accent1) 16%, transparent); }}
        .glass-tight {{ padding:1rem 1.15rem; }}

        .section-title {{ font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:1.5rem;
            margin:0.3rem 0 1.05rem 0; display:flex; align-items:center; gap:0.55rem; }}
        .section-eyebrow {{ font-size:0.75rem; font-weight:700; text-transform:uppercase;
            letter-spacing:0.1em; color:var(--accent2); margin-bottom:0.3rem; }}

        /* KPI */
        .kpi {{ background:var(--surface); border:1px solid var(--surface-border); border-radius:18px;
            padding:1.2rem 1.3rem; transition:transform 0.2s ease; position:relative; overflow:hidden; }}
        .kpi:hover {{ transform:translateY(-3px); }}
        .kpi-icon {{ font-size:1.3rem; margin-bottom:0.35rem; }}
        .kpi-value {{ font-family:'Space Grotesk',sans-serif; font-size:1.7rem; font-weight:700; }}
        .kpi-label {{ font-size:0.78rem; color:var(--muted); text-transform:uppercase; letter-spacing:0.05em; margin-top:0.1rem; }}

        /* BUTTON */
        div.stButton > button {{
            background:linear-gradient(120deg,var(--accent1),var(--accent2)); color:#fff !important;
            font-weight:700; border:none; border-radius:14px; padding:0.75rem 1.6rem;
            box-shadow:0 12px 28px color-mix(in srgb, var(--accent1) 40%, transparent);
            transition:all 0.22s ease; width:100%;
        }}
        div.stButton > button:hover {{ transform:translateY(-2px); box-shadow:0 16px 36px color-mix(in srgb, var(--accent1) 55%, transparent); }}
        div.stButton > button:active {{ transform:translateY(0) scale(0.98); }}
        div.stButton > button p {{ color:#fff !important; }}

        /* BADGES */
        .badge {{ display:inline-flex; align-items:center; gap:0.35rem; font-size:0.78rem; font-weight:700;
            padding:0.32rem 0.85rem; border-radius:999px; margin:0.15rem; }}
        .badge-purple {{ background:color-mix(in srgb, var(--accent1) 18%, transparent); color:var(--accent1);
            border:1px solid color-mix(in srgb, var(--accent1) 40%, transparent); }}
        .badge-cyan {{ background:color-mix(in srgb, var(--accent2) 18%, transparent); color:var(--accent2);
            border:1px solid color-mix(in srgb, var(--accent2) 40%, transparent); }}
        .badge-green {{ background:color-mix(in srgb, var(--success) 18%, transparent); color:var(--success);
            border:1px solid color-mix(in srgb, var(--success) 40%, transparent); }}
        .badge-amber {{ background:color-mix(in srgb, var(--warn) 20%, transparent); color:var(--warn);
            border:1px solid color-mix(in srgb, var(--warn) 45%, transparent); }}
        .badge-red {{ background:color-mix(in srgb, var(--danger) 20%, transparent); color:var(--danger);
            border:1px solid color-mix(in srgb, var(--danger) 45%, transparent); }}
        .badge-pulse {{ animation: pulse 2s ease-in-out infinite; }}

        /* RESULT CARD */
        .result-card {{ border-radius:28px; padding:2.8rem 2.4rem; text-align:center; position:relative;
            overflow:hidden; margin-bottom:1.6rem; animation: popIn 0.5s cubic-bezier(.26,1.36,.44,1); }}
        .result-card.risk-high {{
            background:linear-gradient(135deg, color-mix(in srgb, var(--danger) 75%, black 12%), color-mix(in srgb, var(--accent3) 65%, black 10%));
            box-shadow:0 30px 70px -18px color-mix(in srgb, var(--danger) 50%, transparent);
        }}
        .result-card.risk-low {{
            background:linear-gradient(135deg, color-mix(in srgb, var(--success) 75%, black 10%), color-mix(in srgb, var(--accent2) 65%, black 10%));
            box-shadow:0 30px 70px -18px color-mix(in srgb, var(--success) 50%, transparent);
        }}
        .result-eyebrow {{ font-size:0.85rem; font-weight:700; letter-spacing:0.12em; text-transform:uppercase; color:rgba(255,255,255,0.88); }}
        .result-value {{ font-family:'Space Grotesk',sans-serif; font-size:2.8rem; font-weight:700; color:#fff; margin:0.35rem 0; }}
        .result-sub {{ color:rgba(255,255,255,0.88); font-size:1rem; }}

        /* LOADING */
        .ai-loading-box {{ text-align:center; padding:2.4rem 1rem; }}
        .ai-orb {{ width:72px; height:72px; border-radius:50%; margin:0 auto 1.2rem auto;
            background:conic-gradient(from 0deg, var(--accent1), var(--accent2), var(--accent3), var(--accent1));
            animation: spin 1.3s linear infinite; box-shadow:0 0 44px color-mix(in srgb, var(--accent1) 45%, transparent); }}
        @keyframes spin {{ to {{ transform:rotate(360deg); }} }}
        .ai-loading-text {{ font-size:1.05rem; font-weight:600; }}
        .pbar-track {{ width:100%; height:8px; border-radius:999px; background:var(--surface-border); overflow:hidden; }}
        .pbar-fill {{ height:100%; border-radius:999px; background:linear-gradient(90deg,var(--accent1),var(--accent2)); transition:width 0.3s ease; }}

        /* RECOMMENDATION CARD */
        .rec-card {{ border-left:4px solid var(--accent1); background:var(--surface);
            border-radius:14px; padding:1rem 1.2rem; margin-bottom:0.7rem; }}
        .rec-card.warn {{ border-left-color:var(--warn); }}
        .rec-card.danger {{ border-left-color:var(--danger); }}
        .rec-card.good {{ border-left-color:var(--success); }}

        /* DIVIDER */
        .divider-grad {{ height:1px; border:none; margin:2.2rem 0;
            background:linear-gradient(90deg, transparent, var(--surface-border), transparent); }}

        /* FOOTER */
        .footer {{ margin-top:2.4rem; padding:2.2rem; border-radius:22px; background:var(--surface);
            border:1px solid var(--surface-border); text-align:center; }}
        .footer-badge {{ display:inline-block; padding:0.32rem 0.85rem; border-radius:999px; margin:0.2rem;
            background:color-mix(in srgb, var(--accent1) 12%, transparent);
            border:1px solid color-mix(in srgb, var(--accent1) 30%, transparent); font-size:0.78rem; font-weight:600; }}

        .stTabs [data-baseweb="tab-list"] {{ gap:4px; }}
        .stTabs [data-baseweb="tab"] {{ border-radius:12px; padding:0.5rem 1rem; background:var(--surface);
            border:1px solid var(--surface-border); }}
        .stTabs [aria-selected="true"] {{ background:linear-gradient(120deg,var(--accent1),var(--accent2)) !important; color:#fff !important; }}

        section[data-testid="stSidebar"] {{ background:linear-gradient(180deg, var(--bg-c), var(--bg-a));
            border-right:1px solid var(--surface-border); }}

        .logo-box {{ display:flex; align-items:center; gap:0.6rem; padding:0.9rem; border-radius:16px;
            background:var(--surface); border:1px solid var(--surface-border); margin-bottom:1rem; }}
        .logo-mark {{ width:34px; height:34px; border-radius:10px;
            background:linear-gradient(135deg,var(--accent1),var(--accent2));
            display:flex; align-items:center; justify-content:center; font-size:1.1rem; }}

        ::-webkit-scrollbar {{ width:8px; height:8px; }}
        ::-webkit-scrollbar-thumb {{ background:var(--surface-border); border-radius:8px; }}

        @media (max-width:768px) {{
            .hero-title {{ font-size:2rem; }}
            .result-value {{ font-size:2rem; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ==================================================================================
# DATA / MODEL PIPELINE — EXACT REPLICA OF THE NOTEBOOK
# ==================================================================================
@st.cache_data(show_spinner=False)
def load_raw_data(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


@st.cache_resource(show_spinner=False)
def train_model(_raw_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Reproduces, in the exact order of the notebook:
        df1['Attrition'] = 1/0
        fit StandardScaler on ALL numeric columns (incl. Attrition) of the
            FULL dataset, before any split
        drop 5 high-correlation columns
        x / y split, train_test_split(random_state=10)
        y cast to int (produces classes {0, 2} due to the scaling above)
        make_pipeline(OneHotEncoder(), LogisticRegression()).fit(x_train, y_train)
    """
    if CategoryOneHotEncoder is None:
        raise ImportError(
            "The 'category_encoders' package is required to reproduce the "
            "notebook's exact encoding step. Install it with: "
            f"pip install category_encoders. Original error: {_ENCODER_IMPORT_ERROR}"
        )

    df1 = _raw_df.copy()
    df1["Attrition"] = df1["Attrition"].apply(lambda v: 1 if v == "Yes" else 0)

    numerical_columns = list(df1.select_dtypes(include=["float", "int"]).columns)
    scaler = StandardScaler()
    df1[numerical_columns] = scaler.fit_transform(df1[numerical_columns])

    df1 = df1.drop(columns=DROP_HIGH_CORR_COLUMNS)

    x = df1.drop(["Attrition"], axis="columns")
    y = df1.Attrition

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=10
    )
    y_train = y_train.astype("int")
    y_test = y_test.astype("int")

    model = make_pipeline(CategoryOneHotEncoder(), LogisticRegression(max_iter=1000))
    model.fit(x_train, y_train)

    accuracy = float(model.score(x_test, y_test))
    classes = list(model.named_steps["logisticregression"].classes_)
    leave_class = max(classes)  # the truncated "Yes" label (2), robust either way
    stay_class = min(classes)   # the truncated "No" label (0)

    # Feature importance (best-effort extraction from fitted coefficients)
    feature_importance: Optional[pd.DataFrame] = None
    try:
        ohe = model.named_steps["onehotencoder"]
        lr = model.named_steps["logisticregression"]
        try:
            encoded_cols = list(ohe.get_feature_names_out())
        except Exception:
            encoded_cols = list(ohe.feature_names)
        coefs = lr.coef_[0]
        fi_df = pd.DataFrame({"feature": encoded_cols, "coef": coefs})
        fi_df["abs_coef"] = fi_df["coef"].abs()

        rows = []
        for cat in CATEGORICAL_FEATURES:
            mask = fi_df["feature"].str.startswith(cat)
            if mask.any():
                rows.append({
                    "feature": cat,
                    "coef": fi_df.loc[mask, "coef"].mean(),
                    "abs_coef": fi_df.loc[mask, "abs_coef"].mean(),
                })
        numeric_mask = ~fi_df["feature"].apply(
            lambda f: any(f.startswith(c) for c in CATEGORICAL_FEATURES)
        )
        rows.extend(fi_df[numeric_mask].to_dict("records"))
        feature_importance = pd.DataFrame(rows).sort_values("abs_coef", ascending=False)
    except Exception:
        feature_importance = None

    return {
        "model": model,
        "scaler": scaler,
        "numerical_columns": numerical_columns,
        "feature_columns": list(x.columns),
        "accuracy": accuracy,
        "leave_class": leave_class,
        "stay_class": stay_class,
        "n_train": len(x_train),
        "n_test": len(x_test),
        "feature_importance": feature_importance,
    }


def build_model_input(raw_input: Dict[str, Any], model_info: Dict[str, Any]) -> pd.DataFrame:
    """
    Scale the numeric portion of a raw employee record with the SAME fitted
    StandardScaler used in training, then assemble a single-row DataFrame
    with columns matching x_train exactly (categorical columns pass through
    unscaled, to be one-hot encoded by the pipeline itself).
    """
    scaler: StandardScaler = model_info["scaler"]
    numerical_columns: List[str] = model_info["numerical_columns"]
    feature_columns: List[str] = model_info["feature_columns"]

    # Build a full row across every column the scaler was originally fit on.
    # Columns dropped from x (Attrition, JobLevel, etc.) get a neutral
    # placeholder — StandardScaler scales each column independently, so this
    # placeholder never influences the scaled value of any other column.
    full_row = {col: raw_input.get(col, 0.0) for col in numerical_columns}
    full_row_df = pd.DataFrame([full_row], columns=numerical_columns)
    scaled_values = scaler.transform(full_row_df)[0]
    scaled_map = dict(zip(numerical_columns, scaled_values))

    final_row: Dict[str, Any] = {}
    for col in feature_columns:
        if col in CATEGORICAL_FEATURES:
            final_row[col] = raw_input[col]
        else:
            final_row[col] = scaled_map[col]

    return pd.DataFrame([final_row], columns=feature_columns)


def predict_attrition(model_info: Dict[str, Any], raw_input: Dict[str, Any]) -> Dict[str, float]:
    """Run the exact fitted pipeline and return interpreted probabilities."""
    model = model_info["model"]
    input_df = build_model_input(raw_input, model_info)

    predicted_class = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]
    classes = list(model.named_steps["logisticregression"].classes_)

    leave_idx = classes.index(model_info["leave_class"])
    stay_idx = classes.index(model_info["stay_class"])

    return {
        "will_leave": bool(predicted_class == model_info["leave_class"]),
        "proba_leave": float(proba[leave_idx]),
        "proba_stay": float(proba[stay_idx]),
    }


# ==================================================================================
# ANALYTICAL INSIGHTS (illustrative, rule-based — layered on top of the raw prediction)
# ==================================================================================
def compute_insights(raw_df: pd.DataFrame, raw_input: Dict[str, Any],
                    pred: Dict[str, float]) -> Dict[str, Any]:
    dept_avg_income = raw_df.loc[raw_df["Department"] == raw_input["Department"], "MonthlyIncome"].mean()

    notes: List[Tuple[str, str]] = []  # (severity, text)
    if raw_input["OverTime"] == "Yes":
        notes.append(("danger", "Employee regularly works overtime — a strong historical attrition signal."))
    if raw_input["MonthlyIncome"] < dept_avg_income:
        notes.append(("warn", f"Monthly income (₹{raw_input['MonthlyIncome']:,}) is below the "
                                f"{raw_input['Department']} department average (₹{dept_avg_income:,.0f})."))
    if raw_input["YearsAtCompany"] <= 2:
        notes.append(("warn", "Short tenure at the company — early-tenure employees show higher attrition."))
    if raw_input["JobSatisfaction"] <= 2:
        notes.append(("danger", "Low job satisfaction reported by the employee."))
    if raw_input["EnvironmentSatisfaction"] <= 2:
        notes.append(("warn", "Low environment satisfaction reported."))
    if raw_input["WorkLifeBalance"] <= 2:
        notes.append(("warn", "Poor work-life balance reported."))
    if raw_input["YearsSinceLastPromotion"] >= 5:
        notes.append(("warn", "No promotion in 5+ years — stagnant career progression."))
    if raw_input["NumCompaniesWorked"] >= 5:
        notes.append(("warn", "History of frequent job changes across companies."))
    if raw_input["BusinessTravel"] == "Travel_Frequently":
        notes.append(("warn", "Frequent business travel may contribute to burnout."))
    if not notes:
        notes.append(("good", "No major risk indicators detected across key workforce signals."))

    engagement_inputs = [
        raw_input["EnvironmentSatisfaction"], raw_input["JobSatisfaction"],
        raw_input["JobInvolvement"], raw_input["WorkLifeBalance"],
    ]
    engagement_score = float(np.clip((np.mean(engagement_inputs) - 1) / 3 * 100, 0, 100))
    stability_score = float(np.clip(raw_input["YearsAtCompany"] / 15 * 100, 0, 100))
    risk_score = pred["proba_leave"] * 100
    confidence = max(pred["proba_leave"], pred["proba_stay"]) * 100

    return {
        "notes": notes,
        "engagement_score": engagement_score,
        "stability_score": stability_score,
        "risk_score": risk_score,
        "confidence": confidence,
        "dept_avg_income": dept_avg_income,
    }


# ==================================================================================
# UI FRAGMENTS
# ==================================================================================
def render_topnav() -> None:
    st.markdown(
        """
        <div class="topnav">
            <div class="brand"><span class="brand-dot"></span> ATTRITION IQ</div>
            <div class="nav-pill">● Model Online</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero">
            <span class="hero-eyebrow">✦ Workforce Intelligence Engine</span>
            <div class="hero-title">👨‍💼 AI Employee Attrition<br>Intelligence Platform</div>
            <div class="hero-sub">
                Predict employee attrition using machine learning and workforce
                analytics — surfacing the risk signals HR teams need before
                talent walks out the door.
            </div>
            <div class="hero-badges">
                <span class="hero-badge">🤖 AI Powered</span>
                <span class="hero-badge">🧬 Machine Learning</span>
                <span class="hero-badge">📊 HR Analytics</span>
                <span class="hero-badge">🔮 Predictive Intelligence</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(model_info: Dict[str, Any], raw_df: pd.DataFrame) -> None:
    with st.sidebar:
        st.markdown(
            """
            <div class="logo-box">
                <div class="logo-mark">🧠</div>
                <div><b>Attrition IQ</b><br><span class="muted" style="font-size:0.75rem;">Workforce Intelligence</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### 🧭 Navigation")
        st.radio(
            "Jump to", ["Predict", "Analytics", "Model Info"],
            label_visibility="collapsed", key="nav_choice",
        )

        st.markdown('<hr class="divider-grad">', unsafe_allow_html=True)

        st.markdown("### 📋 Project Information")
        st.markdown(
            """
            <div class="glass glass-tight">
                An enterprise HR analytics tool that estimates the likelihood
                of an employee leaving the organization, using a supervised
                machine learning model trained on historical workforce data.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### 🧠 Model Information")
        st.markdown(
            f"""
            <div class="glass glass-tight">
                <b>Algorithm:</b> Logistic Regression<br>
                <b>Encoding:</b> One-Hot (category_encoders)<br>
                <b>Scaling:</b> StandardScaler (all numeric features)<br>
                <b>Accuracy:</b> {model_info['accuracy']*100:.2f}%<br>
                <b>Train / Test:</b> {model_info['n_train']:,} / {model_info['n_test']:,}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### 🗂️ Dataset")
        st.markdown(
            f"""
            <div class="glass glass-tight">
                <b>Source:</b> HR-Employee-Attrition.csv<br>
                <b>Total Employees:</b> {len(raw_df):,}<br>
                <b>Attrition Rate:</b> {(raw_df['Attrition']=='Yes').mean()*100:.1f}%<br>
                <b>Features Used:</b> {len(model_info['feature_columns'])}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### 👨‍💻 Developer")
        st.markdown(
            """
            <div class="glass glass-tight">
                <b>Manav Sharma</b><br>
                <span class="muted">ML Engineer · People Analytics</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### 🧰 Tech Stack")
        st.markdown(
            """
            <div class="glass glass-tight">
                Python · Pandas · NumPy<br>Scikit-Learn · Category Encoders<br>Streamlit · Plotly
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<hr class="divider-grad">', unsafe_allow_html=True)

        st.markdown("### 🕘 Prediction History")
        if st.session_state.history:
            for h in reversed(st.session_state.history[-5:]):
                risk_tag = "🔴" if h["will_leave"] else "🟢"
                st.markdown(
                    f"""
                    <div class="glass glass-tight" style="margin-bottom:0.5rem;">
                        {risk_tag} <b>{h['risk_score']:.0f}% risk</b> — {h['role']}<br>
                        <span class="muted" style="font-size:0.75rem;">{h['department']} · {h['time']}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            hist_df = pd.DataFrame(st.session_state.history)
            buf = io.StringIO()
            hist_df.to_csv(buf, index=False)
            st.download_button("⬇️ Download History (CSV)", data=buf.getvalue(),
                                file_name="attrition_prediction_history.csv", mime="text/csv",
                                use_container_width=True)
        else:
            st.markdown('<div class="glass glass-tight muted">No predictions yet this session.</div>',
                        unsafe_allow_html=True)

        if st.button("🔄 Reset Inputs", use_container_width=True):
            for key in list(st.session_state.keys()):
                if key.startswith("inp_"):
                    del st.session_state[key]
            st.session_state.prediction_result = None
            st.rerun()


def render_kpi(icon: str, value: str, label: str) -> str:
    return f"""
        <div class="kpi">
            <div class="kpi-icon">{icon}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-label">{label}</div>
        </div>
    """


def render_kpi_row(raw_df: pd.DataFrame, model_info: Dict[str, Any]) -> None:
    st.markdown('<div class="section-title">📊 Platform Overview</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(render_kpi("👥", f"{len(raw_df):,}", "Total Employees"), unsafe_allow_html=True)
    with c2:
        st.markdown(render_kpi("🎯", f"{model_info['accuracy']*100:.1f}%", "Prediction Accuracy"), unsafe_allow_html=True)
    with c3:
        st.markdown(render_kpi("🧬", f"{len(model_info['feature_columns'])}", "Features Used"), unsafe_allow_html=True)
    with c4:
        st.markdown(render_kpi("🤖", "Logistic Regression", "Model Type"), unsafe_allow_html=True)


def render_gauge(value: float, title: str, color: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value,
        number={"suffix": "%", "font": {"size": 26, "family": "Space Grotesk"}},
        title={"text": title, "font": {"size": 13}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": THEME["muted"]},
            "bar": {"color": color}, "bgcolor": "rgba(0,0,0,0)", "borderwidth": 0,
            "steps": [
                {"range": [0, 40], "color": "rgba(52,211,153,0.16)"},
                {"range": [40, 70], "color": "rgba(251,191,36,0.16)"},
                {"range": [70, 100], "color": "rgba(248,113,113,0.16)"},
            ],
        },
    ))
    fig.update_layout(height=220, margin=dict(l=15, r=15, t=40, b=10),
                    paper_bgcolor="rgba(0,0,0,0)", font_color=THEME["text"])
    return fig


# ==================================================================================
# INPUT FORM (grouped sections)
# ==================================================================================
def render_input_form(raw_df: pd.DataFrame) -> Dict[str, Any]:
    st.markdown('<div class="section-title">🧾 Employee Profile</div>', unsafe_allow_html=True)

    tabs = st.tabs(["👤 Personal", "💼 Job", "💰 Compensation", "🌿 Work Environment",
                    "📈 Performance", "🧳 Lifestyle"])

    inputs: Dict[str, Any] = {}

    with tabs[0]:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            inputs["Age"] = st.slider("🎂 Age", 18, 60, st.session_state.get("inp_Age", 35), key="inp_Age",
                                    help="Employee's age in years.")
            inputs["Gender"] = st.radio("⚧ Gender", ["Male", "Female"], horizontal=True,
                                        index=["Male", "Female"].index(st.session_state.get("inp_Gender", "Male")),
                                        key="inp_Gender")
        with c2:
            inputs["MaritalStatus"] = st.selectbox("💍 Marital Status", ["Single", "Married", "Divorced"],
                                                    index=["Single", "Married", "Divorced"].index(
                                                        st.session_state.get("inp_MaritalStatus", "Single")),
                                                    key="inp_MaritalStatus")
            inputs["DistanceFromHome"] = st.slider("🏠 Distance From Home (km)", 1, 29,
                                                    st.session_state.get("inp_DistanceFromHome", 9),
                                                    key="inp_DistanceFromHome",
                                                    help="Commuting distance from home to office.")
        st.markdown("</div>", unsafe_allow_html=True)

    with tabs[1]:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            inputs["Department"] = st.selectbox(
                "🏢 Department", ["Sales", "Research & Development", "Human Resources"],
                index=["Sales", "Research & Development", "Human Resources"].index(
                    st.session_state.get("inp_Department", "Research & Development")),
                key="inp_Department")
            inputs["JobRole"] = st.selectbox(
                "🧑‍💻 Job Role",
                ["Sales Executive", "Research Scientist", "Laboratory Technician",
                "Manufacturing Director", "Healthcare Representative", "Manager",
                "Sales Representative", "Research Director", "Human Resources"],
                key="inp_JobRole")
            inputs["EducationField"] = st.selectbox(
                "🎓 Education Field",
                ["Life Sciences", "Other", "Medical", "Marketing", "Technical Degree", "Human Resources"],
                key="inp_EducationField")
        with c2:
            inputs["BusinessTravel"] = st.selectbox(
                "✈️ Business Travel", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"],
                key="inp_BusinessTravel", help="How frequently the employee travels for work.")
            inputs["Education"] = st.select_slider(
                "📚 Education Level", options=[1, 2, 3, 4, 5],
                value=st.session_state.get("inp_Education", 3), key="inp_Education",
                help="1 = Below College, 5 = Doctorate")
        st.markdown("</div>", unsafe_allow_html=True)

    with tabs[2]:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            inputs["MonthlyIncome"] = st.number_input("💵 Monthly Income (₹)", min_value=1000, max_value=20000,
                                                        value=st.session_state.get("inp_MonthlyIncome", 6500),
                                                        step=100, key="inp_MonthlyIncome")
            inputs["DailyRate"] = st.number_input("📅 Daily Rate (₹)", min_value=100, max_value=1500,
                                                    value=st.session_state.get("inp_DailyRate", 800),
                                                    step=10, key="inp_DailyRate")
            inputs["HourlyRate"] = st.number_input("⏱️ Hourly Rate (₹)", min_value=30, max_value=100,
                                                    value=st.session_state.get("inp_HourlyRate", 65),
                                                    step=1, key="inp_HourlyRate")
        with c2:
            inputs["MonthlyRate"] = st.number_input("📈 Monthly Rate (₹)", min_value=2000, max_value=27000,
                                                    value=st.session_state.get("inp_MonthlyRate", 14300),
                                                    step=100, key="inp_MonthlyRate")
            inputs["PercentSalaryHike"] = st.slider("📊 Percent Salary Hike", 11, 25,
                                                    st.session_state.get("inp_PercentSalaryHike", 15),
                                                    key="inp_PercentSalaryHike")
            inputs["StockOptionLevel"] = st.select_slider("📦 Stock Option Level", options=[0, 1, 2, 3],
                                                            value=st.session_state.get("inp_StockOptionLevel", 1),
                                                            key="inp_StockOptionLevel")
        st.markdown("</div>", unsafe_allow_html=True)

    with tabs[3]:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            inputs["EnvironmentSatisfaction"] = st.select_slider(
                "🌿 Environment Satisfaction", options=[1, 2, 3, 4],
                value=st.session_state.get("inp_EnvironmentSatisfaction", 3),
                key="inp_EnvironmentSatisfaction", help="1 = Low, 4 = Very High")
            inputs["JobInvolvement"] = st.select_slider(
                "🎯 Job Involvement", options=[1, 2, 3, 4],
                value=st.session_state.get("inp_JobInvolvement", 3), key="inp_JobInvolvement")
        with c2:
            inputs["WorkLifeBalance"] = st.select_slider(
                "⚖️ Work-Life Balance", options=[1, 2, 3, 4],
                value=st.session_state.get("inp_WorkLifeBalance", 3), key="inp_WorkLifeBalance")
            inputs["OverTime"] = st.radio("🕑 Works Overtime?", ["No", "Yes"], horizontal=True,
                                        key="inp_OverTime")
        st.markdown("</div>", unsafe_allow_html=True)

    with tabs[4]:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            inputs["JobSatisfaction"] = st.select_slider(
                "😊 Job Satisfaction", options=[1, 2, 3, 4],
                value=st.session_state.get("inp_JobSatisfaction", 3), key="inp_JobSatisfaction")
            inputs["RelationshipSatisfaction"] = st.select_slider(
                "🤝 Relationship Satisfaction", options=[1, 2, 3, 4],
                value=st.session_state.get("inp_RelationshipSatisfaction", 3),
                key="inp_RelationshipSatisfaction")
        with c2:
            inputs["YearsSinceLastPromotion"] = st.slider("🏅 Years Since Last Promotion", 0, 15,
                                                            st.session_state.get("inp_YearsSinceLastPromotion", 2),
                                                            key="inp_YearsSinceLastPromotion")
            inputs["TrainingTimesLastYear"] = st.slider("📘 Training Sessions Last Year", 0, 6,
                                                        st.session_state.get("inp_TrainingTimesLastYear", 3),
                                                        key="inp_TrainingTimesLastYear")
        st.markdown("</div>", unsafe_allow_html=True)

    with tabs[5]:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            inputs["NumCompaniesWorked"] = st.slider("🏢 Number of Companies Worked At", 0, 9,
                                                        st.session_state.get("inp_NumCompaniesWorked", 2),
                                                        key="inp_NumCompaniesWorked")
        with c2:
            inputs["YearsAtCompany"] = st.slider("📆 Years At Company", 0, 40,
                                                    st.session_state.get("inp_YearsAtCompany", 5),
                                                    key="inp_YearsAtCompany")
        st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("⚙️ Advanced / System Fields (dataset-constant, included for pipeline fidelity)"):
        st.caption(
            "These fields are constant across the entire training dataset "
            "(e.g. every employee has Over18 = 'Y') and were still part of "
            "the notebook's feature set — kept here for exact prediction "
            "fidelity, not because they carry predictive signal."
        )
        inputs["EmployeeCount"] = 1.0
        inputs["StandardHours"] = 80.0
        inputs["Over18"] = "Y"
        inputs["EmployeeNumber"] = st.number_input(
            "Employee Number (ID)", min_value=1, max_value=3000,
            value=st.session_state.get("inp_EmployeeNumber", 1025), key="inp_EmployeeNumber")

    return inputs


def run_ai_analysis(model_info: Dict[str, Any], raw_df: pd.DataFrame, inputs: Dict[str, Any]) -> None:
    messages = [
        "🧠 Loading employee profile...",
        "📊 Evaluating workforce patterns...",
        "🤖 Running prediction model...",
        "📈 Calculating attrition probability...",
        "✅ Analysis complete.",
    ]
    placeholder = st.empty()
    progress = st.empty()
    for i, msg in enumerate(messages):
        pct = int(((i + 1) / len(messages)) * 100)
        placeholder.markdown(
            f"""<div class="ai-loading-box"><div class="ai-orb"></div>
            <div class="ai-loading-text">{msg}</div></div>""",
            unsafe_allow_html=True,
        )
        progress.markdown(f'<div class="pbar-track"><div class="pbar-fill" style="width:{pct}%;"></div></div>',
                        unsafe_allow_html=True)
        time.sleep(0.45)

    try:
        pred = predict_attrition(model_info, inputs)
    except Exception as exc:  # noqa: BLE001
        placeholder.empty()
        progress.empty()
        st.error(f"❌ Prediction failed: {exc}")
        return

    placeholder.empty()
    progress.empty()

    insights = compute_insights(raw_df, inputs, pred)

    st.session_state.prediction_result = pred
    st.session_state.final_inputs = dict(inputs)
    st.session_state.insights = insights
    st.session_state.history.append({
        "will_leave": pred["will_leave"],
        "risk_score": insights["risk_score"],
        "role": inputs["JobRole"],
        "department": inputs["Department"],
        "time": datetime.now().strftime("%H:%M:%S"),
    })


def render_results() -> None:
    pred = st.session_state.prediction_result
    inputs = st.session_state.final_inputs
    insights = st.session_state.insights

    if pred["will_leave"]:
        st.markdown(
            f"""
            <div class="result-card risk-high">
                <div class="result-eyebrow">🔴 High Attrition Risk</div>
                <div class="result-value">{insights['risk_score']:.1f}% Probability</div>
                <div class="result-sub">Confidence: {insights['confidence']:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="result-card risk-low">
                <div class="result-eyebrow">🟢 Low Attrition Risk</div>
                <div class="result-value">{100-insights['risk_score']:.1f}% Likely to Stay</div>
                <div class="result-sub">Confidence: {insights['confidence']:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    g1, g2, g3 = st.columns(3)
    with g1:
        st.plotly_chart(render_gauge(insights["risk_score"], "Risk Score", THEME["danger"]),
                        use_container_width=True, config={"displayModeBar": False})
    with g2:
        st.plotly_chart(render_gauge(insights["confidence"], "AI Confidence", THEME["accent2"]),
                        use_container_width=True, config={"displayModeBar": False})
    with g3:
        label = "Engagement Score" if not pred["will_leave"] else "Stability Score"
        val = insights["engagement_score"] if not pred["will_leave"] else insights["stability_score"]
        st.plotly_chart(render_gauge(val, label, THEME["success"]),
                        use_container_width=True, config={"displayModeBar": False})

    st.markdown('<div class="section-title">💡 AI Insights</div>', unsafe_allow_html=True)
    severity_class = {"danger": "danger", "warn": "warn", "good": "good"}
    severity_icon = {"danger": "🔴", "warn": "🟡", "good": "🟢"}
    for sev, text in insights["notes"]:
        st.markdown(
            f'<div class="rec-card {severity_class[sev]}">{severity_icon[sev]} {text}</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-title">🎯 Recommended Actions</div>', unsafe_allow_html=True)
    if pred["will_leave"]:
        st.markdown(
            """
            <div class="rec-card danger">🚨 <b>Recommended HR Action:</b> Schedule a 1:1 retention
            conversation within the next two weeks to understand underlying concerns.</div>
            <div class="rec-card warn">💡 <b>Retention Strategy:</b> Review compensation against
            department benchmarks, revisit workload/overtime balance, and discuss growth or
            promotion pathways.</div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="rec-card good">✅ <b>Recommendation:</b> Continue current engagement
            practices. Consider this employee for mentorship or leadership development
            opportunities to sustain long-term retention.</div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-title">📋 Employee Summary</div>', unsafe_allow_html=True)
    cols = st.columns(6)
    summary = [
        ("👤", f"{inputs['Age']} yrs", "Age"),
        ("🏢", inputs["Department"], "Department"),
        ("🧑‍💻", inputs["JobRole"], "Role"),
        ("💵", f"₹{inputs['MonthlyIncome']:,}", "Income"),
        ("📆", f"{inputs['YearsAtCompany']} yrs", "Tenure"),
        ("🕑", inputs["OverTime"], "Overtime"),
    ]
    for col, (icon, val, label) in zip(cols, summary):
        with col:
            st.markdown(render_kpi(icon, str(val), label), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    report = (
        f"ATTRITION IQ — Employee Risk Report\n"
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        f"Department: {inputs['Department']}\nJob Role: {inputs['JobRole']}\n"
        f"Age: {inputs['Age']}\nMonthly Income: Rs. {inputs['MonthlyIncome']:,}\n"
        f"Years At Company: {inputs['YearsAtCompany']}\nOverTime: {inputs['OverTime']}\n\n"
        f"Prediction: {'High Attrition Risk' if pred['will_leave'] else 'Low Attrition Risk'}\n"
        f"Risk Score: {insights['risk_score']:.1f}%\n"
        f"AI Confidence: {insights['confidence']:.1f}%\n"
    )
    st.download_button("⬇️ Export Prediction Report (.txt)", data=report,
                        file_name="employee_attrition_report.txt", mime="text/plain",
                        use_container_width=True)


# ==================================================================================
# ANALYTICS
# ==================================================================================
def render_analytics(raw_df: pd.DataFrame, model_info: Dict[str, Any]) -> None:
    st.markdown('<hr class="divider-grad">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📊 Workforce Analytics Dashboard</div>', unsafe_allow_html=True)

    layout = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter, sans-serif", color=THEME["text"]))

    tabs = st.tabs(["📉 Attrition", "🏢 Department", "💰 Salary", "🎂 Age", "⚧ Gender",
                    "✈️ Travel", "😊 Satisfaction", "📈 Income", "🔥 Correlation", "🧬 Feature Impact"])

    with tabs[0]:
        counts = raw_df["Attrition"].value_counts()
        fig = px.pie(values=counts.values, names=counts.index, hole=0.55,
                    color_discrete_sequence=[THEME["success"], THEME["danger"]],
                    title="Overall Attrition Distribution")
        fig.update_layout(**layout)
        st.plotly_chart(fig, use_container_width=True)

    with tabs[1]:
        dept_attr = raw_df.groupby(["Department", "Attrition"]).size().reset_index(name="count")
        fig = px.bar(dept_attr, x="Department", y="count", color="Attrition", barmode="group",
                    color_discrete_sequence=[THEME["success"], THEME["danger"]],
                    title="Attrition by Department")
        fig.update_layout(**layout)
        st.plotly_chart(fig, use_container_width=True)

    with tabs[2]:
        fig = px.histogram(raw_df, x="MonthlyIncome", nbins=40, color="Attrition",
                            color_discrete_sequence=[THEME["success"], THEME["danger"]],
                            title="Monthly Income Distribution by Attrition")
        fig.update_layout(**layout, bargap=0.05)
        st.plotly_chart(fig, use_container_width=True)

    with tabs[3]:
        fig = px.histogram(raw_df, x="Age", nbins=30, color="Attrition",
                            color_discrete_sequence=[THEME["success"], THEME["danger"]],
                            title="Age Distribution by Attrition")
        fig.update_layout(**layout, bargap=0.05)
        st.plotly_chart(fig, use_container_width=True)

    with tabs[4]:
        gender_attr = raw_df.groupby(["Gender", "Attrition"]).size().reset_index(name="count")
        fig = px.bar(gender_attr, x="Gender", y="count", color="Attrition", barmode="group",
                    color_discrete_sequence=[THEME["success"], THEME["danger"]],
                    title="Attrition by Gender")
        fig.update_layout(**layout)
        st.plotly_chart(fig, use_container_width=True)

    with tabs[5]:
        travel_attr = raw_df.groupby(["BusinessTravel", "Attrition"]).size().reset_index(name="count")
        fig = px.bar(travel_attr, x="BusinessTravel", y="count", color="Attrition", barmode="group",
                    color_discrete_sequence=[THEME["success"], THEME["danger"]],
                    title="Attrition by Business Travel Frequency")
        fig.update_layout(**layout)
        st.plotly_chart(fig, use_container_width=True)

    with tabs[6]:
        sat_attr = raw_df.groupby(["JobSatisfaction", "Attrition"]).size().reset_index(name="count")
        fig = px.bar(sat_attr, x="JobSatisfaction", y="count", color="Attrition", barmode="group",
                    color_discrete_sequence=[THEME["success"], THEME["danger"]],
                    title="Job Satisfaction vs Attrition")
        fig.update_layout(**layout)
        st.plotly_chart(fig, use_container_width=True)

    with tabs[7]:
        fig = px.box(raw_df, x="Department", y="MonthlyIncome", color="Attrition",
                    color_discrete_sequence=[THEME["success"], THEME["danger"]],
                    title="Monthly Income by Department & Attrition")
        fig.update_layout(**layout)
        st.plotly_chart(fig, use_container_width=True)

    with tabs[8]:
        numeric_df = raw_df.select_dtypes(include=["float", "int"])
        corr = numeric_df.corr()
        fig = px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                        title="Correlation Heatmap (Numeric Features)")
        fig.update_layout(**layout, height=650)
        st.plotly_chart(fig, use_container_width=True)

    with tabs[9]:
        fi = model_info.get("feature_importance")
        if fi is not None and not fi.empty:
            fig = px.bar(fi.head(20), x="abs_coef", y="feature", orientation="h",
                        color="abs_coef", color_continuous_scale="Sunset",
                        title="Feature Impact on Attrition Prediction (|coefficient|)")
            fig.update_layout(**layout, coloraxis_showscale=False,
                            yaxis=dict(categoryorder="total ascending"))
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Derived directly from the fitted LogisticRegression coefficients. "
                    "Categorical variables are averaged across their one-hot columns for readability.")
        else:
            st.info("Feature importance could not be extracted from this encoder version.")


def render_footer(model_info: Dict[str, Any]) -> None:
    st.markdown('<div class="section-title">🧠 About the Model</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            f"""
            <div class="glass">
                <b>Model:</b> Employee Attrition Classifier<br>
                <b>Algorithm:</b> Logistic Regression (scikit-learn)<br>
                <b>Encoding:</b> One-Hot Encoding via <code>category_encoders</code><br>
                <b>Scaling:</b> StandardScaler across all numeric features<br>
                <b>Pipeline:</b> <code>make_pipeline(OneHotEncoder(), LogisticRegression())</code>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="glass">
                <b>Features:</b> {len(model_info['feature_columns'])} workforce attributes<br>
                <b>Target:</b> Attrition (Yes/No)<br>
                <b>Accuracy:</b> {model_info['accuracy']*100:.2f}%<br>
                <b>Split:</b> 80/20, random_state=10
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="footer">
            <div style="font-weight:700; margin-bottom:0.8rem;">🛠️ Built With</div>
            <span class="footer-badge">🐍 Python</span>
            <span class="footer-badge">🐼 Pandas</span>
            <span class="footer-badge">🔢 NumPy</span>
            <span class="footer-badge">🤖 Scikit-Learn</span>
            <span class="footer-badge">🎈 Streamlit</span>
            <span class="footer-badge">📈 Plotly</span>
            <div class="muted" style="margin-top:1rem; font-size:0.85rem;">
                Designed &amp; developed by <b>Manav Sharma</b> · © 2026 Attrition IQ
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==================================================================================
# MAIN
# ==================================================================================
def init_session_state() -> None:
    if "prediction_result" not in st.session_state:
        st.session_state.prediction_result = None
    if "final_inputs" not in st.session_state:
        st.session_state.final_inputs = None
    if "insights" not in st.session_state:
        st.session_state.insights = None
    if "history" not in st.session_state:
        st.session_state.history = []


def main() -> None:
    try:
        raw_df = load_raw_data(CSV_PATH)
    except FileNotFoundError:
        st.error(f"❌ Could not find '{CSV_PATH}'. Place it next to app.py.")
        st.stop()
    except Exception as exc:  # noqa: BLE001
        st.error(f"❌ Failed to load dataset: {exc}")
        st.stop()

    try:
        model_info = train_model(raw_df)
    except ImportError as exc:
        st.error(f"❌ {exc}")
        st.stop()
    except Exception as exc:  # noqa: BLE001
        st.error(f"❌ Model training failed: {exc}")
        st.stop()

    init_session_state()
    inject_css()
    render_topnav()
    render_hero()
    render_sidebar(model_info, raw_df)
    render_kpi_row(raw_df, model_info)

    inputs = render_input_form(raw_df)

    st.markdown("<br>", unsafe_allow_html=True)
    _, mid, _ = st.columns([1, 1.3, 1])
    with mid:
        predict_clicked = st.button("🔮 Predict Attrition Risk", use_container_width=True)

    if predict_clicked:
        run_ai_analysis(model_info, raw_df, inputs)

    if st.session_state.prediction_result is not None:
        st.markdown('<hr class="divider-grad">', unsafe_allow_html=True)
        render_results()

    render_analytics(raw_df, model_info)
    st.markdown('<hr class="divider-grad">', unsafe_allow_html=True)
    render_footer(model_info)


if __name__ == "__main__":
    main()