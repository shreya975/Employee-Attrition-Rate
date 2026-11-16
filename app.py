"""
EMPLOYEE ATTRITION INTELLIGENCE
AI-Powered Workforce Analytics & Predictive HR Intelligence

Run:
    pip install streamlit pandas numpy plotly scikit-learn joblib
    streamlit run app.py

Expected files next to app.py (or in ./data and ./models):
    - the IBM-style HR attrition CSV (any *.csv containing an 'Attrition' column)
    - employee_attrition_model.pkl  (saved sklearn pipeline, 30 input features)
"""

import datetime as dt
import hashlib
import itertools
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Employee Attrition Intelligence",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# 1. CONSTANTS
# =============================================================================
BASE = Path(__file__).resolve().parent
KEYS = itertools.count()

C_ACC, C_CYAN, C_OK, C_WARN, C_BAD = "#7C5CFC", "#22D3EE", "#10B981", "#F59E0B", "#F43F5E"
STATUS_COLORS = {"Stayed": C_OK, "Left": C_BAD}
SEQ = [C_ACC, C_CYAN, C_OK, C_WARN, C_BAD, "#A78BFA", "#38BDF8"]

IBM_FEATURES = [
    "Age", "BusinessTravel", "DailyRate", "Department", "DistanceFromHome", "Education",
    "EducationField", "EnvironmentSatisfaction", "Gender", "HourlyRate", "JobInvolvement",
    "JobLevel", "JobRole", "JobSatisfaction", "MaritalStatus", "MonthlyIncome", "MonthlyRate",
    "NumCompaniesWorked", "OverTime", "PercentSalaryHike", "PerformanceRating",
    "RelationshipSatisfaction", "StockOptionLevel", "TotalWorkingYears",
    "TrainingTimesLastYear", "WorkLifeBalance", "YearsAtCompany", "YearsInCurrentRole",
    "YearsSinceLastPromotion", "YearsWithCurrManager",
]

# Assumes the standard IBM HR data dictionary for coded columns.
_LVL = {1: "Low", 2: "Medium", 3: "High", 4: "Very High"}
LABELS = {
    "Education": {1: "Below College", 2: "College", 3: "Bachelor", 4: "Master", 5: "Doctor"},
    "EnvironmentSatisfaction": _LVL,
    "JobInvolvement": _LVL,
    "JobSatisfaction": _LVL,
    "RelationshipSatisfaction": _LVL,
    "WorkLifeBalance": {1: "Bad", 2: "Good", 3: "Better", 4: "Best"},
    "PerformanceRating": {1: "Low", 2: "Good", 3: "Excellent", 4: "Outstanding"},
}

FALLBACK_OPTS = {
    "BusinessTravel": ["Non-Travel", "Travel_Rarely", "Travel_Frequently"],
    "Department": ["Human Resources", "Research & Development", "Sales"],
    "EducationField": ["Human Resources", "Life Sciences", "Marketing", "Medical", "Other", "Technical Degree"],
    "Gender": ["Female", "Male"],
    "JobRole": ["Healthcare Representative", "Human Resources", "Laboratory Technician", "Manager",
                "Manufacturing Director", "Research Director", "Research Scientist",
                "Sales Executive", "Sales Representative"],
    "MaritalStatus": ["Divorced", "Married", "Single"],
    "OverTime": ["No", "Yes"],
}

# Notebook results typed in by the project owner; shown only when held-out data is unavailable, never as deployed-model metrics.
NOTEBOOK = {
    "Standard Logistic Regression": dict(Accuracy=.8878, Precision=.7692, Recall=.4255, F1=.5479, AUC=.8296, CV=.5400),
    "Balanced Logistic Regression": dict(Accuracy=.7823, Precision=.3976, Recall=.7021, F1=.5077, AUC=.8240, CV=.5135),
}
PAGES = [
    ("Overview", "⌂"), ("Workforce Analytics", "◫"), ("Attrition Analysis", "◔"),
    ("Employee Explorer", "⌕"), ("Predict Attrition", "✦"), ("Model Performance", "◎"),
    ("Business Insights", "✧"), ("Settings / About", "⚙"),
]

# =============================================================================
# 2. STYLING
# =============================================================================
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root{--bg:#0B1020;--bg2:#111827;--card:#151E30;--acc:#7C5CFC;--cyan:#22D3EE;--ok:#10B981;--warn:#F59E0B;--bad:#F43F5E;--text:#F8FAFC;--muted:#94A3B8;--bd:rgba(148,163,184,.14);}
html,body,[class*="css"],.stApp,.stMarkdown,button,input,textarea,select{font-family:'Inter',system-ui,sans-serif!important;}
.stApp{background:radial-gradient(1100px 560px at 88% -8%,rgba(124,92,252,.20),transparent 60%),radial-gradient(900px 500px at -8% 108%,rgba(34,211,238,.10),transparent 60%),#0B1020;color:var(--text);}
header[data-testid="stHeader"]{background:transparent;}
#MainMenu,footer{visibility:hidden;}
.block-container{padding-top:2.2rem;padding-bottom:3rem;max-width:1450px;}
h1,h2,h3,h4{color:var(--text);letter-spacing:-.02em;}
p,li,label,span{color:inherit;}
section[data-testid="stSidebar"]{background:linear-gradient(180deg,#0E1428 0%,#0B1020 100%);border-right:1px solid var(--bd);}
section[data-testid="stSidebar"] .block-container{padding-top:1.2rem;}
.brand{display:flex;gap:.7rem;align-items:center;padding:.4rem .3rem 1.2rem;}
.logo{width:38px;height:38px;border-radius:11px;display:flex;align-items:center;justify-content:center;font-weight:800;color:#fff;background:linear-gradient(135deg,#7C5CFC,#22D3EE);box-shadow:0 8px 24px rgba(124,92,252,.45);}
.brand-t{font-weight:700;font-size:.92rem;line-height:1.2;color:var(--text);}
.brand-s{font-size:.72rem;color:var(--muted);}
.nav-cap{font-size:.68rem;letter-spacing:.12em;text-transform:uppercase;color:#64748B;margin:.2rem .4rem .5rem;}
.stButton{width:100%;}
.stButton>button,.stDownloadButton>button,.stFormSubmitButton>button{width:100%;border-radius:12px;border:1px solid rgba(124,92,252,.5);background:linear-gradient(135deg,#7C5CFC,#5B8DEF);color:#fff;font-weight:600;padding:.6rem 1rem;transition:all .2s ease;box-shadow:0 8px 24px rgba(124,92,252,.28);}
.stButton>button:hover,.stDownloadButton>button:hover,.stFormSubmitButton>button:hover{transform:translateY(-2px);box-shadow:0 14px 32px rgba(124,92,252,.42);border-color:#A78BFA;color:#fff;}
.stFormSubmitButton>button{letter-spacing:.08em;padding:.85rem 1rem;font-size:.95rem;}
section[data-testid="stSidebar"] .stButton>button{justify-content:flex-start;text-align:left;background:transparent;border:1px solid transparent;color:var(--muted);font-weight:500;box-shadow:none;padding:.55rem .8rem;}
section[data-testid="stSidebar"] .stButton>button:hover{background:rgba(124,92,252,.10);color:var(--text);transform:translateX(3px);box-shadow:none;}
section[data-testid="stSidebar"] .stButton>button[kind="primary"],section[data-testid="stSidebar"] .stButton>button[data-testid="stBaseButton-primary"]{background:linear-gradient(90deg,rgba(124,92,252,.30),rgba(34,211,238,.10));border:1px solid rgba(124,92,252,.45);color:#fff;}
.profile{margin-top:1.4rem;padding:.9rem 1rem;border-radius:14px;background:rgba(21,30,48,.7);border:1px solid var(--bd);font-size:.78rem;color:var(--muted);line-height:1.5;}
.profile b{color:var(--text);font-weight:600;}
.hdr{display:flex;justify-content:space-between;align-items:flex-end;gap:1rem;flex-wrap:wrap;margin-bottom:1.4rem;}
.hdr h1{font-size:2rem;font-weight:800;margin:0;padding:0;background:linear-gradient(90deg,#F8FAFC,#A78BFA 70%,#22D3EE);-webkit-background-clip:text;-webkit-text-fill-color:transparent;}
.hdr p{color:var(--muted);margin:.3rem 0 0;font-size:.95rem;}
.chips{display:flex;gap:.45rem;flex-wrap:wrap;}
.chip{font-size:.72rem;padding:.3rem .7rem;border-radius:999px;background:rgba(148,163,184,.08);border:1px solid var(--bd);color:#CBD5E1;}
.kpi-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:1rem;margin:.2rem 0 1.4rem;}
.kpi{position:relative;overflow:hidden;padding:1.1rem 1.2rem;border-radius:18px;background:linear-gradient(145deg,rgba(21,30,48,.95),rgba(17,24,39,.85));border:1px solid var(--bd);box-shadow:0 10px 30px rgba(0,0,0,.30);transition:all .25s ease;backdrop-filter:blur(10px);}
.kpi:before{content:"";position:absolute;inset:0 auto 0 0;width:3px;background:var(--c);opacity:.9;}
.kpi:after{content:"";position:absolute;right:-40px;top:-40px;width:120px;height:120px;border-radius:50%;background:radial-gradient(circle,var(--c),transparent 70%);opacity:.14;}
.kpi:hover{transform:translateY(-4px);border-color:rgba(124,92,252,.45);box-shadow:0 18px 40px rgba(0,0,0,.42);}
.kpi-label{font-size:.72rem;text-transform:uppercase;letter-spacing:.09em;color:var(--muted);font-weight:600;}
.kpi-value{font-size:1.95rem;font-weight:800;margin:.35rem 0 .15rem;color:var(--text);letter-spacing:-.03em;}
.kpi-sub{font-size:.76rem;color:#8EA0B8;}
div[data-testid="stPlotlyChart"]{background:rgba(21,30,48,.62);border:1px solid var(--bd);border-radius:18px;padding:.5rem .6rem;box-shadow:0 8px 26px rgba(0,0,0,.26);transition:border-color .25s,transform .25s;}
div[data-testid="stPlotlyChart"]:hover{border-color:rgba(124,92,252,.40);}
.glass{padding:1.3rem 1.5rem;border-radius:18px;background:linear-gradient(145deg,rgba(21,30,48,.85),rgba(17,24,39,.70));border:1px solid var(--bd);backdrop-filter:blur(12px);margin:.4rem 0 1.2rem;}
.glass h3{margin:.0 0 .7rem;font-size:1.05rem;}
.glass li{margin:.35rem 0;color:#CBD5E1;line-height:1.55;font-size:.92rem;}
.glass li b,.insight b{color:#fff;}
.insight{margin:.3rem 0 1.1rem;padding:.8rem 1rem;border-radius:12px;font-size:.86rem;line-height:1.55;color:#CBD5E1;background:rgba(124,92,252,.07);border-left:3px solid var(--acc);}
.note{margin:.4rem 0 1rem;padding:.75rem 1rem;border-radius:12px;font-size:.82rem;line-height:1.5;color:#FCD9A0;background:rgba(245,158,11,.08);border:1px solid rgba(245,158,11,.25);}
.empty{padding:2.5rem 1.5rem;text-align:center;border-radius:18px;border:1px dashed rgba(148,163,184,.3);background:rgba(21,30,48,.5);color:var(--muted);}
.empty h3{color:var(--text);margin-bottom:.4rem;}
.sec{font-size:.74rem;text-transform:uppercase;letter-spacing:.12em;color:var(--cyan);font-weight:700;margin:1.2rem 0 .5rem;}
.rec{padding:1rem 1.2rem;border-radius:14px;background:rgba(21,30,48,.75);border:1px solid var(--bd);margin-bottom:.8rem;transition:all .2s;}
.rec:hover{border-color:rgba(124,92,252,.5);transform:translateX(3px);}
.rec h4{margin:0 0 .3rem;font-size:.95rem;}
.rec p{margin:.15rem 0;font-size:.85rem;color:#B6C2D4;line-height:1.5;}
.pill{display:inline-block;padding:.2rem .65rem;border-radius:999px;font-size:.72rem;font-weight:700;letter-spacing:.04em;}
div[data-baseweb="select"]>div,.stTextInput input,.stNumberInput input{background:#0F172A!important;border-color:var(--bd)!important;color:var(--text)!important;border-radius:10px!important;}
div[data-testid="stExpander"]{border:1px solid var(--bd);border-radius:14px;background:rgba(21,30,48,.45);}
div[data-testid="stForm"]{border:1px solid var(--bd);border-radius:18px;background:rgba(21,30,48,.45);padding:1.2rem;}
div[data-testid="stDataFrame"]{border:1px solid var(--bd);border-radius:14px;overflow:hidden;}
@media (max-width:768px){.hdr h1{font-size:1.5rem}.kpi-value{font-size:1.5rem}.block-container{padding-left:1rem;padding-right:1rem}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# =============================================================================
# 3. UI HELPERS
# =============================================================================
def show(fig, h=340):
    """Render a Plotly figure with the app theme."""
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#CBD5E1", size=12), height=h,
        margin=dict(l=10, r=10, t=56, b=10),
        title=dict(font=dict(size=15, color="#F8FAFC"), x=0.02),
        hoverlabel=dict(bgcolor="#0F172A", font_color="#F8FAFC", bordercolor="#7C5CFC"),
        legend=dict(bgcolor="rgba(0,0,0,0)", title_text=""),
    )
    fig.update_xaxes(gridcolor="rgba(148,163,184,.08)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(148,163,184,.08)", zeroline=False)
    cfg = {"displayModeBar": False}
    key = f"pc_{next(KEYS)}"
    try:
        st.plotly_chart(fig, width="stretch", config=cfg, key=key)
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, config=cfg, key=key)


def table(df, height=420):
    try:
        st.dataframe(df, width="stretch", height=height, hide_index=True)
    except TypeError:
        st.dataframe(df, use_container_width=True, height=height, hide_index=True)


def header(title, sub, chips=None):
    chips_html = "".join(f'<span class="chip">{c}</span>' for c in (chips or []))
    st.markdown(
        f'<div class="hdr"><div><h1>{title}</h1><p>{sub}</p></div><div class="chips">{chips_html}</div></div>',
        unsafe_allow_html=True,
    )


def kpi_row(items):
    html = '<div class="kpi-grid">'
    for label, value, sub, color in items:
        html += (f'<div class="kpi" style="--c:{color}"><div class="kpi-label">{label}</div>'
                 f'<div class="kpi-value">{value}</div><div class="kpi-sub">{sub}</div></div>')
    st.markdown(html + "</div>", unsafe_allow_html=True)


def insight(text):
    st.markdown(f'<div class="insight">{text}</div>', unsafe_allow_html=True)


def note(text):
    st.markdown(f'<div class="note">{text}</div>', unsafe_allow_html=True)


def section(text):
    st.markdown(f'<div class="sec">{text}</div>', unsafe_allow_html=True)


def empty_state(title, msg):
    st.markdown(f'<div class="empty"><h3>{title}</h3><div>{msg}</div></div>', unsafe_allow_html=True)


def goto(page):
    st.session_state.page = page


# =============================================================================
# 4. DATA & MODEL LOADING (cached)
# =============================================================================
@st.cache_data(show_spinner="Loading workforce dataset…")
def load_data():
    """Load the full HR dataset; never mistake the held-out test CSV for the workforce data."""
    required = set(IBM_FEATURES) | {"Attrition"}
    preferred_names = (
        "HR-Employee-Attrition.csv",
        "WA_Fn-UseC_-HR-Employee-Attrition.csv",
        "HR-Employee-Attrition(1).csv",
    )
    dirs = [BASE, BASE / "data", BASE / "dataset", Path.cwd(), Path.cwd() / "data"]
    candidates, seen = [], set()

    # Prefer the known full-dataset names, then allow another compatible full CSV.
    for d in dirs:
        if not d.exists():
            continue
        for name in preferred_names:
            candidate = d / name
            if candidate.is_file() and candidate not in seen:
                candidates.append(candidate)
                seen.add(candidate)
        for candidate in sorted(d.glob("*.csv")):
            if candidate in seen or candidate.name.lower() == HELDOUT_FILENAME.lower():
                continue
            candidates.append(candidate)
            seen.add(candidate)

    last_err = None
    for candidate in candidates:
        try:
            raw = pd.read_csv(candidate)
        except Exception as exc:  # noqa: BLE001
            last_err = f"{candidate.name}: {exc}"
            continue
        missing = sorted(required - set(raw.columns))
        if missing:
            last_err = f"{candidate.name} is missing columns: {', '.join(missing)}"
            continue
        if raw.empty:
            last_err = f"{candidate.name} contains no employee records."
            continue
        labels = raw["Attrition"].astype(str).str.strip().str.lower()
        if not labels.isin({"yes", "no", "1", "0", "true", "false"}).all():
            last_err = f"{candidate.name} has unexpected values in the Attrition column."
            continue
        return prep(raw), str(candidate), dt.datetime.fromtimestamp(candidate.stat().st_mtime), None

    return None, None, None, last_err or (
        "No compatible full HR attrition CSV was found. Add HR-Employee-Attrition.csv next to app.py."
    )


def prep(raw):
    df = raw.copy()
    a = df["Attrition"]
    df["AttritionFlag"] = a.astype(str).str.strip().str.lower().isin(["yes", "1", "true"]).astype(int)
    df["Status"] = np.where(df["AttritionFlag"] == 1, "Left", "Stayed")
    if "EmployeeNumber" in df:
        df["EmpRef"] = df["EmployeeNumber"].astype(str).map(
            lambda v: "EMP-" + hashlib.sha1(v.encode()).hexdigest()[:6].upper())
    else:
        df["EmpRef"] = [f"EMP-{i:05d}" for i in range(len(df))]
    for c, m in LABELS.items():
        if c in df:
            df[c + "Label"] = pd.Categorical(df[c].map(m), categories=list(m.values()), ordered=True)
    df["AgeGroup"] = pd.cut(df["Age"], [17, 25, 35, 45, 55, 120], labels=["18–25", "26–35", "36–45", "46–55", "56+"])
    df["IncomeBand"] = pd.cut(df["MonthlyIncome"], [-1, 3000, 5000, 8000, 12000, 10**9],
                              labels=["<3K", "3–5K", "5–8K", "8–12K", "12K+"])
    df["TenureBand"] = pd.cut(df["YearsAtCompany"], [-1, 1, 3, 5, 10, 200], labels=["0–1", "2–3", "4–5", "6–10", "10+"])
    df["PromoBand"] = pd.cut(df["YearsSinceLastPromotion"], [-1, 0, 1, 3, 7, 200], labels=["0", "1", "2–3", "4–7", "8+"])
    return df


@st.cache_resource(show_spinner="Loading prediction model…")
def load_model():
    """Returns (model, path, error, version_warning)."""
    import warnings
    cands = [BASE / "employee_attrition_model.pkl", BASE / "models" / "employee_attrition_model.pkl",
             BASE / "model" / "employee_attrition_model.pkl", Path.cwd() / "employee_attrition_model.pkl"]
    for p in cands:
        if p.exists():
            try:
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    m = joblib.load(p)
                vw = "; ".join(sorted({str(x.message).strip()[:240] for x in w if "version" in str(x.message).lower()}))
                if not hasattr(m, "predict_proba"):
                    return None, str(p), "Loaded object has no predict_proba().", ""
                return m, str(p), None, vw
            except Exception as e:  # noqa: BLE001
                return None, str(p), (f"Could not load the model ({type(e).__name__}): {e}. This is often caused by a "
                                      "scikit-learn version different from the one used for training; install the training version."), ""
    return None, None, "employee_attrition_model.pkl was not found next to app.py.", ""


def model_features(model):
    names = getattr(model, "feature_names_in_", None)
    features = [str(n) for n in names] if names is not None else list(IBM_FEATURES)
    if len(features) != len(set(features)):
        raise ValueError("The saved model contains duplicate input feature names.")
    return features


# =============================================================================
# 5. ANALYTICS HELPERS
# =============================================================================
def rate_table(df, col, sort=False):
    g = df.groupby(col, observed=True)["AttritionFlag"].agg(["mean", "sum", "count"]).reset_index()
    g.columns = [col, "Rate", "Left", "Employees"]
    g["Rate"] = g["Rate"] * 100
    if sort:
        g = g.sort_values("Rate", ascending=False)
    g[col] = g[col].astype(str)
    return g


def count_order(df, col):
    if isinstance(df[col].dtype, pd.CategoricalDtype):
        present = set(df[col].dropna().unique())
        return [str(c) for c in df[col].cat.categories if c in present]
    return df[col].astype(str).value_counts().index.tolist()


def rate_fig(df, col, title, sort=False, horizontal=False, h=340):
    t = rate_table(df, col, sort)
    overall = df["AttritionFlag"].mean() * 100
    cd = np.stack([t["Left"], t["Employees"]], axis=-1)
    kw = dict(marker=dict(color=t["Rate"], colorscale=[[0, C_CYAN], [.5, C_ACC], [1, C_BAD]], cmin=0,
                          cmax=max(float(t["Rate"].max()), 1), line=dict(width=0)),
              text=[f"{r:.1f}%" for r in t["Rate"]], textposition="outside", cliponaxis=False, customdata=cd)
    if horizontal:
        fig = go.Figure(go.Bar(x=t["Rate"], y=t[col], orientation="h", **kw,
                               hovertemplate="<b>%{y}</b><br>Attrition rate: %{x:.1f}%<br>Left: %{customdata[0]} of %{customdata[1]}<extra></extra>"))
        fig.add_vline(x=overall, line_dash="dot", line_color="#94A3B8",
                      annotation_text=f"Overall {overall:.1f}%", annotation_font_color="#94A3B8")
        fig.update_yaxes(autorange="reversed")
        fig.update_xaxes(title="Attrition rate (%)", range=[0, float(t["Rate"].max()) * 1.2 + 1])
    else:
        fig = go.Figure(go.Bar(x=t[col], y=t["Rate"], **kw,
                               hovertemplate="<b>%{x}</b><br>Attrition rate: %{y:.1f}%<br>Left: %{customdata[0]} of %{customdata[1]}<extra></extra>"))
        fig.add_hline(y=overall, line_dash="dot", line_color="#94A3B8",
                      annotation_text=f"Overall {overall:.1f}%", annotation_font_color="#94A3B8")
        fig.update_yaxes(title="Attrition rate (%)", range=[0, float(t["Rate"].max()) * 1.2 + 1])
    fig.update_layout(title=title, showlegend=False)
    show(fig, h)
    return t


def stacked_counts(df, col, title, horizontal=False, h=340):
    t = df.groupby([col, "Status"], observed=True).size().reset_index(name="Employees")
    t[col] = t[col].astype(str)
    order = count_order(df, col)
    if horizontal:
        fig = px.bar(t, y=col, x="Employees", color="Status", orientation="h", barmode="stack",
                     color_discrete_map=STATUS_COLORS, category_orders={col: order}, title=title)
        fig.update_yaxes(autorange="reversed", title="")
    else:
        fig = px.bar(t, x=col, y="Employees", color="Status", barmode="stack",
                     color_discrete_map=STATUS_COLORS, category_orders={col: order}, title=title)
        fig.update_xaxes(title="")
    show(fig, h)


def hist(df, col, title, nbins=20, h=340):
    fig = px.histogram(df, x=col, color="Status", nbins=nbins, barmode="stack",
                       color_discrete_map=STATUS_COLORS, title=title)
    fig.update_traces(marker_line_width=0)
    fig.update_yaxes(title="Employees")
    show(fig, h)


def top_group(t, min_n=20):
    t = t[t["Employees"] >= min_n]
    return None if t.empty else t.sort_values("Rate", ascending=False).iloc[0]


def interpret(t, col, overall):
    r = top_group(t, 20)
    if r is None:
        return "Groups are too small (<20 employees) for a reliable comparison."
    return (f"Highest observed attrition: <b>{r[col]}</b> at <b>{r['Rate']:.1f}%</b> "
            f"({int(r['Left'])} of {int(r['Employees'])}) versus <b>{overall:.1f}%</b> overall. "
            "This is an observed association in this dataset, not evidence of cause; small groups can swing widely.")


def top_segments(df, k=3, min_n=30):
    rows = []
    for c in ("Department", "JobRole", "OverTime", "MaritalStatus", "BusinessTravel", "AgeGroup", "IncomeBand"):
        if c in df:
            for _, r in rate_table(df, c).iterrows():
                if r["Employees"] >= min_n:
                    rows.append((c, r[c], r["Rate"], int(r["Left"]), int(r["Employees"])))
    rows.sort(key=lambda x: -x[2])
    return rows[:k]


def grp(df, mask):
    n = int(mask.sum())
    return (df.loc[mask, "AttritionFlag"].mean() * 100 if n else float("nan")), n


VIEW_COLS = ["EmpRef", "Age", "Gender", "MaritalStatus", "Department", "JobRole", "EducationField",
             "MonthlyIncome", "TotalWorkingYears", "YearsAtCompany", "YearsSinceLastPromotion",
             "OverTime", "BusinessTravel", "JobSatisfaction", "WorkLifeBalance", "Status"]


def view(df):
    return df[[c for c in VIEW_COLS if c in df]].rename(columns={"Status": "Attrition"})


def csv_bytes(df):
    return view(df).to_csv(index=False).encode("utf-8")


def need_data(df, err):
    if df is None:
        empty_state("Dataset unavailable",
                    f"{err}<br>Place the HR attrition CSV next to <code>app.py</code> (or in a <code>data/</code> folder).")
        st.stop()


# =============================================================================
# 6. PAGES
# =============================================================================
def page_overview(df, src, refreshed):
    header("Workforce Intelligence Overview",
           "A comprehensive view of workforce health, employee retention, and attrition patterns.",
           [dt.datetime.now().strftime("%A, %d %B %Y"), f"Dataset: loaded · {len(df):,} records",
            f"Source: {Path(src).name}", f"Data file updated {refreshed:%d %b %Y, %H:%M}"])
    n = len(df)
    left = int(df["AttritionFlag"].sum())
    stayed = n - left
    rate = left / n * 100
    kpi_row([
        ("Total Employees", f"{n:,}", "Records in dataset", C_ACC),
        ("Employees Stayed", f"{stayed:,}", f"{stayed / n * 100:.1f}% of workforce", C_OK),
        ("Employees Left", f"{left:,}", f"{rate:.1f}% of workforce", C_BAD),
        ("Attrition Rate", f"{rate:.1f}%", "Left ÷ total employees", C_WARN),
        ("Retention Rate", f"{100 - rate:.1f}%", "Stayed ÷ total employees", C_CYAN),
        ("Avg Monthly Income", f"{df['MonthlyIncome'].mean():,.0f}", "Mean across all employees", C_ACC),
    ])
    c1, c2, c3 = st.columns(3)
    with c1:
        fig = go.Figure(go.Pie(labels=["Stayed", "Left"], values=[stayed, left], hole=0.7,
                               marker=dict(colors=[C_OK, C_BAD], line=dict(color="#0B1020", width=3)),
                               textinfo="percent", hovertemplate="<b>%{label}</b><br>%{value:,} employees (%{percent})<extra></extra>"))
        fig.update_layout(title="Employee Attrition", showlegend=True,
                          annotations=[dict(text=f"<b>{rate:.1f}%</b><br>attrition", x=.5, y=.5, showarrow=False,
                                            font=dict(size=18, color="#F8FAFC"))])
        show(fig)
    with c2:
        t = df.groupby("Department").size().reset_index(name="Employees").sort_values("Employees", ascending=False)
        fig = px.bar(t, x="Department", y="Employees", title="Employees by Department", color_discrete_sequence=[C_ACC])
        fig.update_xaxes(title="")
        show(fig)
    with c3:
        rate_fig(df, "Department", "Attrition Rate by Department", sort=True)
    c1, c2 = st.columns(2)
    with c1:
        hist(df, "Age", "Age Distribution", nbins=22)
    with c2:
        hist(df, "MonthlyIncome", "Monthly Income Distribution", nbins=30)
    c1, c2 = st.columns(2)
    with c1:
        rate_fig(df, "Gender", "Attrition Rate by Gender")
    with c2:
        col = "JobSatisfactionLabel" if "JobSatisfactionLabel" in df else "JobSatisfaction"
        stacked_counts(df, col, "Job Satisfaction Distribution")

    segs = top_segments(df)
    dept = top_group(rate_table(df, "Department"), 20)
    bullets = [f"<li><b>{rate:.1f}%</b> of employees ({left:,} of {n:,}) have left; the retention rate is <b>{100 - rate:.1f}%</b>.</li>"]
    if dept is not None:
        bullets.append(f"<li><b>{dept['Department']}</b> has the highest department attrition rate at <b>{dept['Rate']:.1f}%</b> "
                       f"({int(dept['Left'])} of {int(dept['Employees'])}).</li>")
    if segs:
        s = "".join(f"<li><b>{c}: {v}</b> — {r:.1f}% observed attrition ({l} of {m}), {r - rate:+.1f} pts vs overall.</li>"
                    for c, v, r, l, m in segs)
        bullets.append(f"<li>Segments HR may want to investigate further (highest observed rates, groups of 30+):<ul>{s}</ul></li>")
    st.markdown(f'<div class="glass"><h3>Executive Summary</h3><ul>{"".join(bullets)}</ul></div>', unsafe_allow_html=True)
    insight("All figures are computed from the loaded dataset. Segment rates describe observed association only and do not establish causes of attrition.")


def wf_filters(df):
    with st.expander("Filters", expanded=True):
        a = st.columns(4)
        dep = a[0].multiselect("Department", sorted(df["Department"].unique()), key="wf_dep")
        gen = a[1].multiselect("Gender", sorted(df["Gender"].unique()), key="wf_gen")
        role = a[2].multiselect("Job Role", sorted(df["JobRole"].unique()), key="wf_role")
        edu = a[3].multiselect("Education Field", sorted(df["EducationField"].unique()), key="wf_edu")
        b = st.columns(3)
        amin, amax = int(df["Age"].min()), int(df["Age"].max())
        age = b[0].slider("Age range", amin, amax, (amin, amax), key="wf_age")
        mar = b[1].multiselect("Marital Status", sorted(df["MaritalStatus"].unique()), key="wf_mar")
        bt = b[2].multiselect("Business Travel", sorted(df["BusinessTravel"].unique()), key="wf_bt")
    m = df["Age"].between(*age)
    for col, sel in (("Department", dep), ("Gender", gen), ("JobRole", role), ("EducationField", edu),
                     ("MaritalStatus", mar), ("BusinessTravel", bt)):
        if sel:
            m &= df[col].isin(sel)
    return df[m]


def page_workforce(df):
    header("Workforce Analytics", "Explore workforce composition with dynamic, multi-dimensional filters.")
    f = wf_filters(df)
    if f.empty:
        empty_state("No employees match these filters", "Broaden or reset the filters above.")
        return
    n, left = len(f), int(f["AttritionFlag"].sum())
    kpi_row([
        ("Headcount", f"{n:,}", f"{n / len(df) * 100:.1f}% of dataset", C_ACC),
        ("Employees Left", f"{left:,}", f"{left / n * 100:.1f}% attrition in selection", C_BAD),
        ("Avg Age", f"{f['Age'].mean():.1f}", "Years", C_CYAN),
        ("Avg Monthly Income", f"{f['MonthlyIncome'].mean():,.0f}", "Mean in selection", C_OK),
        ("Avg Tenure", f"{f['YearsAtCompany'].mean():.1f}", "Years at company", C_WARN),
    ])
    c1, c2 = st.columns(2)
    with c1:
        t = f.groupby("Department").size().reset_index(name="Employees")
        show(px.bar(t, x="Department", y="Employees", title="Workforce by Department", color_discrete_sequence=[C_ACC]))
    with c2:
        stacked_counts(f, "JobRole", "Employees by Job Role", horizontal=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        stacked_counts(f, "AgeGroup", "Age Group Distribution")
    with c2:
        hist(f, "MonthlyIncome", "Salary Distribution", nbins=25)
    with c3:
        g = f.groupby("Gender").size().reset_index(name="Employees")
        fig = go.Figure(go.Pie(labels=g["Gender"], values=g["Employees"], hole=.6,
                               marker=dict(colors=[C_ACC, C_CYAN, C_WARN], line=dict(color="#0B1020", width=3))))
        fig.update_layout(title="Gender Distribution")
        show(fig)
    c1, c2, c3 = st.columns(3)
    with c1:
        hist(f, "TotalWorkingYears", "Experience Distribution (Total Working Years)", nbins=20)
    with c2:
        stacked_counts(f, "JobSatisfactionLabel" if "JobSatisfactionLabel" in f else "JobSatisfaction", "Job Satisfaction")
    with c3:
        stacked_counts(f, "WorkLifeBalanceLabel" if "WorkLifeBalanceLabel" in f else "WorkLifeBalance", "Work-Life Balance")
    c1, c2 = st.columns(2)
    with c1:
        stacked_counts(f, "OverTime", "Overtime Distribution")
    with c2:
        stacked_counts(f, "EducationField", "Education Field Distribution", horizontal=True)
    section("Workforce records")
    table(view(f), 400)
    st.download_button("Download filtered records (CSV)", csv_bytes(f), "workforce_filtered.csv", "text/csv", key="dl_wf")


def page_attrition(df):
    header("Attrition Analysis", "Understand where and among whom turnover is concentrated.")
    n, left = len(df), int(df["AttritionFlag"].sum())
    overall = left / n * 100
    d = top_group(rate_table(df, "Department"), 20)
    lf = df[df["AttritionFlag"] == 1]
    kpi_row([
        ("Total Attrition Cases", f"{left:,}", "Employees who left", C_BAD),
        ("Overall Attrition Rate", f"{overall:.1f}%", "Left ÷ total", C_WARN),
        ("Highest Attrition Dept", d["Department"] if d is not None else "n/a",
         f"{d['Rate']:.1f}% observed rate" if d is not None else "", C_ACC),
        ("Avg Age (Left)", f"{lf['Age'].mean():.1f}" if len(lf) else "n/a", "Years", C_CYAN),
        ("Avg Income (Left)", f"{lf['MonthlyIncome'].mean():,.0f}" if len(lf) else "n/a",
         f"vs {df.loc[df['AttritionFlag'] == 0, 'MonthlyIncome'].mean():,.0f} for stayers", C_OK),
    ])
    note("Every chart shows <b>attrition rate = employees who left ÷ employees in that group</b> (not raw counts). Hover for counts.")
    lab = lambda c: c + "Label" if c + "Label" in df else c  # noqa: E731
    specs = [
        ("Attrition Rate by Department", "Department", True, False),
        ("Attrition Rate by Job Role", "JobRole", True, True),
        ("Age Group vs Attrition", "AgeGroup", False, False),
        ("Monthly Income Band vs Attrition", "IncomeBand", False, False),
        ("Overtime vs Attrition", "OverTime", False, False),
        ("Job Satisfaction vs Attrition", lab("JobSatisfaction"), False, False),
        ("Work-Life Balance vs Attrition", lab("WorkLifeBalance"), False, False),
        ("Years at Company vs Attrition", "TenureBand", False, False),
        ("Years Since Last Promotion vs Attrition", "PromoBand", False, False),
        ("Business Travel vs Attrition", "BusinessTravel", True, False),
    ]
    for i in range(0, len(specs), 2):
        cols = st.columns(2)
        for col, (title, c, sort, horiz) in zip(cols, specs[i:i + 2]):
            with col:
                t = rate_fig(df, c, title, sort=sort, horizontal=horiz, h=380 if horiz else 340)
                insight(interpret(t, c, overall))


def page_explorer(df):
    header("Employee Explorer", "Search and filter individual employee records. Identifiers are anonymized.")
    a = st.columns([2, 1, 1, 1])
    q = a[0].text_input("Search (reference, department, role, field)", key="ex_q", placeholder="e.g. Sales, Scientist, EMP-1A2B3C")
    dep = a[1].selectbox("Department", ["All"] + sorted(df["Department"].unique()), key="ex_dep")
    role = a[2].selectbox("Job Role", ["All"] + sorted(df["JobRole"].unique()), key="ex_role")
    stat = a[3].selectbox("Attrition status", ["All", "Left", "Stayed"], key="ex_stat")
    b = st.columns(3)
    amin, amax = int(df["Age"].min()), int(df["Age"].max())
    imin, imax = int(df["MonthlyIncome"].min()), int(df["MonthlyIncome"].max())
    ymin, ymax = int(df["YearsAtCompany"].min()), int(df["YearsAtCompany"].max())
    age = b[0].slider("Age", amin, amax, (amin, amax), key="ex_age")
    inc = b[1].slider("Monthly income", imin, imax, (imin, imax), key="ex_inc")
    yrs = b[2].slider("Years at company", ymin, ymax, (ymin, ymax), key="ex_yrs")
    m = df["Age"].between(*age) & df["MonthlyIncome"].between(*inc) & df["YearsAtCompany"].between(*yrs)
    if dep != "All":
        m &= df["Department"] == dep
    if role != "All":
        m &= df["JobRole"] == role
    if stat != "All":
        m &= df["Status"] == stat
    if q.strip():
        hay = (df["EmpRef"] + " " + df["Department"] + " " + df["JobRole"] + " " + df["EducationField"]).str.lower()
        m &= hay.str.contains(q.strip().lower(), regex=False)
    f = df[m]
    st.caption(f"{len(f):,} of {len(df):,} employees match")
    if f.empty:
        empty_state("No matching employees", "Try clearing the search or widening the filters.")
        return
    table(view(f), 520)
    st.download_button("Download filtered records (CSV)", csv_bytes(f), "employee_explorer.csv", "text/csv", key="dl_ex")


# ---- Prediction -------------------------------------------------------------
def opts(df, col):
    if df is not None and col in df:
        return sorted(df[col].dropna().astype(str).unique())
    return FALLBACK_OPTS[col]


def coded(label, key, col, default):
    m = LABELS[col]
    return st.selectbox(label, list(m.keys()), index=list(m.keys()).index(default),
                        format_func=lambda v: f"{v} – {m[v]}", key=key)


def validate(r):
    errs = []
    if r["YearsAtCompany"] > r["TotalWorkingYears"]:
        errs.append("Years at Company cannot exceed Total Working Years.")
    for k in ("YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager"):
        if r[k] > r["YearsAtCompany"]:
            errs.append(f"{k} cannot exceed Years at Company.")
    if r["TotalWorkingYears"] > r["Age"] - 14:
        errs.append("Total Working Years is inconsistent with Age (check both values).")
    return errs


def band(p):
    if p < 0.20:
        return "Low Risk", C_OK
    if p < 0.50:
        return "Moderate Risk", C_WARN
    return "High Risk", C_BAD


def positive_class(classes):
    """Find the class representing an employee who left across common label formats."""
    values = list(classes)
    for value in values:
        if str(value).strip().lower() in {"1", "yes", "true", "left", "attrition"}:
            return value
    raise ValueError(f"Could not identify the positive (Attrition=Yes) class in model classes: {values}")


def run_model(model, features, row):
    X = pd.DataFrame([row], columns=list(features))
    classes = list(model.classes_)
    pos = positive_class(classes)
    proba = model.predict_proba(X)[0]
    p_leave = float(proba[classes.index(pos)])
    pred = model.predict(X)[0]
    return p_leave, pred == pos


def report_text(r):
    lines = ["EMPLOYEE ATTRITION INTELLIGENCE — PREDICTION REPORT",
             f"Generated: {r['ts']}", "",
             f"Predicted status : {r['status']}",
             f"Attrition probability : {r['p'] * 100:.1f}%",
             f"Retention probability : {(1 - r['p']) * 100:.1f}%",
             f"Risk band (application-defined) : {r['band']}", "",
             "Risk bands are application-defined probability ranges (Low <20%, Moderate 20–50%, High ≥50%),",
             "not validated HR risk categories. The model output is a statistical estimate, not a decision.", "",
             "INPUTS"]
    lines += [f"  {k}: {v}" for k, v in r["row"].items()]
    return "\n".join(lines).encode("utf-8")


def page_predict(df):
    header("Predict Employee Attrition", "Score an employee profile with the saved Logistic Regression pipeline.")
    model, mpath, merr, vwarn = load_model()
    if model is None:
        empty_state("Prediction model unavailable",
                    f"{merr}<br>Add <code>employee_attrition_model.pkl</code> next to <code>app.py</code>. "
                    "No predictions are generated without the real model.")
        return
    if vwarn:
        note(f"Model loaded with a scikit-learn version warning: {vwarn}")
    features = model_features(model)
    if len(features) != 30:
        note(f"The loaded model reports {len(features)} input features (expected 30). Proceeding with the model's own feature list.")

    with st.form("predict_form"):
        section("A · Personal Information")
        c = st.columns(3)
        age = c[0].number_input("Age", 18, 65, 35, key="p_age")
        gender = c[1].selectbox("Gender", opts(df, "Gender"), key="p_gen")
        marital = c[2].selectbox("Marital Status", opts(df, "MaritalStatus"), key="p_mar")
        c = st.columns(3)
        with c[0]:
            edu = coded("Education", "p_edu", "Education", 3)
        edf = c[1].selectbox("Education Field", opts(df, "EducationField"), key="p_edf")
        dist = c[2].number_input("Distance From Home", 1, 60, 8, key="p_dist")

        section("B · Job Information")
        c = st.columns(3)
        dept = c[0].selectbox("Department", opts(df, "Department"), key="p_dep")
        role = c[1].selectbox("Job Role", opts(df, "JobRole"), key="p_role")
        jl = c[2].number_input("Job Level", 1, 5, 2, key="p_jl")
        c = st.columns(3)
        bt = c[0].selectbox("Business Travel", opts(df, "BusinessTravel"), key="p_bt")
        with c[1]:
            jinv = coded("Job Involvement", "p_ji", "JobInvolvement", 3)
        with c[2]:
            jsat = coded("Job Satisfaction", "p_js", "JobSatisfaction", 3)
        c = st.columns(3)
        with c[0]:
            esat = coded("Environment Satisfaction", "p_es", "EnvironmentSatisfaction", 3)
        with c[1]:
            rsat = coded("Relationship Satisfaction", "p_rs", "RelationshipSatisfaction", 3)

        section("C · Compensation")
        c = st.columns(3)
        inc = c[0].number_input("Monthly Income", 1000, 30000, 5000, step=100, key="p_inc")
        dr = c[1].number_input("Daily Rate", 100, 1500, 800, key="p_dr")
        hr = c[2].number_input("Hourly Rate", 30, 100, 65, key="p_hr")
        c = st.columns(3)
        mr = c[0].number_input("Monthly Rate", 2000, 27000, 14000, step=100, key="p_mr")
        hike = c[1].number_input("Percent Salary Hike", 0, 30, 14, key="p_hike")
        stock = c[2].number_input("Stock Option Level", 0, 3, 1, key="p_stock")

        section("D · Career Information")
        c = st.columns(4)
        twy = c[0].number_input("Total Working Years", 0, 45, 10, key="p_twy")
        yac = c[1].number_input("Years at Company", 0, 45, 5, key="p_yac")
        ycr = c[2].number_input("Years in Current Role", 0, 20, 3, key="p_ycr")
        ysp = c[3].number_input("Years Since Last Promotion", 0, 20, 1, key="p_ysp")
        c = st.columns(3)
        ycm = c[0].number_input("Years With Current Manager", 0, 20, 3, key="p_ycm")
        ncw = c[1].number_input("Number of Companies Worked", 0, 12, 2, key="p_ncw")
        trn = c[2].number_input("Training Times Last Year", 0, 10, 3, key="p_trn")

        section("E · Work Environment")
        c = st.columns(3)
        ot = c[0].selectbox("OverTime", opts(df, "OverTime"), key="p_ot")
        with c[1]:
            wlb = coded("Work-Life Balance", "p_wlb", "WorkLifeBalance", 3)
        with c[2]:
            perf = coded("Performance Rating", "p_perf", "PerformanceRating", 3)
        submitted = st.form_submit_button("ANALYZE ATTRITION RISK")

    if submitted:
        row = {
            "Age": age, "BusinessTravel": bt, "DailyRate": dr, "Department": dept, "DistanceFromHome": dist,
            "Education": edu, "EducationField": edf, "EnvironmentSatisfaction": esat, "Gender": gender,
            "HourlyRate": hr, "JobInvolvement": jinv, "JobLevel": jl, "JobRole": role, "JobSatisfaction": jsat,
            "MaritalStatus": marital, "MonthlyIncome": inc, "MonthlyRate": mr, "NumCompaniesWorked": ncw,
            "OverTime": ot, "PercentSalaryHike": hike, "PerformanceRating": perf,
            "RelationshipSatisfaction": rsat, "StockOptionLevel": stock, "TotalWorkingYears": twy,
            "TrainingTimesLastYear": trn, "WorkLifeBalance": wlb, "YearsAtCompany": yac,
            "YearsInCurrentRole": ycr, "YearsSinceLastPromotion": ysp, "YearsWithCurrManager": ycm,
        }
        errs = validate(row)
        absent = [f for f in features if f not in row]
        if absent:
            st.error(f"The model expects features this form does not collect: {', '.join(absent)}. Prediction was not run.")
        elif errs:
            for e in errs:
                st.error(e)
        else:
            try:
                with st.spinner("Scoring employee profile…"):
                    p, leave = run_model(model, features, {k: row[k] for k in features})
                b, bc = band(p)
                st.session_state.last_pred = dict(row={k: row[k] for k in features}, p=p, band=b, bc=bc,
                                                  status="Likely to Leave" if leave else "Likely to Stay",
                                                  ts=dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            except Exception as e:  # noqa: BLE001
                st.error(f"Prediction failed: {e}")

    r = st.session_state.get("last_pred")
    if r:
        section("Prediction result")
        g, k = st.columns([1, 1])
        with g:
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=r["p"] * 100, number=dict(suffix="%", font=dict(size=44, color="#F8FAFC")),
                title=dict(text="Attrition probability", font=dict(size=14, color="#94A3B8")),
                gauge=dict(axis=dict(range=[0, 100], tickcolor="#64748B"), bar=dict(color=r["bc"], thickness=.28),
                           bgcolor="rgba(0,0,0,0)", borderwidth=0,
                           steps=[dict(range=[0, 20], color="rgba(16,185,129,.18)"),
                                  dict(range=[20, 50], color="rgba(245,158,11,.18)"),
                                  dict(range=[50, 100], color="rgba(244,63,94,.18)")])))
            show(fig, 320)
        with k:
            kpi_row([("Predicted Status", r["status"], "Model class prediction", r["bc"]),
                     ("Risk Level", r["band"], "Application-defined band", r["bc"]),
                     ("Attrition Probability", f"{r['p'] * 100:.1f}%", "From predict_proba", C_BAD),
                     ("Retention Probability", f"{(1 - r['p']) * 100:.1f}%", "From predict_proba", C_OK)])
        base = f" The dataset's observed attrition rate is {df['AttritionFlag'].mean() * 100:.1f}%." if df is not None else ""
        insight("Risk bands are <b>application-defined</b> probability ranges (Low &lt;20%, Moderate 20–50%, High ≥50%), "
                "not validated HR risk categories. The model's recall on the positive class is limited in the notebook "
                f"results, so a low score does not guarantee retention.{base}")
        with st.expander("Input summary", expanded=False):
            s = pd.DataFrame({"Feature": list(r["row"].keys()), "Value": [str(v) for v in r["row"].values()]})
            table(s, 400)
        st.download_button("Download prediction report", report_text(r), "attrition_prediction_report.txt",
                           "text/plain", key="dl_pred")


# ---- Model performance ---------------------------------------------------------
TARGET = "Attrition"
EXPECTED_TEST_ROWS = 294
HELDOUT_FILENAME = "heldout_test_data.csv"
UNAVAILABLE_MSG = "Held-out evaluation data unavailable. Upload the test dataset to verify model performance."


def find_heldout():
    """Look ONLY for the exported held-out test file. Never falls back to the full dataset."""
    for d in (BASE, BASE / "data", BASE / "dataset", Path.cwd(), Path.cwd() / "data"):
        p = d / HELDOUT_FILENAME
        if p.exists():
            try:
                return pd.read_csv(p), str(p), None
            except Exception as e:  # noqa: BLE001
                return None, str(p), f"Could not read {p.name}: {e}"
    return None, None, None


def validate_heldout(raw, features):
    """Validate a held-out file. Returns (X, y, error)."""
    if TARGET not in raw.columns:
        return None, None, f"The test file must contain the actual labels in a column named '{TARGET}'."
    missing = [f for f in features if f not in raw.columns]
    if missing:
        return None, None, f"The test file is missing model features: {', '.join(missing)}."
    if len(raw) != EXPECTED_TEST_ROWS:
        return None, None, (f"The file has {len(raw):,} rows, but the held-out test set must contain exactly "
                            f"{EXPECTED_TEST_ROWS} records. Full or training datasets are not accepted.")
    y = raw[TARGET]
    # Normalize both numeric and text labels without silently accepting unknown values.
    y = y.astype(str).str.strip().str.lower().map({
        "yes": 1, "no": 0, "1": 1, "0": 0, "true": 1, "false": 0,
    })
    if y.isna().any():
        return None, None, f"'{TARGET}' must contain only 0/1 (or Yes/No) labels."
    return raw[list(features)].copy(), y.astype(int), None


def evaluate_heldout(_model, X, y, features):
    from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,
                                 precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve)
    classes = list(_model.classes_)
    pos = positive_class(classes)
    Xf = X[list(features)]
    proba = _model.predict_proba(Xf)[:, classes.index(pos)]
    pred = (np.asarray(_model.predict(Xf)) == pos).astype(int)
    out = dict(
        n=int(len(y)), n_pos=int(y.sum()),
        Accuracy=accuracy_score(y, pred), Precision=precision_score(y, pred, zero_division=0),
        Recall=recall_score(y, pred, zero_division=0), F1=f1_score(y, pred, zero_division=0),
        cm=confusion_matrix(y, pred, labels=[0, 1]),
        report=classification_report(y, pred, labels=[0, 1], target_names=["Stayed", "Left"],
                                     output_dict=True, zero_division=0),
        AUC=None, fpr=None, tpr=None, prec_c=None, rec_c=None,
    )
    if y.nunique() == 2:
        out["AUC"] = roc_auc_score(y, proba)
        out["fpr"], out["tpr"], _ = roc_curve(y, proba)
        out["prec_c"], out["rec_c"], _ = precision_recall_curve(y, proba)
    return out


def insample_check(_model, X, features):
    classes = list(_model.classes_)
    pos = positive_class(classes)
    Xf = X[list(features)]
    proba = _model.predict_proba(Xf)[:, classes.index(pos)]
    pred = (np.asarray(_model.predict(Xf)) == pos).astype(int)
    return dict(n=int(len(Xf)), mean_p=float(proba.mean()), pred_rate=float(pred.mean()))


def cm_fig(cm, title):
    z = [[cm["TN"], cm["FP"]], [cm["FN"], cm["TP"]]]
    fig = go.Figure(go.Heatmap(z=z, x=["Pred: Stayed", "Pred: Left"], y=["Actual: Stayed", "Actual: Left"],
                               text=z, texttemplate="%{text}", textfont=dict(size=20),
                               colorscale=[[0, "#151E30"], [1, C_ACC]], showscale=False,
                               hovertemplate="%{y} / %{x}: %{z}<extra></extra>"))
    fig.update_layout(title=title)
    fig.update_yaxes(autorange="reversed")
    return fig


def page_model(df):
    header("Model Performance", "Held-out evaluation of the deployed Logistic Regression attrition classifier.")
    model, mpath, merr, vwarn = load_model()
    if model is None:
        empty_state("Model not loaded", merr or "")
        return
    if vwarn:
        note(f"Model loaded with a scikit-learn version warning: {vwarn}. Install the version used for training "
             "if results look wrong.")
    features = model_features(model)

    # ---- held-out data source (never the full / training dataset) ----
    disk_raw, disk_path, disk_err = find_heldout()
    with st.expander("Held-out test data source",
                     expanded=(disk_raw is None and st.session_state.get("ho_upload") is None)):
        st.caption(f"Expected: {EXPECTED_TEST_ROWS} held-out records with the {len(features)} model features and the "
                   f"actual labels in '{TARGET}' (file name: {HELDOUT_FILENAME}).")
        up = st.file_uploader("Upload held-out test dataset (CSV)", type="csv", key="ho_upload")
    raw, src, rerr = None, None, disk_err
    if up is not None:
        try:
            raw, src, rerr = pd.read_csv(up), f"Uploaded file: {up.name}", None
        except Exception as e:  # noqa: BLE001
            rerr = f"Could not read the uploaded file: {e}"
    elif disk_raw is not None:
        raw, src = disk_raw, disk_path

    X = y = res = None
    verr = rerr
    if raw is not None and verr is None:
        X, y, verr = validate_heldout(raw, features)
    if X is not None:
        try:
            res = evaluate_heldout(model, X, y, tuple(features))
        except Exception as e:  # noqa: BLE001
            verr = (f"Evaluation failed ({type(e).__name__}): {e}. This can happen when the model was saved with a "
                    "different scikit-learn version than the one installed.")

    if res is None:
        if verr:
            st.error(verr)
        empty_state("Held-out evaluation unavailable", UNAVAILABLE_MSG)
        with st.expander("Notebook experiment reference (user-reported, not computed by this app)"):
            note("These figures were typed in from the notebook and are <b>not verified</b> against the deployed model.")
            ref = pd.DataFrame(NOTEBOOK).T.reset_index().rename(columns={"index": "Experiment", "CV": "CV mean F1"})
            table(ref.round(4), 150)
    else:
        st.caption(f"Evaluated the deployed model on {res['n']} held-out records ({res['n_pos']} leavers) · "
                   f"source: {Path(src.split(': ')[-1]).name if src else '—'}")
        auc = res["AUC"]
        kpi_row([("Accuracy", f"{res['Accuracy'] * 100:.2f}%", "Held-out test set", C_ACC),
                 ("Precision", f"{res['Precision'] * 100:.2f}%", "Left class", C_CYAN),
                 ("Recall", f"{res['Recall'] * 100:.2f}%", "Left class", C_WARN),
                 ("F1-Score", f"{res['F1'] * 100:.2f}%", "Left class", C_OK),
                 ("ROC-AUC", f"{auc:.4f}" if auc is not None else "n/a", "Held-out test set", C_ACC)])
        cm = res["cm"]
        cmd = dict(TN=int(cm[0][0]), FP=int(cm[0][1]), FN=int(cm[1][0]), TP=int(cm[1][1]))
        c1, c2 = st.columns(2)
        with c1:
            show(cm_fig(cmd, "Confusion Matrix (held-out)"))
        with c2:
            if res["fpr"] is not None:
                fig = go.Figure()
                fig.add_scatter(x=res["fpr"], y=res["tpr"], mode="lines", line=dict(color=C_ACC, width=3), name="Model")
                fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="#64748B", dash="dot"), name="Chance")
                fig.update_layout(title=f"ROC Curve (AUC {auc:.4f})", xaxis_title="False positive rate",
                                  yaxis_title="True positive rate")
                show(fig)
            else:
                empty_state("ROC unavailable", "The test labels contain only one class.")
        c1, c2 = st.columns(2)
        with c1:
            vals = [res["Accuracy"], res["Precision"], res["Recall"], res["F1"], auc if auc is not None else 0]
            names = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
            fig = go.Figure(go.Bar(x=names, y=vals, marker_color=[C_ACC, C_CYAN, C_WARN, C_OK, C_ACC],
                                   text=[f"{v:.3f}" for v in vals], textposition="outside"))
            fig.update_layout(title="Deployed Model Metrics (held-out)", showlegend=False)
            fig.update_yaxes(range=[0, 1.1])
            show(fig)
        with c2:
            if res["rec_c"] is not None:
                fig = go.Figure(go.Scatter(x=res["rec_c"], y=res["prec_c"], mode="lines",
                                           line=dict(color=C_CYAN, width=3), fill="tozeroy",
                                           fillcolor="rgba(34,211,238,.10)"))
                fig.update_layout(title="Precision vs Recall Curve", xaxis_title="Recall", yaxis_title="Precision")
                fig.update_yaxes(range=[0, 1.05])
                show(fig)
        section("Classification report")
        rep = res["report"]
        rows = []
        for k in ("Stayed", "Left", "macro avg", "weighted avg"):
            r = rep[k]
            rows.append([k, r["precision"], r["recall"], r["f1-score"], int(r["support"])])
        rows.append(["accuracy", np.nan, np.nan, rep["accuracy"], res["n"]])
        rdf = pd.DataFrame(rows, columns=["Class", "Precision", "Recall", "F1-score", "Support"]).round(4)
        table(rdf, 260)

        section("Why accuracy alone is not enough")
        base = max(res["n_pos"], res["n"] - res["n_pos"]) / res["n"] * 100
        st.markdown(
            f'<div class="glass"><ul><li>Attrition is an <b>imbalanced</b> problem: predicting the majority class for every employee '
            f'would score <b>{base:.1f}%</b> accuracy on this test set while catching none of the {res["n_pos"]} leavers.</li>'
            f'<li>On this test set the deployed model finds <b>{res["Recall"] * 100:.1f}%</b> of leavers (recall) with '
            f'<b>{res["Precision"] * 100:.1f}%</b> of its “Left” predictions correct (precision).</li>'
            "<li>The right balance depends on the cost of a missed leaver versus a wasted retention conversation; read precision, recall, F1 and ROC-AUC together.</li></ul></div>",
            unsafe_allow_html=True)

    section("Logistic Regression coefficient analysis")
    try:
        names_out = model[:-1].get_feature_names_out()
        coef = model[-1].coef_[0]
        cdf = pd.DataFrame({"Feature": [str(n).split("__", 1)[-1] for n in names_out], "Coefficient": coef})
        top = pd.concat([cdf.nlargest(10, "Coefficient"), cdf.nsmallest(10, "Coefficient")]).drop_duplicates()
        top = top.sort_values("Coefficient")
        fig = go.Figure(go.Bar(x=top["Coefficient"], y=top["Feature"], orientation="h",
                               marker_color=[C_BAD if v > 0 else C_CYAN for v in top["Coefficient"]],
                               hovertemplate="<b>%{y}</b><br>Coefficient: %{x:.3f}<extra></extra>"))
        fig.update_layout(title="Largest coefficients (positive = higher predicted attrition odds)")
        show(fig, 560)
        insight("Coefficients apply to standardized numeric and one-hot encoded features, so magnitudes are comparable "
                "within the model but describe association learned by the model, not causal effects.")
    except Exception as e:  # noqa: BLE001
        note(f"Coefficients could not be extracted from this model object ({e}).")

    section("In-Sample Sanity Check — Not a Model Performance Metric.")
    if df is None:
        empty_state("Unavailable", "The full dataset is required for this check.")
    else:
        with st.expander("Run full-dataset prediction check"):
            note("This scores the <b>entire loaded dataset</b>, which overlaps the training data. It only confirms the "
                 "saved pipeline runs end-to-end on every row and must <b>not</b> be quoted as model performance.")
            if st.button("Run sanity check", key="run_sanity"):
                try:
                    r = insample_check(model, df[list(features)], tuple(features))
                    kpi_row([("Rows scored", f"{r['n']:,}", "Pipeline ran without errors", C_OK),
                             ("Mean predicted probability", f"{r['mean_p'] * 100:.1f}%", "Average P(Left)", C_CYAN),
                             ("Predicted leaver share", f"{r['pred_rate'] * 100:.1f}%", "Class predictions", C_WARN),
                             ("Observed attrition rate", f"{df['AttritionFlag'].mean() * 100:.1f}%", "Dataset label", C_ACC)])
                except Exception as e:  # noqa: BLE001
                    st.error(f"Sanity check failed ({type(e).__name__}): {e}")


# ---- Business insights ---------------------------------------------------------
def page_insights(df):
    header("Business Insights", "Dataset-driven observations and rule-based suggestions for HR investigation.")
    n = len(df)
    overall = df["AttritionFlag"].mean() * 100
    cards, recs = [], []

    ot_y, ny = grp(df, df["OverTime"] == "Yes")
    ot_n, nn = grp(df, df["OverTime"] == "No")
    if ny and nn:
        cards.append(("Overtime", f"{ot_y:.1f}% vs {ot_n:.1f}%", f"Attrition with vs without overtime ({ny} / {nn} employees)", C_WARN))
        if ot_y - ot_n >= 5:
            recs.append(("Review overtime and workload distribution",
                         f"Employees working overtime show {ot_y:.1f}% observed attrition vs {ot_n:.1f}% without ({ot_y - ot_n:+.1f} pts).",
                         "Consider reviewing workload allocation and overtime patterns, and gather qualitative feedback."))

    d = top_group(rate_table(df, "Department"), 20)
    if d is not None:
        cards.append(("Highest-attrition department", d["Department"], f"{d['Rate']:.1f}% observed ({int(d['Left'])} of {int(d['Employees'])})", C_BAD))
        if d["Rate"] - overall >= 2:
            recs.append((f"Investigate the {d['Department']} department",
                         f"Observed attrition is {d['Rate']:.1f}% vs {overall:.1f}% overall.",
                         "Consider stay interviews and a review of role design, management practices and growth paths there."))

    if "JobSatisfaction" in df:
        lo, nlo = grp(df, df["JobSatisfaction"] <= 2)
        hi, nhi = grp(df, df["JobSatisfaction"] >= 3)
        if nlo and nhi:
            cards.append(("Job satisfaction", f"{lo:.1f}% vs {hi:.1f}%", f"Attrition at low (1–2) vs high (3–4) satisfaction ({nlo} / {nhi})", C_CYAN))
            if lo - hi >= 3:
                recs.append(("Evaluate employee satisfaction",
                             f"Lower-satisfaction employees show {lo:.1f}% attrition vs {hi:.1f}% for higher-satisfaction employees.",
                             "Consider pulse surveys and follow-ups with lower-satisfaction groups."))

    med_l = df.loc[df["AttritionFlag"] == 1, "MonthlyIncome"].median()
    med_s = df.loc[df["AttritionFlag"] == 0, "MonthlyIncome"].median()
    if not np.isnan(med_l) and not np.isnan(med_s):
        cards.append(("Median monthly income", f"{med_l:,.0f} vs {med_s:,.0f}", "Employees who left vs stayed", C_OK))
        if med_s > 0 and (med_s - med_l) / med_s >= 0.05:
            recs.append(("Analyze compensation patterns",
                         f"Median income of leavers ({med_l:,.0f}) is {(med_s - med_l) / med_s * 100:.0f}% below that of stayers ({med_s:,.0f}).",
                         "Consider benchmarking pay by role and level; income also correlates with seniority, so compare like-for-like."))

    pl, npl = grp(df, df["YearsSinceLastPromotion"] >= 5)
    ph, nph = grp(df, df["YearsSinceLastPromotion"] < 5)
    if npl and nph:
        cards.append(("Promotion gap", f"{pl:.1f}% vs {ph:.1f}%", f"5+ yrs vs <5 yrs since promotion ({npl} / {nph})", C_ACC))
        if pl - ph >= 3:
            recs.append(("Review career development opportunities",
                         f"Employees 5+ years since promotion show {pl:.1f}% attrition vs {ph:.1f}%.",
                         "Consider reviewing promotion cadence and development paths for long-tenured, non-promoted staff."))

    ten_s, nts = grp(df, df["YearsAtCompany"] <= 2)
    ten_l, ntl = grp(df, df["YearsAtCompany"] > 2)
    if nts and ntl:
        cards.append(("Early tenure", f"{ten_s:.1f}% vs {ten_l:.1f}%", f"≤2 yrs vs >2 yrs at company ({nts} / {ntl})", C_WARN))
        if ten_s - ten_l >= 5:
            recs.append(("Strengthen early-tenure support",
                         f"Employees with ≤2 years at the company show {ten_s:.1f}% attrition vs {ten_l:.1f}%.",
                         "Consider reviewing onboarding, mentoring and early check-ins."))

    tf, ntf = grp(df, df["BusinessTravel"] == "Travel_Frequently")
    to, nto = grp(df, df["BusinessTravel"] != "Travel_Frequently")
    if ntf and nto:
        cards.append(("Frequent travel", f"{tf:.1f}% vs {to:.1f}%", f"Frequent travellers vs others ({ntf} / {nto})", C_CYAN))
        if tf - to >= 5:
            recs.append(("Examine travel load",
                         f"Frequent travellers show {tf:.1f}% attrition vs {to:.1f}% for others.",
                         "Consider reviewing travel expectations and recovery time for frequent travellers."))

    kpi_row(cards[:6] if len(cards) >= 6 else cards)
    if len(cards) > 6:
        kpi_row(cards[6:])

    section("Rule-based suggestions for HR investigation")
    note("Suggestions fire when an observed gap passes an <b>application-defined threshold</b> (e.g. ≥5 percentage points). "
         "They flag areas worth investigating and are <b>not proven causes</b> of turnover.")
    if not recs:
        empty_state("No thresholds exceeded", "None of the rule thresholds were met in this dataset.")
    for title, evidence, suggestion in recs:
        st.markdown(f'<div class="rec"><h4>{title}</h4><p><b>Evidence:</b> {evidence}</p><p><b>Suggestion:</b> {suggestion}</p></div>',
                    unsafe_allow_html=True)

    section("Supporting analysis")
    with st.expander("Overtime and satisfaction"):
        a, b = st.columns(2)
        with a:
            rate_fig(df, "OverTime", "Attrition Rate by Overtime")
        with b:
            rate_fig(df, "JobSatisfactionLabel" if "JobSatisfactionLabel" in df else "JobSatisfaction", "Attrition Rate by Job Satisfaction")
    with st.expander("Compensation"):
        fig = px.box(df, x="Status", y="MonthlyIncome", color="Status", color_discrete_map=STATUS_COLORS,
                     title="Monthly Income: Left vs Stayed", points=False)
        fig.update_layout(showlegend=False)
        show(fig, 380)
        rate_fig(df, "IncomeBand", "Attrition Rate by Income Band")
    with st.expander("Employee experience"):
        a, b = st.columns(2)
        with a:
            rate_fig(df, "TenureBand", "Attrition Rate by Years at Company")
        with b:
            rate_fig(df, "PromoBand", "Attrition Rate by Years Since Promotion")


def page_about(df, src, refreshed, derr):
    header("Settings / About", "Application status, data sources and limitations.")
    model, mpath, merr, vwarn = load_model()
    ho, hpath, herr = find_heldout()
    import sklearn
    rows = [
        ("Dataset", f"Loaded · {len(df):,} rows × {df.shape[1]} columns" if df is not None else f"Unavailable — {derr}"),
        ("Dataset file", src or "—"),
        ("Model", f"Loaded · {type(model).__name__}" if model is not None else f"Unavailable — {merr}"),
        ("Model file", mpath or "—"),
        ("Model load warning", vwarn or "None"),
        ("Held-out test file", hpath or "Not found (upload on the Model Performance page)"),
        ("Python libs", f"streamlit {st.__version__} · pandas {pd.__version__} · scikit-learn {sklearn.__version__}"),
    ]
    st.markdown('<div class="glass"><h3>System status</h3><ul>' +
                "".join(f"<li><b>{k}:</b> {v}</li>" for k, v in rows) + "</ul></div>", unsafe_allow_html=True)
    st.markdown(
        '<div class="glass"><h3>Limitations</h3><ul>'
        "<li>The dataset is a static snapshot; no historical trends are shown.</li>"
        "<li>Coded fields (satisfaction, education, etc.) are labelled using the standard IBM HR data dictionary.</li>"
        "<li>Model metrics are computed only from the 294-record held-out test file; without it, no performance metrics are shown.</li>"
        "<li>Predictions are statistical estimates and should support, not replace, HR judgement. Risk bands are application-defined.</li>"
        "<li>Employee identifiers are hashed into anonymous references.</li></ul></div>", unsafe_allow_html=True)
    if st.button("Clear cached data and model", key="clear_cache"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("Caches cleared. Reloading on next interaction.")


# =============================================================================
# 7. APP SHELL
# =============================================================================
if "page" not in st.session_state:
    st.session_state.page = "Overview"

with st.sidebar:
    st.markdown('<div class="brand"><div class="logo">E</div><div><div class="brand-t">Employee Attrition Intelligence</div>'
                '<div class="brand-s">Workforce analytics</div></div></div><div class="nav-cap">Navigation</div>',
                unsafe_allow_html=True)
    for name, icon in PAGES:
        st.button(f"{icon}  {name}", key=f"nav_{name}", on_click=goto, args=(name,),
                  type="primary" if st.session_state.page == name else "secondary")
    st.markdown('<div class="profile"><b>Data Science Analytics Platform</b><br>Powered by Machine Learning</div>',
                unsafe_allow_html=True)

DF, SRC, REFRESHED, DERR = load_data()
page = st.session_state.page

if page == "Settings / About":
    page_about(DF, SRC, REFRESHED, DERR)
elif page == "Predict Attrition":
    page_predict(DF)
elif page == "Model Performance":
    page_model(DF)
else:
    need_data(DF, DERR)
    if page == "Overview":
        page_overview(DF, SRC, REFRESHED)
    elif page == "Workforce Analytics":
        page_workforce(DF)
    elif page == "Attrition Analysis":
        page_attrition(DF)
    elif page == "Employee Explorer":
        page_explorer(DF)
    elif page == "Business Insights":
        page_insights(DF)
