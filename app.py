"""
Heart Disease Prediction — Streamlit App
Predict + Model Performance only. Premium obsidian & crimson aesthetic.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as goR
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_curve, auc)
import warnings, os

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="CardioSense · AI",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# GLOBAL CSS — Obsidian / Crimson / Gold theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;600&display=swap');

/* ─── Reset & root ─────────────────────────── */
:root {
    --bg-0:       #090B10;
    --bg-1:       #0E1118;
    --bg-2:       #141720;
    --bg-3:       #1C2030;
    --border:     rgba(255,255,255,0.07);
    --border-h:   rgba(255,255,255,0.14);
    --crimson:    #C8374A;
    --crimson-l:  #E8455B;
    --crimson-gl: rgba(200,55,74,0.18);
    --gold:       #D4A843;
    --gold-gl:    rgba(212,168,67,0.14);
    --jade:       #3ECFA0;
    --jade-gl:    rgba(62,207,160,0.14);
    --text-hi:    #EEF0F6;
    --text-md:    #9BA3BB;
    --text-lo:    #5C6278;
    --font-head:  'DM Serif Display', Georgia, serif;
    --font-body:  'DM Sans', system-ui, sans-serif;
    --font-mono:  'JetBrains Mono', monospace;
    --rad-sm:     8px;
    --rad-md:     14px;
    --rad-lg:     22px;
    --rad-xl:     32px;
    --shadow-glow-c: 0 0 40px rgba(200,55,74,0.20);
    --shadow-glow-j: 0 0 40px rgba(62,207,160,0.18);
    --shadow-card: 0 4px 32px rgba(0,0,0,0.55);
}

/* ─── App shell ─────────────────────────────── */
.stApp {
    background: var(--bg-0) !important;
    font-family: var(--font-body);
}

/* Animated radial glow behind content */
.stApp::before {
    content: '';
    position: fixed;
    top: -20%;
    left: 50%;
    transform: translateX(-50%);
    width: 800px;
    height: 500px;
    background: radial-gradient(ellipse, rgba(200,55,74,0.07) 0%, transparent 70%);
    pointer-events: none;
    z-index: 0;
}

/* ─── Sidebar ────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: var(--bg-1) !important;
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] > div {
    padding-top: 2rem;
}

/* ─── Sidebar nav labels ─────────────────────── */
.stRadio label {
    color: var(--text-md) !important;
    font-family: var(--font-body);
    font-size: 14px;
    font-weight: 500;
    letter-spacing: 0.03em;
}

/* ─── Headings ───────────────────────────────── */
h1 {
    font-family: var(--font-head) !important;
    color: var(--text-hi) !important;
    font-size: 2.8rem !important;
    letter-spacing: -0.02em;
    line-height: 1.15;
    margin-bottom: 0.25rem !important;
}
h2, h3 {
    font-family: var(--font-head) !important;
    color: var(--text-hi) !important;
    letter-spacing: -0.01em;
}

/* ─── Metrics ────────────────────────────────── */
div[data-testid="stMetric"] {
    background: var(--bg-2);
    border: 1px solid var(--border);
    border-radius: var(--rad-md);
    padding: 22px 26px 18px;
    position: relative;
    overflow: hidden;
    box-shadow: var(--shadow-card);
    transition: border-color 0.25s;
}
div[data-testid="stMetric"]:hover {
    border-color: var(--border-h);
}
div[data-testid="stMetric"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--crimson), var(--gold));
}
div[data-testid="stMetric"] label {
    color: var(--text-lo) !important;
    font-family: var(--font-mono) !important;
    font-size: 11px !important;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: var(--text-hi) !important;
    font-family: var(--font-head) !important;
    font-size: 2rem !important;
}

/* ─── Form inputs ────────────────────────────── */
.stSelectbox label,
.stSlider label,
.stNumberInput label {
    color: var(--text-md) !important;
    font-size: 12px !important;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    font-family: var(--font-mono) !important;
}
div[data-baseweb="select"] > div {
    background: var(--bg-2) !important;
    border-color: var(--border) !important;
    border-radius: var(--rad-sm) !important;
    color: var(--text-hi) !important;
}
div[data-baseweb="select"] > div:hover {
    border-color: var(--crimson) !important;
}

/* ─── Slider ─────────────────────────────────── */
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background: var(--crimson) !important;
    border-color: var(--crimson) !important;
    box-shadow: 0 0 0 4px var(--crimson-gl) !important;
}

/* ─── Primary button ─────────────────────────── */
.stButton > button[kind="primaryFormSubmit"],
.stButton > button {
    background: linear-gradient(135deg, var(--crimson) 0%, #8B1F2C 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: var(--rad-md) !important;
    padding: 14px 40px !important;
    font-family: var(--font-body) !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    letter-spacing: 0.04em;
    box-shadow: 0 4px 20px rgba(200,55,74,0.35) !important;
    transition: all 0.3s ease !important;
    width: 100%;
}
.stButton > button:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 8px 30px rgba(200,55,74,0.5) !important;
}

/* ─── Tabs ───────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-2);
    border-radius: var(--rad-md);
    padding: 4px;
    gap: 4px;
    border: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-lo) !important;
    border-radius: var(--rad-sm) !important;
    font-family: var(--font-mono) !important;
    font-size: 12px !important;
    letter-spacing: 0.08em;
    padding: 10px 20px !important;
    transition: all 0.2s;
}
.stTabs [aria-selected="true"] {
    background: var(--bg-3) !important;
    color: var(--text-hi) !important;
    border-bottom: 2px solid var(--crimson) !important;
}

/* ─── Dataframe ──────────────────────────────── */
.stDataFrame, iframe {
    border-radius: var(--rad-md) !important;
    overflow: hidden;
    border: 1px solid var(--border) !important;
}

/* ─── Divider ────────────────────────────────── */
hr { border-color: var(--border) !important; margin: 28px 0 !important; }

/* ─── Custom prediction cards ────────────────── */
.result-safe, .result-risk {
    border-radius: var(--rad-xl);
    padding: 40px 32px 36px;
    text-align: center;
    position: relative;
    overflow: hidden;
    box-shadow: var(--shadow-card);
}
.result-safe {
    background: linear-gradient(135deg, rgba(62,207,160,0.10) 0%, rgba(9,11,16,0.95) 100%);
    border: 1px solid rgba(62,207,160,0.25);
    box-shadow: var(--shadow-glow-j);
}
.result-risk {
    background: linear-gradient(135deg, rgba(200,55,74,0.12) 0%, rgba(9,11,16,0.95) 100%);
    border: 1px solid rgba(200,55,74,0.28);
    box-shadow: var(--shadow-glow-c);
}
.result-safe::after, .result-risk::after {
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(ellipse at top, rgba(255,255,255,0.03) 0%, transparent 65%);
    pointer-events: none;
}
.result-verdict {
    font-family: var(--font-head);
    font-size: 2rem;
    margin: 0 0 8px;
    letter-spacing: -0.01em;
}
.result-safe  .result-verdict { color: #3ECFA0; }
.result-risk  .result-verdict { color: #E8455B; }
.result-prob {
    font-family: var(--font-mono);
    font-size: 13px;
    color: var(--text-md);
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.result-sub {
    font-size: 12px;
    color: var(--text-lo);
    margin-top: 14px;
    letter-spacing: 0.04em;
}

/* ─── Section label ──────────────────────────── */
.section-eyebrow {
    font-family: var(--font-mono);
    font-size: 10px;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: var(--crimson);
    margin-bottom: 6px;
}

/* ─── Input group card ───────────────────────── */
.input-group {
    background: var(--bg-2);
    border: 1px solid var(--border);
    border-radius: var(--rad-lg);
    padding: 28px 24px 20px;
    position: relative;
}
.input-group-title {
    font-family: var(--font-mono);
    font-size: 10px;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--gold);
    margin-bottom: 18px;
    display: block;
}

/* ─── Sidebar brand ──────────────────────────── */
.sidebar-brand {
    font-family: var(--font-head);
    font-size: 1.5rem;
    color: var(--text-hi);
    margin-bottom: 4px;
}
.sidebar-tagline {
    font-family: var(--font-mono);
    font-size: 10px;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--crimson);
    margin-bottom: 0;
}

/* ─── Scrollbar ──────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-1); }
::-webkit-scrollbar-thumb { background: var(--bg-3); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--crimson); }

/* ─── Alert / info boxes ─────────────────────── */
.stAlert { border-radius: var(--rad-md) !important; }

/* ─── Form container ─────────────────────────── */
div[data-testid="stForm"] {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# DATA LOADING & MODEL TRAINING
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    possible_paths = [
        os.path.join(os.path.dirname(__file__), "dataset", "heart.csv"),
        os.path.join(os.path.dirname(__file__), "heart.csv"),
        "dataset/heart.csv",
        "heart.csv",
    ]
    for p in possible_paths:
        if os.path.exists(p):
            return pd.read_csv(p)
    st.error("❌ heart.csv not found. Place it in the project directory.")
    st.stop()


@st.cache_resource
def train_model(df_hash):
    data = load_data().copy()
    cat_cols = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        encoders[col] = le

    X = data.drop("HeartDisease", axis=1)
    y = data["HeartDisease"]

    scaler = StandardScaler()
    num_cols = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
    X[num_cols] = scaler.fit_transform(X[num_cols])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42
    )
    model.fit(X_train, y_train)

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    acc     = accuracy_score(y_test, y_pred)
    report  = classification_report(y_test, y_pred, output_dict=True)
    cm      = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    feature_imp = pd.Series(
        model.feature_importances_, index=X.columns
    ).sort_values(ascending=False)

    return dict(
        model=model, encoders=encoders, scaler=scaler,
        num_cols=num_cols, cat_cols=cat_cols,
        accuracy=acc, report=report,
        confusion_matrix=cm, fpr=fpr, tpr=tpr, roc_auc=roc_auc,
        feature_importance=feature_imp,
        X_test=X_test, y_test=y_test, y_pred=y_pred,
    )


# ─────────────────────────────────────────────
# PLOTLY DARK THEME
# ─────────────────────────────────────────────
def dark_fig(fig, title=""):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(20,23,32,0.7)",
        font=dict(family="DM Sans, sans-serif", color="#9BA3BB", size=12),
        title=dict(
            text=title,
            font=dict(family="DM Serif Display, serif", size=19, color="#EEF0F6"),
            x=0.02, xanchor="left",
        ),
        margin=dict(l=48, r=24, t=56, b=40),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(255,255,255,0.06)"),
    )
    fig.update_xaxes(
        gridcolor="rgba(255,255,255,0.05)",
        zeroline=False,
        tickfont=dict(family="JetBrains Mono, monospace", size=11),
    )
    fig.update_yaxes(
        gridcolor="rgba(255,255,255,0.05)",
        zeroline=False,
        tickfont=dict(family="JetBrains Mono, monospace", size=11),
    )
    return fig


PALETTE = ["#C8374A", "#D4A843", "#3ECFA0", "#6E8EF7", "#A78BFA", "#F97316"]


# ─────────────────────────────────────────────
# LOAD DATA & MODEL
# ─────────────────────────────────────────────
df        = load_data()
artefacts = train_model(hash(df.to_csv()))
model     = artefacts["model"]


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:0 8px 28px;">
        <div class="sidebar-brand">CardioSense</div>
        <div class="sidebar-tagline">AI Cardiac Risk Engine</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "nav",
        ["🔮  Predict", "📈  Model Performance"],
        label_visibility="collapsed",
    )

    st.markdown("---")

    # Mini accuracy badge in sidebar
    acc_pct = artefacts["accuracy"] * 100
    st.markdown(f"""
    <div style="background:rgba(200,55,74,0.08);border:1px solid rgba(200,55,74,0.22);
                border-radius:12px;padding:16px 18px;margin-top:8px;">
        <div style="font-family:'JetBrains Mono',monospace;font-size:10px;
                    letter-spacing:0.18em;text-transform:uppercase;color:#C8374A;
                    margin-bottom:8px;">Model Accuracy</div>
        <div style="font-family:'DM Serif Display',serif;font-size:2rem;color:#EEF0F6;
                    line-height:1;">{acc_pct:.1f}%</div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:10px;
                    color:#5C6278;margin-top:4px;">Gradient Boosting · n=200</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.caption("Built with Streamlit · scikit-learn · Plotly")


# ═════════════════════════════════════════════
# PAGE: PREDICT
# ═════════════════════════════════════════════
if "Predict" in page:
    st.markdown('<div class="section-eyebrow">Clinical Assessment</div>', unsafe_allow_html=True)
    st.markdown("# Cardiac Risk Predictor")
    st.markdown(
        '<p style="color:var(--text-md);font-size:15px;margin-bottom:2rem;max-width:600px;">'
        "Enter the patient's clinical parameters below. The model evaluates "
        "11 biomarkers to estimate the probability of heart disease."
        "</p>",
        unsafe_allow_html=True,
    )

    # ── Split into three styled groups ──────────────────────────────────────
    with st.form("prediction_form"):
        # GROUP 1 — Demographics
        st.markdown("""
        <div style="background:var(--bg-2);border:1px solid var(--border);
                    border-radius:18px;padding:28px 24px 10px;margin-bottom:18px;">
            <span style="font-family:'JetBrains Mono',monospace;font-size:10px;
                         letter-spacing:0.2em;text-transform:uppercase;
                         color:#D4A843;">01 · Demographics</span>
        """, unsafe_allow_html=True)
        d1, d2, d3 = st.columns(3)
        with d1:
            age = st.slider("Age (years)", 28, 77, 54)
        with d2:
            sex = st.selectbox("Biological Sex", ["M", "F"],
                               format_func=lambda x: "Male" if x == "M" else "Female")
        with d3:
            fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1],
                                      format_func=lambda x: "Yes" if x == 1 else "No")
        st.markdown("</div>", unsafe_allow_html=True)

        # GROUP 2 — Cardiac Vitals
        st.markdown("""
        <div style="background:var(--bg-2);border:1px solid var(--border);
                    border-radius:18px;padding:28px 24px 10px;margin-bottom:18px;">
            <span style="font-family:'JetBrains Mono',monospace;font-size:10px;
                         letter-spacing:0.2em;text-transform:uppercase;
                         color:#3ECFA0;">02 · Cardiac Vitals</span>
        """, unsafe_allow_html=True)
        v1, v2, v3, v4 = st.columns(4)
        with v1:
            resting_bp = st.slider("Resting BP (mm Hg)", 80, 200, 130)
        with v2:
            cholesterol = st.slider("Cholesterol (mg/dl)", 0, 603, 200)
        with v3:
            max_hr = st.slider("Max Heart Rate", 60, 202, 140)
        with v4:
            oldpeak = st.slider("Oldpeak (ST depression)", -2.6, 6.2, 0.0, step=0.1)
        st.markdown("</div>", unsafe_allow_html=True)

        # GROUP 3 — ECG & Symptoms
        st.markdown("""
        <div style="background:var(--bg-2);border:1px solid var(--border);
                    border-radius:18px;padding:28px 24px 10px;margin-bottom:28px;">
            <span style="font-family:'JetBrains Mono',monospace;font-size:10px;
                         letter-spacing:0.2em;text-transform:uppercase;
                         color:#C8374A;">03 · ECG & Symptoms</span>
        """, unsafe_allow_html=True)
        e1, e2, e3, e4 = st.columns(4)
        with e1:
            chest_pain = st.selectbox(
                "Chest Pain Type", ["ASY", "ATA", "NAP", "TA"],
                help="ASY=Asymptomatic · ATA=Atypical Angina · NAP=Non-Anginal · TA=Typical Angina",
            )
        with e2:
            resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
        with e3:
            exercise_angina = st.selectbox(
                "Exercise Angina", ["N", "Y"],
                format_func=lambda x: "Yes" if x == "Y" else "No",
            )
        with e4:
            st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])
        st.markdown("</div>", unsafe_allow_html=True)

        # ── Submit ───────────────────────────────────────────────────────────
        col_btn = st.columns([1, 2, 1])
        with col_btn[1]:
            submitted = st.form_submit_button("🫀  Run Cardiac Risk Assessment")

    # ── Result ───────────────────────────────────────────────────────────────
    if submitted:
        input_data = pd.DataFrame({
            "Age":            [age],
            "Sex":            [sex],
            "ChestPainType":  [chest_pain],
            "RestingBP":      [resting_bp],
            "Cholesterol":    [cholesterol],
            "FastingBS":      [fasting_bs],
            "RestingECG":     [resting_ecg],
            "MaxHR":          [max_hr],
            "ExerciseAngina": [exercise_angina],
            "Oldpeak":        [oldpeak],
            "ST_Slope":       [st_slope],
        })

        for col in artefacts["cat_cols"]:
            input_data[col] = artefacts["encoders"][col].transform(input_data[col])
        input_data[artefacts["num_cols"]] = artefacts["scaler"].transform(
            input_data[artefacts["num_cols"]]
        )

        prediction = model.predict(input_data)[0]
        proba      = model.predict_proba(input_data)[0]

        st.markdown("---")

        r1, r2 = st.columns([1.1, 1])

        with r1:
            if prediction == 1:
                st.markdown(f"""
                <div class="result-risk">
                    <div style="font-size:3rem;margin-bottom:12px;">⚠️</div>
                    <div class="result-verdict">Elevated Risk Detected</div>
                    <div class="result-prob">Disease probability · {proba[1]*100:.1f}%</div>
                    <div class="result-sub">
                        Statistical model output — consult a qualified cardiologist.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-safe">
                    <div style="font-size:3rem;margin-bottom:12px;">✅</div>
                    <div class="result-verdict">Low Risk Profile</div>
                    <div class="result-prob">Disease probability · {proba[1]*100:.1f}%</div>
                    <div class="result-sub">
                        Statistical model output — consult a qualified cardiologist.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # ── Key parameters recap ─────────────────────────────────────────
            st.markdown("""
            <div style="margin-top:20px;background:var(--bg-2);border:1px solid var(--border);
                        border-radius:14px;padding:20px 22px;">
                <div style="font-family:'JetBrains Mono',monospace;font-size:10px;
                            letter-spacing:0.2em;text-transform:uppercase;
                            color:#5C6278;margin-bottom:14px;">Input Summary</div>
            """, unsafe_allow_html=True)
            params = {
                "Age": age, "Sex": sex, "Chest Pain": chest_pain,
                "Resting BP": f"{resting_bp} mmHg", "Cholesterol": f"{cholesterol} mg/dl",
                "Max HR": max_hr, "Oldpeak": oldpeak, "ST Slope": st_slope,
            }
            rows = ""
            for k, v in params.items():
                rows += (
                    f'<div style="display:flex;justify-content:space-between;'
                    f'padding:7px 0;border-bottom:1px solid rgba(255,255,255,0.04);">'
                    f'<span style="color:var(--text-lo);font-size:12px;">{k}</span>'
                    f'<span style="color:var(--text-hi);font-family:\'JetBrains Mono\',monospace;'
                    f'font-size:12px;">{v}</span></div>'
                )
            st.markdown(rows + "</div>", unsafe_allow_html=True)

        with r2:
            # ── Gauge ────────────────────────────────────────────────────────
            bar_color = "#C8374A" if prediction == 1 else "#3ECFA0"
            glow_color = "rgba(200,55,74,0.15)" if prediction == 1 else "rgba(62,207,160,0.12)"
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=proba[1] * 100,
                number=dict(
                    suffix="%",
                    font=dict(size=52, family="DM Serif Display, serif", color="#EEF0F6"),
                ),
                gauge=dict(
                    axis=dict(
                        range=[0, 100],
                        tickcolor="#5C6278",
                        tickfont=dict(family="JetBrains Mono, monospace", size=10),
                        nticks=6,
                    ),
                    bar=dict(color=bar_color, thickness=0.28),
                    bgcolor="rgba(20,23,32,0.8)",
                    borderwidth=0,
                    steps=[
                        dict(range=[0,  33], color="rgba(62,207,160,0.08)"),
                        dict(range=[33, 66], color="rgba(212,168,67,0.08)"),
                        dict(range=[66,100], color="rgba(200,55,74,0.10)"),
                    ],
                    threshold=dict(
                        line=dict(color="rgba(255,255,255,0.35)", width=2),
                        thickness=0.85, value=50,
                    ),
                ),
            ))
            fig_gauge = dark_fig(fig_gauge, "Risk Score")
            fig_gauge.update_layout(height=300, margin=dict(l=24, r=24, t=52, b=12))
            st.plotly_chart(fig_gauge, use_container_width=True)

            # ── Probability bar chart ─────────────────────────────────────────
            fig_proba = go.Figure()
            fig_proba.add_trace(go.Bar(
                x=["No Disease", "Disease"],
                y=[proba[0] * 100, proba[1] * 100],
                marker=dict(
                    color=["#3ECFA0", "#C8374A"],
                    line=dict(width=0),
                ),
                text=[f"{proba[0]*100:.1f}%", f"{proba[1]*100:.1f}%"],
                textposition="outside",
                textfont=dict(family="JetBrains Mono, monospace", size=13, color="#EEF0F6"),
                width=0.45,
            ))
            fig_proba.update_layout(yaxis=dict(range=[0, 115], title="Probability (%)", ticksuffix="%"))
            fig_proba = dark_fig(fig_proba, "Class Probabilities")
            fig_proba.update_layout(height=280, showlegend=False,
                                    margin=dict(l=24, r=24, t=52, b=12))
            st.plotly_chart(fig_proba, use_container_width=True)


# ═════════════════════════════════════════════
# PAGE: MODEL PERFORMANCE
# ═════════════════════════════════════════════
elif "Model Performance" in page:
    st.markdown('<div class="section-eyebrow">Evaluation Report</div>', unsafe_allow_html=True)
    st.markdown("# Model Performance")
    st.markdown(
        '<p style="color:var(--text-md);font-size:15px;margin-bottom:2rem;">'
        "Comprehensive diagnostics for the Gradient Boosting classifier trained on 80% of the dataset."
        "</p>",
        unsafe_allow_html=True,
    )

    # ── Top metric row ────────────────────────────────────────────────────────
    report = artefacts["report"]
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy",         f"{artefacts['accuracy']*100:.1f}%")
    m2.metric("Precision",        f"{report['weighted avg']['precision']*100:.1f}%")
    m3.metric("Recall",           f"{report['weighted avg']['recall']*100:.1f}%")
    m4.metric("F1 Score",         f"{report['weighted avg']['f1-score']*100:.1f}%")
    m5.metric("ROC-AUC",          f"{artefacts['roc_auc']:.3f}")

    st.markdown("---")

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["  Confusion & ROC  ", "  Feature Importance  ", "  Full Report  "])

    # ── TAB 1: Confusion Matrix + ROC ─────────────────────────────────────────
    with tab1:
        ca, cb = st.columns(2)

        with ca:
            cm = artefacts["confusion_matrix"]
            # Annotate with percentages
            total = cm.sum()
            annot = np.array([
                [f"{cm[i,j]}\n({cm[i,j]/total*100:.1f}%)" for j in range(2)]
                for i in range(2)
            ])
            fig_cm = go.Figure(go.Heatmap(
                z=cm,
                x=["Predicted: No Disease", "Predicted: Disease"],
                y=["Actual: No Disease",    "Actual: Disease"],
                text=annot,
                texttemplate="%{text}",
                textfont=dict(family="JetBrains Mono, monospace", size=14),
                colorscale=[
                    [0.0, "#0E1118"],
                    [0.4, "#3B1F35"],
                    [1.0, "#C8374A"],
                ],
                showscale=False,
                hoverongaps=False,
            ))
            fig_cm.update_layout(
                xaxis=dict(side="bottom"),
                yaxis=dict(autorange="reversed"),
            )
            fig_cm = dark_fig(fig_cm, "Confusion Matrix")
            fig_cm.update_layout(height=400)
            st.plotly_chart(fig_cm, use_container_width=True)

        with cb:
            fpr, tpr = artefacts["fpr"], artefacts["tpr"]
            fig_roc = go.Figure()
            # Shaded AUC area
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr, mode="lines",
                name=f"AUC = {artefacts['roc_auc']:.3f}",
                line=dict(color="#C8374A", width=3),
                fill="tozeroy",
                fillcolor="rgba(200,55,74,0.08)",
            ))
            # Diagonal baseline
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines",
                name="Random baseline",
                line=dict(color="#5C6278", width=1.5, dash="dot"),
            ))
            # Operating point (closest to top-left)
            dist = np.sqrt(fpr**2 + (1 - tpr)**2)
            idx  = np.argmin(dist)
            fig_roc.add_trace(go.Scatter(
                x=[fpr[idx]], y=[tpr[idx]], mode="markers",
                name="Optimal threshold",
                marker=dict(color="#D4A843", size=12, symbol="diamond",
                            line=dict(color="#EEF0F6", width=1.5)),
            ))
            fig_roc.update_layout(
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
            )
            fig_roc = dark_fig(fig_roc, f"ROC Curve · AUC = {artefacts['roc_auc']:.3f}")
            fig_roc.update_layout(height=400)
            st.plotly_chart(fig_roc, use_container_width=True)

        # ── Predicted probability distribution ──────────────────────────────
        st.markdown("---")
        y_test    = artefacts["y_test"]
        y_pred    = artefacts["y_pred"]
        X_test_df = artefacts["X_test"]
        y_proba_all = model.predict_proba(X_test_df)[:, 1]

        df_dist = pd.DataFrame({
            "Probability": y_proba_all,
            "Actual":      ["Disease" if v == 1 else "No Disease" for v in y_test],
        })
        fig_dist = px.histogram(
            df_dist, x="Probability", color="Actual",
            barmode="overlay", opacity=0.75,
            nbins=40,
            color_discrete_map={"Disease": "#C8374A", "No Disease": "#3ECFA0"},
        )
        fig_dist.add_vline(x=0.5, line_dash="dot", line_color="#D4A843", line_width=1.5,
                           annotation_text=" Decision boundary",
                           annotation_font=dict(color="#D4A843",
                                                family="JetBrains Mono, monospace", size=11))
        fig_dist = dark_fig(fig_dist, "Predicted Probability Distribution")
        fig_dist.update_layout(height=320)
        st.plotly_chart(fig_dist, use_container_width=True)

    # ── TAB 2: Feature Importance ─────────────────────────────────────────────
    with tab2:
        fi      = artefacts["feature_importance"]
        fi_norm = fi / fi.max()

        # ── Colour-coded horizontal bars ─────────────────────────────────────
        colors_fi = [
            "#C8374A" if v >= 0.7 else ("#D4A843" if v >= 0.4 else "#3ECFA0")
            for v in fi_norm.values
        ]
        fig_fi = go.Figure(go.Bar(
            x=fi.values,
            y=fi.index,
            orientation="h",
            marker=dict(color=colors_fi, line=dict(width=0)),
            text=[f"{v:.4f}" for v in fi.values],
            textposition="outside",
            textfont=dict(family="JetBrains Mono, monospace", size=11, color="#9BA3BB"),
        ))
        fig_fi = dark_fig(fig_fi, "Feature Importances — Gradient Boosting")
        fig_fi.update_layout(
            yaxis=dict(autorange="reversed"),
            xaxis_title="Importance Score",
            yaxis_title="",
            height=420,
        )
        st.plotly_chart(fig_fi, use_container_width=True)

        # ── Legend for color bands ───────────────────────────────────────────
        st.markdown("""
        <div style="display:flex;gap:20px;margin-top:-8px;padding:14px 0;">
            <span style="font-family:'JetBrains Mono',monospace;font-size:11px;color:#5C6278;">
                <span style="color:#C8374A;">■</span> High impact (&gt;70%)
            </span>
            <span style="font-family:'JetBrains Mono',monospace;font-size:11px;color:#5C6278;">
                <span style="color:#D4A843;">■</span> Medium impact (40–70%)
            </span>
            <span style="font-family:'JetBrains Mono',monospace;font-size:11px;color:#5C6278;">
                <span style="color:#3ECFA0;">■</span> Lower impact (&lt;40%)
            </span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # ── Sunburst of top features ─────────────────────────────────────────
        top_n  = 8
        fi_top = fi.head(top_n)
        fig_sun = px.bar_polar(
            r=fi_top.values,
            theta=fi_top.index,
            color=fi_top.values,
            color_continuous_scale=["#1C2030", "#D4A843", "#C8374A"],
            template="plotly_dark",
        )
        fig_sun.update_coloraxes(showscale=False)
        fig_sun.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            polar=dict(
                bgcolor="rgba(20,23,32,0.6)",
                radialaxis=dict(
                    visible=True,
                    gridcolor="rgba(255,255,255,0.06)",
                    tickfont=dict(family="JetBrains Mono, monospace", size=9),
                ),
                angularaxis=dict(
                    tickfont=dict(family="DM Sans, sans-serif", size=11, color="#9BA3BB"),
                    gridcolor="rgba(255,255,255,0.06)",
                ),
            ),
            title=dict(
                text="Top 8 Features — Polar View",
                font=dict(family="DM Serif Display, serif", size=18, color="#EEF0F6"),
            ),
            font=dict(family="DM Sans, sans-serif"),
            height=420,
        )
        st.plotly_chart(fig_sun, use_container_width=True)

    # ── TAB 3: Classification Report ──────────────────────────────────────────
    with tab3:
        # ── Styled classification report ─────────────────────────────────────
        report_df = pd.DataFrame(report).transpose().round(4)
        st.dataframe(
            report_df.style
                .background_gradient(cmap="RdYlGn", axis=None, vmin=0, vmax=1)
                .format("{:.3f}", subset=["precision", "recall", "f1-score"])
                .format("{:.0f}", subset=["support"]),
            use_container_width=True,
            height=240,
        )

        st.markdown("---")

        # ── Per-class metric comparison bars ─────────────────────────────────
        classes = ["0", "1"]
        metrics_data = {
            "Precision": [report[c]["precision"] for c in classes],
            "Recall":    [report[c]["recall"]    for c in classes],
            "F1-Score":  [report[c]["f1-score"]  for c in classes],
        }
        fig_bar = go.Figure()
        metric_colors = ["#3ECFA0", "#C8374A", "#D4A843"]
        for (metric, values), color in zip(metrics_data.items(), metric_colors):
            fig_bar.add_trace(go.Bar(
                name=metric,
                x=["No Disease (0)", "Disease (1)"],
                y=values,
                marker=dict(color=color, line=dict(width=0)),
                text=[f"{v:.3f}" for v in values],
                textposition="outside",
                textfont=dict(family="JetBrains Mono, monospace", size=12),
            ))
        fig_bar.update_layout(
            barmode="group",
            yaxis=dict(range=[0, 1.15], title="Score", tickformat=".0%"),
        )
        fig_bar = dark_fig(fig_bar, "Per-Class Metrics Breakdown")
        fig_bar.update_layout(height=380)
        st.plotly_chart(fig_bar, use_container_width=True)
        