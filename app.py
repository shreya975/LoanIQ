import time
import numpy as np
import plotly.graph_objects as go
import streamlit as st

# ==========================================================================================
# LOANIQ  —  Credit Turn-Down Risk Intelligence
# ------------------------------------------------------------------------------------------
# HONESTY NOTE (read before editing):
# Target variable is TURNFEAR from the Federal Reserve's 2019 Survey of Consumer Finances:
# a household is coded 1 if, in the past 5 years, it was DENIED credit (TURNDOWN=1) or wanted
# to apply for credit but didn't because it expected to be denied (FEARDENIAL=1). This is a
# genuine, real, binary label — so unlike a pure clustering demo, this app trains and reports
# a real supervised classifier against it, with real held-out accuracy/precision/recall/ROC-AUC.
#
# What this still is NOT: a live bank underwriting system. TURNFEAR reflects self-reported
# survey history/expectation, not an actual algorithmic lending decision, and the model was
# fit on public 2019 survey microdata, not any bank's proprietary applicant data. The
# "Methodology" page states this plainly.
#
# Model: Logistic Regression (class_weight="balanced", StandardScaler features), chosen over
# a higher-raw-accuracy Random Forest specifically because its coefficients let every
# prediction be explained by contributing factors — the same "adverse action reason" logic
# real lenders are required to provide. All numbers below (coefficients, intercept, scaler
# params, evaluation metrics) come directly from that fitted model on an 80/20 held-out split,
# random_state=42.
# ==========================================================================================

st.set_page_config(
    page_title="LoanIQ™ — Credit Turn-Down Risk Intelligence",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ------------------------------------------------------------------------------------------
# MODEL ARTIFACTS — fitted LogisticRegression(class_weight="balanced", random_state=42)
# Trained on an 80/20 stratified split of the 28,885-household SCF 2019 sample.
# ------------------------------------------------------------------------------------------
FEATURES = ["AGE", "MALE", "MARRIED", "KIDS", "EDCL", "EMPLOYED", "OWNS_HOME",
            "INCCAT", "NWCAT", "ASSETCAT", "DEBT2INC", "LEVRATIO", "CCBAL",
            "LATE", "LATE60", "BNKRUPLAST5"]

COEF = np.array([-0.1286, 0.0782, -0.0064, 0.2330, -0.2039, 0.0106, -0.3447,
                  -0.2149, -0.3336, -0.3468, 0.3006, -0.0016, 0.2867,
                  0.2833, 0.0514, 0.1898])
INTERCEPT = -0.62289838

SCALER_MEAN = np.array([53.2136, 0.7757, 0.6245, 0.7563, 3.0743, 0.7510, 0.6812,
                         3.6500, 3.0735, 3.6659, 1.0667, 0.6269, 2560.5806,
                         0.1100, 0.0411, 0.0178])
SCALER_SCALE = np.array([16.1580, 0.4171, 0.4842, 1.1302, 1.0086, 0.4324, 0.4660,
                          1.8049, 1.5080, 1.8568, 1.6886, 2.2385, 6514.7684,
                          0.3128, 0.1985, 0.1322])

# same clip caps applied to raw inputs before scaling, exactly as during training
CAPS = {"DEBT2INC": 10.0, "LEVRATIO": 18.911, "CCBAL": 41000.0}

# real, held-out evaluation metrics (20% test split, n=5,777)
METRICS = {
    "accuracy": 73.3, "precision": 35.4, "recall": 80.6, "f1": 49.2, "roc_auc": 82.8,
    "tn": 3491, "fp": 1361, "fn": 179, "tp": 746, "n_test": 5777, "base_rate": 16.0,
}
N_HOUSEHOLDS = 28885

EDCL_OPTIONS = [
    ("No high school diploma", 1), ("High school diploma / GED", 2),
    ("Some college", 3), ("College degree or higher", 4),
]
INCCAT_OPTIONS = [
    ("0th – 20th percentile", 1), ("20th – 39.9th percentile", 2),
    ("40th – 59.9th percentile", 3), ("60th – 79.9th percentile", 4),
    ("80th – 89.9th percentile", 5), ("90th – 100th percentile", 6),
]
NWCAT_OPTIONS = [
    ("Below 25th percentile", 1), ("25th – 49.9th percentile", 2),
    ("50th – 74.9th percentile", 3), ("75th – 89.9th percentile", 4),
    ("90th – 100th percentile", 5),
]
ASSETCAT_OPTIONS = [
    ("0th – 20th percentile", 1), ("20th – 39.9th percentile", 2),
    ("40th – 59.9th percentile", 3), ("60th – 79.9th percentile", 4),
    ("80th – 89.9th percentile", 5), ("90th – 100th percentile", 6),
]

FRIENDLY_NAMES = {
    "AGE": "Age", "MALE": "Male head of household", "MARRIED": "Married / partnered",
    "KIDS": "Number of dependents", "EDCL": "Education level", "EMPLOYED": "Currently employed",
    "OWNS_HOME": "Owns primary residence", "INCCAT": "Income percentile", "NWCAT": "Net worth percentile",
    "ASSETCAT": "Asset percentile", "DEBT2INC": "Debt-to-income ratio", "LEVRATIO": "Leverage ratio",
    "CCBAL": "Credit card balance", "LATE": "History of late payments",
    "LATE60": "History of 60+ day late payments", "BNKRUPLAST5": "Bankruptcy in last 5 years",
}


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def predict_risk(raw_values: dict):
    """Reproduces the fitted LogisticRegression exactly: clip -> StandardScaler -> sigmoid(w·z+b)."""
    x = np.array([raw_values[f] for f in FEATURES], dtype=float)
    for i, f in enumerate(FEATURES):
        if f in CAPS:
            x[i] = min(x[i], CAPS[f])
    z = (x - SCALER_MEAN) / SCALER_SCALE
    logit = np.dot(COEF, z) + INTERCEPT
    prob = float(sigmoid(logit))
    contributions = COEF * z  # per-feature contribution to the logit, real model math
    return prob, contributions, z


# ------------------------------------------------------------------------------------------
# SESSION STATE
# ------------------------------------------------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "landing"
if "result" not in st.session_state:
    st.session_state.result = None
if "inputs" not in st.session_state:
    st.session_state.inputs = None


def go_to(page):
    st.session_state.page = page


# ==========================================================================================
# STYLE
# ==========================================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

:root{
  --void:#04060B; --navy:#0A0F1E; --navy-2:#0E1526; --navy-3:#121A30;
  --glass: rgba(255,255,255,0.045); --glass-hi: rgba(255,255,255,0.08);
  --border: rgba(255,255,255,0.09); --border-hi: rgba(255,255,255,0.18);
  --royal:#3B6FE0; --royal-2:#6E8FFF; --emerald:#17C982; --gold:#D4AF37; --amber:#F2A93B; --red:#E5484D;
  --text-hi:#F4F6FB; --text-mid:#A9B3C9; --text-low:#6B7690;
  --radius: 18px; --shadow: 0 20px 60px -20px rgba(0,0,0,0.6);
}
html, body, [class*="css"]{ font-family:'Inter', sans-serif; }

#MainMenu {visibility:hidden;}
header[data-testid="stHeader"]{ background:transparent; height:0; }
div[data-testid="stToolbar"], div[data-testid="stDecoration"], div[data-testid="stStatusWidget"]{ display:none; }
footer {visibility:hidden; height:0;}
.block-container{ padding-top:0rem; padding-bottom:2rem; max-width:1180px; }
div[data-testid="stApp"]{ background:var(--void); }

.stApp{
  background:
    radial-gradient(1200px 700px at 12% -10%, rgba(59,111,224,0.20), transparent 60%),
    radial-gradient(1000px 650px at 88% 8%, rgba(23,201,130,0.14), transparent 55%),
    radial-gradient(900px 600px at 50% 100%, rgba(212,175,55,0.08), transparent 55%),
    linear-gradient(180deg, #04060B 0%, #060911 40%, #04060B 100%);
  background-attachment: fixed;
}
.aurora-orb{ position:fixed; border-radius:50%; filter:blur(90px); z-index:0; pointer-events:none; animation: floaty 18s ease-in-out infinite; }
@keyframes floaty{ 0%,100%{ transform:translate(0,0) scale(1);} 50%{ transform:translate(30px,-40px) scale(1.08);} }

h1,h2,h3, .disp{ font-family:'Space Grotesk', sans-serif; color:var(--text-hi); letter-spacing:-0.02em; }
.mono{ font-family:'JetBrains Mono', monospace; }
p, span, label, li { color:var(--text-mid); }

.glass{ background:var(--glass); border:1px solid var(--border); border-radius:var(--radius);
  backdrop-filter: blur(18px); -webkit-backdrop-filter: blur(18px); box-shadow: var(--shadow); }
.glass:hover{ border-color:var(--border-hi); }

.st-key-topnav{
  position:sticky; top:12px; z-index:999; background:rgba(8,11,20,0.65); border:1px solid var(--border);
  border-radius:100px; backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px);
  padding:8px 10px 8px 22px; margin: 14px auto 30px auto; max-width:1000px; box-shadow: var(--shadow);
}
.st-key-topnav .stButton > button{ background:transparent; border:none; color:var(--text-mid);
  font-family:'Space Grotesk', sans-serif; font-weight:500; font-size:0.86rem;
  padding:8px 16px; border-radius:100px; transition:all .2s ease; }
.st-key-topnav .stButton > button:hover{ color:var(--text-hi); background:var(--glass-hi); }
.navbrand{ font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:1.05rem; color:var(--text-hi);
  display:flex; align-items:center; gap:8px; }
.navbrand .dot{ color:var(--gold); }

.stButton > button{ border-radius:12px; font-family:'Space Grotesk', sans-serif; font-weight:600;
  transition: all .25s cubic-bezier(.2,.8,.2,1); }
.st-key-cta .stButton > button, .st-key-analyze_btn .stButton > button{
  background:linear-gradient(135deg, var(--royal) 0%, #5A4FD6 55%, var(--gold) 130%);
  color:white; border:none; padding:16px 30px; font-size:1.02rem; box-shadow: 0 10px 40px -8px rgba(59,111,224,0.55); }
.st-key-cta .stButton > button:hover, .st-key-analyze_btn .stButton > button:hover{
  transform: translateY(-2px) scale(1.015); box-shadow: 0 16px 50px -8px rgba(59,111,224,0.75); }
.st-key-secondary_btn .stButton > button, .st-key-secondary_btn2 .stButton > button{
  background:var(--glass); color:var(--text-hi); border:1px solid var(--border-hi); padding:14px 26px; }
.st-key-secondary_btn .stButton > button:hover, .st-key-secondary_btn2 .stButton > button:hover{ background:var(--glass-hi); }

.st-key-hero{ padding:70px 10px 30px 10px; text-align:center; position:relative; z-index:2; }
.eyebrow{ display:inline-flex; align-items:center; gap:8px; font-family:'JetBrains Mono',monospace;
  font-size:0.72rem; letter-spacing:0.14em; text-transform:uppercase; color:var(--gold);
  background:rgba(212,175,55,0.08); border:1px solid rgba(212,175,55,0.3); padding:7px 16px;
  border-radius:100px; margin-bottom:26px; }
.hero-title{ font-size:clamp(2.6rem, 6vw, 4.6rem); font-weight:700; line-height:1.03; margin:0 0 22px 0;
  background:linear-gradient(135deg, #FFFFFF 20%, var(--royal-2) 65%, var(--gold) 110%);
  -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent; }
.hero-sub{ font-size:1.12rem; max-width:660px; margin:0 auto 36px auto; line-height:1.65; color:var(--text-mid); }

.kpi-card{ padding:22px 20px; text-align:left; height:100%; }
.kpi-val{ font-family:'JetBrains Mono',monospace; font-size:1.9rem; font-weight:700; color:var(--text-hi); }
.kpi-lbl{ font-size:0.78rem; color:var(--text-low); text-transform:uppercase; letter-spacing:0.06em; margin-top:4px;}

.section-eyebrow{ font-family:'JetBrains Mono',monospace; font-size:0.72rem; letter-spacing:0.12em;
  text-transform:uppercase; color:var(--royal-2); margin-bottom:8px; }
.section-title{ font-size:1.7rem; font-weight:700; margin-bottom:6px; }
.section-sub{ color:var(--text-low); font-size:0.94rem; margin-bottom:24px; }

.form-card{ padding:26px 26px 10px 26px; margin-bottom:22px; }
.form-card .fc-head{ display:flex; align-items:center; gap:10px; margin-bottom:18px; }
.form-card .fc-icon{ font-size:1.3rem; }
.form-card .fc-title{ font-family:'Space Grotesk',sans-serif; font-weight:600; color:var(--text-hi); font-size:1.05rem;}
.form-card .fc-sub{ font-size:0.8rem; color:var(--text-low); margin:-12px 0 16px 0; }

div[data-testid="stSlider"] label, div[data-testid="stSelectbox"] label, div[data-testid="stRadio"] label,
div[data-testid="stNumberInput"] label, div[data-testid="stCheckbox"] label{
  color:var(--text-mid) !important; font-family:'Space Grotesk',sans-serif; font-size:0.9rem; font-weight:500; }
div[data-baseweb="select"] > div{ background:var(--navy-2) !important; border-color:var(--border) !important; border-radius:10px !important; }
div[data-testid="stSlider"] [data-baseweb="slider"] > div > div{ background:var(--royal) !important; }
div[data-testid="stNumberInput"] input{ background:var(--navy-2) !important; color:var(--text-hi) !important; border-radius:10px !important; }

.disclaimer{ border:1px solid rgba(242,169,59,0.35); background:rgba(242,169,59,0.06); border-radius:14px;
  padding:16px 20px; font-size:0.86rem; color:#E4C486; margin: 18px 0 30px 0; line-height:1.55; }
.disclaimer b{ color:var(--gold); }

.result-card{ padding:44px 40px; text-align:center; position:relative; overflow:hidden; }
.seg-badge{ display:inline-block; padding:8px 22px; border-radius:100px; font-family:'JetBrains Mono',monospace;
  font-size:0.78rem; letter-spacing:0.08em; text-transform:uppercase; margin-bottom:18px; border:1px solid;
  animation: pulseglow 2.4s ease-in-out infinite; }
@keyframes pulseglow{ 0%,100%{ box-shadow:0 0 0 0 rgba(255,255,255,0.0);} 50%{ box-shadow:0 0 26px 2px currentColor;} }
.seg-name{ font-size:clamp(2rem,4.4vw,3.1rem); font-weight:700; margin:6px 0 10px 0; }
.seg-tag{ color:var(--text-mid); font-size:1.02rem; margin-bottom:6px; }

.insight-card{ padding:20px 22px; height:100%; }
.insight-card .num{ font-family:'JetBrains Mono',monospace; color:var(--gold); font-size:0.78rem; margin-bottom:10px;}
.insight-card p{ color:var(--text-hi); font-size:0.92rem; line-height:1.55; margin:0; }

.load-step{ display:flex; align-items:center; gap:14px; padding:14px 6px; font-family:'JetBrains Mono',monospace;
  font-size:0.95rem; color:var(--text-low); transition:all .3s ease; }
.load-step.active{ color:var(--text-hi); }
.load-dot{ width:9px; height:9px; border-radius:50%; background:var(--border-hi); flex-shrink:0; }
.load-step.active .load-dot{ background:var(--emerald); box-shadow:0 0 12px 3px rgba(23,201,130,0.6); }

.footer{ text-align:center; padding:50px 10px 20px 10px; color:var(--text-low); font-size:0.8rem; }
hr.hair{ border:none; border-top:1px solid var(--border); margin:10px 0 30px 0; }
</style>

<div class="aurora-orb" style="width:420px;height:420px;top:-120px;left:-100px;background:radial-gradient(circle, rgba(59,111,224,0.35), transparent 70%);"></div>
<div class="aurora-orb" style="width:380px;height:380px;top:200px;right:-140px;background:radial-gradient(circle, rgba(23,201,130,0.28), transparent 70%); animation-delay:-6s;"></div>
<div class="aurora-orb" style="width:300px;height:300px;bottom:-100px;left:40%;background:radial-gradient(circle, rgba(212,175,55,0.22), transparent 70%); animation-delay:-11s;"></div>
""", unsafe_allow_html=True)


# ==========================================================================================
# SHARED COMPONENTS
# ==========================================================================================
def render_nav():
    with st.container(key="topnav"):
        c0, c1, c2, c3, c4 = st.columns([1.3, 1, 1, 1, 1])
        with c0:
            st.markdown('<div class="navbrand">◆ LoanIQ<span class="dot">™</span></div>', unsafe_allow_html=True)
        with c1:
            if st.button("Home", key="nav_home", use_container_width=True):
                go_to("landing")
        with c2:
            if st.button("Risk Analysis", key="nav_analyze", use_container_width=True):
                go_to("analyze")
        with c3:
            if st.button("Model Analytics", key="nav_analytics", use_container_width=True):
                go_to("analytics")
        with c4:
            if st.button("Methodology", key="nav_about", use_container_width=True):
                go_to("about")


def glass_open(extra_class=""):
    st.markdown(f'<div class="glass {extra_class}">', unsafe_allow_html=True)


def glass_close():
    st.markdown("</div>", unsafe_allow_html=True)


def kpi_card(value, label):
    return f"""<div class="glass kpi-card"><div class="kpi-val">{value}</div><div class="kpi-lbl">{label}</div></div>"""


def render_footer():
    st.markdown("""
    <div class="footer">
      LoanIQ is an educational data-science demo built on the public Federal Reserve
      <span class="mono">Survey of Consumer Finances (2019)</span> · Logistic regression model,
      evaluated on a held-out test split · Not a live lending or underwriting system.
    </div>
    """, unsafe_allow_html=True)


# ==========================================================================================
# PAGE — LANDING
# ==========================================================================================
def page_landing():
    render_nav()
    with st.container(key="hero"):
        st.markdown(f"""
        <div class="eyebrow">◆ Supervised Machine Learning · Federal Reserve SCF 2019</div>
        <div class="hero-title">LoanIQ™</div>
        <div class="hero-sub">Estimate a household's likelihood of being denied credit — or avoiding applying
        for fear of denial — using a logistic regression model trained and evaluated on {N_HOUSEHOLDS:,} real
        U.S. households from the Federal Reserve's Survey of Consumer Finances.</div>
        """, unsafe_allow_html=True)

        cta_col = st.columns([1, 0.6, 1])[1]
        with cta_col:
            with st.container(key="cta"):
                if st.button("Run Risk Analysis →", key="cta_btn", use_container_width=True):
                    go_to("analyze")

    st.markdown('<div style="height:10px;"></div>', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(kpi_card(f"{METRICS['roc_auc']:.1f}%", "ROC-AUC (held-out test)"), unsafe_allow_html=True)
    k2.markdown(kpi_card(f"{METRICS['recall']:.1f}%", "Recall on Turn-Down Cases"), unsafe_allow_html=True)
    k3.markdown(kpi_card(f"{METRICS['accuracy']:.1f}%", "Overall Accuracy"), unsafe_allow_html=True)
    k4.markdown(kpi_card(f"{METRICS['base_rate']:.1f}%", "Sample Turn-Down Rate"), unsafe_allow_html=True)

    st.markdown('<div style="height:50px;"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="section-eyebrow">What "turn-down risk" means here</div>
    <div class="section-title">A real, evaluated model — with real limits</div>
    <div class="section-sub" style="max-width:760px;">The model predicts <b>TURNFEAR</b>: whether a household
    reported being denied credit in the past 5 years, or wanted to apply but didn't because it expected to be
    turned down. That's a genuine label in the survey, so this is a real supervised classifier with real
    accuracy — not a fabricated "approval engine." It is still a research model on public survey data, not a
    bank's live underwriting system. See Methodology for the full disclosure.</div>
    """, unsafe_allow_html=True)
    render_footer()


# ==========================================================================================
# PAGE — ANALYZE (input form)
# ==========================================================================================
def page_analyze():
    render_nav()
    st.markdown("""
    <div class="section-eyebrow">Step 1 of 2</div>
    <div class="section-title">Build a household profile</div>
    <div class="section-sub">These 16 fields are exactly the features the model was trained on — nothing
    else affects the result.</div>
    """, unsafe_allow_html=True)

    with st.container(key="form1"):
        glass_open("form-card")
        st.markdown('<div class="fc-head"><span class="fc-icon">👤</span><span class="fc-title">Applicant Information</span></div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.slider("Age", 18, 95, 40)
            gender = st.selectbox("Gender of household head", ["Male", "Female"])
        with c2:
            married = st.selectbox("Marital status", ["Married / partnered", "Not married"])
            kids = st.slider("Number of dependents", 0, 7, 0)
        with c3:
            edcl_label = st.selectbox("Education level", [o[0] for o in EDCL_OPTIONS], index=2)
            employed = st.selectbox("Employment status", ["Employed", "Not currently employed"])
        glass_close()

    with st.container(key="form2"):
        glass_open("form-card")
        st.markdown('<div class="fc-head"><span class="fc-icon">💵</span><span class="fc-title">Financial Information</span></div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            inc_label = st.selectbox("Annual income percentile", [o[0] for o in INCCAT_OPTIONS], index=2)
            debt2inc = st.number_input("Debt-to-income ratio", 0.0, 25.0, 1.1, step=0.1,
                                        help="Total debt ÷ annual income. Capped at 10.0 to match training data.")
            ccbal = st.number_input("Credit card balance carried ($)", 0, 60000, 2500, step=100)
        with c2:
            late = st.selectbox("Any late payments in the past year?", ["No", "Yes"])
            late60 = st.selectbox("Any payments 60+ days late?", ["No", "Yes"])
            bankrupt = st.selectbox("Bankruptcy filed in last 5 years?", ["No", "Yes"])
        glass_close()

    with st.container(key="form3"):
        glass_open("form-card")
        st.markdown('<div class="fc-head"><span class="fc-icon">🏠</span><span class="fc-title">Property &amp; Assets</span></div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            owns_home = st.selectbox("Residence type", ["Owns primary residence", "Rents / other"])
        with c2:
            nw_label = st.selectbox("Net worth percentile", [o[0] for o in NWCAT_OPTIONS], index=2)
        with c3:
            asset_label = st.selectbox("Total assets percentile", [o[0] for o in ASSETCAT_OPTIONS], index=2)
        levratio = st.slider("Leverage ratio (total debt ÷ total assets)", 0.0, 20.0, 0.6, step=0.1)
        glass_close()

    st.markdown("""
    <div class="disclaimer"><b>What this does not do:</b> LoanIQ does not access any real credit bureau,
    bank, or applicant data. It estimates statistical similarity to survey households who reported being
    denied credit or avoiding applying out of fear of denial — it is not a live lending decision.</div>
    """, unsafe_allow_html=True)

    center = st.columns([1, 0.7, 1])[1]
    with center:
        with st.container(key="analyze_btn"):
            run = st.button("Run Risk Analysis →", key="run_btn", use_container_width=True)

    if run:
        raw = {
            "AGE": age,
            "MALE": 1 if gender == "Male" else 0,
            "MARRIED": 1 if married == "Married / partnered" else 0,
            "KIDS": kids,
            "EDCL": dict(EDCL_OPTIONS)[edcl_label],
            "EMPLOYED": 1 if employed == "Employed" else 0,
            "OWNS_HOME": 1 if owns_home == "Owns primary residence" else 0,
            "INCCAT": dict(INCCAT_OPTIONS)[inc_label],
            "NWCAT": dict(NWCAT_OPTIONS)[nw_label],
            "ASSETCAT": dict(ASSETCAT_OPTIONS)[asset_label],
            "DEBT2INC": debt2inc,
            "LEVRATIO": levratio,
            "CCBAL": ccbal,
            "LATE": 1 if late == "Yes" else 0,
            "LATE60": 1 if late60 == "Yes" else 0,
            "BNKRUPLAST5": 1 if bankrupt == "Yes" else 0,
        }

        steps = [
            "Reading application…",
            "Standardizing inputs (z-score vs. 28,885-household sample)…",
            "Scoring logistic regression model…",
            "Computing feature contributions…",
            "Building risk report…",
        ]
        ph = st.empty()
        for i in range(len(steps)):
            html = '<div class="glass" style="padding:26px 30px;">'
            for j, s2 in enumerate(steps):
                cls = "active" if j <= i else ""
                mark = "●" if j <= i else "○"
                html += f'<div class="load-step {cls}"><span class="load-dot"></span>{mark} {s2}</div>'
            html += "</div>"
            ph.markdown(html, unsafe_allow_html=True)
            time.sleep(0.4)
        ph.empty()

        prob, contributions, z = predict_risk(raw)
        st.session_state.inputs = raw
        st.session_state.result = dict(prob=prob, contributions=contributions, z=z)
        go_to("result")
        st.rerun()

    render_footer()


# ==========================================================================================
# PAGE — RESULT
# ==========================================================================================
def page_result():
    render_nav()
    if st.session_state.result is None:
        st.info("Run a risk analysis first.")
        if st.button("Go to Risk Analysis"):
            go_to("analyze")
        return

    res = st.session_state.result
    prob = res["prob"]
    contributions = res["contributions"]
    pct = prob * 100
    elevated = prob >= 0.5
    label = "Elevated Turn-Down Risk" if elevated else "Low Turn-Down Risk"
    color = "#E5484D" if elevated else "#17C982"

    with st.container(key="resultcard"):
        glass_open("result-card")
        st.markdown(f"""
        <div class="seg-badge" style="border-color:{color}; color:{color};">Model Prediction</div>
        <div class="seg-name" style="color:{color};">{label}</div>
        <div class="seg-tag">Estimated probability of credit turn-down: <b style="color:{color};">{pct:.1f}%</b></div>
        <div style="color:var(--text-low); font-size:0.85rem; margin-top:8px;">
          Sample-wide turn-down rate is {METRICS['base_rate']:.1f}%, for comparison.
        </div>
        """, unsafe_allow_html=True)
        glass_close()

    st.markdown('<div style="height:26px;"></div>', unsafe_allow_html=True)

    g1, g2 = st.columns(2)
    with g1:
        with st.container(key="gaugebox"):
            glass_open("form-card")
            st.markdown('<div class="fc-title">Turn-Down Risk Gauge</div>', unsafe_allow_html=True)
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=pct,
                number={"suffix": "%", "font": {"color": "#F4F6FB", "family": "JetBrains Mono"}},
                gauge={"axis": {"range": [0, 100], "tickcolor": "#6B7690"},
                       "bar": {"color": color}, "bgcolor": "rgba(255,255,255,0.03)", "borderwidth": 0,
                       "threshold": {"line": {"color": "#F4F6FB", "width": 2}, "value": METRICS["base_rate"]},
                       "steps": [{"range": [0, 30], "color": "rgba(23,201,130,0.12)"},
                                 {"range": [30, 60], "color": "rgba(242,169,59,0.12)"},
                                 {"range": [60, 100], "color": "rgba(229,72,77,0.12)"}]},
            ))
            fig.update_layout(height=260, margin=dict(t=10, b=10, l=20, r=20),
                               paper_bgcolor="rgba(0,0,0,0)", font_color="#A9B3C9")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            st.caption("White line marks the 16.0% sample-wide turn-down rate. Note: because the model "
                       "is trained with balanced class weights to prioritize recall, this percentage is a "
                       "relative risk score, not a calibrated real-world probability — see Methodology.")
            glass_close()

    with g2:
        with st.container(key="contribbox"):
            glass_open("form-card")
            st.markdown('<div class="fc-title">Top Contributing Factors</div>', unsafe_allow_html=True)
            order = np.argsort(-np.abs(contributions))[:6]
            names = [FRIENDLY_NAMES[FEATURES[i]] for i in order]
            vals = [contributions[i] for i in order]
            colors = ["#E5484D" if v > 0 else "#17C982" for v in vals]
            fig2 = go.Figure(go.Bar(
                x=vals, y=names, orientation="h", marker=dict(color=colors),
                text=[f"{v:+.2f}" for v in vals], textposition="outside",
                textfont=dict(color="#F4F6FB", family="JetBrains Mono", size=10),
            ))
            fig2.update_layout(height=260, margin=dict(t=10, b=10, l=10, r=40),
                                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                font_color="#A9B3C9", xaxis=dict(visible=False),
                                yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
            st.caption("Red pushes risk up, green pulls it down — same logic used for real adverse-action reason codes.")
            glass_close()

    with st.container(key="allcontrib"):
        glass_open("form-card")
        st.markdown('<div class="fc-title">Full Feature Contribution Breakdown</div>', unsafe_allow_html=True)
        order_all = np.argsort(contributions)
        names_all = [FRIENDLY_NAMES[FEATURES[i]] for i in order_all]
        vals_all = [contributions[i] for i in order_all]
        colors_all = ["#E5484D" if v > 0 else "#17C982" for v in vals_all]
        fig3 = go.Figure(go.Bar(x=vals_all, y=names_all, orientation="h", marker=dict(color=colors_all)))
        fig3.update_layout(height=420, margin=dict(t=10, b=10, l=10, r=20),
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            font_color="#A9B3C9",
                            xaxis=dict(title="Contribution to log-odds", gridcolor="rgba(255,255,255,0.06)",
                                       zerolinecolor="rgba(255,255,255,0.3)"))
        st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})
        glass_close()

    st.markdown("""
    <div class="disclaimer" style="margin-top:10px;"><b>Reminder:</b> this estimates statistical
    similarity to survey households who reported credit denial or fear of denial. It is not a credit
    score, a real loan decision, or financial advice.</div>
    """, unsafe_allow_html=True)

    b1, b2 = st.columns(2)
    with b1:
        with st.container(key="secondary_btn"):
            if st.button("← Analyze Another Profile", key="another_btn", use_container_width=True):
                go_to("analyze")
    with b2:
        with st.container(key="secondary_btn2"):
            if st.button("View Model Analytics →", key="pop_btn", use_container_width=True):
                go_to("analytics")

    render_footer()


# ==========================================================================================
# PAGE — MODEL ANALYTICS
# ==========================================================================================
def page_analytics():
    render_nav()
    st.markdown(f"""
    <div class="section-eyebrow">Held-Out Test Set · {METRICS['n_test']:,} Households</div>
    <div class="section-title">Model Analytics</div>
    <div class="section-sub">Real evaluation results from an 80/20 stratified train/test split
    (random_state=42) — not illustrative numbers.</div>
    """, unsafe_allow_html=True)

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.markdown(kpi_card(f"{METRICS['accuracy']:.1f}%", "Accuracy"), unsafe_allow_html=True)
    k2.markdown(kpi_card(f"{METRICS['precision']:.1f}%", "Precision"), unsafe_allow_html=True)
    k3.markdown(kpi_card(f"{METRICS['recall']:.1f}%", "Recall"), unsafe_allow_html=True)
    k4.markdown(kpi_card(f"{METRICS['f1']:.1f}%", "F1 Score"), unsafe_allow_html=True)
    k5.markdown(kpi_card(f"{METRICS['roc_auc']:.1f}%", "ROC-AUC"), unsafe_allow_html=True)

    st.markdown('<div style="height:20px;"></div>', unsafe_allow_html=True)
    a1, a2 = st.columns(2)
    with a1:
        with st.container(key="cmbox"):
            glass_open("form-card")
            st.markdown('<div class="fc-title">Confusion Matrix</div>', unsafe_allow_html=True)
            z = [[METRICS["tn"], METRICS["fp"]], [METRICS["fn"], METRICS["tp"]]]
            fig = go.Figure(go.Heatmap(
                z=z, x=["Predicted: No Turn-Down", "Predicted: Turn-Down"],
                y=["Actual: No Turn-Down", "Actual: Turn-Down"],
                text=z, texttemplate="%{text}", textfont=dict(size=16, color="#04060B", family="JetBrains Mono"),
                colorscale=[[0, "#0E1526"], [1, "#3B6FE0"]], showscale=False,
            ))
            fig.update_layout(height=320, margin=dict(t=10, b=10, l=10, r=10),
                               paper_bgcolor="rgba(0,0,0,0)", font_color="#A9B3C9")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            st.caption("Class-balanced training deliberately favors catching real turn-down cases (recall 80.6%) over minimizing false alarms — precision is lower as a direct trade-off.")
            glass_close()

    with a2:
        with st.container(key="featimp"):
            glass_open("form-card")
            st.markdown('<div class="fc-title">Model Coefficients (Standardized)</div>', unsafe_allow_html=True)
            order = np.argsort(np.abs(COEF))
            names = [FRIENDLY_NAMES[f] for f in np.array(FEATURES)[order]]
            vals = COEF[order]
            colors = ["#E5484D" if v > 0 else "#17C982" for v in vals]
            fig2 = go.Figure(go.Bar(x=vals, y=names, orientation="h", marker=dict(color=colors)))
            fig2.update_layout(height=420, margin=dict(t=10, b=10, l=10, r=20),
                                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                font_color="#A9B3C9",
                                xaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.3)"))
            st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
            st.caption("Positive = associated with higher turn-down risk. Owning a home, higher net worth/asset/income percentiles, and more education are the strongest protective factors.")
            glass_close()

    render_footer()


# ==========================================================================================
# PAGE — METHODOLOGY / ABOUT
# ==========================================================================================
def page_about():
    render_nav()
    st.markdown("""
    <div class="section-eyebrow">Full Disclosure</div>
    <div class="section-title">Methodology &amp; What This Tool Is (and Isn't)</div>
    """, unsafe_allow_html=True)

    with st.container(key="about1"):
        glass_open("form-card")
        st.markdown(f"""
        <div class="fc-title">✅ What LoanIQ is</div>
        <ul style="margin-top:10px; line-height:1.9;">
          <li>A <b>logistic regression</b> classifier (<span class="mono">class_weight="balanced"</span>,
          <span class="mono">random_state=42</span>) predicting <span class="mono">TURNFEAR</span> from the
          Federal Reserve's 2019 Survey of Consumer Finances ({N_HOUSEHOLDS:,} households).</li>
          <li><span class="mono">TURNFEAR</span> = 1 if a household was denied credit in the past 5 years, or
          wanted to apply but didn't because it expected denial — a genuine, real label in the data.</li>
          <li>16 features spanning demographics, income/wealth percentiles, debt ratios, and payment/bankruptcy
          history — standardized with <span class="mono">StandardScaler</span> before fitting.</li>
          <li>Evaluated honestly on a held-out 20% test split: {METRICS['accuracy']:.1f}% accuracy,
          {METRICS['recall']:.1f}% recall on turn-down cases, {METRICS['roc_auc']:.1f}% ROC-AUC.</li>
          <li>Logistic regression was chosen over a higher-raw-accuracy Random Forest (which reached ~79.5%
          accuracy / 91.0% ROC-AUC) specifically so every prediction can be explained by transparent,
          signed feature contributions — mirroring real adverse-action requirements in lending.</li>
        </ul>
        """, unsafe_allow_html=True)
        glass_close()

    with st.container(key="about2"):
        glass_open("form-card")
        st.markdown("""
        <div class="fc-title">🚫 What LoanIQ is not</div>
        <ul style="margin-top:10px; line-height:1.9;">
          <li>It is <b>not connected to any bank, credit bureau, or live lending system</b>. It never sees
          real applicant or credit-report data.</li>
          <li><span class="mono">TURNFEAR</span> mixes actual denials with self-reported <i>fear</i> of denial
          (households that never applied) — so a high score reflects statistical resemblance to that mixed
          group, not a guaranteed real-world credit denial.</li>
          <li>Because the positive class is only 16% of the sample, the model was tuned to prioritize recall,
          which means <b>most positive predictions are false positives</b> (precision 35.4%) — a known,
          disclosed trade-off, not a hidden flaw.</li>
          <li>That same balanced-class training means the displayed percentage is a <b>relative risk score,
          not a calibrated probability</b> — an average household scores roughly 35%, well above the true
          16% population base rate. Compare scores to each other and to the base-rate line, not as literal
          real-world odds.</li>
          <li>It is a demonstration built on 2019 public survey microdata — real underwriting uses proprietary
          data, credit bureau reports, and is subject to fair-lending regulation this project does not
          implement.</li>
        </ul>
        """, unsafe_allow_html=True)
        glass_close()

    render_footer()


# ==========================================================================================
# ROUTER
# ==========================================================================================
PAGES = {
    "landing": page_landing, "analyze": page_analyze, "result": page_result,
    "analytics": page_analytics, "about": page_about,
}
PAGES.get(st.session_state.page, page_landing)()