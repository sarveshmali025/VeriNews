"""
app.py
======
Owner : Member 1
Job   : Streamlit UI — layout, styling, section rendering.
        Imports everything from model/ and utils/.
        Contains ZERO prediction logic — only display code.

UI Order (after Analyze is clicked):
  1. Input  (text area + file upload)
  2. VERDICT  — large centred FAKE/REAL panel
  3. Model Comparison  — ML card vs Manual card
  4. Word Analysis  — highlighted text
  5. Explanation  — why this verdict?
  6. Math Walkthrough  — TF-IDF then Naive Bayes step by step
  7. Confusion Matrix + Metrics
  [Always visible at bottom]
  8. TF-IDF Formula Reference
  9. Naive Bayes Formula Reference
"""

import io
import sys
import os

# ── Allow imports from project root ──────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from model.ml_model     import load_model
from model.manual_model import predict_manual, highlight_text
from utils.tfidf_manual import get_tfidf_walkthrough
from utils.nb_manual    import get_nb_steps
from utils.explain      import build_ml_explanation, build_manual_explanation
from utils.gauge        import render_gauge
from utils.verify       import check_agreement, render_verify_panel

# ══════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title = "VeriNews",
    page_icon  = "🔍",
    layout     = "wide",
)

# ══════════════════════════════════════════════════════════════
#  CSS  — dark theme, Space Mono + Syne, purple accent #7c6af7
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

/* ── Base ────────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family   : 'Syne', sans-serif;
    background    : #0b0b12;
    color         : #e4e4f0;
}
#MainMenu, footer, header { visibility: hidden; }

/* ── Hero ────────────────────────────────────────────────── */
.hero {
    text-align  : center;
    padding     : 3.5rem 1rem 2.5rem;
    background  : radial-gradient(ellipse at 50% -10%,
                    rgba(124,106,247,.14) 0%, transparent 65%);
}
.hero h1 {
    font-size      : 3.8rem;
    font-weight    : 800;
    color          : #ffffff;
    letter-spacing : -2px;
    line-height    : 1.1;
    margin-bottom  : 0.5rem;
}
.hero h1 .ac { color: #7c6af7; }
.hero .sub {
    font-family : 'Space Mono', monospace;
    font-size   : 0.82rem;
    color       : #505080;
    margin-top  : 0.4rem;
}
.hero .tags {
    margin-top : 1.2rem;
    display    : flex;
    gap        : 0.5rem;
    justify-content : center;
    flex-wrap  : wrap;
}
.tag {
    background    : #13131f;
    border        : 1px solid #252540;
    border-radius : 20px;
    padding       : 0.22rem 0.85rem;
    font-family   : 'Space Mono', monospace;
    font-size     : 0.65rem;
    color         : #7c6af7;
}

/* ── Section heading ─────────────────────────────────────── */
.sec {
    display     : flex;
    align-items : center;
    gap         : 0.7rem;
    margin      : 0.2rem 0 1.3rem;
}
.sec .ic {
    width           : 2.1rem;
    height          : 2.1rem;
    border-radius   : 9px;
    background      : #13131f;
    border          : 1px solid #252540;
    display         : flex;
    align-items     : center;
    justify-content : center;
    font-size       : 1rem;
    flex-shrink     : 0;
}
.sec .lbl {
    font-family    : 'Syne', sans-serif;
    font-size      : 1.15rem;
    font-weight    : 800;
    color          : #ffffff;
    letter-spacing : -0.3px;
}
.sec .sub {
    font-family    : 'Space Mono', monospace;
    font-size      : 0.62rem;
    color          : #404065;
    text-transform : uppercase;
    letter-spacing : 1.5px;
    margin-left    : 0.15rem;
}

/* ── Input ───────────────────────────────────────────────── */
.stTextArea textarea {
    background    : #12121e !important;
    border        : 1.5px solid #252540 !important;
    border-radius : 14px !important;
    color         : #e4e4f0 !important;
    font-family   : 'Space Mono', monospace !important;
    font-size     : 0.87rem !important;
    padding       : 1.1rem !important;
    line-height   : 1.8 !important;
}
.stTextArea textarea:focus {
    border-color : #7c6af7 !important;
    box-shadow   : 0 0 0 3px rgba(124,106,247,.14) !important;
}
.stButton > button {
    background    : linear-gradient(135deg, #7c6af7 0%, #5548cc 100%) !important;
    color         : #fff !important;
    border        : none !important;
    border-radius : 12px !important;
    font-family   : 'Syne', sans-serif !important;
    font-weight   : 800 !important;
    font-size     : 1.05rem !important;
    padding       : 0.75rem 2rem !important;
    width         : 100%;
    box-shadow    : 0 4px 22px rgba(124,106,247,.32) !important;
    letter-spacing: 0.3px;
}
.stButton > button:hover {
    transform  : translateY(-2px) !important;
    box-shadow : 0 8px 30px rgba(124,106,247,.5) !important;
}

/* ── Main Verdict panel ──────────────────────────────────── */
.verdict {
    border-radius : 20px;
    padding       : 2.8rem 2rem 2.2rem;
    text-align    : center;
    position      : relative;
    overflow      : hidden;
    margin-bottom : 0.3rem;
}
.verdict.fake {
    background : linear-gradient(140deg, #1c0808 0%, #12121e 55%);
    border     : 1.5px solid rgba(247,74,74,.38);
    box-shadow : 0 0 70px rgba(247,74,74,.07);
}
.verdict.real {
    background : linear-gradient(140deg, #071812 0%, #12121e 55%);
    border     : 1.5px solid rgba(74,247,160,.30);
    box-shadow : 0 0 70px rgba(74,247,160,.05);
}
.verdict.uncertain {
    background : linear-gradient(140deg, #18140a 0%, #12121e 55%);
    border     : 1.5px solid rgba(247,200,74,.28);
    box-shadow : 0 0 70px rgba(247,200,74,.05);
}
.verdict .v-eye {
    font-family    : 'Space Mono', monospace;
    font-size      : 0.65rem;
    text-transform : uppercase;
    letter-spacing : 3px;
    color          : #404068;
    margin-bottom  : 0.9rem;
}
.verdict .v-label {
    font-family    : 'Syne', sans-serif;
    font-weight    : 800;
    font-size      : 5rem;
    line-height    : 1;
    letter-spacing : -3px;
    margin-bottom  : 0.5rem;
}
.verdict.fake .v-label {
    color       : #f74a4a;
    text-shadow : 0 0 50px rgba(247,74,74,.45);
}
.verdict.real .v-label {
    color       : #4af7a0;
    text-shadow : 0 0 50px rgba(74,247,160,.40);
}
.verdict.uncertain .v-label {
    color       : #f7c84a;
    text-shadow : 0 0 50px rgba(247,200,74,.40);
}
.verdict .v-conf {
    font-family : 'Space Mono', monospace;
    font-size   : 1rem;
    color       : #8080b0;
    margin-bottom: 1.5rem;
}
.verdict .v-bar-wrap {
    max-width     : 300px;
    margin        : 0 auto 0.5rem;
    background    : #1c1c30;
    border-radius : 8px;
    height        : 10px;
    overflow      : hidden;
}
.verdict.fake .v-bar-fill      { background: linear-gradient(90deg,#b02020,#f74a4a); height:10px; border-radius:8px; }
.verdict.real .v-bar-fill      { background: linear-gradient(90deg,#20b060,#4af7a0); height:10px; border-radius:8px; }
.verdict.uncertain .v-bar-fill { background: linear-gradient(90deg,#b09020,#f7c84a); height:10px; border-radius:8px; }
.verdict .v-note {
    font-family : 'Space Mono', monospace;
    font-size   : 0.68rem;
    color       : #303055;
    margin-top  : 0.4rem;
}

/* ── Model comparison cards ──────────────────────────────── */
.mcard {
    background    : #12121e;
    border        : 1px solid #202038;
    border-radius : 16px;
    padding       : 1.5rem 1.6rem;
    position      : relative;
    overflow      : hidden;
    height        : 100%;
}
.mcard::after {
    content       : '';
    position      : absolute;
    top           : 0; left : 0;
    width         : 100%; height: 3px;
    border-radius : 16px 16px 0 0;
}
.mcard.fake::after      { background: linear-gradient(90deg,#f74a4a,#b02020); }
.mcard.real::after      { background: linear-gradient(90deg,#4af7a0,#20b060); }
.mcard.uncertain::after { background: linear-gradient(90deg,#f7c84a,#b09020); }

.mcard .mc-eye {
    font-family    : 'Space Mono', monospace;
    font-size      : 0.60rem;
    text-transform : uppercase;
    letter-spacing : 2px;
    color          : #404065;
    margin-bottom  : 0.8rem;
}
.mcard .mc-verdict {
    font-family    : 'Syne', sans-serif;
    font-size      : 2.3rem;
    font-weight    : 800;
    line-height    : 1;
    margin-bottom  : 0.35rem;
}
.mcard.fake .mc-verdict      { color: #f74a4a; }
.mcard.real .mc-verdict      { color: #4af7a0; }
.mcard.uncertain .mc-verdict { color: #f7c84a; }
.mcard .mc-conf {
    font-family   : 'Space Mono', monospace;
    font-size     : 0.78rem;
    color         : #606090;
    margin-bottom : 1rem;
}
.mcard .mc-bar-bg { background:#1c1c2e; border-radius:5px; height:6px; overflow:hidden; margin-bottom:0.35rem; }
.mcard.fake .mc-bar-fill      { background:#f74a4a; height:6px; border-radius:5px; }
.mcard.real .mc-bar-fill      { background:#4af7a0; height:6px; border-radius:5px; }
.mcard.uncertain .mc-bar-fill { background:#f7c84a; height:6px; border-radius:5px; }
.mcard .mc-note { font-family:'Space Mono',monospace; font-size:0.62rem; color:#30304a; }

/* ── Highlighted text ────────────────────────────────────── */
.hi-box {
    background    : #12121e;
    border        : 1px solid #202038;
    border-radius : 14px;
    padding       : 1.4rem 1.6rem;
    font-family   : 'Space Mono', monospace;
    font-size     : 0.86rem;
    color         : #c8c8e0;
    line-height   : 2.1;
}
.fake-word {
    background    : rgba(247,74,74,.14);
    color         : #f79595;
    border-radius : 5px;
    padding       : 2px 6px;
    border-bottom : 1.5px solid rgba(247,74,74,.55);
    cursor        : default;
    font-weight   : 700;
}
.real-word {
    background    : rgba(74,247,160,.11);
    color         : #7af7b5;
    border-radius : 5px;
    padding       : 2px 6px;
    border-bottom : 1.5px solid rgba(74,247,160,.45);
    cursor        : default;
    font-weight   : 700;
}
.leg-row { display:flex; gap:1rem; margin-bottom:1rem; flex-wrap:wrap; }
.leg { display:inline-flex; align-items:center; gap:0.4rem; border-radius:20px; padding:0.25rem 0.9rem; font-family:'Space Mono',monospace; font-size:0.67rem; }
.leg-f { background:rgba(247,74,74,.11);  color:#f79595; border:1px solid rgba(247,74,74,.22); }
.leg-r { background:rgba(74,247,160,.09); color:#7af7b5; border:1px solid rgba(74,247,160,.18); }

/* ── Why / Explanation cards ─────────────────────────────── */
.why {
    background    : #12121e;
    border        : 1px solid #202038;
    border-radius : 14px;
    padding       : 1.4rem 1.6rem;
    margin-bottom : 0.7rem;
}
.why .w-title {
    font-family   : 'Syne', sans-serif;
    font-size     : 0.92rem;
    font-weight   : 800;
    color         : #9080f7;
    margin-bottom : 0.7rem;
}
.why .w-body {
    font-family : 'Space Mono', monospace;
    font-size   : 0.80rem;
    color       : #8080a8;
    line-height : 1.85;
}
.why .w-body span { color: #e4e4f0; font-weight: 700; }

/* ── TF-IDF calc boxes ───────────────────────────────────── */
.cbox {
    background    : #12121e;
    border        : 1px solid #202038;
    border-left   : 3px solid #7c6af7;
    border-radius : 12px;
    padding       : 1.15rem 1.5rem;
    margin-bottom : 0.6rem;
}
.cbox .cw {
    font-family   : 'Syne', sans-serif;
    font-size     : 1rem;
    font-weight   : 800;
    color         : #9080f7;
    margin-bottom : 0.5rem;
}
.cbox .cl {
    font-family : 'Space Mono', monospace;
    font-size   : 0.80rem;
    color       : #606090;
    margin      : 0.2rem 0;
}
.cbox .cl span { color: #e4e4f0; font-weight: 700; }

/* ── NB step boxes ───────────────────────────────────────── */
.nbbox {
    background    : #12121e;
    border        : 1px solid #202038;
    border-left   : 3px solid #5548cc;
    border-radius : 12px;
    padding       : 1.2rem 1.6rem;
    margin-bottom : 0.6rem;
}
.nbbox .nb-t {
    font-family   : 'Syne', sans-serif;
    font-size     : 0.9rem;
    font-weight   : 800;
    color         : #7c6af7;
    margin-bottom : 0.6rem;
}
.nbbox .nb-l {
    font-family : 'Space Mono', monospace;
    font-size   : 0.80rem;
    color       : #606090;
    margin      : 0.28rem 0;
    line-height : 1.6;
}
.nbbox .nb-l span   { color: #e4e4f0; font-weight: 700; }
.nbbox .nb-l.hl     { color:#b0b0d0; border-top:1px solid #1c1c30; padding-top:0.6rem; margin-top:0.5rem; }

/* ── Formula reference boxes (always visible) ────────────── */
.fbox {
    background    : #12121e;
    border        : 1px solid #202038;
    border-radius : 14px;
    padding       : 1.4rem 1.6rem;
    height        : 100%;
}
.fbox .ft {
    font-family   : 'Syne', sans-serif;
    font-size     : 1rem;
    font-weight   : 800;
    color         : #7c6af7;
    margin-bottom : 0.8rem;
}
.fbox .feq {
    font-family   : 'Space Mono', monospace;
    font-size     : 0.82rem;
    color         : #d0d0f0;
    background    : #0b0b12;
    border        : 1px solid #252540;
    border-radius : 8px;
    padding       : 0.6rem 1rem;
    margin-bottom : 0.8rem;
    font-weight   : 700;
    letter-spacing: 0.3px;
}
.fbox .fn {
    font-family : 'Space Mono', monospace;
    font-size   : 0.72rem;
    color       : #505078;
    line-height : 1.75;
}
.fbox .fn span { color: #9090b8; }

/* ── Metric table ────────────────────────────────────────── */
.met {
    background    : #12121e;
    border        : 1px solid #202038;
    border-radius : 12px;
    padding       : 1.2rem 1.5rem;
    margin-bottom : 0.6rem;
}
.met .mt { font-family:'Syne',sans-serif; font-size:.88rem; font-weight:800; color:#7c6af7; margin-bottom:.8rem; }
.met .mr { display:flex; justify-content:space-between; align-items:center; padding:.4rem 0; border-bottom:1px solid #181828; font-family:'Space Mono',monospace; font-size:.78rem; }
.met .mr:last-child { border-bottom:none; }
.met .mr .k { color:#505078; }
.met .mr .v { color:#e4e4f0; font-weight:700; }

/* ── Part label ──────────────────────────────────────────── */
.part-lbl {
    font-family    : 'Syne', sans-serif;
    font-size      : 0.9rem;
    font-weight    : 800;
    color          : #9080f7;
    margin-bottom  : 0.9rem;
    margin-top     : 0.2rem;
    padding-left   : 0.3rem;
    border-left    : 3px solid #7c6af7;
    padding-left   : 0.7rem;
}

/* ── Empty placeholder ───────────────────────────────────── */
.empty {
    height          : 200px;
    background      : #0e0e1a;
    border          : 1.5px dashed #1e1e36;
    border-radius   : 16px;
    display         : flex;
    align-items     : center;
    justify-content : center;
    flex-direction  : column;
    color           : #252545;
    font-family     : 'Space Mono', monospace;
    font-size       : 0.78rem;
    text-align      : center;
    gap             : 0.7rem;
}

hr { border-color: #181828 !important; margin: 2rem 0 !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  LOAD MODEL  (trains once, cached for the session)
# ══════════════════════════════════════════════════════════════
@st.cache_resource
def init():
    return load_model()

model, vectorizer, data, conf_matrix, accuracy = init()
N = len(data)   # total documents in training corpus


# ══════════════════════════════════════════════════════════════
#  SMALL RENDER HELPERS  (UI only — no logic)
# ══════════════════════════════════════════════════════════════

def section(icon: str, label: str, sublabel: str = ""):
    """Render a styled section heading."""
    sub = f'<span class="sub">— {sublabel}</span>' if sublabel else ""
    st.markdown(
        f'<div class="sec"><div class="ic">{icon}</div>'
        f'<span class="lbl">{label}</span>{sub}</div>',
        unsafe_allow_html=True,
    )


def render_verdict(prediction: str, confidence: float):
    """Large centred verdict panel — most prominent element on page."""
    cls  = prediction.lower()
    icon = "🔴" if prediction == "FAKE" else ("🟢" if prediction == "REAL" else "🟡")
    bar  = int(confidence)
    st.markdown(f"""
    <div class="verdict {cls}">
        <div class="v-eye">ML Model Verdict — Multinomial Naive Bayes</div>
        <div class="v-label">{icon}&nbsp;{prediction}</div>
        <div class="v-conf">Confidence &nbsp;·&nbsp; {confidence}%</div>
        <div class="v-bar-wrap">
            <div class="v-bar-fill" style="width:{bar}%"></div>
        </div>
        <div class="v-note">{bar}% posterior probability · based on TF-IDF + Naive Bayes</div>
    </div>""", unsafe_allow_html=True)


def render_model_card(label: str, prediction: str, confidence: float):
    """Compact comparison card for ML or Manual model."""
    cls  = prediction.lower()
    icon = "🔴" if prediction == "FAKE" else ("🟢" if prediction == "REAL" else "🟡")
    bar  = int(confidence)
    st.markdown(f"""
    <div class="mcard {cls}">
        <div class="mc-eye">{label}</div>
        <div class="mc-verdict">{icon} {prediction}</div>
        <div class="mc-conf">Confidence: {confidence}%</div>
        <div class="mc-bar-bg">
            <div class="mc-bar-fill" style="width:{bar}%"></div>
        </div>
        <div class="mc-note">{bar}% certainty score</div>
    </div>""", unsafe_allow_html=True)


def render_confusion_matrix(cm):
    """Generate confusion matrix as a dark-themed matplotlib PNG buffer."""
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor("#0e0e1a")
    ax.set_facecolor("#0e0e1a")
    ax.imshow(cm, cmap="RdPu", aspect="auto", vmin=0)

    labels = ["FAKE", "REAL"]
    ax.set_xticks([0, 1]);  ax.set_yticks([0, 1])
    ax.set_xticklabels(labels, color="#c4c4e0", fontsize=11, fontfamily="monospace")
    ax.set_yticklabels(labels, color="#c4c4e0", fontsize=11, fontfamily="monospace")
    ax.set_xlabel("Predicted", color="#606090", fontsize=10, fontfamily="monospace")
    ax.set_ylabel("Actual",    color="#606090", fontsize=10, fontfamily="monospace")
    ax.set_title("Confusion Matrix — Multinomial Naive Bayes",
                 color="#e4e4f8", fontsize=11, fontfamily="monospace", pad=14)
    ax.tick_params(colors="#c4c4e0")
    for s in ax.spines.values():
        s.set_edgecolor("#252540")

    quad = [["TN","FP"],["FN","TP"]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i,       str(cm[i][j]), ha="center", va="center",
                    color="#ffffff", fontsize=20, fontweight="bold", fontfamily="monospace")
            ax.text(j, i+0.36, quad[i][j],    ha="center", va="center",
                    color="#9080f7", fontsize=9,  fontfamily="monospace")

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor="#0e0e1a")
    buf.seek(0)
    plt.close(fig)
    return buf


# ══════════════════════════════════════════════════════════════
#  HERO HEADER
# ══════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="hero">
    <h1>🔍 Veri<span class="ac">News</span></h1>
    <p class="sub"> NLP Project &nbsp;·&nbsp;
       TF-IDF + Multinomial Naive Bayes + Rule-Based Model</p>
    <div class="tags">
        <span class="tag">Python</span>
        <span class="tag">Streamlit</span>
        <span class="tag">TF-IDF</span>
        <span class="tag">Naive Bayes</span>
        <span class="tag">{N:,} Articles</span>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════
#  SECTION 1 — INPUT
# ══════════════════════════════════════════════════════════════
section("📋", "Input", "enter or upload a news article")

col_text, col_up = st.columns([3, 1], gap="medium")

with col_text:
    news_input = st.text_area(
        label="news_article",
        label_visibility="collapsed",
        placeholder=(
            "Paste any news article or headline here…\n"
            "e.g. SHOCKING: Scientists discover miracle cure "
            "that doctors don't want you to know!"
        ),
        height=175,
    )
    wc = len(news_input.split()) if news_input.strip() else 0
    st.caption(f"📝 {wc} words · {len(news_input)} characters")

with col_up:
    st.markdown("<div style='height:0.3rem'></div>", unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Upload .txt or .csv",
        type=["txt", "csv"],
        help="For CSV, the first text column is used.",
    )
    if uploaded:
        if uploaded.name.endswith(".csv"):
            df_up      = pd.read_csv(uploaded)
            news_input = " ".join(
                df_up.iloc[:, 0].dropna().astype(str).tolist()
            )[:3000]
            st.success(f"✅ {len(df_up)} rows loaded from CSV")
        else:
            news_input = uploaded.read().decode("utf-8", errors="ignore")[:3000]
            st.success("✅ Text file loaded")

analyze = st.button("🔍 Analyze Article")

# ══════════════════════════════════════════════════════════════
#  ANALYSIS BLOCK  (runs when button is clicked)
# ══════════════════════════════════════════════════════════════
if analyze and not news_input.strip():
    st.warning("⚠️ Please enter or upload some text first.")

elif analyze and news_input.strip():

    # ── 1. Run all models / calculations ─────────────────────
    nb_steps      = get_nb_steps(news_input, model, vectorizer)
    manual_result = predict_manual(news_input)
    top_words, total_words = get_tfidf_walkthrough(news_input, vectorizer, top_n=8)

    ml_pred  = "FAKE" if nb_steps["prediction"] == 0 else "REAL"
    ml_conf  = nb_steps["prob_fake"]  if ml_pred == "FAKE" else nb_steps["prob_real"]
    man_pred = manual_result["prediction"]
    man_conf = manual_result["confidence"]

    top_word_names = [w for w, _ in top_words[:5]]

    st.markdown("---")

    # ════════════════════════════════════════════════════════
    #  SECTION 2 — MAIN VERDICT  (gauge)
    # ════════════════════════════════════════════════════════
    section("🎯", "Verdict", "primary result")
    render_gauge(ml_pred, ml_conf)

    st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)

    # ── Agreement / Verification check ───────────────────────
    agreement = check_agreement(ml_pred, ml_conf, man_pred)
    render_verify_panel(news_input, ml_pred, ml_conf, man_pred, man_conf, agreement)

    st.markdown("---")

    # ════════════════════════════════════════════════════════
    #  SECTION 3 — MODEL COMPARISON
    # ════════════════════════════════════════════════════════
    section("⚖️", "Model Comparison", "ml model vs rule-based model")

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        render_model_card("ML Model · Multinomial Naive Bayes", ml_pred, ml_conf)
    with c2:
        render_model_card("Manual Model · Rule-Based Keywords", man_pred, man_conf)

    st.markdown("---")

    # ════════════════════════════════════════════════════════
    #  SECTION 4 — WORD ANALYSIS
    # ════════════════════════════════════════════════════════
    section("🎨", "Word Analysis", "highlighted keywords in article")

    st.markdown("""
    <div class="leg-row">
        <span class="leg leg-f">🔴 Fake-indicating word</span>
        <span class="leg leg-r">🟢 Real / credible word</span>
    </div>""", unsafe_allow_html=True)

    highlighted = highlight_text(news_input[:1500])
    st.markdown(f'<div class="hi-box">{highlighted}</div>', unsafe_allow_html=True)

    st.markdown("---")

    # ════════════════════════════════════════════════════════
    #  SECTION 5 — EXPLANATION
    # ════════════════════════════════════════════════════════
    section("💡", "Explanation", "why this verdict?")

    ml_exp  = build_ml_explanation(ml_pred, ml_conf, nb_steps, top_word_names)
    man_exp = build_manual_explanation(manual_result)

    st.markdown(f"""
    <div class="why">
        <div class="w-title">🤖 ML Model (Naive Bayes) — Reasoning</div>
        <div class="w-body">{ml_exp}</div>
    </div>
    <div class="why">
        <div class="w-title">📋 Manual Model (Rule-Based) — Reasoning</div>
        <div class="w-body">{man_exp}</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ════════════════════════════════════════════════════════
    #  SECTION 6 — MATHEMATICAL WALKTHROUGH
    # ════════════════════════════════════════════════════════
    section("🧮", "Mathematical Walkthrough", "step-by-step calculations")

    # ── Part A: TF-IDF ──────────────────────────────────────
    st.markdown('<div class="part-lbl">Part A &nbsp;— &nbsp;TF-IDF Calculation (Top Words by Score)</div>',
                unsafe_allow_html=True)

    for word, v in top_words:
        tf_r  = round(v["tf"],    6)
        idf_r = round(v["idf"],   4)
        tf_r2 = round(v["tfidf"], 6)
        st.markdown(f"""
        <div class="cbox">
            <div class="cw">"{word}"</div>
            <div class="cl">
                TF &nbsp;&nbsp;&nbsp;&nbsp; =
                count(<span>{v['count']}</span>) ÷ total_words(<span>{total_words}</span>)
                = <span>{tf_r}</span>
            </div>
            <div class="cl">
                IDF &nbsp;&nbsp;&nbsp; =
                log((<span>{N}</span>+1) / (df+1)) + 1
                = <span>{idf_r}</span>
            </div>
            <div class="cl">
                TF-IDF =
                <span>{tf_r}</span> × <span>{idf_r}</span>
                = <span>{tf_r2}</span>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

    # ── Part B: Naive Bayes ──────────────────────────────────
    st.markdown('<div class="part-lbl">Part B &nbsp;— &nbsp;Naive Bayes Probability Walkthrough</div>',
                unsafe_allow_html=True)

    st.markdown(f"""
    <div class="nbbox">
        <div class="nb-t">Step 1 — Class Prior Probabilities
            <span style="font-weight:400;color:#404065;font-size:0.72rem">
            &nbsp; log P(C) from training data distribution
            </span>
        </div>
        <div class="nb-l">
            log P(FAKE) &nbsp;= <span>{nb_steps['log_prior_fake']}</span>
            &nbsp;&nbsp;|&nbsp;&nbsp;
            log P(REAL) &nbsp;= <span>{nb_steps['log_prior_real']}</span>
        </div>
        <div class="nb-l">Training corpus: <span>{N:,}</span> documents total</div>
    </div>

    <div class="nbbox">
        <div class="nb-t">Step 2 — Log-Likelihood Scores
            <span style="font-weight:400;color:#404065;font-size:0.72rem">
            &nbsp; log P(C) + Σ log P(wᵢ|C) for all words in document
            </span>
        </div>
        <div class="nb-l">
            log P(doc | FAKE) + log P(FAKE) &nbsp;= <span>{nb_steps['log_prob_fake']}</span>
        </div>
        <div class="nb-l">
            log P(doc | REAL) + log P(REAL) &nbsp;= <span>{nb_steps['log_prob_real']}</span>
        </div>
    </div>

    <div class="nbbox">
        <div class="nb-t">Step 3 — Final Posterior Probabilities
            <span style="font-weight:400;color:#404065;font-size:0.72rem">
            &nbsp; after softmax normalisation
            </span>
        </div>
        <div class="nb-l">P(FAKE | document) &nbsp;= <span>{nb_steps['prob_fake']}%</span></div>
        <div class="nb-l">P(REAL | document) &nbsp;= <span>{nb_steps['prob_real']}%</span></div>
        <div class="nb-l hl">
            → Predicted class:
            <span>{"FAKE 🔴" if nb_steps['prediction'] == 0 else "REAL 🟢"}</span>
            &nbsp;(class with the higher posterior wins)
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ════════════════════════════════════════════════════════
    #  SECTION 7 — CONFUSION MATRIX + METRICS
    # ════════════════════════════════════════════════════════
    section("📊", "Model Evaluation", "confusion matrix & performance metrics")

    tn, fp, fn, tp = conf_matrix.ravel()
    precision = round(tp / (tp + fp) * 100, 2) if (tp + fp) > 0 else 0
    recall    = round(tp / (tp + fn) * 100, 2) if (tp + fn) > 0 else 0
    f1        = round(2*precision*recall / (precision+recall), 2) if (precision+recall) > 0 else 0

    cm_col, stats_col = st.columns([3, 2], gap="large")
    with cm_col:
        st.image(render_confusion_matrix(conf_matrix), use_container_width=True)
    with stats_col:
        st.markdown(f"""
        <div class="met">
            <div class="mt">Performance Metrics</div>
            <div class="mr"><span class="k">Accuracy</span>  <span class="v">{round(accuracy*100,2)}%</span></div>
            <div class="mr"><span class="k">Precision</span> <span class="v">{precision}%</span></div>
            <div class="mr"><span class="k">Recall</span>    <span class="v">{recall}%</span></div>
            <div class="mr"><span class="k">F1 Score</span>  <span class="v">{f1}%</span></div>
        </div>
        <div class="met">
            <div class="mt">Matrix Values</div>
            <div class="mr"><span class="k">True Negatives  (TN)</span> <span class="v">{tn:,}</span></div>
            <div class="mr"><span class="k">False Positives (FP)</span> <span class="v">{fp:,}</span></div>
            <div class="mr"><span class="k">False Negatives (FN)</span> <span class="v">{fn:,}</span></div>
            <div class="mr"><span class="k">True Positives  (TP)</span> <span class="v">{tp:,}</span></div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  DEFAULT STATE  (before Analyze is clicked)
# ══════════════════════════════════════════════════════════════
else:
    st.markdown("---")
    section("🎯", "Verdict", "waiting for input")
    st.markdown("""
    <div class="empty">
        <div style="font-size:2.8rem">🤖</div>
        <div>Enter an article above and click Analyze</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")
    section("📊", "Model Evaluation", "confusion matrix — pre-computed on 20% test split")
    cm_col, _ = st.columns([3, 2])
    with cm_col:
        st.image(render_confusion_matrix(conf_matrix), use_container_width=True)
    st.caption("Confusion matrix computed once on held-out test data. Does not change per article.")


# ══════════════════════════════════════════════════════════════
#  ALWAYS VISIBLE — FORMULA REFERENCE  (TF-IDF + Naive Bayes)
# ══════════════════════════════════════════════════════════════
st.markdown("---")
section("📐", "Formula Reference", "tf-idf")

f1c, f2c, f3c = st.columns(3, gap="medium")
with f1c:
    st.markdown("""
    <div class="fbox">
        <div class="ft">Term Frequency (TF)</div>
        <div class="feq">TF(t,d) = count(t,d) / total_words(d)</div>
        <div class="fn">
            <span>count(t,d)</span> — times word t appears in doc d<br>
            <span>total_words(d)</span> — total tokens in document d<br>
            Normalises by document length so longer docs don't dominate.
        </div>
    </div>""", unsafe_allow_html=True)
with f2c:
    st.markdown(f"""
    <div class="fbox">
        <div class="ft">Inverse Document Frequency (IDF)</div>
        <div class="feq">IDF(t) = log((N+1) / (df(t)+1)) + 1</div>
        <div class="fn">
            <span>N</span> — total documents in corpus = {N:,}<br>
            <span>df(t)</span> — number of docs containing term t<br>
            Rare words get high IDF; common words get low IDF.
        </div>
    </div>""", unsafe_allow_html=True)
with f3c:
    st.markdown("""
    <div class="fbox">
        <div class="ft">TF-IDF Score</div>
        <div class="feq">TF-IDF(t,d) = TF(t,d) × IDF(t)</div>
        <div class="fn">
            Higher score → word is more unique & important in this doc.<br>
            Common words like <span>"the"</span>, <span>"is"</span> have low IDF → low TF-IDF.<br>
            Rare but frequent words in doc get highest score.
        </div>
    </div>""", unsafe_allow_html=True)

st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)
section("⚙️", "Formula Reference", "naive bayes")

n1c, n2c, n3c = st.columns(3, gap="medium")
with n1c:
    st.markdown("""
    <div class="fbox">
        <div class="ft">Bayes' Theorem</div>
        <div class="feq">P(C|d) = P(C) × P(d|C) / P(d)</div>
        <div class="fn">
            <span>P(C)</span> — prior: class frequency in training set<br>
            <span>P(d|C)</span> — likelihood: how probable is this doc in class C?<br>
            <span>P(d)</span> — evidence: same for all classes, can be ignored.
        </div>
    </div>""", unsafe_allow_html=True)
with n2c:
    st.markdown("""
    <div class="fbox">
        <div class="ft">Naive Bayes Classifier</div>
        <div class="feq">P(C|d) ∝ P(C) × ∏ P(wᵢ|C)</div>
        <div class="fn">
            <span>"Naive"</span> = words are treated as independent features.<br>
            <span>P(wᵢ|C)</span> — probability of each word given class C.<br>
            Multiply for all words in document, pick highest class.
        </div>
    </div>""", unsafe_allow_html=True)
with n3c:
    st.markdown("""
    <div class="fbox">
        <div class="ft">Log-Likelihood (Practical Form)</div>
        <div class="feq">log P(C|d) = log P(C) + Σ log P(wᵢ|C)</div>
        <div class="fn">
            <span>Why log?</span> Multiplying many tiny probabilities causes<br>
            floating-point underflow. Log converts × into +.<br>
            <span>argmax</span> over log scores gives the final prediction.
        </div>
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("""
<div style="text-align:center; font-family:'Space Mono',monospace;
            font-size:0.65rem; color:#252545; padding:0.5rem 0 1rem; line-height:2.2;">
    Fake News Detector &nbsp;·&nbsp;
    TF-IDF + Multinomial Naive Bayes + Rule-Based Model &nbsp;·&nbsp;
    Built with Streamlit<br>
    NLP End-Semester Project
</div>""", unsafe_allow_html=True)
