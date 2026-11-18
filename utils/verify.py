"""
utils/verify.py
===============
Verification panel — always shows verify buttons.
Highlighted yellow ONLY when models disagree or confidence < 60%.
Neutral/calm style when both models agree.
"""

import urllib.parse
import streamlit as st
import streamlit.components.v1 as components


def check_agreement(ml_pred: str, ml_conf: float, man_pred: str) -> bool:
    """True = agree + high confidence. False = uncertain."""
    return (ml_pred == man_pred) and (ml_conf >= 60.0)


def _build_query(text: str, max_words: int = 11) -> str:
    words = text.strip().split()[:max_words]
    return urllib.parse.quote(" ".join(words))


def render_verify_panel(
    text     : str,
    ml_pred  : str,
    ml_conf  : float,
    man_pred : str,
    man_conf : float,
    agreement: bool,
):
    query      = _build_query(text)
    url_google = f"https://www.google.com/search?q={query}"
    url_bbc    = f"https://www.google.com/search?q=site:bbc.com+{query}"
    url_ap     = f"https://www.google.com/search?q=site:apnews.com+{query}"

    if agreement:
        # ── CALM — both agree, just show buttons quietly ───────
        verdict_color = "#4af7a0" if ml_pred == "REAL" else "#f74a4a"
        icon          = "✅"
        status_text   = f"Both models agree — <strong style='color:{verdict_color}'>{ml_pred}</strong>"
        sub_text      = f"ML: {ml_pred} ({ml_conf}%)&nbsp;&nbsp;·&nbsp;&nbsp;Manual: {man_pred} ({man_conf}%)&nbsp;&nbsp;·&nbsp;&nbsp;High confidence"
        box_bg        = "rgba(255,255,255,0.03)"
        box_border    = "rgba(255,255,255,0.08)"
        title_color   = "#c0c0e0"
        btn_label     = "Cross-check anyway"
        height        = 200
    else:
        # ── ALERT — uncertain, highlight strongly ──────────────
        if ml_pred != man_pred:
            reason = f"ML says <strong style='color:#f0f0a0'>{ml_pred}</strong> but Manual says <strong style='color:#f0f0a0'>{man_pred}</strong> — models disagree."
        else:
            reason = f"Confidence is only <strong style='color:#f0f0a0'>{ml_conf}%</strong> — below the 60% threshold."
        icon          = "⚠️"
        status_text   = "UNCERTAIN — Verification Recommended"
        sub_text      = reason + "<br>Please cross-check with a trusted source before sharing."
        box_bg        = "rgba(247,200,74,0.07)"
        box_border    = "rgba(247,200,74,0.40)"
        title_color   = "#f7c84a"
        btn_label     = "Verify this article on"
        height        = 210

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
      * {{ margin:0; padding:0; box-sizing:border-box; }}
      body {{ background:transparent; font-family: 'Courier New', monospace; }}

      .box {{
        background   : {box_bg};
        border       : 1.5px solid {box_border};
        border-radius: 16px;
        padding      : 1.2rem 1.5rem 1.1rem;
        text-align   : center;
      }}

      .top-row {{
        display        : flex;
        align-items    : center;
        justify-content: center;
        gap            : 0.6rem;
        margin-bottom  : 0.45rem;
      }}

      .title {{
        color      : {title_color};
        font-size  : 0.95rem;
        font-weight: 800;
      }}

      .sub {{
        color        : #707088;
        font-size    : 0.73rem;
        line-height  : 1.7;
        margin-bottom: 1rem;
      }}

      .btn-row {{
        display        : flex;
        gap            : 0.6rem;
        flex-wrap      : wrap;
        justify-content: center;
        margin-top     : 0.2rem;
      }}

      .btn-label {{
        color        : #404060;
        font-size    : 0.62rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 0.55rem;
      }}

      a.btn {{
        border-radius  : 10px;
        padding        : 0.45rem 1.1rem;
        font-size      : 0.74rem;
        text-decoration: none;
        display        : inline-flex;
        align-items    : center;
        gap            : 0.4rem;
        font-family    : 'Courier New', monospace;
        transition     : opacity 0.2s;
      }}
      a.btn:hover {{ opacity: 0.8; }}

      .btn-g {{ background:#1e1e30; border:1px solid #353555; color:#e0e0f0; }}
      .btn-b {{ background:#1a1e1a; border:1px solid #2a3a2a; color:#80d080; }}
      .btn-a {{ background:#1e1a1a; border:1px solid #3a2a2a; color:#d08080; }}
    </style>
    </head>
    <body>
    <div class="box">
      <div class="top-row">
        <span style="font-size:1.1rem">{icon}</span>
        <span class="title">{status_text}</span>
      </div>
      <div class="sub">{sub_text}</div>
      <div class="btn-label">{btn_label}</div>
      <div class="btn-row">
        <a class="btn btn-g" href="{url_google}" target="_blank">🌐 Google Search</a>
        <a class="btn btn-b" href="{url_bbc}"    target="_blank">📡 BBC News</a>
        <a class="btn btn-a" href="{url_ap}"     target="_blank">📰 AP News</a>
      </div>
    </div>
    </body>
    </html>
    """
    components.html(html, height=height)
