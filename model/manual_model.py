"""
model/manual_model.py
=====================
Owner : Member 3
Job   : Rule-based fake news classifier using curated keyword lists.
        Works completely independently of the ML model.
        Returns: prediction (FAKE/REAL/UNCERTAIN), confidence %, scores,
                 matched words, and a human-readable explanation.

How it works:
  - Two keyword dictionaries: FAKE_KEYWORDS and REAL_KEYWORDS
  - Each keyword has a weight (1–3) based on how strongly it signals fake/real
  - Scores are summed, compared, and converted to a confidence percentage
  - highlight_text() wraps matched words in HTML <span> tags for the UI
"""

# ─────────────────────────────────────────────────────────────
# FAKE-indicating keywords  (sensational, conspiratorial, etc.)
# weight 3 = very strong signal, 2 = moderate, 1 = mild
# ─────────────────────────────────────────────────────────────
FAKE_KEYWORDS: dict[str, int] = {
    # Clickbait / sensational
    "shocking"         : 3,  "bombshell"        : 3,  "breaking"         : 2,
    "unbelievable"     : 3,  "exposed"          : 2,  "leaked"           : 2,
    "exclusive"        : 2,  "viral"            : 2,  "urgent"           : 2,
    "alert"            : 2,  "warning"          : 1,  "secret"           : 2,
    "hidden"           : 2,  "banned"           : 2,  "censored"         : 2,
    "suppressed"       : 2,  "truth"            : 1,  "wakeup"           : 2,
    "woke"             : 1,  "agenda"           : 2,  "scam"             : 3,
    "hoax"             : 3,  "fraud"            : 2,  "corrupt"          : 2,
    "corruption"       : 2,  "conspiracy"       : 3,  "illuminati"       : 3,
    "satanic"          : 3,  "ritual"           : 2,  "trafficking"      : 2,
    # Political propaganda
    "globalist"        : 3,  "deep state"       : 3,  "new world order"  : 3,
    "elites"           : 2,  "mainstream media" : 2,  "fake news"        : 1,
    "lamestream"       : 3,  "socialist"        : 2,  "communist"        : 2,
    "invasion"         : 2,  "replacement"      : 2,
    # Health disinfo / pseudo-science
    "miracle"          : 3,  "cure"             : 2,  "doctors hate"     : 3,
    "big pharma"       : 3,  "vaccine kills"    : 3,  "toxic"            : 2,
    "poison"           : 2,  "chemtrails"       : 3,  "microchip"        : 2,
    "5g"               : 2,  "radiation"        : 1,  "detox"            : 2,
    "natural cure"     : 3,
    # Emotional manipulation / share-bait
    "must share"       : 3,  "share before deleted": 3, "spread the word": 2,
    "wake up sheeple"  : 3,  "sheeple"          : 3,  "brainwashed"      : 3,
    "dont be fooled"   : 2,  "open your eyes"   : 2,
}

# ─────────────────────────────────────────────────────────────
# REAL-indicating keywords  (neutral, journalistic, factual)
# ─────────────────────────────────────────────────────────────
REAL_KEYWORDS: dict[str, int] = {
    # Sourcing & attribution
    "reuters"          : 3,  "associated press" : 3,  "bbc"              : 3,
    "npr"              : 3,  "according to"     : 2,  "officials said"   : 2,
    "government said"  : 2,  "spokesperson"     : 2,  "confirmed"        : 2,
    "announced"        : 1,  "statement"        : 2,  "report"           : 1,
    "survey"           : 2,  "study"            : 2,  "research"         : 2,
    "published"        : 2,  "peer reviewed"    : 3,  "journal"          : 2,
    "data shows"       : 2,  "statistics"       : 2,  "evidence"         : 2,
    "analysis"         : 2,  "experts"          : 2,  "scientists"       : 2,
    "researchers"      : 2,  "university"       : 2,  "institute"        : 2,
    # Governance / formal institutions
    "parliament"       : 2,  "congress"         : 2,  "senate"           : 2,
    "president"        : 1,  "minister"         : 1,  "legislation"      : 2,
    "policy"           : 2,  "regulation"       : 2,  "department"       : 2,
    "agency"           : 2,  "committee"        : 2,  "court"            : 2,
    "ruling"           : 2,  "verdict"          : 2,  "signed"           : 2,
    # Measured, balanced language
    "however"          : 1,  "although"         : 1,  "nevertheless"     : 1,
    "despite"          : 1,  "on the other hand": 2,  "in contrast"      : 2,
    "analysis suggests": 2,  "percent"          : 1,  "million"          : 1,
    "billion"          : 1,  "quarterly"        : 2,
}


# ─────────────────────────────────────────────────────────────
# INTERNAL: score a text against one keyword dictionary
# ─────────────────────────────────────────────────────────────
def _score_text(text_lower: str, keyword_dict: dict) -> tuple[int, list[str]]:
    """Return (total_weight, list_of_matched_keywords)."""
    score, matched = 0, []
    for phrase, weight in keyword_dict.items():
        if phrase in text_lower:
            score  += weight
            matched.append(phrase)
    return score, matched


# ─────────────────────────────────────────────────────────────
# MAIN PREDICTION FUNCTION
# ─────────────────────────────────────────────────────────────
def predict_manual(text: str) -> dict:
    """
    Classify `text` as FAKE, REAL, or UNCERTAIN using keyword scoring.

    Returns
    -------
    dict with keys:
        prediction  str    "FAKE" | "REAL" | "UNCERTAIN"
        confidence  float  0–95  (never claims 100%)
        fake_score  int    total fake keyword weight
        real_score  int    total real keyword weight
        fake_pct    float  fake share of combined score
        real_pct    float  real share of combined score
        fake_words  list   matched fake keywords
        real_words  list   matched real keywords
        explanation str    plain-text reasoning sentence
    """
    text_lower = text.lower()
    fake_score, fake_words = _score_text(text_lower, FAKE_KEYWORDS)
    real_score, real_words = _score_text(text_lower, REAL_KEYWORDS)
    total = fake_score + real_score

    # ── No keywords matched ───────────────────────────────────
    if total == 0:
        return {
            "prediction" : "UNCERTAIN",
            "confidence" : 50.0,
            "fake_score" : 0,
            "real_score" : 0,
            "fake_pct"   : 50.0,
            "real_pct"   : 50.0,
            "fake_words" : [],
            "real_words" : [],
            "explanation": "No strong fake or real indicators found. Manual model is uncertain.",
        }

    # ── Compute confidence ────────────────────────────────────
    fake_pct   = round((fake_score / total) * 100, 1)
    real_pct   = round((real_score / total) * 100, 1)
    raw_conf   = max(fake_pct, real_pct)
    # Mild dampening: never claim 100% from keyword scoring alone
    confidence = round(50 + (raw_conf - 50) * 0.9, 1)
    confidence = min(confidence, 95.0)

    prediction = "FAKE" if fake_score >= real_score else "REAL"

    # ── Build explanation sentence ────────────────────────────
    if prediction == "FAKE" and fake_words:
        top = fake_words[:4]
        explanation = (
            f"Trigger words detected: {', '.join(repr(w) for w in top)}. "
            f"These are strongly associated with sensationalism or misinformation."
        )
    elif prediction == "REAL" and real_words:
        top = real_words[:4]
        explanation = (
            f"Credibility markers detected: {', '.join(repr(w) for w in top)}. "
            f"These suggest sourced, factual reporting."
        )
    else:
        explanation = "Prediction based on overall keyword weight balance."

    return {
        "prediction" : prediction,
        "confidence" : confidence,
        "fake_score" : fake_score,
        "real_score" : real_score,
        "fake_pct"   : fake_pct,
        "real_pct"   : real_pct,
        "fake_words" : fake_words,
        "real_words" : real_words,
        "explanation": explanation,
    }


# ─────────────────────────────────────────────────────────────
# WORD HIGHLIGHTER  — for the UI "Word Analysis" section
# ─────────────────────────────────────────────────────────────
def highlight_text(text: str) -> str:
    """
    Wrap fake-indicator words in <span class='fake-word'> (red)
    and real-indicator words in <span class='real-word'> (green).
    Only single-word keywords are highlighted (multi-word phrases skipped
    to keep token splitting simple).

    Returns an HTML string safe to pass to st.markdown(..., unsafe_allow_html=True).
    """
    fake_single = {k for k in FAKE_KEYWORDS if " " not in k}
    real_single = {k for k in REAL_KEYWORDS  if " " not in k}

    tokens = []
    for word in text.split():
        clean = word.lower().strip(".,!?\"'();:—")
        if clean in fake_single:
            tokens.append(
                f'<span class="fake-word" title="Fake indicator: {clean}">{word}</span>'
            )
        elif clean in real_single:
            tokens.append(
                f'<span class="real-word" title="Real indicator: {clean}">{word}</span>'
            )
        else:
            tokens.append(word)

    return " ".join(tokens)
