"""
utils/explain.py
================
Owner : Member 5
Job   : Assemble human-readable explanation strings from model outputs.
        No prediction logic here — only text/HTML construction.
        Called by app.py to fill the "Why this verdict?" section.
"""


def build_ml_explanation(
    ml_pred        : str,
    ml_conf        : float,
    nb_steps       : dict,
    top_word_names : list[str],
) -> str:
    """
    Build the HTML explanation paragraph for the ML (Naive Bayes) prediction.

    Parameters
    ----------
    ml_pred        : "FAKE" or "REAL"
    ml_conf        : confidence percentage (e.g. 91.3)
    nb_steps       : dict returned by utils.nb_manual.get_nb_steps()
    top_word_names : list of top TF-IDF words (strings)

    Returns
    -------
    str — HTML string with <span> highlights (rendered by st.markdown)
    """
    words_str = ", ".join(f"<span>{w}</span>" for w in top_word_names)

    if ml_pred == "FAKE":
        return (
            f"Naive Bayes assigned this article a FAKE posterior of "
            f"<span>{ml_conf}%</span>. "
            f"The full log-score for FAKE was <span>{nb_steps['log_prob_fake']}</span>, "
            f"versus <span>{nb_steps['log_prob_real']}</span> for REAL. "
            f"The highest-scoring TF-IDF words — {words_str} — "
            f"drove the FAKE classification. These words appear frequently in "
            f"sensational or misleading articles in the training data."
        )
    else:
        return (
            f"Naive Bayes assigned this article a REAL posterior of "
            f"<span>{ml_conf}%</span>. "
            f"The full log-score for REAL was <span>{nb_steps['log_prob_real']}</span>, "
            f"versus <span>{nb_steps['log_prob_fake']}</span> for FAKE. "
            f"Key TF-IDF words — {words_str} — "
            f"suggest credible, journalistic language. These patterns "
            f"are strongly associated with verified news in the training data."
        )


def build_manual_explanation(manual_result: dict) -> str:
    """
    Build the HTML explanation paragraph for the rule-based manual model.

    Parameters
    ----------
    manual_result : dict returned by model.manual_model.predict_manual()

    Returns
    -------
    str — HTML string with <span> highlights
    """
    return (
        f"{manual_result['explanation']} &nbsp;"
        f"Fake keyword score: <span>{manual_result['fake_score']}</span> &nbsp;|&nbsp; "
        f"Real keyword score: <span>{manual_result['real_score']}</span>."
    )
