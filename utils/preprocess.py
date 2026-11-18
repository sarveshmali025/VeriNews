"""
utils/preprocess.py
===================
Owner : Member 5
Job   : Clean raw news text before feeding to ML model or manual TF-IDF.

Steps:
  1. Lowercase everything
  2. Remove URLs  (http://...)
  3. Remove all non-letter characters (numbers, punctuation)
  4. Collapse extra whitespace
"""

import re


def clean_text(text: str) -> str:
    """
    Return a cleaned, lowercase version of the input text.

    Parameters
    ----------
    text : str  — raw news article or headline

    Returns
    -------
    str  — cleaned text, letters only, single-space separated
    """
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)        # remove URLs
    text = re.sub(r'[^a-zA-Z ]', '', text)     # keep letters + spaces only
    text = re.sub(r'\s+', ' ', text).strip()   # collapse whitespace
    return text
