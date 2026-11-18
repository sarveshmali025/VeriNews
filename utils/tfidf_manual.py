"""
utils/tfidf_manual.py
=====================
Owner : Member 4
FULLY MANUAL TF-IDF — zero sklearn.

Two responsibilities:
  1. BUILD the TF-IDF vectorizer from scratch (used by ml_model.py)
  2. UI walkthrough functions (used by app.py for explanation display)

Formulas:
  TF(t, d)     = count(t, d) / total_words(d)
  IDF(t)       = log((N + 1) / (df(t) + 1)) + 1
  TF-IDF(t, d) = TF(t, d) x IDF(t)
"""

import math
from utils.preprocess import clean_text


# ══════════════════════════════════════════════════════════════
#  PART 1 — MANUAL TF-IDF VECTORIZER CLASS
# ══════════════════════════════════════════════════════════════

class ManualTfidfVectorizer:
    """
    Scratch TF-IDF vectorizer. No sklearn used.
    Matches sklearn smooth IDF formula exactly.
    """

    def __init__(self, max_features: int = 5000):
        self.max_features = max_features
        self.vocabulary_  = {}   # word -> index
        self.idf_         = {}   # word -> IDF float
        self._N           = 0    # total docs seen at fit time

    # ── Step 1: build vocabulary ──────────────────────────────
    def build_vocab(self, corpus):
        """Count df per word, keep top max_features by df."""
        df_counts = {}
        for doc in corpus:
            for word in set(doc.split()):
                df_counts[word] = df_counts.get(word, 0) + 1

        sorted_words     = sorted(df_counts.items(), key=lambda x: x[1], reverse=True)
        top_words        = sorted_words[:self.max_features]
        self.vocabulary_ = {word: idx for idx, (word, _) in enumerate(top_words)}
        self._df_counts  = dict(top_words)

    # ── Step 2: compute IDF for all vocab words ───────────────
    def compute_idf(self, N: int):
        """IDF(t) = log((N+1)/(df(t)+1)) + 1"""
        self._N   = N
        self.idf_ = {}
        for word, df in self._df_counts.items():
            self.idf_[word] = math.log((N + 1) / (df + 1)) + 1

    # ── Step 3: fit + transform in one shot ───────────────────
    def fit_transform(self, corpus):
        """Build vocab+IDF from corpus, return TF-IDF matrix."""
        self.build_vocab(corpus)
        self.compute_idf(len(corpus))
        return [self._vectorize(doc) for doc in corpus]

    # ── Step 4: transform new docs with fitted vocab ──────────
    def transform(self, corpus):
        """Convert new docs to TF-IDF vectors using fitted vocab."""
        return [self._vectorize(doc) for doc in corpus]

    # ── Internal: one doc -> one vector ───────────────────────
    def _vectorize(self, doc):
        """Fixed-length TF-IDF vector for one document."""
        words       = doc.split()
        total_words = max(len(words), 1)
        vector      = [0.0] * len(self.vocabulary_)

        word_count = {}
        for w in words:
            word_count[w] = word_count.get(w, 0) + 1

        for word, count in word_count.items():
            if word in self.vocabulary_:
                idx          = self.vocabulary_[word]
                tf           = count / total_words
                idf          = self.idf_[word]
                vector[idx]  = tf * idf

        return vector


# ══════════════════════════════════════════════════════════════
#  PART 2 — UI WALKTHROUGH FUNCTIONS  (app.py uses these)
#  API identical to before — app.py needs zero changes
# ══════════════════════════════════════════════════════════════

def compute_tf(count: int, total_words: int) -> float:
    """TF(t, d) = count(t, d) / total_words(d)"""
    return count / total_words


def get_idf(word: str, vectorizer) -> float | None:
    """Get IDF for a word from ManualTfidfVectorizer. Returns None if OOV."""
    if isinstance(getattr(vectorizer, 'idf_', None), dict):
        return vectorizer.idf_.get(word, None)
    # sklearn fallback
    vocab = getattr(vectorizer, 'vocabulary_', {})
    idf_v = getattr(vectorizer, 'idf_', [])
    if word in vocab:
        return float(idf_v[vocab[word]])
    return None


def compute_tfidf(tf: float, idf: float) -> float:
    """TF-IDF(t, d) = TF(t, d) x IDF(t)"""
    return tf * idf


def get_tfidf_walkthrough(text: str, vectorizer, top_n: int = 8):
    """
    Compute TF/IDF/TF-IDF per word for UI display.
    Returns top-N words by TF-IDF score + total word count.
    """
    cleaned     = clean_text(text)
    words       = cleaned.split()
    total_words = max(len(words), 1)

    word_count = {}
    for w in words:
        word_count[w] = word_count.get(w, 0) + 1

    scores = {}
    for word, count in word_count.items():
        idf = get_idf(word, vectorizer)
        if idf is not None:
            tf    = compute_tf(count, total_words)
            tfidf = compute_tfidf(tf, idf)
            scores[word] = {"tf": tf, "idf": idf, "tfidf": tfidf, "count": count}

    top_words = sorted(scores.items(), key=lambda x: x[1]["tfidf"], reverse=True)[:top_n]
    return top_words, total_words
