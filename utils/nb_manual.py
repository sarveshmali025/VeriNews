"""
utils/nb_manual.py
==================
Owner : Member 4
FULLY MANUAL Naive Bayes — zero sklearn.

Two responsibilities:
  1. train_nb()   — train NB from TF-IDF vectors + labels
  2. predict()    — predict class + return all intermediate values for UI
  3. get_nb_steps() — same API as before so app.py needs zero changes

Math implemented:
  Prior       : log P(C) = log(count(C) / N)
  Likelihood  : log P(w|C) with Laplace smoothing
  Posterior   : log P(C|doc) = log P(C) + sum(log P(w|C) * tfidf_weight)
  Decision    : argmax over classes
"""

import math
from utils.preprocess import clean_text


# ══════════════════════════════════════════════════════════════
#  PART 1 — MANUAL NAIVE BAYES MODEL CLASS
# ══════════════════════════════════════════════════════════════

class ManualNaiveBayes:
    """
    Multinomial Naive Bayes from scratch.
    Uses TF-IDF vectors as input (non-negative floats).
    Laplace smoothing to handle zero probabilities.
    Log-space arithmetic to prevent underflow.
    """

    def __init__(self):
        self.class_log_prior_ = {}    # class -> log P(C)
        self._log_likelihood  = {}    # class -> list of log P(w|C) per feature
        self._vocab_size      = 0
        self._classes         = []

    # ── Train ─────────────────────────────────────────────────
    def train_nb(self, X: list, y: list) -> None:
        """
        Fit the model.

        Parameters
        ----------
        X : list of vectors  (each vector = TF-IDF floats, fixed length)
        y : list of labels   (0 = FAKE, 1 = REAL)
        """
        N            = len(y)
        self._classes    = sorted(set(y))           # [0, 1]
        self._vocab_size = len(X[0])                # feature count

        for cls in self._classes:
            # ── Prior: log P(C) ───────────────────────────────
            cls_count               = sum(1 for label in y if label == cls)
            self.class_log_prior_[cls] = math.log(cls_count / N)

            # ── Collect all TF-IDF vectors for this class ─────
            cls_vectors = [X[i] for i, label in enumerate(y) if label == cls]

            # ── Sum feature values across all class documents ─
            # feature_sums[j] = total weight of feature j in class cls
            feature_sums = [0.0] * self._vocab_size
            for vec in cls_vectors:
                for j, val in enumerate(vec):
                    feature_sums[j] += val

            total_weight = sum(feature_sums)

            # ── Log likelihood with Laplace smoothing ─────────
            # log P(w_j | C) = log((sum_j + 1) / (total + vocab_size))
            self._log_likelihood[cls] = [
                math.log((feature_sums[j] + 1) / (total_weight + self._vocab_size))
                for j in range(self._vocab_size)
            ]

    # ── Predict single document ───────────────────────────────
    def predict_one(self, vector: list) -> tuple:
        """
        Predict class for one TF-IDF vector.

        Returns
        -------
        (prediction, log_scores, probabilities)
          prediction   : int   0 = FAKE, 1 = REAL
          log_scores   : dict  class -> full log posterior
          probabilities: dict  class -> normalised probability 0-1
        """
        log_scores = {}

        for cls in self._classes:
            # log P(C|doc) = log P(C) + sum(tfidf_j * log P(w_j|C))
            score = self.class_log_prior_[cls]
            ll    = self._log_likelihood[cls]
            for j, tfidf_val in enumerate(vector):
                if tfidf_val > 0:
                    score += tfidf_val * ll[j]
            log_scores[cls] = score

        # ── Softmax normalisation for probabilities ────────────
        # Shift by max for numerical stability
        max_score = max(log_scores.values())
        exp_scores = {cls: math.exp(s - max_score) for cls, s in log_scores.items()}
        total_exp  = sum(exp_scores.values())
        probs      = {cls: exp_scores[cls] / total_exp for cls in self._classes}

        prediction = max(log_scores, key=log_scores.get)
        return prediction, log_scores, probs

    # ── Batch predict (used in evaluation) ────────────────────
    def predict(self, X: list) -> list:
        """Return predicted class for each vector in X."""
        return [self.predict_one(vec)[0] for vec in X]


# ══════════════════════════════════════════════════════════════
#  PART 2 — UI WALKTHROUGH FUNCTION
#  Same API as before — app.py needs zero changes
# ══════════════════════════════════════════════════════════════

def get_nb_steps(text: str, model, vectorizer) -> dict:
    """
    Run manual NB on text, return all intermediate values for UI display.

    Parameters
    ----------
    text       : str                  — raw news text
    model      : ManualNaiveBayes     — trained model
    vectorizer : ManualTfidfVectorizer — fitted vectorizer

    Returns
    -------
    dict with keys:
        prediction      int    0=FAKE, 1=REAL
        prob_fake       float  P(FAKE|doc) as %
        prob_real       float  P(REAL|doc) as %
        log_prob_fake   float  log posterior for FAKE
        log_prob_real   float  log posterior for REAL
        log_prior_fake  float  log P(FAKE) prior
        log_prior_real  float  log P(REAL) prior
    """
    cleaned = clean_text(text)
    vector  = vectorizer.transform([cleaned])[0]

    prediction, log_scores, probs = model.predict_one(vector)

    return {
        "prediction"    : int(prediction),
        "prob_fake"     : round(probs.get(0, 0.0) * 100, 2),
        "prob_real"     : round(probs.get(1, 0.0) * 100, 2),
        "log_prob_fake" : round(log_scores.get(0, 0.0), 4),
        "log_prob_real" : round(log_scores.get(1, 0.0), 4),
        "log_prior_fake": round(model.class_log_prior_.get(0, 0.0), 4),
        "log_prior_real": round(model.class_log_prior_.get(1, 0.0), 4),
    }
