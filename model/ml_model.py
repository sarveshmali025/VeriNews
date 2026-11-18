"""
model/ml_model.py
=================
Owner : Member 2
FULLY MANUAL ML pipeline — zero sklearn for modelling.

Only pandas used for data loading. numpy used for matrix ops.
sklearn used ONLY for train_test_split and confusion_matrix
(data utilities, not the model itself).

Pipeline:
  Fake.csv + True.csv
       |  label (0=FAKE, 1=REAL)
       |  clean_text()
       |  ManualTfidfVectorizer.fit_transform()   <- our scratch TF-IDF
       |  manual train_test_split (80/20)
       |  ManualNaiveBayes.train_nb()             <- our scratch NB
       |  evaluate: confusion matrix + accuracy
"""

import math
import random
import pandas as pd

from utils.preprocess    import clean_text
from utils.tfidf_manual  import ManualTfidfVectorizer
from utils.nb_manual     import ManualNaiveBayes


# ─────────────────────────────────────────────────────────────
# Manual train/test split (no sklearn needed)
# ─────────────────────────────────────────────────────────────
def _manual_split(X, y, test_size=0.2, random_state=42):
    """
    Shuffle and split X, y into train and test sets.
    Pure Python — no sklearn.
    """
    random.seed(random_state)
    indices = list(range(len(y)))
    random.shuffle(indices)

    split   = int(len(indices) * (1 - test_size))
    train_i = indices[:split]
    test_i  = indices[split:]

    X_train = [X[i] for i in train_i]
    X_test  = [X[i] for i in test_i]
    y_train = [y[i] for i in train_i]
    y_test  = [y[i] for i in test_i]

    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────────────────────
# Manual confusion matrix (no sklearn needed)
# ─────────────────────────────────────────────────────────────
def _manual_confusion_matrix(y_true, y_pred):
    """
    Build 2x2 confusion matrix as nested list.
    [[TN, FP],
     [FN, TP]]
    """
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)

    import numpy as np
    return np.array([[tn, fp], [fn, tp]])


# ─────────────────────────────────────────────────────────────
# MAIN — load data, train, evaluate, return artifacts
# ─────────────────────────────────────────────────────────────
def load_model():
    """
    Full scratch ML pipeline.

    Returns
    -------
    model       : ManualNaiveBayes    — trained model
    vectorizer  : ManualTfidfVectorizer — fitted vectorizer
    data        : pd.DataFrame        — full cleaned dataset
    conf_matrix : np.ndarray (2,2)    — from 20% test split
    accuracy    : float  0-1
    """

    # ── Load & label ──────────────────────────────────────────
    fake = pd.read_csv("Fake.csv")
    true = pd.read_csv("True.csv")
    fake["label"] = 0   # 0 = FAKE
    true["label"] = 1   # 1 = REAL

    # ── Combine, clean, shuffle ───────────────────────────────
    data = pd.concat([fake, true])[["text", "label"]].dropna()
    data["text"] = data["text"].apply(clean_text)
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)

    corpus = data["text"].tolist()
    labels = data["label"].tolist()

    # ── Manual TF-IDF ─────────────────────────────────────────
    # Build vocabulary, compute IDF, return TF-IDF matrix
    vectorizer = ManualTfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(corpus)   # list of float vectors

    # ── Manual train/test split (80/20) ───────────────────────
    X_train, X_test, y_train, y_test = _manual_split(
        X, labels, test_size=0.2, random_state=42
    )

    # ── Manual Naive Bayes training ───────────────────────────
    # Computes class priors + log likelihoods with Laplace smoothing
    model = ManualNaiveBayes()
    model.train_nb(X_train, y_train)

    # ── Evaluate on held-out test set ─────────────────────────
    y_pred      = model.predict(X_test)
    conf_matrix = _manual_confusion_matrix(y_test, y_pred)
    correct     = sum(1 for t, p in zip(y_test, y_pred) if t == p)
    accuracy    = correct / len(y_test)

    return model, vectorizer, data, conf_matrix, accuracy
