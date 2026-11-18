# 🔍 VeriNews — Dual-Engine Fake News Detection & Explainable NLP

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![NLP](https://img.shields.io/badge/NLP-TF--IDF%20%2B%20Naive%20Bayes-purple.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()

> **VeriNews** is an Explainable AI (XAI) and Natural Language Processing (NLP) platform designed to detect misinformation and fake news. Unlike opaque black-box models, VeriNews features a **dual-engine verification architecture** combining an algorithmic Machine Learning pipeline built from scratch with a rule-based linguistic heuristic engine, complete with transparent mathematical walkthroughs and real-time fact-checking verification.

---

## 🌟 Key Features

- **🧠 Dual-Engine Verification**:
  - **Scratch ML Engine**: Fully custom implementation of **TF-IDF Vectorization** and **Multinomial Naive Bayes** (built in pure Python & NumPy without depending on modeling black-boxes).
  - **Rule-Based Heuristic Engine**: Lexical analysis leveraging weighted keyword dictionaries for sensationalism, conspiratorial cues, and neutral journalistic language.
  - **Cross-Engine Consensus**: Highlights agreement or flags discrepancies between statistical predictions and linguistic indicators.

- **📊 Explainable AI (XAI) & Mathematical Transparency**:
  - Exposes every mathematical computation step-by-step:
    - **Term Frequency (TF)** & **Inverse Document Frequency (IDF)** calculation tables.
    - **Log-Priors**, **Laplace-smoothed Likelihoods**, and **Log-Posterior Probabilities**.
  - Highlights top TF-IDF driving terms for both Fake and Real classifications.

- **🎨 Interactive Word Analysis**:
  - Real-time in-text visual token highlighting:
    - <span style="color:#ff6b6b">■</span> Red markers for sensational or conspiratorial signals.
    - <span style="color:#51cf66">■</span> Green markers for objective, journalistic, or credible signals.

- **🎯 Analog Risk Gauge & Verdict Panel**:
  - Visual gauge meter indicating risk levels:
    - `LOW RISK — LIKELY REAL`
    - `HIGH RISK — LIKELY FAKE`
    - `UNCERTAIN / REVIEW NEEDED`

- **🌐 Instant One-Click Fact-Checking**:
  - Pre-builds targeted queries and provides quick external verification links to **Google News**, **Reuters**, **Snopes**, and **FactCheck.org**.

- **📈 Confusion Matrix & Model Performance**:
  - In-app evaluation metrics including True Positives, True Negatives, False Positives, False Negatives, and overall accuracy.

- **📓 Complete Exploratory Notebook**:
  - Includes `fake_news_detection.ipynb` demonstrating the end-to-end data science lifecycle: exploratory data analysis (EDA), text preprocessing, manual vs. library benchmarking, and validation.

---

## 🏗️ Architecture & Pipeline

```text
                             ┌─────────────────────────┐
                             │    Raw News Article     │
                             │ (Headline / Full Text)  │
                             └────────────┬────────────┘
                                          │
                               ┌──────────┴──────────┐
                               ▼                     ▼
                     ┌──────────────────┐   ┌──────────────────┐
                     │ Text Cleaning &  │   │ Lexical Keyword  │
                     │  Preprocessing   │   │     Analysis     │
                     └─────────┬────────┘   └────────┬─────────┘
                               │                     │
                               ▼                     ▼
                     ┌──────────────────┐   ┌──────────────────┐
                     │ Manual TF-IDF    │   │  Heuristic Rule  │
                     │  Vectorization   │   │     Scorer       │
                     └─────────┬────────┘   └────────┬─────────┘
                               │                     │
                               ▼                     ▼
                     ┌──────────────────┐   ┌──────────────────┐
                     │ Manual Naive     │   │  Linguistic      │
                     │ Bayes Classifier │   │  Confidence      │
                     └─────────┬────────┘   └────────┬─────────┘
                               │                     │
                               └──────────┬──────────┘
                                          │
                                          ▼
                            ┌───────────────────────────┐
                            │  Cross-Model Agreement    │
                            │    & Gauge Assessment     │
                            └─────────────┬─────────────┘
                                          │
                                          ▼
                            ┌───────────────────────────┐
                            │  Interactive Streamlit UI │
                            │  - Verdict & Gauge        │
                            │  - Token Highlighter      │
                            │  - Math Walkthrough       │
                            │  - Fact-Check Shortcuts   │
                            └───────────────────────────┘
```

---

## 📐 Mathematical Formulation

### 1. Term Frequency — Inverse Document Frequency (TF-IDF)
For a term $t$ in document $d$:

$$\text{TF}(t, d) = \frac{\text{count}(t, d)}{\text{total words in } d}$$

$$\text{IDF}(t) = \ln\left(\frac{N + 1}{\text{DF}(t) + 1}\right) + 1$$

$$\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t)$$

### 2. Multinomial Naive Bayes (Log-Space with Laplace Smoothing)
For class $C \in \{\text{FAKE}, \text{REAL}\}$:

$$\log P(C \mid d) \propto \log P(C) + \sum_{t \in d} \left( \text{TF-IDF}(t, d) \times \log P(t \mid C) \right)$$

Where likelihood with additive smoothing ($\alpha = 1.0$) is:

$$P(t \mid C) = \frac{\text{count}(t, C) + \alpha}{\sum_{t'} \text{count}(t', C) + \alpha \cdot |V|}$$

---

## 📁 Project Structure

```text
Updated_VeriNews/
├── app.py                      # Primary Streamlit web application (UI & visualization)
├── app1.py                     # Alternative / baseline UI implementation
├── fake_news_detection.ipynb   # Jupyter notebook with research, EDA & training
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules for clean repository state
├── Fake.csv                    # Dataset: Fake news articles
├── True.csv                    # Dataset: Verified true news articles
├── model/
│   ├── manual_model.py         # Rule-based heuristic classifier & keyword engine
│   └── ml_model.py             # Scratch ML training & inference pipeline
└── utils/
    ├── explain.py              # Natural language generation for model reasoning
    ├── gauge.py                # Matplotlib-based analog credibility gauge
    ├── nb_manual.py            # From-scratch Naive Bayes implementation
    ├── preprocess.py           # Text cleaning, normalization & tokenization
    ├── tfidf_manual.py         # From-scratch TF-IDF vectorizer & math breakdown
    └── verify.py               # Cross-engine verification & fact-checking queries
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13
- Git installed on your system

### 1. Clone the Repository
```bash
git clone https://github.com/sarveshmali025/verinews.git
cd verinews
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
streamlit run app.py
```
Open your browser and navigate to `http://localhost:8501`.

---

## 💻 Usage Guide

1. **Input Article**: Paste any news headline or paragraph into the input text area, or upload a `.txt` file.
2. **Click "Analyze Article"**:
   - View the overall **Credibility Verdict** and **Analog Gauge**.
   - Review **Model Comparison** (ML Naive Bayes vs. Heuristic Rule-Based).
   - Inspect **Word Analysis** to see which specific terms triggered the verdict.
   - Explore the **Math Walkthrough** to follow the exact probability and TF-IDF calculations.
   - Use the **Verification Panel** to cross-reference with Reuters, Snopes, or Google Fact Check with a single click.

---

## 📊 Dataset Reference

The models in this repository are trained on the **ISOT Fake News Dataset** (University of Victoria), comprising over 40,000 articles:
- **`Fake.csv`**: Contains unreliable, fabricated, and sensational news articles.
- **`True.csv`**: Contains verified factual reports sourced primarily from Reuters.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!
Feel free to open an issue or submit a Pull Request.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

<p align="center">
  Developed by <a href="https://github.com/sarveshmali025">Sarvesh Mali</a>
</p>
