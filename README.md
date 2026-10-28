# AspectSense AI — Enterprise ABSA Platform (Enhanced Edition)

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.66-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.9-orange.svg)](https://scikit-learn.org/)
[![Pytest](https://img.shields.io/badge/Pytest-21%20Passed-brightgreen.svg)]()

> **Master of Computer Applications (MCA) Project — Natural Language Processing & Machine Learning**  
> An upgraded, production-grade Aspect-Based Sentiment Analysis & Consumer Opinion Intelligence platform.

---

## 🌟 Upgraded Capabilities & New Features

### 1. Premium SaaS Interface (Phase 1)
- Redesigned executive dashboard featuring KPI metric cards, clean visual hierarchy, consistent typography, and responsive grid layouts.
- Live session activity tracking, aspect mention counters, and real-time polarity breakdown.

### 2. Enhanced Aspect Extraction & Ambiguity Detection (Phase 2 & 9)
- Multi-clause syntactic segmentation isolating distinct opinion phrases connected by contrast conjunctions (`but`, `however`, `yet`).
- **Feature Explainability Attribution**: Highlights individual token weights driving model predictions.
- **Calibrated Uncertainty Warning**: Automatically flags predictions with low confidence ($<50\%$) as ambiguous rather than presenting them as certain.

### 3. Smart Review Insights & Dissatisfaction Radar (Phase 5)
- Automated qualitative findings and polarity climate metrics computed from actual review distributions.
- **High Negative Risk Radar** prioritizing component defects (e.g., Battery drain, Customer Support response times).

### 4. Product Comparison Lab (Phase 4)
- Compare multiple products or competitor SKUs side-by-side.
- Generates an Aspect-by-Aspect Delight vs. Complaint matrix (`🟢 Mostly Positive`, `🔴 Mostly Negative`, `🟡 Mixed`).

### 5. Advanced Bulk CSV Analysis (Phase 7)
- Upload spreadsheets with automatic encoding detection.
- Review column selector, duplicate review detection, and optional deduplication.
- Dual export capability: Download analyzed predictions as **CSV** or structured **JSON**.

### 6. Interactive Sentiment Explorer (Phase 3)
- Multi-faceted filtering across products, aspect categories, sentiment polarities, confidence sliders, and free-text regex search.

### 7. Review Session History (Phase 6)
- Keeps track of all single-review analyses conducted during the session.
- Filterable and searchable history with one-click clear session confirmation.

### 8. Model Performance Center & SVM Baseline (Phase 8)
- Benchmarks **Multinomial Logistic Regression** against **Linear SVM** (SGD modified huber loss).
- Complete confusion matrix, precision, recall, and macro F1 scores evaluated on held-out test splits.

### 9. Presentation Demo Mode (Phase 10)
- Preloaded with 7 targeted scenarios for college viva presentations (multi-aspect conflicting, negation preservation, low-confidence, out-of-domain).

---

## 📂 Project Architecture

```text
AspectSense-AI/
├── app.py                      # Upgraded Streamlit Enterprise Dashboard (10 Modules)
├── requirements.txt            # Python dependencies
├── README.md                   # Complete documentation
├── config/
│   └── aspects.json            # Configurable taxonomy & negation dictionaries
├── data/
│   ├── sample_reviews.csv      # Multi-product benchmark reviews (Apex, Zenith, NovaBook, Pulse)
│   └── aspect_annotations.csv  # 125 ground-truth aspect-level annotations
├── models/
│   ├── sentiment_classifier.joblib      # Logistic Regression model
│   ├── svm_sentiment_classifier.joblib  # Linear SVM baseline model
│   ├── tfidf_vectorizer.joblib          # TF-IDF N-gram feature vectorizer
│   └── training_metrics.joblib          # Evaluation metrics & confusion matrices
├── src/
│   ├── preprocessing.py        # Tokenization, cleaning, negation-preserving stopwords
│   ├── aspect_extraction.py    # Clause splitting & keyword matching
│   ├── sentiment_model.py      # Logistic Regression & Linear SVM models with explainability
│   ├── insights.py             # Smart Insights generation engine
│   ├── product_comparison.py   # Product Comparison Lab engine
│   ├── analytics.py            # Summary aggregation and pivots
│   ├── evaluation.py           # Aspect & sentiment evaluation
│   └── utils.py                # CSV validation and formatting
├── scripts/
│   └── train_model.py          # Standalone training script
├── tests/
│   ├── test_preprocessing.py   # Preprocessing unit tests
│   ├── test_aspect_extraction.py # Aspect extraction unit tests
│   ├── test_sentiment_model.py # Model inference, SVM, and explainability tests
│   └── test_enhanced_features.py # Product comparison & Smart Insights tests
└── docs/
    ├── project_report.md       # MCA Academic Project Report
    ├── methodology.md          # Technical NLP & ML Methodology
    └── viva_questions.md       # 25+ Viva Examination Questions & Answers
```

---

## 💻 Execution Commands (Windows PowerShell)

### 1. Run Unit Tests (21 Automated Tests)
```powershell
python -m pytest tests
```

### 2. Retrain Models (Logistic Regression + Linear SVM)
```powershell
python -m scripts.train_model
```

### 3. Launch Web Application
```powershell
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

---

## 📊 Evaluation Benchmark Results

| Model | Held-out Accuracy | Macro F1-Score | Inference Speed |
|---|---|---|---|
| **Linear SVM (SGD Modified Huber)** | **84.00%** | **0.8234** | < 2ms |
| **Multinomial Logistic Regression** | **80.00%** | **0.7929** | < 2ms |
| **Aspect Extraction (Ground Truth)** | **Precision: 89.09%** | **Recall: 79.03%** | **F1: 83.76%** |
