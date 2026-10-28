# AspectSense AI — Comprehensive Academic Project Report

**Degree:** Master of Computer Applications (MCA)  
**Subject:** Natural Language Processing & Machine Learning  
**Project Title:** AspectSense AI — Aspect-Based Sentiment Analysis of Customer Reviews  

---

### Project Metadata & Title Page Details
- **Student Name:** `[Student Name Placeholder]`
- **Registration / Roll Number:** `[Registration Number Placeholder]`
- **Internal Guide:** `[Guide Name & Designation Placeholder]`
- **Department:** Department of Computer Applications
- **Institution / College:** `[College / University Name Placeholder]`
- **Academic Year:** 2025–2026

---

## 1. Abstract
Online customer feedback represents one of the most voluminous and valuable sources of market intelligence for e-commerce platforms and consumer brands. However, conventional sentiment analysis methodologies restrict classification to document or sentence levels, yielding a single aggregated polarity (Positive, Negative, or Neutral). Such coarse representations obscure granular feedback when reviews address multiple conflicting product attributes (e.g., *"The camera is phenomenal, but the battery drains in four hours"*). 

This project presents **AspectSense AI**, an end-to-end Aspect-Based Sentiment Analysis (ABSA) system designed to identify distinct target entities within unstructured feedback and compute targeted polarities for each aspect independently. The architecture couples syntactic clause segmentation with Negation-Preserving Text Normalization, Term Frequency-Inverse Document Frequency (TF-IDF) feature engineering, and a multi-class Logistic Regression classifier. The application features an interactive Streamlit web dashboard for real-time review testing, batch CSV uploads, Plotly-powered executive analytics, and model evaluation against held-out splits and ground-truth aspect annotations.

---

## 2. Introduction
Sentiment analysis, also termed opinion mining, is a foundational domain within Natural Language Processing (NLP). With the rapid proliferation of user-generated content across digital storefronts, social media channels, and support forums, organizations require automated systems capable of distilling consumer sentiment into actionable product insights.

Traditional sentiment analysis systems treat a textual review as a monolithic entity. If a customer writes *"The screen is gorgeous, but delivery was delayed by two weeks"*, a standard binary or ternary classifier often predicts a neutral or conflicting score, masking the praise for product design while ignoring the logistical failure. Aspect-Based Sentiment Analysis (ABSA) reformulates this problem into a fine-grained task: identifying *what* the customer is talking about (aspect terms) and *how* they feel about that specific attribute (aspect sentiment).

---

## 3. Problem Statement
To design, implement, and validate an automated Natural Language Processing application capable of:
1. Extracting explicit product aspect categories (such as Camera, Battery, Display, Performance, Price, Quality, Delivery, and Customer Service) from unstructured customer reviews.
2. Segmenting multi-clause sentences to isolate the specific opinion clause modifying each detected aspect.
3. Classifying sentiment polarity (*Positive*, *Negative*, *Neutral*) at the aspect level rather than the review level.
4. Handling negation patterns (e.g., *"not bad"*, *"never fails"*) to prevent polarity inversion errors.
5. Providing an interactive graphical interface and executive analytics dashboard for business decision-making.

---

## 4. Objectives
- Implement a robust text preprocessing pipeline that cleans noise, handles contractions, and preserves crucial negation markers.
- Formulate a configurable aspect taxonomy stored in JSON format for easy domain adaptability.
- Implement syntactic clause splitting to link opinion words directly to their governing aspects.
- Train and evaluate a Scikit-learn TF-IDF + Logistic Regression model on a dedicated review corpus.
- Prevent data leakage by isolating training and held-out test distributions prior to feature vectorization.
- Implement bulk CSV analysis with real-time progress tracking and downloadable reports.
- Visualize aspect-level sentiment breakdowns, mention frequencies, and negative dissatisfaction risks using interactive Plotly charts.

---

## 5. Scope of the Project
- **Target Domain:** Consumer electronics, smartphones, and e-commerce product reviews.
- **Aspect Categories Covered:** 8 canonical categories: Camera, Battery, Display, Performance, Price, Quality, Delivery, Customer Service.
- **Sentiment Classes:** 3 classes: Positive, Negative, Neutral.
- **Operational Environment:** Local academic execution, zero external API keys, zero paid cloud dependencies.

---

## 6. Literature Review
1. **Bing Liu (2012):** *Sentiment Analysis and Opinion Mining*. Liu formalized the sentiment target and aspect extraction problem, establishing the framework of identifying aspect terms, opinion holders, and sentiment polarities.
2. **SemEval ABSA Benchmarks (Pontiki et al., 2014, 2016):** The Semantic Evaluation workshop standardized benchmarks for Aspect Category Detection (ACD) and Aspect-Level Sentiment Classification (ALSC) across laptop and restaurant domains.
3. **Manning & Schütze (1999):** *Foundations of Statistical Natural Language Processing*. Foundational text discussing n-gram tokenization, TF-IDF weighting, and statistical text classification.
4. **Pedregosa et al. (2011):** *Scikit-learn: Machine Learning in Python*. Documented efficient, scalable implementations of logistic regression, TF-IDF representations, and cross-validation pipelines.

---

## 7. Existing System vs. Proposed System

| Dimension | Existing Conventional System | Proposed AspectSense AI System |
|---|---|---|
| **Granularity** | Coarse review-level classification. | Fine-grained aspect-level clause classification. |
| **Conflicting Sentiments** | Collapses mixed reviews into Neutral or arbitrary polarity. | Resolves multiple opposing sentiments simultaneously. |
| **Negation Handling** | Often strips negations during stopword filtering. | Explicitly preserves negations (*not, never, barely*). |
| **Explainability** | Black-box or uninterpretable averages. | Transparent clause-opinion matching with confidence probabilities. |
| **Actionability** | Only indicates overall satisfaction. | PINPOINTS specific component failures (e.g., 60% negative on Battery). |

---

## 8. System Requirements

### Hardware Requirements
- **Processor:** Intel Core i3 / AMD Ryzen 3 or higher.
- **RAM:** Minimum 4 GB (8 GB recommended).
- **Storage:** Minimum 500 MB free hard disk space.

### Software Requirements
- **Operating System:** Windows 10/11, Linux, or macOS.
- **Programming Language:** Python 3.11 / Python 3.13.
- **Core Libraries:** Streamlit, Pandas, NumPy, NLTK, Scikit-learn, Plotly, Joblib, Pytest.

---

## 9. System Architecture & Methodology

```
+-------------------------------------------------------------+
|                     User Interface                          |
|             (Streamlit Web Application)                     |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|                 1. NLP Preprocessing Engine                 |
|  - Contraction Expansion  - Regex Word Tokenization         |
|  - Lowercasing            - Negation-Preserving Stopwords   |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|              2. Aspect Extraction & Clause Splitting         |
|  - Configurable Taxonomy (config/aspects.json)              |
|  - Conjunction Clause Segmentation (but, however, yet)      |
|  - Word-Boundary Regex Keyword Association                  |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|             3. TF-IDF Feature Extraction Pipeline           |
|  - Unigrams & Bigrams (1,2 N-grams)                         |
|  - Sublinear Term Frequency Scaling                         |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|             4. Logistic Regression Classification           |
|  - Multinomial Softmax Log-Loss Formulation                 |
|  - Balanced Class Weighting                                 |
|  - Calibrated Probability Distribution                      |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|             5. Aggregation, Visuals & Reporting             |
|  - Aspect-Level Risk Radar - Plotly Interactive Charts      |
|  - Confusion Matrices      - Batch CSV Export               |
+-------------------------------------------------------------+
```

---

## 10. Algorithmic Formulations

### 10.1 TF-IDF Feature Weighting
Term Frequency-Inverse Document Frequency quantifies word significance within an individual opinion snippet relative to the entire training corpus:

$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$

Where:
$$\text{TF}(t, d) = 1 + \log(\text{count}(t, d)) \quad \text{if count}(t, d) > 0$$
$$\text{IDF}(t, D) = \log\left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1$$

### 10.2 Multinomial Logistic Regression
For class $k \in \{\text{negative}, \text{neutral}, \text{positive}\}$ and input feature vector $\mathbf{x} \in \mathbb{R}^V$:

$$P(y = k \mid \mathbf{x}) = \frac{\exp(\mathbf{w}_k^T \mathbf{x} + b_k)}{\sum_{j=1}^K \exp(\mathbf{w}_j^T \mathbf{x} + b_j)}$$

The model is trained by minimizing the regularized multi-class cross-entropy loss:

$$\mathcal{L}(\mathbf{W}) = -\frac{1}{N} \sum_{i=1}^N \sum_{k=1}^K y_{ik} \log(P(y_i = k \mid \mathbf{x}_i)) + \frac{1}{2C} \sum_{k=1}^K \|\mathbf{w}_k\|_2^2$$

---

## 11. Experimental Results & Performance Verification

Actual performance metrics obtained from model execution on the held-out test split:

| Evaluation Metric | Measured Value |
|---|---|
| **Sentiment Test Accuracy** | **80.00%** |
| **Sentiment Macro F1-Score** | **0.7929** |
| **Aspect Extraction Precision** | **89.09%** |
| **Aspect Extraction Recall** | **79.03%** |
| **Aspect Extraction F1-Score** | **83.76%** |
| **Ground-Truth Sentiment Accuracy** | **91.20%** |
| **Ground-Truth Sentiment Macro F1** | **0.9172** |
| **Total Vocabulary Features** | **821 N-grams** |

---

## 12. Limitations
1. **Rule-Based Aspect Lexicon:** Extraction depends on predefined keywords in `config/aspects.json`. Unseen colloquial expressions require manual configuration updates.
2. **Implicit Sarcasm:** Sentences without explicit sentiment adjectives (e.g., *"My phone doubled as a hand warmer in winter"*) can be misclassified.
3. **Complex Dependency Parsing:** Very long compound sentences lacking punctuation may associate sentiment adjectives with adjacent aspects.

---

## 13. Future Enhancements
- Integration of Contextual Pre-trained Transformers (e.g., DeBERTa or RoBERTa-ABSA) for zero-shot aspect boundary detection.
- Dependency parse trees using spaCy for syntactically guided subject-modifier extraction.
- Multilingual aspect analysis for cross-border e-commerce feedback.

---

## 14. Conclusion
AspectSense AI successfully achieves fine-grained aspect-based sentiment analysis, providing a transparent, explainable, and runnable machine learning solution. The system demonstrates key syllabus concepts of NLP text tokenization, negation preservation, TF-IDF feature weighting, multinomial logistic classification, and full lifecycle metric validation.
