# AspectSense AI — Technical Methodology & Pipeline Documentation

This document describes the exact algorithmic methodology, architectural decisions, and data processing flow in AspectSense AI.

---

## 1. Data Pipeline & Schema Definition

The project utilizes two primary data artifacts located in `data/`:
1. `sample_reviews.csv`: 100 benchmark consumer reviews spanning smartphones, tablets, and consumer electronics, covering positive, negative, neutral, and mixed-sentiment samples.
2. `aspect_annotations.csv`: 125 granular aspect annotations linking specific aspect categories, matched aspect terms, opinion phrases, and ground-truth sentiment labels.

### Predefined Aspect Categories (`config/aspects.json`)
The system recognizes 8 core product dimensions:
- `camera`: Camera, lens, photos, megapixels, portrait, video, night mode.
- `battery`: Battery life, charging speed, drain, backup, mAh, standby.
- `display`: Screen, AMOLED, OLED, brightness, refresh rate, resolution.
- `performance`: Speed, processor, RAM, lag, gaming, heating, stutter.
- `price`: Cost, pricing, value for money, cheap, expensive, affordable.
- `quality`: Build quality, material, durability, finish, sturdy, design.
- `delivery`: Shipping, courier, packaging, dispatch, damaged box.
- `customer service`: Support, helpline, warranty, replacement, refund.

---

## 2. Text Preprocessing Engine (`src/preprocessing.py`)

Standard NLP pipelines frequently strip all stopwords indiscriminately. However, in sentiment analysis, words like *"not"*, *"never"*, and *"barely"* are stopwords in standard lists but critical modifiers in opinion mining. Stripping *"not"* converts *"not good"* into *"good"*, causing severe polarity inversion.

### Pipeline Stages:
1. **Contraction Standardization:** Regex replacement maps `can't` $\to$ `cannot`, `won't` $\to$ `will not`, `n't` $\to$ `not`.
2. **Noise Cleansing:** HTML tags, URL strings, and extraneous punctuation are removed.
3. **Word Tokenization:** Lowercased regex token matching `\b[a-z0-9]+(?:'[a-z]+)?\b`.
4. **Negation-Preserving Stopword Filtering:** Standard English functional stopwords are filtered, but terms in `DEFAULT_NEGATIONS` are retained.

---

## 3. Clause Splitting & Aspect Extraction (`src/aspect_extraction.py`)

A single review may contain distinct clauses with opposing sentiments:
$$\text{"The camera is excellent, but the battery drains quickly."}$$

### Splitting Heuristics:
1. Sentence boundaries (`.`, `;`, `!`, `?`) and contrast conjunctions (`but`, `however`, `although`, `yet`, `while`) are used as clause delimiter regexes.
2. For each aspect category, candidate keywords are matched using regex word boundaries `\b<keyword>\b` sorted descending by length to prioritize composite terms (e.g., *"battery life"* before *"battery"*).
3. The specific clause containing the aspect keyword is isolated as the governing `opinion_context`.

---

## 4. Machine Learning Classification (`src/sentiment_model.py`)

### Feature Representation: TF-IDF
- **N-gram Range:** $(1, 2)$ unigrams and bigrams. Bigrams allow the model to learn compound phrases such as `"not good"`, `"very fast"`, `"rapid charging"`.
- **Sublinear TF:** Replaces raw term count $tf$ with $1 + \log(tf)$ to dampen the influence of repeated words in long reviews.

### Classifier: Multinomial Logistic Regression
- **Class Weights:** Configured as `'balanced'` to adjust weights inversely proportional to class frequencies:
  $$w_c = \frac{N}{K \cdot N_c}$$
- **Optimization:** L-BFGS solver with L2 regularization ($C = 1.5$) ensuring stable convergence.

---

## 5. Separate Evaluation Protocols (`src/evaluation.py`)

To prevent metric conflation:
1. **Aspect Extraction Evaluation:** Assesses whether the system correctly extracted the exact aspects mentioned in the ground-truth annotations (reporting Precision, Recall, F1).
2. **Aspect Sentiment Evaluation:** Assesses whether the model assigned the correct sentiment to the isolated aspect clause (reporting Accuracy, Macro F1, and Confusion Matrix).
