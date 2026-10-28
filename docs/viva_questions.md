# AspectSense AI — Viva Examination Questions & Model Answers

This comprehensive guide prepares an MCA postgraduate student for project defense, internal/external viva examinations, and NLP/ML conceptual questions.

---

### Q1: What is Natural Language Processing (NLP)?
**Answer:** Natural Language Processing (NLP) is an interdisciplinary subfield of artificial intelligence, computer science, and linguistics concerned with enabling computers to understand, interpret, process, and generate human language in a meaningful and computationally useful way.

---

### Q2: What is Aspect-Based Sentiment Analysis (ABSA)?
**Answer:** Aspect-Based Sentiment Analysis (ABSA) is a fine-grained opinion mining task that decomposes text to identify specific target entities or attributes (aspects) mentioned by the user and determines the individual sentiment polarity associated with each aspect separately.

---

### Q3: How does ABSA differ from ordinary (document/sentence-level) sentiment analysis?
**Answer:** Standard sentiment analysis assigns a single overall polarity label (Positive, Negative, Neutral) to an entire document or sentence. In contrast, ABSA analyzes multiple aspects simultaneously. For example, in *"Great camera but awful battery"*, ordinary analysis outputs an ambiguous Neutral, whereas ABSA correctly identifies `Camera -> Positive` and `Battery -> Negative`.

---

### Q4: What is text tokenization?
**Answer:** Tokenization is the foundational NLP preprocessing step of splitting an unbroken string of written text into discrete linguistic units called tokens (typically words, punctuation, or subwords).

---

### Q5: Why are stopwords removed in NLP pipelines?
**Answer:** Stopwords (e.g., *"is"*, *"at"*, *"which"*, *"the"*) are high-frequency functional words that carry little semantic or topical discriminative information. Removing them reduces feature dimensionality and prevents common grammatical words from dominating term frequency statistics.

---

### Q6: Why did you preserve negation words during stopword removal?
**Answer:** Standard stopword lists include words like *"not"*, *"never"*, *"no"*, and *"without"*. If stripped, a phrase like *"not good"* becomes *"good"*, causing severe polarity inversion. Preserving negation words ensures that sentiment valence is maintained.

---

### Q7: What is the difference between Stemming and Lemmatization?
**Answer:** 
- **Stemming:** A crude heuristic process that chops off suffixes (e.g., Porter stemmer turns *"caring"* into *"car"*), often producing non-dictionary roots.
- **Lemmatization:** A morphological analysis utilizing vocabulary and grammatical context to return the legitimate dictionary base form, known as the lemma (e.g., *"better"* $\to$ *"good"*, *"caring"* $\to$ *"care"*).

---

### Q8: What is Bag-of-Words (BoW)?
**Answer:** Bag-of-Words is a text representation technique where a document is modeled as an unordered collection of its words, disregarding grammar and word order but keeping track of word multiplicity (frequency counts).

---

### Q9: What is TF-IDF and how is it calculated?
**Answer:** TF-IDF stands for Term Frequency-Inverse Document Frequency. It evaluates how important a word is to a document within a collection.
- $\text{TF}(t, d)$: How frequently term $t$ occurs in document $d$.
- $\text{IDF}(t, D) = \log\left(\frac{|D|}{\text{DF}(t)}\right)$: Measures how rare the term is across the entire corpus $D$.
- $\text{TF-IDF} = \text{TF} \times \text{IDF}$.

---

### Q10: Why did you choose Logistic Regression instead of deep neural networks as your classifier?
**Answer:** Logistic Regression is an optimal baseline for an MCA academic project:
1. Highly computationally efficient, training in seconds on CPU without GPUs.
2. Highly explainable through learnable feature coefficients.
3. Provides well-calibrated probabilistic class outputs via the Softmax function.
4. Performs exceptionally well on sparse, high-dimensional TF-IDF matrices without overfitting small sample sizes.

---

### Q11: What is the difference between Precision and Recall?
**Answer:**
- **Precision:** $\frac{TP}{TP + FP}$ — Out of all instances the model predicted as positive, how many were actually positive? (Measures exactness).
- **Recall:** $\frac{TP}{TP + FN}$ — Out of all actual positive instances in the data, how many did the model correctly identify? (Measures completeness).

---

### Q12: What is the F1-Score and why is it preferred over Accuracy for imbalanced datasets?
**Answer:** The F1-Score is the harmonic mean of Precision and Recall:
$$F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
Accuracy can be misleading when classes are imbalanced (e.g., in a 95% positive dataset, a dummy classifier predicting all positive achieves 95% accuracy but zero recall on negative cases). F1 balances both error types.

---

### Q13: What is Macro F1 vs. Micro F1?
**Answer:**
- **Macro F1:** Computes the F1-score independently for each class and calculates the unweighted average. It gives equal importance to all classes regardless of support.
- **Micro F1:** Aggregates total true positives, false negatives, and false positives globally, which tends to be heavily dominated by majority classes.

---

### Q14: What is a Confusion Matrix?
**Answer:** A Confusion Matrix is an $N \times N$ contingency table summarizing the prediction outcomes of a classification model, cross-tabulating actual ground-truth classes (rows) against predicted classes (columns) to display True Positives, False Positives, True Negatives, and False Negatives.

---

### Q15: How does your aspect extraction algorithm work?
**Answer:** AspectSense AI uses a hybrid syntactic-lexical approach:
1. Sentences are split into semantic clauses using punctuation and contrast conjunctions (`but`, `however`, `although`, `yet`).
2. Regex word boundaries match keywords defined in `config/aspects.json`.
3. The clause containing the aspect keyword is isolated as the aspect-specific opinion context and passed to the sentiment classifier.

---

### Q16: How do you prevent Data Leakage during model training?
**Answer:** Data leakage is prevented by splitting the dataset into training (75%) and test (25%) subsets *before* any feature extraction. The `TfidfVectorizer` is fitted (`fit_transform`) exclusively on training data and then merely transforms (`transform`) the held-out test data.

---

### Q17: What is Overfitting and how do you combat it in Logistic Regression?
**Answer:** Overfitting occurs when a model memorizes idiosyncrasies and noise in the training set rather than learning generalizable patterns, resulting in high training accuracy but poor validation performance. In Logistic Regression, it is mitigated using L2 regularization ($C$ parameter) which penalizes excessively large feature weights.

---

### Q18: What are N-grams and why did you use unigram + bigram (1, 2)?
**Answer:** An N-gram is a contiguous sequence of $n$ items from a given text. Using $(1, 2)$ n-grams allows the model to capture single words (`"good"`, `"slow"`) as well as word pairs (`"not good"`, `"fast charging"`, `"battery drain"`), which are vital for capturing negations and compound attributes.

---

### Q19: Why do you report aspect extraction metrics separately from aspect sentiment metrics?
**Answer:** Conflating the two conceals error sources. A system could have 100% sentiment classification accuracy on provided text snippets but miss 50% of the aspects mentioned in reviews. Separating extraction metrics (Precision, Recall) from sentiment metrics (Accuracy, Macro F1) provides a transparent, scientifically honest evaluation.

---

### Q20: What are the limitations of keyword-based aspect extraction?
**Answer:** 
1. Vocabulary coverage: Any aspect synonym not listed in `config/aspects.json` is overlooked.
2. Polysemy: A word may have different meanings in different contexts (e.g., *"the charge on my credit card"* vs. *"battery charge"*).
3. Implicit aspects: Fails to detect implicit references that don't name the aspect (e.g., *"it slipped out of my hand and cracked"* implies build quality).

---

### Q21: What is the role of Streamlit in this project?
**Answer:** Streamlit is an open-source Python framework used to build rapid, reactive, and interactive web applications for machine learning and data science without needing separate frontend languages (HTML/CSS/JS).

---

### Q22: Why use Joblib instead of Pickle for model serialization?
**Answer:** `joblib.dump` and `joblib.load` are optimized specifically for Python objects containing large NumPy numerical arrays and Scikit-learn estimators, serializing and deserializing much faster and more compactly than standard Python `pickle`.

---

### Q23: How can Deep Learning (LSTM, GRU) improve this system?
**Answer:** Recurrent Neural Networks (RNNs) such as LSTMs (Long Short-Term Memory) or GRUs (Gated Recurrent Units) process words sequentially, maintaining hidden memory states that capture long-range dependencies and complex word ordering beyond fixed n-gram windows.

---

### Q24: How would a Pretrained Transformer (BERT) improve Aspect-Based Sentiment Analysis?
**Answer:** Transformers utilize self-attention mechanisms to generate dynamic contextual embeddings for every word based on its full bi-directional sentence context. A fine-tuned BERT model (or ABSA-specialized variants like DeBERTa-ABSA) can formulate aspect extraction as token classification (Sequence Tagging with BIO tags) and achieve superior zero-shot generalization.

---

### Q25: How does this project help a commercial enterprise?
**Answer:** Businesses receive thousands of reviews weekly. Instead of reading each one or looking at an ambiguous 3-star rating, AspectSense AI provides an executive dashboard highlighting exact product issues (e.g., *"Battery dissatisfaction increased to 64% after the latest firmware update"*), enabling engineering and QA teams to address specific flaws directly.
