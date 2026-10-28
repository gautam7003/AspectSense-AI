# Serialized Machine Learning Artifacts

This directory stores the trained model components produced by running the model training pipeline:

```bash
python -m scripts.train_model
```

### Artifact Files:
1. `sentiment_classifier.joblib`: Serialized Scikit-learn `LogisticRegression` model trained with balanced class weights for 3-class sentiment prediction (`positive`, `negative`, `neutral`).
2. `tfidf_vectorizer.joblib`: Fitted `TfidfVectorizer` capturing unigram and bigram token features.
3. `training_metrics.joblib`: Evaluation metrics, confusion matrix, classification report, and test sample counts from the held-out evaluation split.
