"""
Machine Learning Sentiment Classification Models for AspectSense AI.
Provides TF-IDF Vectorization, Logistic Regression, Linear SVM baseline,
artifact serialization, calibrated probabilistic inference, and feature explainability.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

from src.preprocessing import preprocess_for_model


DEFAULT_MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models"
)
VECTORIZER_FILE = "tfidf_vectorizer.joblib"
MODEL_FILE = "sentiment_classifier.joblib"
SVM_MODEL_FILE = "svm_sentiment_classifier.joblib"
METRICS_FILE = "training_metrics.joblib"


class SentimentClassifier:
    """
    Manages the machine learning lifecycle for aspect sentiment classification:
    preprocessing, TF-IDF vectorization, Logistic Regression fitting, Linear SVM baseline (SGDClassifier modified huber),
    evaluation, saving, loading, predicting, and explainability.
    """

    def __init__(self, model_dir: Optional[str] = None):
        self.model_dir = model_dir or DEFAULT_MODEL_DIR
        self.vectorizer_path = os.path.join(self.model_dir, VECTORIZER_FILE)
        self.model_path = os.path.join(self.model_dir, MODEL_FILE)
        self.svm_model_path = os.path.join(self.model_dir, SVM_MODEL_FILE)
        self.metrics_path = os.path.join(self.model_dir, METRICS_FILE)
        
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.model: Optional[LogisticRegression] = None
        self.svm_model: Optional[Any] = None
        self.classes: List[str] = ["negative", "neutral", "positive"]
        self.training_metrics: Dict[str, Any] = {}

    def is_trained(self) -> bool:
        """Returns True if model and vectorizer are loaded and ready."""
        return self.vectorizer is not None and self.model is not None

    def load_artifacts(self) -> bool:
        """Loads serialized TF-IDF vectorizer and classifiers from disk."""
        if os.path.exists(self.vectorizer_path) and os.path.exists(self.model_path):
            try:
                self.vectorizer = joblib.load(self.vectorizer_path)
                self.model = joblib.load(self.model_path)
                if hasattr(self.model, "classes_"):
                    self.classes = list(self.model.classes_)
                if os.path.exists(self.svm_model_path):
                    self.svm_model = joblib.load(self.svm_model_path)
                if os.path.exists(self.metrics_path):
                    self.training_metrics = joblib.load(self.metrics_path)
                return True
            except Exception as e:
                print(f"Error loading model artifacts: {e}")
                return False
        return False

    def train(
        self,
        df: pd.DataFrame,
        text_col: str = "review_text",
        label_col: str = "sentiment",
        test_size: float = 0.25,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """
        Trains TF-IDF + Logistic Regression AND Linear SVM baseline on provided DataFrame.
        Guarantees strict train-test split before fitting vectorizer to avoid data leakage.
        """
        df = df.dropna(subset=[text_col, label_col]).copy()
        df = df[df[label_col].isin(["positive", "negative", "neutral"])].copy()
        
        if len(df) < 10:
            raise ValueError("Insufficient training data. Need at least 10 valid labeled samples.")

        # Train-test split before feature extraction to prevent leakage
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=df[label_col]
        )

        # Preprocess text
        X_train_clean = [preprocess_for_model(t) for t in train_df[text_col]]
        X_test_clean = [preprocess_for_model(t) for t in test_df[text_col]]
        # Convert explicitly to standard Python string list / numpy array to avoid pyarrow indexing bugs
        y_train = np.array(train_df[label_col].tolist(), dtype=object)
        y_test = np.array(test_df[label_col].tolist(), dtype=object)

        # TF-IDF Vectorizer with unigrams and bigrams
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True
        )
        
        X_train_vec = self.vectorizer.fit_transform(X_train_clean)
        X_test_vec = self.vectorizer.transform(X_test_clean)

        # 1. Fit Logistic Regression
        self.model = LogisticRegression(
            C=1.5,
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state
        )
        self.model.fit(X_train_vec, y_train)
        self.classes = list(self.model.classes_)

        y_pred = self.model.predict(X_test_vec)
        acc = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        conf_mat = confusion_matrix(y_test, y_pred, labels=self.classes).tolist()

        # 2. Fit Linear SVM Baseline via SGDClassifier(loss='modified_huber')
        # Modified Huber loss provides calibrated probability estimates smoothly without CV wrapper overhead
        self.svm_model = SGDClassifier(
            loss="modified_huber",
            penalty="l2",
            alpha=1e-3,
            max_iter=2000,
            random_state=random_state,
            class_weight="balanced"
        )
        self.svm_model.fit(X_train_vec, y_train)

        svm_y_pred = self.svm_model.predict(X_test_vec)
        svm_acc = accuracy_score(y_test, svm_y_pred)
        svm_macro_f1 = f1_score(y_test, svm_y_pred, average="macro", zero_division=0)
        svm_report = classification_report(y_test, svm_y_pred, output_dict=True, zero_division=0)
        svm_conf_mat = confusion_matrix(y_test, svm_y_pred, labels=self.classes).tolist()

        self.training_metrics = {
            "accuracy": float(acc),
            "macro_f1": float(macro_f1),
            "classification_report": report,
            "confusion_matrix": conf_mat,
            "classes": self.classes,
            "train_samples": len(train_df),
            "test_samples": len(test_df),
            "total_vocab_size": len(self.vectorizer.vocabulary_),
            "svm_metrics": {
                "accuracy": float(svm_acc),
                "macro_f1": float(svm_macro_f1),
                "classification_report": svm_report,
                "confusion_matrix": svm_conf_mat
            }
        }

        # Save artifacts
        os.makedirs(self.model_dir, exist_ok=True)
        joblib.dump(self.vectorizer, self.vectorizer_path)
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.svm_model, self.svm_model_path)
        joblib.dump(self.training_metrics, self.metrics_path)

        return self.training_metrics

    def predict_sentiment(self, text: str, model_type: str = "logistic_regression") -> Dict[str, Any]:
        """
        Predicts sentiment and class probabilities for a single text context.
        """
        if not self.is_trained():
            raise RuntimeError("Model is not loaded or trained. Call load_artifacts() or train().")

        cleaned = preprocess_for_model(text)
        if not cleaned:
            return {
                "sentiment": "neutral",
                "confidence": 0.33,
                "probabilities": {c: 0.33 for c in self.classes},
                "explanation": {"top_features": [], "summary": "Empty or neutral text."}
            }

        vec = self.vectorizer.transform([cleaned])
        
        # Choose classifier
        active_model = self.svm_model if (model_type == "svm" and self.svm_model is not None) else self.model
        probas = active_model.predict_proba(vec)[0]
        pred_idx = np.argmax(probas)
        pred_label = self.classes[pred_idx]
        conf = float(probas[pred_idx])
        prob_dict = {cls_name: round(float(prob), 4) for cls_name, prob in zip(self.classes, probas)}

        # Explainability feature contributions for Logistic Regression
        explanation = self.explain_prediction(cleaned, pred_label)

        return {
            "sentiment": pred_label,
            "confidence": round(conf, 4),
            "probabilities": prob_dict,
            "explanation": explanation
        }

    def explain_prediction(self, cleaned_text: str, predicted_label: str) -> Dict[str, Any]:
        """
        Extracts TF-IDF features present in the query and looks up their learned weights (coefficients)
        in the Logistic Regression model for the predicted class.
        """
        if not self.is_trained() or self.model is None or not hasattr(self.model, "coef_"):
            return {
                "top_features": [],
                "summary": "Explainability features unavailable for this model configuration."
            }

        vocab = self.vectorizer.vocabulary_
        tokens = cleaned_text.split()
        
        # Collect unigrams and bigrams present in the text
        ngrams_in_text = []
        for i in range(len(tokens)):
            ngrams_in_text.append(tokens[i])
            if i + 1 < len(tokens):
                ngrams_in_text.append(f"{tokens[i]} {tokens[i+1]}")

        target_class_idx = self.classes.index(predicted_label)
        coefs = self.model.coef_[target_class_idx]

        feature_impacts = []
        seen = set()
        for ng in ngrams_in_text:
            if ng in vocab and ng not in seen:
                seen.add(ng)
                feat_idx = vocab[ng]
                weight = float(coefs[feat_idx])
                feature_impacts.append({
                    "feature": ng,
                    "weight": round(weight, 3),
                    "impact": "Supports Prediction" if weight > 0 else "Opposes / Neutral"
                })

        # Sort features by weight descending
        feature_impacts.sort(key=lambda x: abs(x["weight"]), reverse=True)

        if not feature_impacts:
            summary = "Prediction driven by general sentence structure and baseline class distribution."
        else:
            top_kw = feature_impacts[0]["feature"]
            top_wt = feature_impacts[0]["weight"]
            summary = f"Key driver: '{top_kw}' (weight {top_wt:+.2f}) strongly inclined model towards {predicted_label}."

        return {
            "top_features": feature_impacts[:5],
            "summary": summary
        }
