"""
Unit tests for Sentiment Classification Model, inference, SVM baseline, and feature explainability.
"""

import pytest
import os
import pandas as pd
from src.sentiment_model import SentimentClassifier


@pytest.fixture
def classifier():
    clf = SentimentClassifier()
    loaded = clf.load_artifacts()
    assert loaded is True, "Failed to load pre-trained artifacts for test."
    return clf


def test_positive_prediction(classifier):
    res = classifier.predict_sentiment("The camera is phenomenal and takes crisp photos.")
    assert res["sentiment"] == "positive"
    assert res["confidence"] > 0.4
    assert "positive" in res["probabilities"]


def test_negative_prediction(classifier):
    res = classifier.predict_sentiment("Battery life is terrible and drains quickly.")
    assert res["sentiment"] == "negative"
    assert res["confidence"] > 0.4


def test_negation_prediction(classifier):
    res = classifier.predict_sentiment("The camera is not good at all, photos are noisy.")
    assert res["sentiment"] == "negative"


def test_empty_prediction(classifier):
    res = classifier.predict_sentiment("")
    assert res["sentiment"] == "neutral"


def test_svm_model_prediction(classifier):
    res = classifier.predict_sentiment("Super fast processor with zero lag during heavy gaming.", model_type="svm")
    assert res["sentiment"] == "positive"
    assert res["confidence"] > 0.4


def test_explainability_feature_contributions(classifier):
    res = classifier.predict_sentiment("The camera is phenomenal and battery is terrible.")
    explanation = res.get("explanation", {})
    assert "top_features" in explanation
    assert "summary" in explanation
    assert len(explanation["top_features"]) > 0


def test_missing_model_handling():
    clf = SentimentClassifier(model_dir="/non/existent/path")
    assert clf.is_trained() is False
    with pytest.raises(RuntimeError):
        clf.predict_sentiment("Test review")
