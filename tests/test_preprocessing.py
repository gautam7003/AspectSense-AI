"""
Unit tests for text preprocessing and normalization routines.
"""

import pytest
from src.preprocessing import clean_text, tokenize, remove_stopwords, preprocess_for_model


def test_clean_text_contractions_and_whitespace():
    raw = "The phone can't work properly and won't charge!  "
    cleaned = clean_text(raw)
    assert "cannot" in cleaned
    assert "will not" in cleaned
    assert "  " not in cleaned


def test_clean_text_empty_and_none():
    assert clean_text("") == ""
    assert clean_text(None) == ""
    assert clean_text("   ") == ""


def test_tokenize_standard_review():
    text = "Battery life is great."
    tokens = tokenize(text)
    assert tokens == ["battery", "life", "is", "great"]


def test_stopwords_preserves_negations():
    tokens = ["the", "screen", "is", "not", "good", "at", "all"]
    filtered = remove_stopwords(tokens, preserve_negations=True)
    assert "not" in filtered
    assert "screen" in filtered
    assert "good" in filtered
    assert "the" not in filtered
    assert "is" not in filtered


def test_preprocess_for_model_pipeline():
    review = "The camera is NOT bad, but charging is slow."
    processed = preprocess_for_model(review)
    assert "not" in processed
    assert "camera" in processed
    assert "bad" in processed
    assert "slow" in processed
