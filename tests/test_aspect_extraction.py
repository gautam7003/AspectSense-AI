"""
Unit tests for Aspect Extraction and Clause Association Engine.
"""

import pytest
from src.aspect_extraction import AspectExtractor


@pytest.fixture
def extractor():
    return AspectExtractor()


def test_single_aspect_extraction(extractor):
    text = "The camera takes gorgeous daylight photos."
    aspects = extractor.extract_aspects(text)
    assert len(aspects) >= 1
    categories = [a["aspect_category"] for a in aspects]
    assert "camera" in categories


def test_multiple_aspects_mixed_sentiment(extractor):
    text = "The display is beautiful, but the battery drains quickly."
    aspects = extractor.extract_aspects(text)
    categories = [a["aspect_category"] for a in aspects]
    assert "display" in categories
    assert "battery" in categories


def test_three_aspects_review(extractor):
    text = "The camera is excellent, battery life is poor, and the price is reasonable."
    aspects = extractor.extract_aspects(text)
    categories = [a["aspect_category"] for a in aspects]
    assert "camera" in categories
    assert "battery" in categories
    assert "price" in categories


def test_empty_or_whitespace_input(extractor):
    assert extractor.extract_aspects("") == []
    assert extractor.extract_aspects("   ") == []
    assert extractor.extract_aspects(None) == []


def test_negation_detection(extractor):
    clause1 = "The screen is not bright enough"
    clause2 = "The screen is bright and clear"
    assert extractor.has_negation(clause1) is True
    assert extractor.has_negation(clause2) is False


def test_clause_splitting(extractor):
    compound = "Fast delivery, but customer support was unhelpful."
    clauses = extractor.split_into_clauses(compound)
    assert len(clauses) >= 2
