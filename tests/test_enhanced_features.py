"""
Unit tests for Product Comparison Lab and Smart Review Insights engines.
"""

import pytest
import pandas as pd
from src.product_comparison import compare_products
from src.insights import generate_smart_insights


def test_compare_products_empty():
    res = compare_products([])
    assert res["products"] == []
    assert res["product_summary"].empty


def test_compare_products_valid():
    sample_records = [
        {
            "product_name": "Product A",
            "review_id": 1,
            "aspects": [
                {"aspect_category": "camera", "display_name": "Camera", "sentiment": "positive", "confidence": 0.85, "opinion_context": "great camera"},
                {"aspect_category": "battery", "display_name": "Battery", "sentiment": "negative", "confidence": 0.90, "opinion_context": "drains fast"}
            ]
        },
        {
            "product_name": "Product B",
            "review_id": 2,
            "aspects": [
                {"aspect_category": "camera", "display_name": "Camera", "sentiment": "positive", "confidence": 0.88, "opinion_context": "crisp shots"},
                {"aspect_category": "battery", "display_name": "Battery", "sentiment": "positive", "confidence": 0.82, "opinion_context": "lasts all day"}
            ]
        }
    ]
    res = compare_products(sample_records)
    assert "Product A" in res["products"]
    assert "Product B" in res["products"]
    assert len(res["product_summary"]) == 2
    assert not res["aspect_comparison_table"].empty


def test_generate_smart_insights_structure():
    agg_data = {
        "total_reviews": 10,
        "total_aspect_mentions": 15,
        "sentiment_counts": {"positive": 10, "negative": 4, "neutral": 1},
        "aspect_distribution": {"Camera": 8, "Battery": 5, "Display": 2},
        "highest_negative_aspects": [
            {"aspect": "Battery", "negative_pct": 60.0, "negative_count": 3, "total": 5}
        ],
        "aspect_sentiment_matrix": pd.DataFrame({
            "display_name": ["Camera", "Battery"],
            "positive": [7, 2],
            "negative": [1, 3],
            "neutral": [0, 0],
            "Total": [8, 5]
        })
    }
    insights = generate_smart_insights(agg_data)
    assert len(insights) >= 2
    titles = [i["title"] for i in insights]
    assert any("Positive" in t or "Camera" in t or "Battery" in t for t in titles)
