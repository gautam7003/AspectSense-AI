"""
Analytics and Aggregation Engine for AspectSense AI.
Aggregates bulk review aspect predictions into metrics, distributions, and summary tables.
"""

from typing import Dict, List, Any
import pandas as pd


def aggregate_aspect_sentiment(results_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Computes summary metrics and distributions from a list of aspect prediction records.
    Carefully distinguishes review count from aspect prediction count.
    """
    if not results_list:
        return {
            "total_reviews": 0,
            "total_aspect_mentions": 0,
            "sentiment_counts": {"positive": 0, "negative": 0, "neutral": 0},
            "aspect_distribution": {},
            "aspect_sentiment_matrix": pd.DataFrame(),
            "highest_negative_aspects": []
        }

    total_reviews = len(set(r.get("review_id", i) for i, r in enumerate(results_list)))
    
    # Extract aspect-level rows
    aspect_records = []
    for r in results_list:
        for asp in r.get("aspects", []):
            aspect_records.append({
                "review_id": r.get("review_id"),
                "aspect": asp["aspect_category"],
                "display_name": asp.get("display_name", asp["aspect_category"].title()),
                "sentiment": asp["sentiment"],
                "confidence": asp.get("confidence", 0.0),
                "opinion_context": asp.get("opinion_context", "")
            })

    if not aspect_records:
        return {
            "total_reviews": total_reviews,
            "total_aspect_mentions": 0,
            "sentiment_counts": {"positive": 0, "negative": 0, "neutral": 0},
            "aspect_distribution": {},
            "aspect_sentiment_matrix": pd.DataFrame(),
            "highest_negative_aspects": []
        }

    df_aspects = pd.DataFrame(aspect_records)

    total_aspect_mentions = len(df_aspects)
    
    sentiment_counts = {
        "positive": int((df_aspects["sentiment"] == "positive").sum()),
        "negative": int((df_aspects["sentiment"] == "negative").sum()),
        "neutral": int((df_aspects["sentiment"] == "neutral").sum())
    }

    # Count mentions per aspect
    aspect_dist = df_aspects["display_name"].value_counts().to_dict()

    # Pivot table: Aspect x Sentiment counts
    pivot = pd.crosstab(
        df_aspects["display_name"],
        df_aspects["sentiment"]
    )
    for col in ["positive", "negative", "neutral"]:
        if col not in pivot.columns:
            pivot[col] = 0
    pivot = pivot[["positive", "neutral", "negative"]]
    pivot["Total"] = pivot.sum(axis=1)
    pivot["Negative_Share_%"] = (pivot["negative"] / pivot["Total"] * 100).round(1)

    # Highest negative share aspects (minimum 1 mention)
    neg_ranking = pivot.sort_values(by=["Negative_Share_%", "negative"], ascending=[False, False])
    highest_negative_aspects = [
        {"aspect": idx, "negative_pct": row["Negative_Share_%"], "negative_count": int(row["negative"]), "total": int(row["Total"])}
        for idx, row in neg_ranking.iterrows() if row["negative"] > 0
    ]

    return {
        "total_reviews": total_reviews,
        "total_aspect_mentions": total_aspect_mentions,
        "sentiment_counts": sentiment_counts,
        "aspect_distribution": aspect_dist,
        "aspect_sentiment_matrix": pivot.reset_index(),
        "highest_negative_aspects": highest_negative_aspects,
        "detailed_df": df_aspects
    }
