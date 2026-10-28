"""
Smart Insights Generation Engine for AspectSense AI.
Generates explainable, statistically supported qualitative and quantitative observations
from aspect sentiment prediction distributions.
"""

from typing import List, Dict, Any
import pandas as pd


def generate_smart_insights(aggregated_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Produces transparent, data-grounded insights from aggregated ABSA metrics.
    Avoids hallucinated or unverified conclusions and provides exact percentages and counts.
    """
    insights = []
    total_reviews = aggregated_data.get("total_reviews", 0)
    total_aspects = aggregated_data.get("total_aspect_mentions", 0)
    
    if total_reviews == 0 or total_aspects == 0:
        return [{
            "type": "info",
            "title": "Insufficient Data",
            "message": "Dataset sample is empty. Ingest or analyze reviews to generate analytical insights.",
            "metric": "0 records"
        }]

    if total_reviews < 5:
        insights.append({
            "type": "warning",
            "title": "Small Sample Notice",
            "message": f"Sample size is small ({total_reviews} reviews, {total_aspects} aspects). Statistics show initial indicators rather than statistically significant population trends.",
            "metric": f"N={total_reviews}"
        })

    sentiment_counts = aggregated_data.get("sentiment_counts", {})
    pos = sentiment_counts.get("positive", 0)
    neg = sentiment_counts.get("negative", 0)
    neu = sentiment_counts.get("neutral", 0)

    pos_pct = round(pos / total_aspects * 100, 1)
    neg_pct = round(neg / total_aspects * 100, 1)
    neu_pct = round(neu / total_aspects * 100, 1)

    # 1. Overall polarity climate
    if pos_pct >= 55.0:
        insights.append({
            "type": "success",
            "title": "Predominantly Positive Reception",
            "message": f"Positive aspect feedback leads overall sentiment at {pos_pct}% ({pos} out of {total_aspects} aspect mentions).",
            "metric": f"{pos_pct}% Positive"
        })
    elif neg_pct >= 40.0:
        insights.append({
            "type": "error",
            "title": "Elevated Dissatisfaction Climate",
            "message": f"Negative aspect feedback represents {neg_pct}% ({neg} of {total_aspects} aspect mentions), indicating prevalent consumer dissatisfaction.",
            "metric": f"{neg_pct}% Negative"
        })
    else:
        insights.append({
            "type": "info",
            "title": "Balanced / Multi-Polar Feedback",
            "message": f"Feedback exhibits balanced distribution: {pos_pct}% positive, {neg_pct}% negative, and {neu_pct}% neutral mentions.",
            "metric": "Balanced Distribution"
        })

    # 2. Most frequently discussed aspect
    aspect_dist = aggregated_data.get("aspect_distribution", {})
    if aspect_dist:
        top_aspect, top_count = next(iter(aspect_dist.items()))
        top_share = round(top_count / total_aspects * 100, 1)
        insights.append({
            "type": "info",
            "title": f"Primary Discussion Driver: {top_aspect}",
            "message": f"{top_aspect} is the most discussed aspect, capturing {top_share}% of all aspect mentions ({top_count} mentions).",
            "metric": f"{top_count} Mentions"
        })

    # 3. Component Dissatisfaction Risk (Negative Share)
    highest_neg = aggregated_data.get("highest_negative_aspects", [])
    if highest_neg:
        worst = highest_neg[0]
        if worst["negative_pct"] >= 40.0 and worst["total"] >= 2:
            insights.append({
                "type": "error",
                "title": f"Critical Attention Required: {worst['aspect']}",
                "message": f"{worst['aspect']} exhibits a high negative sentiment ratio of {worst['negative_pct']}% ({worst['negative_count']} negative out of {worst['total']} mentions).",
                "metric": f"{worst['negative_pct']}% Dissatisfaction"
            })

    # 4. Standout Positive Aspect
    matrix = aggregated_data.get("aspect_sentiment_matrix")
    if isinstance(matrix, pd.DataFrame) and not matrix.empty and "positive" in matrix.columns:
        # Calculate positive share
        matrix_calc = matrix.copy()
        matrix_calc["pos_pct"] = (matrix_calc["positive"] / matrix_calc["Total"] * 100).round(1)
        standout = matrix_calc[matrix_calc["Total"] >= 3].sort_values(by="pos_pct", ascending=False)
        if not standout.empty and standout.iloc[0]["pos_pct"] >= 60.0:
            best_row = standout.iloc[0]
            insights.append({
                "type": "success",
                "title": f"Top Customer Delight: {best_row['display_name']}",
                "message": f"Customers express overwhelming satisfaction with {best_row['display_name']} ({best_row['pos_pct']}% positive over {int(best_row['Total'])} mentions).",
                "metric": f"{best_row['pos_pct']}% Positive"
            })

    return insights
