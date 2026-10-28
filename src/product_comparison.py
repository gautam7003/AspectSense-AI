"""
Product Comparison Lab Engine for AspectSense AI.
Compares aspect-level distributions, volume, sentiment ratios, and sentiment profiles across multiple products.
"""

from typing import Dict, List, Any
import pandas as pd


def compare_products(results_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Takes analyzed review items with product names and computes side-by-side product comparisons.
    """
    records = []
    for r in results_list:
        prod = r.get("product_name", "Unknown Product")
        for asp in r.get("aspects", []):
            records.append({
                "product_name": prod,
                "review_id": r.get("review_id"),
                "aspect": asp["aspect_category"],
                "display_name": asp.get("display_name", asp["aspect_category"].title()),
                "sentiment": asp["sentiment"],
                "confidence": asp.get("confidence", 0.0),
                "opinion_context": asp.get("opinion_context", "")
            })

    if not records:
        return {
            "products": [],
            "product_summary": pd.DataFrame(),
            "aspect_comparison_table": pd.DataFrame(),
            "detailed_df": pd.DataFrame()
        }

    df = pd.DataFrame(records)
    products = sorted(df["product_name"].unique().tolist())

    # Summary per product
    summary_rows = []
    for p in products:
        sub = df[df["product_name"] == p]
        tot_revs = len(set(sub["review_id"]))
        tot_asps = len(sub)
        p_pos = int((sub["sentiment"] == "positive").sum())
        p_neg = int((sub["sentiment"] == "negative").sum())
        p_neu = int((sub["sentiment"] == "neutral").sum())

        top_aspect = sub["display_name"].value_counts().index[0] if not sub.empty else "N/A"
        
        # Best and worst aspect by % sentiment
        piv = pd.crosstab(sub["display_name"], sub["sentiment"])
        for col in ["positive", "negative", "neutral"]:
            if col not in piv.columns:
                piv[col] = 0
        piv["Total"] = piv.sum(axis=1)
        piv["Pos_%"] = (piv["positive"] / piv["Total"] * 100).round(1)
        piv["Neg_%"] = (piv["negative"] / piv["Total"] * 100).round(1)

        best_asp = piv.sort_values(by="Pos_%", ascending=False).index[0] if not piv.empty else "N/A"
        worst_asp = piv.sort_values(by="Neg_%", ascending=False).index[0] if not piv.empty else "N/A"

        summary_rows.append({
            "Product": p,
            "Reviews": tot_revs,
            "Aspect Mentions": tot_asps,
            "Positive Mentions": p_pos,
            "Negative Mentions": p_neg,
            "Neutral Mentions": p_neu,
            "Net Positive %": round(p_pos / tot_asps * 100, 1) if tot_asps > 0 else 0.0,
            "Top Discussed Aspect": top_aspect,
            "Highest Delight Aspect": best_asp,
            "Highest Complaint Aspect": worst_asp
        })

    summary_df = pd.DataFrame(summary_rows)

    # Aspect Sentiment Profile Matrix (Aspect vs Products: 'Mostly Positive', 'Mixed', etc.)
    all_aspects = sorted(df["display_name"].unique().tolist())
    matrix_rows = []
    for asp in all_aspects:
        row = {"Aspect": asp}
        for p in products:
            sub = df[(df["product_name"] == p) & (df["display_name"] == asp)]
            if sub.empty:
                row[p] = "No Data"
            else:
                pos = (sub["sentiment"] == "positive").sum()
                neg = (sub["sentiment"] == "negative").sum()
                tot = len(sub)
                pos_pct = pos / tot * 100
                neg_pct = neg / tot * 100

                if pos_pct >= 60.0:
                    status = f"🟢 Mostly Positive ({int(pos_pct)}%)"
                elif neg_pct >= 50.0:
                    status = f"🔴 Mostly Negative ({int(neg_pct)}%)"
                elif pos_pct > neg_pct:
                    status = f"🟡 Lean Positive ({int(pos_pct)}%)"
                elif neg_pct > pos_pct:
                    status = f"🟠 Lean Negative ({int(neg_pct)}%)"
                else:
                    status = f"⚪ Mixed / Neutral ({tot} mentions)"
                row[p] = status
        matrix_rows.append(row)

    aspect_matrix_df = pd.DataFrame(matrix_rows)

    return {
        "products": products,
        "product_summary": summary_df,
        "aspect_comparison_table": aspect_matrix_df,
        "detailed_df": df
    }
