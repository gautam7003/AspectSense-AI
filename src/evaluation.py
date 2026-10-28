"""
Comprehensive Evaluation Engine for AspectSense AI.
Evaluates Aspect Extraction (Precision, Recall, F1) against ground-truth annotations
and Sentiment Classification separately.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Any
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

from src.aspect_extraction import AspectExtractor
from src.sentiment_model import SentimentClassifier


def evaluate_aspect_extraction(
    annotations_path: str,
    extractor: AspectExtractor
) -> Dict[str, Any]:
    """
    Evaluates rule-based aspect extraction against labeled ground-truth aspect annotations.
    Computes Aspect-level Precision, Recall, and F1-score across all categories.
    """
    if not os.path.exists(annotations_path):
        raise FileNotFoundError(f"Annotations file not found at {annotations_path}")

    df = pd.read_csv(annotations_path)
    
    # Group true aspects by review_id
    grouped_true = df.groupby("review_id")["aspect_category"].apply(lambda s: set(s.str.lower())).to_dict()
    # Map review_id to review_text
    review_texts = df.groupby("review_id")["review_text"].first().to_dict()

    tp = 0
    fp = 0
    fn = 0
    
    per_aspect_stats: Dict[str, Dict[str, int]] = {}

    for review_id, text in review_texts.items():
        true_aspects = grouped_true.get(review_id, set())
        detected = extractor.extract_aspects(text)
        pred_aspects = set(d["aspect_category"].lower() for d in detected)

        for asp in pred_aspects:
            if asp not in per_aspect_stats:
                per_aspect_stats[asp] = {"tp": 0, "fp": 0, "fn": 0}
            if asp in true_aspects:
                tp += 1
                per_aspect_stats[asp]["tp"] += 1
            else:
                fp += 1
                per_aspect_stats[asp]["fp"] += 1

        for asp in true_aspects:
            if asp not in per_aspect_stats:
                per_aspect_stats[asp] = {"tp": 0, "fp": 0, "fn": 0}
            if asp not in pred_aspects:
                fn += 1
                per_aspect_stats[asp]["fn"] += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    per_aspect_metrics = {}
    for asp, stats in per_aspect_stats.items():
        p = stats["tp"] / (stats["tp"] + stats["fp"]) if (stats["tp"] + stats["fp"]) > 0 else 0.0
        r = stats["tp"] / (stats["tp"] + stats["fn"]) if (stats["tp"] + stats["fn"]) > 0 else 0.0
        f = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
        per_aspect_metrics[asp] = {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1_score": round(f, 4),
            "support": stats["tp"] + stats["fn"]
        }

    return {
        "overall_precision": round(precision, 4),
        "overall_recall": round(recall, 4),
        "overall_f1": round(f1, 4),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "total_ground_truth_aspects": tp + fn,
        "per_aspect_metrics": per_aspect_metrics
    }


def evaluate_aspect_sentiment(
    annotations_path: str,
    extractor: AspectExtractor,
    classifier: SentimentClassifier
) -> Dict[str, Any]:
    """
    Evaluates end-to-end aspect sentiment accuracy on the annotated dataset.
    Feeds the aspect-specific clause/opinion context into the sentiment model.
    """
    if not os.path.exists(annotations_path):
        raise FileNotFoundError(f"Annotations file not found at {annotations_path}")

    df = pd.read_csv(annotations_path)
    
    y_true = []
    y_pred = []

    for _, row in df.iterrows():
        text = str(row["review_text"])
        target_aspect = str(row["aspect_category"]).lower().strip()
        true_sentiment = str(row["sentiment"]).lower().strip()

        if true_sentiment not in ["positive", "negative", "neutral"]:
            continue

        detected = extractor.extract_aspects(text)
        # Find matched context for target aspect
        context = text
        for item in detected:
            if item["aspect_category"].lower() == target_aspect:
                context = item["opinion_context"]
                break

        res = classifier.predict_sentiment(context)
        y_true.append(true_sentiment)
        y_pred.append(res["sentiment"])

    classes = ["negative", "neutral", "positive"]
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    conf_mat = confusion_matrix(y_true, y_pred, labels=classes).tolist()

    return {
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "classification_report": report,
        "confusion_matrix": conf_mat,
        "classes": classes,
        "total_evaluated": len(y_true)
    }
