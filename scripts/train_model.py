"""
Standalone Training Pipeline Script for AspectSense AI.
Usage:
    python -m scripts.train_model
"""

import os
import sys
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.sentiment_model import SentimentClassifier
from src.aspect_extraction import AspectExtractor
from src.evaluation import evaluate_aspect_extraction, evaluate_aspect_sentiment


def main():
    print("=" * 60)
    print("AspectSense AI — Model Training Pipeline")
    print("=" * 60)

    # File paths
    sample_csv = os.path.join(PROJECT_ROOT, "data", "sample_reviews.csv")
    annotations_csv = os.path.join(PROJECT_ROOT, "data", "aspect_annotations.csv")

    if not os.path.exists(sample_csv):
        print(f"Error: Dataset not found at {sample_csv}")
        sys.exit(1)

    print(f"\n[1/4] Loading training dataset from: {sample_csv}")
    df_reviews = pd.read_csv(sample_csv)
    print(f"      Total records loaded: {len(df_reviews)}")

    # We also leverage aspect_annotations to give the model rich aspect-level sentence pairs
    if os.path.exists(annotations_csv):
        print(f"\n[2/4] Augmenting training context with annotated opinion clauses: {annotations_csv}")
        df_annot = pd.read_csv(annotations_csv)
        # Create training pairs of (opinion_phrase, sentiment) and (review_text, sentiment)
        train_pairs = []
        for _, row in df_reviews.iterrows():
            if str(row["sentiment"]).lower() in ["positive", "negative", "neutral"]:
                train_pairs.append({"text": row["review_text"], "sentiment": row["sentiment"].lower()})
        for _, row in df_annot.iterrows():
            if str(row["sentiment"]).lower() in ["positive", "negative", "neutral"]:
                train_pairs.append({"text": row["opinion_phrase"], "sentiment": row["sentiment"].lower()})
        
        combined_df = pd.DataFrame(train_pairs).drop_duplicates().reset_index(drop=True)
        print(f"      Total training instances after combining: {len(combined_df)}")
    else:
        # Fallback to pure review texts
        combined_df = df_reviews[df_reviews["sentiment"].isin(["positive", "negative", "neutral"])].copy()
        combined_df = combined_df.rename(columns={"review_text": "text"})

    print("\n[3/4] Fitting TF-IDF Vectorizer and Training Logistic Regression Classifier...")
    classifier = SentimentClassifier()
    metrics = classifier.train(
        combined_df,
        text_col="text",
        label_col="sentiment",
        test_size=0.25,
        random_state=42
    )

    print(f"      Model Training Complete!")
    print(f"      Held-out Test Set Accuracy: {metrics['accuracy']:.4f}")
    print(f"      Held-out Test Set Macro F1: {metrics['macro_f1']:.4f}")
    print(f"      Classes: {metrics['classes']}")
    print(f"      Vocabulary size: {metrics['total_vocab_size']}")

    print("\n[4/4] Running Ground-Truth Evaluation on Aspect Annotations...")
    if os.path.exists(annotations_csv):
        extractor = AspectExtractor()
        aspect_eval = evaluate_aspect_extraction(annotations_csv, extractor)
        sentiment_eval = evaluate_aspect_sentiment(annotations_csv, extractor, classifier)

        print("\n--- Aspect Extraction Evaluation ---")
        print(f"Overall Precision : {aspect_eval['overall_precision']:.4f}")
        print(f"Overall Recall    : {aspect_eval['overall_recall']:.4f}")
        print(f"Overall F1-Score  : {aspect_eval['overall_f1']:.4f}")
        print(f"True Positives    : {aspect_eval['true_positives']}")
        print(f"False Positives   : {aspect_eval['false_positives']}")
        print(f"False Negatives   : {aspect_eval['false_negatives']}")

        print("\n--- Aspect Sentiment Classification Evaluation ---")
        print(f"Sentiment Accuracy: {sentiment_eval['accuracy']:.4f}")
        print(f"Sentiment Macro F1: {sentiment_eval['macro_f1']:.4f}")

    print("\n" + "=" * 60)
    print("All artifacts saved successfully to the 'models/' directory.")
    print("=" * 60)


if __name__ == "__main__":
    main()
