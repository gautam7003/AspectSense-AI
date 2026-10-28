"""
Helper utilities and data validation routines for AspectSense AI.
"""

import os
import pandas as pd
from typing import Tuple, Optional


def validate_csv_upload(file_obj) -> Tuple[bool, Optional[pd.DataFrame], str]:
    """
    Validates user uploaded CSV file, checking encoding, empty file conditions,
    and parsing issues.
    """
    if file_obj is None:
        return False, None, "No file uploaded."

    try:
        # Try standard utf-8 then fallback to latin1
        try:
            df = pd.read_csv(file_obj, encoding="utf-8")
        except UnicodeDecodeError:
            file_obj.seek(0)
            df = pd.read_csv(file_obj, encoding="latin1")

        if df.empty:
            return False, None, "Uploaded CSV file is completely empty."

        return True, df, "CSV parsed successfully."
    except Exception as e:
        return False, None, f"Failed to parse CSV file: {str(e)}"


def get_sentiment_badge(sentiment: str) -> str:
    """Returns an HTML badge color-coded by sentiment."""
    sentiment = str(sentiment).lower().strip()
    if sentiment == "positive":
        return '<span style="background-color:#d4edda; color:#155724; padding:3px 8px; border-radius:4px; font-weight:600;">Positive</span>'
    elif sentiment == "negative":
        return '<span style="background-color:#f8d7da; color:#721c24; padding:3px 8px; border-radius:4px; font-weight:600;">Negative</span>'
    else:
        return '<span style="background-color:#e2e3e5; color:#383d41; padding:3px 8px; border-radius:4px; font-weight:600;">Neutral</span>'
