"""
Text Preprocessing module for AspectSense AI.
Handles lowercasing, punctuation cleaning, tokenization, stopword removal with negation preservation,
and text normalization suitable for TF-IDF feature extraction.
"""

import re
import string
from typing import List, Optional

# Standard English stopwords from NLTK/academic NLP standard,
# with negation words explicitly preserved to protect sentiment semantics.
DEFAULT_NEGATIONS = {
    "not", "no", "never", "nor", "neither", "barely", "hardly", "scarcely",
    "cannot", "cant", "don't", "dont", "doesn't", "doesnt", "didn't", "didnt",
    "won't", "wont", "wouldn't", "wouldnt", "shouldn't", "shouldnt", "couldn't",
    "couldnt", "isn't", "isnt", "aren't", "arent", "wasn't", "wasnt", "weren't",
    "werent", "without", "lack", "lacks", "lacking"
}

BASE_STOPWORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
    "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she",
    "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
    "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
    "these", "those", "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
    "the", "and", "if", "or", "because", "as", "until", "while", "of", "at",
    "by", "for", "with", "about", "against", "between", "into", "through",
    "during", "before", "after", "above", "below", "to", "from", "up", "down",
    "in", "out", "on", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "any",
    "both", "each", "few", "more", "most", "other", "some", "such", "only",
    "own", "same", "so", "than", "too", "very", "s", "t", "can", "will",
    "just", "should", "now"
}

# Preserve contrast conjunctions when splitting or analyzing segments
CONTRAST_CONJUNCTIONS = {
    "but", "however", "although", "though", "yet", "still", "nevertheless",
    "nonetheless", "whereas", "while"
}


def clean_text(text: Optional[str]) -> str:
    """
    Cleans raw review text by removing unwanted symbols, extra spaces,
    and standardizing basic contractions.
    
    Args:
        text: Raw text string.
        
    Returns:
        Cleaned text string.
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    
    # Expand or standardize common contractions
    text = re.sub(r"can't|cant", "cannot", text, flags=re.IGNORECASE)
    text = re.sub(r"won't|wont", "will not", text, flags=re.IGNORECASE)
    text = re.sub(r"n't", " not", text, flags=re.IGNORECASE)
    
    # Strip URL links and HTML tags if present
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    
    # Standardize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    """
    Tokenizes text into words while keeping punctuation-separated tokens clean.
    
    Args:
        text: Input string.
        
    Returns:
        List of word tokens in lowercase.
    """
    if not text:
        return []
    cleaned = clean_text(text).lower()
    # Retain alphanumeric characters, hyphens, and single apostrophes
    tokens = re.findall(r"\b[a-z0-9]+(?:'[a-z]+)?\b", cleaned)
    return tokens


def remove_stopwords(tokens: List[str], preserve_negations: bool = True) -> List[str]:
    """
    Removes standard stopwords from a token list while optionally preserving
    critical sentiment negation words.
    
    Args:
        tokens: List of word tokens.
        preserve_negations: If True, words in DEFAULT_NEGATIONS are kept.
        
    Returns:
        Filtered list of tokens.
    """
    filtered = []
    for token in tokens:
        if preserve_negations and token in DEFAULT_NEGATIONS:
            filtered.append(token)
        elif token in BASE_STOPWORDS:
            continue
        else:
            filtered.append(token)
    return filtered


def preprocess_for_model(text: str, preserve_negations: bool = True) -> str:
    """
    Complete end-to-end preprocessing pipeline for machine learning feature extraction.
    
    Args:
        text: Raw review or opinion snippet.
        preserve_negations: Whether to keep negation words.
        
    Returns:
        Cleaned, normalized string ready for TF-IDF vectorization.
    """
    cleaned = clean_text(text)
    if not cleaned:
        return ""
    tokens = tokenize(cleaned)
    filtered = remove_stopwords(tokens, preserve_negations=preserve_negations)
    return " ".join(filtered)
