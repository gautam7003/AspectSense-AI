"""
Aspect Extraction and Clause Association Engine for AspectSense AI.
Supports configurable aspect categories from JSON, multi-clause splitting by conjunctions/punctuation,
negation window detection, and aspect-opinion context extraction.
"""

import json
import os
import re
from typing import Dict, List, Any, Optional


DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config",
    "aspects.json"
)


class AspectExtractor:
    """
    Extracts predefined aspect categories from review text and associates
    opinion contexts and sentiment phrases using rule-based and syntactic clause splitting.
    """

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or DEFAULT_CONFIG_PATH
        self.aspect_categories: Dict[str, Dict[str, Any]] = {}
        self.negation_words: List[str] = []
        self.contrast_conjunctions: List[str] = []
        self.load_config(self.config_path)

    def load_config(self, config_path: str) -> None:
        """Loads aspect categories, keywords, and linguistic markers from configuration."""
        if not os.path.exists(config_path):
            # Fallback default configuration if file is missing
            self.aspect_categories = {
                "camera": {"keywords": ["camera", "photos", "picture", "lens", "video"], "display_name": "Camera & Imaging", "icon": "📸"},
                "battery": {"keywords": ["battery", "charging", "charger", "backup", "drain"], "display_name": "Battery & Power", "icon": "🔋"},
                "display": {"keywords": ["display", "screen", "panel", "amoled", "oled", "brightness"], "display_name": "Display & Screen", "icon": "📱"},
                "performance": {"keywords": ["performance", "speed", "processor", "lag", "gaming", "fast"], "display_name": "Performance & Speed", "icon": "⚡"},
                "price": {"keywords": ["price", "cost", "value", "money", "expensive", "cheap", "affordable"], "display_name": "Price & Value", "icon": "💰"},
                "quality": {"keywords": ["quality", "build", "durability", "material", "finish"], "display_name": "Build & Quality", "icon": "⭐"},
                "delivery": {"keywords": ["delivery", "shipping", "courier", "package", "dispatch"], "display_name": "Shipping & Delivery", "icon": "📦"},
                "customer service": {"keywords": ["customer service", "support", "warranty", "replacement", "refund"], "display_name": "Customer Service", "icon": "🎧"}
            }
            self.negation_words = ["not", "no", "never", "cannot", "hardly", "barely", "doesnt", "didnt", "isnt", "without"]
            self.contrast_conjunctions = ["but", "however", "although", "though", "yet", "still", "while"]
            return

        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.aspect_categories = data.get("aspect_categories", {})
            self.negation_words = data.get("negation_words", [])
            self.contrast_conjunctions = data.get("contrast_conjunctions", [])

    def split_into_clauses(self, text: str) -> List[str]:
        """
        Splits a compound or complex sentence into syntactic/semantic clauses
        based on punctuation (period, comma, semicolon, exclamation) and contrast conjunctions.
        """
        if not text:
            return []

        # Split on standard clause boundaries and contrast conjunctions
        pattern = r"(?:[\.\;\,]|\b(?:but|however|although|though|yet|while|whereas|and)\b)"
        raw_clauses = re.split(pattern, text, flags=re.IGNORECASE)
        
        clauses = []
        for clause in raw_clauses:
            cleaned = clause.strip()
            if cleaned and len(cleaned) > 2:
                clauses.append(cleaned)
        
        # If no clause splits were formed, return original text as single clause
        return clauses if clauses else [text.strip()]

    def extract_aspects(self, text: str) -> List[Dict[str, Any]]:
        """
        Scans review text to detect all mentioned aspect categories, locating their
        specific clause and opinion context.

        Returns:
            List of dictionaries containing:
                - aspect_category: Canonical category (e.g., 'battery')
                - display_name: Pretty name (e.g., 'Battery & Power')
                - icon: Aspect emoji icon
                - matched_keyword: Exact matched keyword (e.g., 'charging')
                - clause: Context clause mentioning the aspect
                - opinion_context: Cleaned opinion phrase / context window
        """
        if not text or not isinstance(text, str):
            return []

        clauses = self.split_into_clauses(text)
        detected_aspects = []
        seen_aspects = set()

        for category, cat_info in self.aspect_categories.items():
            keywords = cat_info.get("keywords", [])
            # Sort keywords by length descending so multi-word terms like "battery life" match before "battery"
            sorted_keywords = sorted(keywords, key=len, reverse=True)

            for kw in sorted_keywords:
                # Word boundary match to avoid false positive substrings (e.g. "expensive" inside an unrelated word)
                kw_pattern = r"\b" + re.escape(kw) + r"\b"
                
                matched = False
                for clause in clauses:
                    if re.search(kw_pattern, clause, flags=re.IGNORECASE):
                        # Extract the opinion context around this clause
                        opinion_context = clause.strip()
                        
                        detected_aspects.append({
                            "aspect_category": category,
                            "display_name": cat_info.get("display_name", category.capitalize()),
                            "icon": cat_info.get("icon", "📌"),
                            "matched_keyword": kw,
                            "clause": clause,
                            "opinion_context": opinion_context
                        })
                        seen_aspects.add(category)
                        matched = True
                        break  # Found best match in this category for this review
                
                if matched:
                    break

        return detected_aspects

    def has_negation(self, clause: str) -> bool:
        """Checks if a clause contains negation words."""
        clause_lower = clause.lower()
        for neg in self.negation_words:
            if re.search(r"\b" + re.escape(neg) + r"\b", clause_lower):
                return True
        return False
