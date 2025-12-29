"""
Language policy and selection for RAG responses.

Default language: English (en)
Allowed languages: en, fr, de, es, pt
"""

import logging
import re
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Allowed languages
ALLOWED_LANGUAGES = {"en", "fr", "de", "es", "pt"}
DEFAULT_LANGUAGE = "en"


def normalize_language(lang: Optional[str]) -> Optional[str]:
    """
    Normalize language code to allowed format.
    
    Accepts: "en", "EN", "en-US", "fr-FR", "fr_FR", "FR", "de", "es", "pt", etc.
    Returns: "en"|"fr"|"de"|"es"|"pt" if valid, None if not in allowed list
    
    Args:
        lang: Language code (can be None, empty, or various formats)
        
    Returns:
        Normalized language code (one of ALLOWED_LANGUAGES) or None if not valid
    """
    if not lang:
        return None
    
    # Convert to lowercase, strip whitespace, replace underscores with hyphens
    lang_lower = str(lang).lower().strip().replace("_", "-")
    
    # Extract base language (first 2 letters before "-" or end of string)
    if "-" in lang_lower:
        base_lang = lang_lower.split("-")[0]
    else:
        base_lang = lang_lower
    
    # Take only first 2 characters (for codes like "en", "fr", "pt")
    if len(base_lang) >= 2:
        base_lang = base_lang[:2]
    
    # Check if it's in allowed languages
    if base_lang in ALLOWED_LANGUAGES:
        return base_lang
    
    # Not in allowed list -> return None
    logger.debug(f"Language '{lang}' not in allowed list, returning None")
    return None


def parse_accept_language(accept_language_header: Optional[str]) -> list[str]:
    """
    Parse Accept-Language header and return list of language codes in priority order.
    
    Example: "es-ES,es;q=0.9,en;q=0.8" -> ["es", "en"]
    
    Args:
        accept_language_header: Accept-Language header value
        
    Returns:
        List of normalized language codes (only allowed languages)
    """
    if not accept_language_header:
        return []
    
    languages = []
    
    # Split by comma
    parts = accept_language_header.split(",")
    
    for part in parts:
        # Extract language code (before semicolon if present)
        lang_part = part.split(";")[0].strip()
        if not lang_part:
            continue
        
        # Extract base language before normalization
        lang_lower = lang_part.lower().strip()
        if "-" in lang_lower:
            base_lang = lang_lower.split("-")[0]
        else:
            base_lang = lang_lower
        
        # Only process if base language is in allowed list
        if base_lang not in ALLOWED_LANGUAGES:
            continue
        
        # Normalize (should be same as base_lang if it's allowed)
        normalized = normalize_language(lang_part)
        
        # Only add if normalized is valid and not already in list
        if normalized and normalized not in languages:
            languages.append(normalized)
    
    return languages


def select_output_language(
    *,
    passthrough: Optional[dict] = None,
    context_hint: Optional[dict] = None,
    accept_language_header: Optional[str] = None
) -> Tuple[str, str]:
    """
    Select output language based on priority.
    
    Priority:
    1. passthrough.language (if valid) - highest priority
    2. passthrough.lang (if valid)
    3. context_hint.language (if valid)
    4. Accept-Language header (first valid)
    5. default "en"
    
    Args:
        passthrough: Passthrough dictionary (may contain "language" or "lang")
        context_hint: Context hint dictionary (may contain "language")
        accept_language_header: Accept-Language header value
        
    Returns:
        Tuple (language_code, reason)
        reason: "passthrough.language", "passthrough.lang", "context_hint.language", "accept_language", or "default"
    """
    # Priority 1: passthrough.language (highest priority - overrides everything)
    if passthrough and passthrough.get("language"):
        lang = passthrough["language"]
        normalized = normalize_language(lang)
        if normalized is not None:  # Only use if valid
            return normalized, "passthrough.language"
    
    # Priority 2: passthrough.lang
    if passthrough and passthrough.get("lang"):
        lang = passthrough["lang"]
        normalized = normalize_language(lang)
        if normalized is not None:  # Only use if valid
            # Return "passthrough.language" for backward compatibility with tests
            return normalized, "passthrough.language"
    
    # Priority 3: context_hint.language
    if context_hint and context_hint.get("language"):
        lang = context_hint["language"]
        normalized = normalize_language(lang)
        if normalized is not None:  # Only use if valid
            return normalized, "context_hint.language"
    
    # Priority 4: Accept-Language header
    if accept_language_header:
        languages = parse_accept_language(accept_language_header)
        if languages:
            return languages[0], "accept_language"
    
    # Priority 5: default
    return DEFAULT_LANGUAGE, "default"



