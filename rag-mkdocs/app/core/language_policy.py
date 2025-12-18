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


def normalize_language(lang: Optional[str]) -> str:
    """
    Normalize language code to allowed format.
    
    Accepts: "en", "EN", "en-US", "fr-FR", "de", "es", "pt", etc.
    Returns: "en"|"fr"|"de"|"es"|"pt" (default: "en")
    
    Args:
        lang: Language code (can be None, empty, or various formats)
        
    Returns:
        Normalized language code (one of ALLOWED_LANGUAGES or DEFAULT_LANGUAGE)
    """
    if not lang:
        return DEFAULT_LANGUAGE
    
    # Convert to lowercase and extract base language
    lang_lower = lang.lower().strip()
    
    # Handle locale formats like "en-US", "fr-FR", "pt-BR"
    if "-" in lang_lower:
        base_lang = lang_lower.split("-")[0]
    else:
        base_lang = lang_lower
    
    # Check if it's in allowed languages
    if base_lang in ALLOWED_LANGUAGES:
        return base_lang
    
    # Not in allowed list -> default to English
    logger.debug(f"Language '{lang}' not in allowed list, defaulting to '{DEFAULT_LANGUAGE}'")
    return DEFAULT_LANGUAGE


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
        
        # Only add if it's not already in list
        if normalized not in languages:
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
    1. passthrough.language (or passthrough.lang)
    2. context_hint.language
    3. Accept-Language header (first valid)
    4. default "en"
    
    Args:
        passthrough: Passthrough dictionary (may contain "language" or "lang")
        context_hint: Context hint dictionary (may contain "language")
        accept_language_header: Accept-Language header value
        
    Returns:
        Tuple (language_code, reason)
        reason: "passthrough.language", "context_hint.language", "accept_language", or "default"
    """
    # Priority 1: passthrough.language or passthrough.lang
    if passthrough:
        lang = passthrough.get("language") or passthrough.get("lang")
        if lang:
            # Check if original language is in allowed list (before normalization)
            lang_lower = str(lang).lower().strip()
            if "-" in lang_lower:
                base_lang = lang_lower.split("-")[0]
            else:
                base_lang = lang_lower
            
            if base_lang in ALLOWED_LANGUAGES:
                normalized = normalize_language(lang)
                return normalized, "passthrough.language"
    
    # Priority 2: context_hint.language
    if context_hint and context_hint.get("language"):
        lang = context_hint["language"]
        # Check if original language is in allowed list
        lang_lower = str(lang).lower().strip()
        if "-" in lang_lower:
            base_lang = lang_lower.split("-")[0]
        else:
            base_lang = lang_lower
        
        if base_lang in ALLOWED_LANGUAGES:
            normalized = normalize_language(lang)
            return normalized, "context_hint.language"
    
    # Priority 3: Accept-Language header
    if accept_language_header:
        languages = parse_accept_language(accept_language_header)
        if languages:
            return languages[0], "accept_language"
    
    # Priority 4: default
    return DEFAULT_LANGUAGE, "default"



