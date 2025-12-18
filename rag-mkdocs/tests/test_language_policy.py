"""
Unit tests for language_policy module.
"""

import pytest

from app.core.language_policy import (
    normalize_language,
    select_output_language,
    parse_accept_language,
)


def test_normalize_language_basic():
    """Test basic language normalization."""
    assert normalize_language("en") == "en"
    assert normalize_language("EN") == "en"
    assert normalize_language("fr") == "fr"
    assert normalize_language("de") == "de"
    assert normalize_language("es") == "es"
    assert normalize_language("pt") == "pt"


def test_normalize_language_locale():
    """Test locale format normalization."""
    assert normalize_language("en-US") == "en"
    assert normalize_language("fr-FR") == "fr"
    assert normalize_language("de-DE") == "de"
    assert normalize_language("es-ES") == "es"
    assert normalize_language("pt-BR") == "pt"
    assert normalize_language("pt-PT") == "pt"


def test_normalize_language_unknown():
    """Test unknown language defaults to English."""
    assert normalize_language("ru") == "en"
    assert normalize_language("zh") == "en"
    assert normalize_language("ja") == "en"
    assert normalize_language("unknown") == "en"


def test_normalize_language_none_empty():
    """Test None and empty string default to English."""
    assert normalize_language(None) == "en"
    assert normalize_language("") == "en"
    assert normalize_language("   ") == "en"


def test_parse_accept_language():
    """Test Accept-Language header parsing."""
    # Single language
    assert parse_accept_language("en") == ["en"]
    assert parse_accept_language("fr") == ["fr"]
    
    # Multiple languages
    assert parse_accept_language("es-ES,es;q=0.9,en;q=0.8") == ["es", "en"]
    assert parse_accept_language("de-DE,de;q=0.9,en;q=0.8") == ["de", "en"]
    
    # With quality values
    assert parse_accept_language("fr;q=0.9,en;q=0.8") == ["fr", "en"]
    
    # Only allowed languages
    assert parse_accept_language("ru-RU,en;q=0.9") == ["en"]
    assert parse_accept_language("zh-CN,fr;q=0.9") == ["fr"]
    
    # Empty/None
    assert parse_accept_language(None) == []
    assert parse_accept_language("") == []


def test_select_output_language_passthrough_priority():
    """Test passthrough.language has highest priority."""
    passthrough = {"language": "fr"}
    context_hint = {"language": "de"}
    accept_language = "es-ES,es;q=0.9"
    
    lang, reason = select_output_language(
        passthrough=passthrough,
        context_hint=context_hint,
        accept_language_header=accept_language
    )
    
    assert lang == "fr"
    assert reason == "passthrough.language"


def test_select_output_language_passthrough_lang():
    """Test passthrough.lang (alternative key)."""
    passthrough = {"lang": "de"}
    
    lang, reason = select_output_language(
        passthrough=passthrough,
        context_hint=None,
        accept_language_header=None
    )
    
    assert lang == "de"
    assert reason == "passthrough.language"


def test_select_output_language_context_hint():
    """Test context_hint.language is used if passthrough absent."""
    context_hint = {"language": "es"}
    accept_language = "fr-FR,fr;q=0.9"
    
    lang, reason = select_output_language(
        passthrough=None,
        context_hint=context_hint,
        accept_language_header=accept_language
    )
    
    assert lang == "es"
    assert reason == "context_hint.language"


def test_select_output_language_accept_language():
    """Test Accept-Language header is used if passthrough and context_hint absent."""
    accept_language = "pt-BR,pt;q=0.9,en;q=0.8"
    
    lang, reason = select_output_language(
        passthrough=None,
        context_hint=None,
        accept_language_header=accept_language
    )
    
    assert lang == "pt"
    assert reason == "accept_language"


def test_select_output_language_default():
    """Test default English if nothing provided."""
    lang, reason = select_output_language(
        passthrough=None,
        context_hint=None,
        accept_language_header=None
    )
    
    assert lang == "en"
    assert reason == "default"


def test_select_output_language_invalid_passthrough():
    """Test invalid passthrough language falls back to next priority."""
    passthrough = {"language": "ru"}  # Not allowed
    context_hint = {"language": "de"}
    
    lang, reason = select_output_language(
        passthrough=passthrough,
        context_hint=context_hint,
        accept_language_header=None
    )
    
    # Should use context_hint since passthrough is invalid
    assert lang == "de"
    assert reason == "context_hint.language"


def test_select_output_language_invalid_all():
    """Test all invalid languages default to English."""
    passthrough = {"language": "ru"}
    context_hint = {"language": "zh"}
    accept_language = "ja-JP,ja;q=0.9"
    
    lang, reason = select_output_language(
        passthrough=passthrough,
        context_hint=context_hint,
        accept_language_header=accept_language
    )
    
    assert lang == "en"
    assert reason == "default"



