"""
Unit tests for centralized settings.
"""

import os
import pytest
from unittest.mock import patch

from app.settings import Settings, get_settings


def test_settings_defaults():
    """Test that settings have correct defaults."""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        settings = Settings()
        assert settings.VECTORSTORE_DIR == "var/vectorstore/faiss_index"
        assert settings.DOCS_PATH == "data/mkdocs_docs"
        assert settings.ENV == "production"
        assert settings.LOG_LEVEL == "INFO"
        assert settings.PROMPT_TEMPLATE_MODE == "legacy"
        assert settings.PROMPT_PRESET == "strict"
        assert settings.CACHE_TTL_SECONDS == 600
        assert settings.CACHE_MAX_SIZE == 500


def test_settings_from_env():
    """Test that settings read from environment variables."""
    env_vars = {
        "OPENAI_API_KEY": "test-key",
        "VECTORSTORE_DIR": "custom/path",
        "LOG_LEVEL": "DEBUG",
        "PROMPT_TEMPLATE_MODE": "jinja",
        "CACHE_TTL_SECONDS": "1200",
    }
    
    with patch.dict(os.environ, env_vars, clear=False):
        # Clear cache to force reload
        get_settings.cache_clear()
        settings = get_settings()
        assert settings.VECTORSTORE_DIR == "custom/path"
        assert settings.LOG_LEVEL == "DEBUG"
        assert settings.PROMPT_TEMPLATE_MODE == "jinja"
        assert settings.CACHE_TTL_SECONDS == 1200


def test_settings_validation():
    """Test that settings validate values correctly."""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        settings = Settings()
        
        # Temperature should be clamped
        assert 0.0 <= settings.PROMPT_DEFAULT_TEMPERATURE <= 1.0
        
        # Top K should be clamped
        assert 1 <= settings.PROMPT_DEFAULT_TOP_K <= 10
        
        # Max tokens should be clamped
        assert 128 <= settings.PROMPT_DEFAULT_MAX_TOKENS <= 4096
        
        # Score threshold should be clamped
        assert 0.0 <= settings.NOT_FOUND_SCORE_THRESHOLD <= 1.0
        
        # Base URL should end with slash
        assert settings.PROMPT_BASE_DOCS_URL.endswith("/")


def test_get_settings_cached():
    """Test that get_settings() is cached."""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        settings1 = get_settings()
        settings2 = get_settings()
        assert settings1 is settings2  # Same instance due to caching


def test_get_rag_api_keys():
    """Test that get_rag_api_keys() returns list."""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        settings = Settings()
        assert isinstance(settings.get_rag_api_keys(), list)
        
        # Test with keys
        with patch.dict(os.environ, {"RAG_API_KEYS": "key1,key2,key3"}):
            get_settings.cache_clear()
            settings = get_settings()
            keys = settings.get_rag_api_keys()
            assert len(keys) == 3
            assert "key1" in keys
            assert "key2" in keys
            assert "key3" in keys


def test_get_rag_api_keys_set():
    """Test that get_rag_api_keys_set() returns set."""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        settings = Settings()
        # Empty -> open mode
        assert settings.get_rag_api_keys_set() == set()
        
        # Test with keys (with spaces)
        with patch.dict(os.environ, {"RAG_API_KEYS": "a,b, c"}):
            get_settings.cache_clear()
            settings = get_settings()
            keys_set = settings.get_rag_api_keys_set()
            assert isinstance(keys_set, set)
            assert len(keys_set) == 3
            assert "a" in keys_set
            assert "b" in keys_set
            assert "c" in keys_set

