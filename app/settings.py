"""
Centralized application settings using Pydantic Settings.

All configuration is loaded from environment variables with sensible defaults.
"""

import logging
import os
from functools import lru_cache
from typing import Literal, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Centralized application settings."""
    
    model_config = SettingsConfigDict(
        env_file=os.getenv("ENV_FILE"),  # Only load .env if ENV_FILE is explicitly set
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )
    
    # Environment
    ENV: Literal["development", "production"] = Field(default="production", description="Environment mode")
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    
    # OpenAI
    OPENAI_API_KEY: str = Field(..., description="OpenAI API key (required)")
    OPENAI_TIMEOUT: int = Field(default=30, description="OpenAI API timeout in seconds")
    OPENAI_LLM_TIMEOUT: int = Field(default=120, description="OpenAI LLM timeout in seconds")
    OPENAI_BATCH_TIMEOUT: int = Field(default=300, description="OpenAI batch operations timeout in seconds")
    OPENAI_MAX_RETRIES: int = Field(default=3, description="OpenAI max retries")
    OPENAI_RETRY_BACKOFF_BASE: float = Field(default=1.5, description="OpenAI retry backoff base")
    
    # Vectorstore
    VECTORSTORE_DIR: str = Field(
        default="var/vectorstore/faiss_index",
        description="Directory for FAISS vectorstore index"
    )
    DOCS_PATH: str = Field(
        default="data/mkdocs_docs",
        description="Path to documentation directory"
    )
    
    # Chunking
    CHUNK_SIZE: int = Field(default=1500, description="Document chunk size")
    CHUNK_OVERLAP: int = Field(default=300, description="Document chunk overlap")
    MIN_CHUNK_SIZE: int = Field(default=200, description="Minimum chunk size")
    
    # Prompt configuration
    PROMPT_TEMPLATE_MODE: Literal["legacy", "jinja"] = Field(
        default="legacy",
        description="Prompt template mode: legacy or jinja"
    )
    PROMPT_TEMPLATE: Optional[str] = Field(default=None, description="Inline Jinja2 template string")
    PROMPT_TEMPLATE_PATH: Optional[str] = Field(default=None, description="Path to Jinja2 template file")
    PROMPT_PRESET: Literal["strict", "support", "developer"] = Field(
        default="strict",
        description="Prompt preset: strict, support, or developer"
    )
    PROMPT_DIR: str = Field(default="app/prompts", description="Directory for prompt templates")
    PROMPT_MAX_CHARS: int = Field(default=40000, description="Maximum characters in rendered prompt")
    PROMPT_VALIDATE_ON_STARTUP: bool = Field(default=True, description="Validate template on startup")
    PROMPT_FAIL_HARD: bool = Field(default=False, description="Fail hard on template errors")
    PROMPT_LOG_RENDERED: bool = Field(default=False, description="Log rendered prompts")
    PROMPT_STRICT_UNDEFINED: bool = Field(default=True, description="Strict undefined variables in Jinja2")
    
    # Prompt behavior
    PROMPT_SUPPORTED_LANGUAGES: str = Field(
        default="en,de,fr,es,pt",
        description="Comma-separated list of supported languages"
    )
    PROMPT_FALLBACK_LANGUAGE: Literal["en", "de", "fr", "es", "pt"] = Field(
        default="en",
        description="Fallback language"
    )
    PROMPT_BASE_DOCS_URL: str = Field(
        default="https://docs.aqtra.io/",
        description="Base URL for documentation"
    )
    PROMPT_NOT_FOUND_MESSAGE: str = Field(
        default="Information not found in documentation database.",
        description="Message when information is not found"
    )
    PROMPT_INCLUDE_SOURCES_IN_TEXT: bool = Field(
        default=True,
        description="Include sources in text"
    )
    PROMPT_MODE: Literal["strict", "helpful"] = Field(
        default="strict",
        description="Prompt mode: strict or helpful"
    )
    PROMPT_DEFAULT_TEMPERATURE: float = Field(default=0.0, description="Default LLM temperature")
    PROMPT_DEFAULT_TOP_K: int = Field(default=4, description="Default top K for retrieval")
    PROMPT_DEFAULT_MAX_TOKENS: int = Field(default=1200, description="Default max tokens for LLM")
    
    # Latency budget settings
    STREAM_FLUSH_EVERY_N_CHARS: int = Field(default=15, description="Flush streaming buffer every N characters after first token (0 = flush immediately, first token always flushed immediately)")
    HISTORY_MAX_CHARS: int = Field(default=6000, description="Maximum characters in chat history for prompt")
    HISTORY_MAX_MESSAGES: int = Field(default=20, description="Maximum number of messages in chat history")
    STREAM_HISTORY_MAX_CHARS: int = Field(default=3000, description="Maximum characters in chat history for streaming prompt (more restrictive)")
    STREAM_HISTORY_MAX_MESSAGES: int = Field(default=10, description="Maximum number of messages in chat history for streaming (more restrictive)")
    STREAM_TOP_K: Optional[int] = Field(default=None, description="Top K for retrieval in streaming mode (overrides DEFAULT_TOP_K if set)")
    
    # RAG behavior
    STRICT_SHORT_CIRCUIT: bool = Field(
        default=True,
        description="Enable short-circuit in strict mode when no relevant sources"
    )
    NOT_FOUND_SCORE_THRESHOLD: float = Field(
        default=0.20,
        description="Score threshold for 'not found' detection"
    )
    
    # Lexical overlap gate for strict mode
    STRICT_LEXICAL_GATE_ENABLED: bool = Field(
        default=True,
        description="Enable lexical overlap gate in strict mode (requires keywords in document)"
    )
    STRICT_LEXICAL_MIN_HITS: int = Field(
        default=1,
        description="Minimum number of keyword hits required for document to pass lexical gate"
    )
    STRICT_LEXICAL_MIN_TOKEN_LEN: int = Field(
        default=4,
        description="Minimum token length for keywords in lexical gate"
    )
    RERANKING_ENABLED: bool = Field(
        default=False,
        description="Enable LLM-based reranking"
    )
    
    # Cache
    CACHE_TTL_SECONDS: int = Field(default=600, description="Cache TTL in seconds")
    CACHE_MAX_SIZE: int = Field(default=500, description="Maximum cache size")
    
    # Rate limiting
    QUERY_RATE_LIMIT: int = Field(default=30, description="Query rate limit per window")
    QUERY_RATE_WINDOW_SECONDS: int = Field(default=60, description="Query rate limit window in seconds")
    UPDATE_RATE_LIMIT: int = Field(default=3, description="Update index rate limit per window")
    UPDATE_RATE_WINDOW_SECONDS: int = Field(default=3600, description="Update index rate limit window in seconds")
    ESCALATE_RATE_LIMIT: int = Field(default=5, description="Escalate rate limit per window")
    ESCALATE_RATE_WINDOW_SECONDS: int = Field(default=3600, description="Escalate rate limit window in seconds")
    
    # API keys
    RAG_API_KEYS: Optional[str] = Field(
        default=None,
        description="Comma-separated list of API keys for /api/answer and /stream"
    )
    UPDATE_API_KEY: Optional[str] = Field(
        default=None,
        description="API key for /update_index endpoint"
    )
    
    # Index management
    INDEX_LOCK_TIMEOUT_SECONDS: int = Field(
        default=300,
        description="Index lock timeout in seconds"
    )
    
    # Database (optional)
    DATABASE_URL: Optional[str] = Field(default=None, description="Database connection URL")
    
    # Zoho Desk (optional)
    ZOHO_CLIENT_ID: Optional[str] = Field(default=None, description="Zoho Desk client ID")
    ZOHO_CLIENT_SECRET: Optional[str] = Field(default=None, description="Zoho Desk client secret")
    ZOHO_REFRESH_TOKEN: Optional[str] = Field(default=None, description="Zoho Desk refresh token")
    ZOHO_ORG_ID: Optional[str] = Field(default=None, description="Zoho Desk organization ID")
    
    # Zoho OAuth (for SalesIQ OAuth flow)
    ZOHO_REDIRECT_URI: Optional[str] = Field(
        default="https://agent.aqtra.io/oauth/callback",
        description="OAuth redirect URI"
    )
    ZOHO_ACCOUNTS_BASE_URL: Optional[str] = Field(
        default="https://accounts.zoho.com",
        description="Zoho Accounts base URL (default DC)"
    )
    ZOHO_SCOPES: Optional[str] = Field(
        default=None,
        description="OAuth scopes (comma-separated, e.g., 'SalesIQ.tickets.READ,SalesIQ.tickets.WRITE')"
    )
    
    @field_validator("PROMPT_DEFAULT_TEMPERATURE")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        """Validate temperature is in valid range."""
        return max(0.0, min(1.0, v))
    
    @field_validator("PROMPT_DEFAULT_TOP_K")
    @classmethod
    def validate_top_k(cls, v: int) -> int:
        """Validate top K is in valid range."""
        return max(1, min(10, v))
    
    @field_validator("PROMPT_DEFAULT_MAX_TOKENS")
    @classmethod
    def validate_max_tokens(cls, v: int) -> int:
        """Validate max tokens is in valid range."""
        return max(128, min(4096, v))
    
    @field_validator("NOT_FOUND_SCORE_THRESHOLD")
    @classmethod
    def validate_score_threshold(cls, v: float) -> float:
        """Validate score threshold is in valid range."""
        return max(0.0, min(1.0, v))
    
    @field_validator("PROMPT_BASE_DOCS_URL")
    @classmethod
    def validate_base_url(cls, v: str) -> str:
        """Ensure base URL ends with slash."""
        if v and not v.endswith("/"):
            return v + "/"
        return v
    
    @field_validator("OPENAI_TIMEOUT", "OPENAI_LLM_TIMEOUT", "OPENAI_BATCH_TIMEOUT")
    @classmethod
    def validate_timeout(cls, v: int) -> int:
        """Validate timeout is positive."""
        return max(1, v)
    
    @field_validator("OPENAI_MAX_RETRIES")
    @classmethod
    def validate_retries(cls, v: int) -> int:
        """Validate retries is non-negative."""
        return max(0, v)
    
    @field_validator("CHUNK_SIZE", "CHUNK_OVERLAP", "MIN_CHUNK_SIZE")
    @classmethod
    def validate_chunk_size(cls, v: int) -> int:
        """Validate chunk size is positive."""
        return max(1, v)
    
    @field_validator("CACHE_TTL_SECONDS", "CACHE_MAX_SIZE")
    @classmethod
    def validate_cache_settings(cls, v: int) -> int:
        """Validate cache settings are positive."""
        return max(1, v)
    
    @field_validator("QUERY_RATE_LIMIT", "UPDATE_RATE_LIMIT", "ESCALATE_RATE_LIMIT")
    @classmethod
    def validate_rate_limit(cls, v: int) -> int:
        """Validate rate limit is positive."""
        return max(1, v)
    
    @field_validator("QUERY_RATE_WINDOW_SECONDS", "UPDATE_RATE_WINDOW_SECONDS", "ESCALATE_RATE_WINDOW_SECONDS", "INDEX_LOCK_TIMEOUT_SECONDS")
    @classmethod
    def validate_window_seconds(cls, v: int) -> int:
        """Validate window/timeout is positive."""
        return max(1, v)
    
    def get_rag_api_keys(self) -> list[str]:
        """Get list of RAG API keys."""
        if not self.RAG_API_KEYS:
            return []
        return [key.strip() for key in self.RAG_API_KEYS.split(",") if key.strip()]
    
    def get_rag_api_keys_set(self) -> set[str]:
        """
        Get set of RAG API keys (for fast lookup).
        
        Returns:
            Set of API keys, or empty set if RAG_API_KEYS is not set (open mode)
        """
        if not self.RAG_API_KEYS:
            return set()
        return {key.strip() for key in self.RAG_API_KEYS.split(",") if key.strip()}


@lru_cache()
def get_settings() -> Settings:
    """
    Get application settings (cached).
    
    Returns:
        Settings instance
    """
    try:
        settings = Settings()
        logger.info("Settings loaded successfully")
        return settings
    except Exception as e:
        logger.error(f"Failed to load settings: {e}")
        raise

