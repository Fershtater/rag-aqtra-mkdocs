"""
Utilities for working with OpenAI API.

Centralized error handling, timeouts and retry logic.
"""

import logging
import os
import time
from typing import Optional, Callable, Any

from langchain_openai import OpenAIEmbeddings, ChatOpenAI

logger = logging.getLogger(__name__)

# Timeout settings (in seconds)
OPENAI_TIMEOUT = int(os.getenv("OPENAI_TIMEOUT", "30"))
# Timeout for LLM requests (Chat API) - default 2 minutes for large contexts
OPENAI_LLM_TIMEOUT = int(os.getenv("OPENAI_LLM_TIMEOUT", "120"))
# Timeout for batch operations (index creation) - default 5 minutes
OPENAI_BATCH_TIMEOUT = int(os.getenv("OPENAI_BATCH_TIMEOUT", "300"))
OPENAI_MAX_RETRIES = int(os.getenv("OPENAI_MAX_RETRIES", "3"))
OPENAI_RETRY_BACKOFF_BASE = float(os.getenv("OPENAI_RETRY_BACKOFF_BASE", "1.5"))


def get_embeddings_client(timeout: Optional[int] = None) -> OpenAIEmbeddings:
    """
    Creates client for OpenAI embeddings with timeout settings.
    
    Args:
        timeout: Timeout in seconds. If not specified, OPENAI_TIMEOUT is used.
                 For batch operations use OPENAI_BATCH_TIMEOUT or pass explicitly.
    
    Returns:
        OpenAIEmbeddings client
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set in environment")
    
    if timeout is None:
        timeout = OPENAI_TIMEOUT
    
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=api_key,
        timeout=timeout,
        max_retries=OPENAI_MAX_RETRIES
    )


def get_chat_llm(
    temperature: float = 0.0,
    model: str = "gpt-4o-mini",
    max_tokens: Optional[int] = None,
    timeout: Optional[int] = None
) -> ChatOpenAI:
    """
    Creates client for OpenAI Chat API with timeout settings.
    
    Args:
        temperature: Generation temperature
        model: OpenAI model
        max_tokens: Maximum number of tokens (optional)
        timeout: Timeout in seconds. If not specified, OPENAI_LLM_TIMEOUT is used.
        
    Returns:
        ChatOpenAI client
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set in environment")
    
    if timeout is None:
        timeout = OPENAI_LLM_TIMEOUT
    
    kwargs = {
        "model": model,
        "temperature": temperature,
        "api_key": api_key,
        "timeout": timeout,
        "max_retries": OPENAI_MAX_RETRIES
    }
    
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    
    return ChatOpenAI(**kwargs)


def with_retries(
    fn: Callable,
    *args,
    max_retries: int = OPENAI_MAX_RETRIES,
    backoff_base: float = OPENAI_RETRY_BACKOFF_BASE,
    **kwargs
) -> Any:
    """
    Executes function with exponential backoff on errors.
    
    Args:
        fn: Function to execute
        *args: Positional arguments
        max_retries: Maximum number of attempts
        backoff_base: Base for exponential backoff
        **kwargs: Named arguments
        
    Returns:
        Function execution result
        
    Raises:
        Last exception if all attempts exhausted
    """
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_exception = e
            if attempt < max_retries - 1:
                wait_time = backoff_base ** attempt
                logger.warning(
                    f"Error calling {fn.__name__} (attempt {attempt + 1}/{max_retries}): {e}. "
                    f"Retrying after {wait_time:.2f}s"
                )
                time.sleep(wait_time)
            else:
                logger.error(
                    f"All attempts exhausted for {fn.__name__}: {e}",
                    exc_info=True
                )
    
    raise last_exception

