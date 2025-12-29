"""
Utilities for working with OpenAI API.

Centralized error handling, timeouts and retry logic.
"""

import asyncio
import logging
import os
import time
from typing import Optional, Callable, Any, AsyncIterator, List

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


class CachedEmbeddings(OpenAIEmbeddings):
    """OpenAIEmbeddings wrapper with embedding cache."""
    
    def embed_query(self, text: str) -> List[float]:
        """Embed query with cache support."""
        from app.infra.embedding_cache import embedding_cache
        
        # Check cache
        embedding_model = self.model or "text-embedding-3-small"
        cached_embedding = embedding_cache.get(text, embedding_model)
        
        if cached_embedding:
            logger.debug(f"Using cached embedding for query: {text[:50]}...")
            return cached_embedding
        
        # Compute embedding
        embedding = super().embed_query(text)
        
        # Cache it
        embedding_cache.set(text, embedding, embedding_model)
        logger.debug(f"Computed and cached embedding for query: {text[:50]}...")
        
        return embedding


def get_embeddings_client(timeout: Optional[int] = None) -> OpenAIEmbeddings:
    """
    Creates client for OpenAI embeddings with timeout settings and cache support.
    
    Args:
        timeout: Timeout in seconds. If not specified, OPENAI_TIMEOUT is used.
                 For batch operations use OPENAI_BATCH_TIMEOUT or pass explicitly.
    
    Returns:
        OpenAIEmbeddings client (with cache support)
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set in environment")
    
    if timeout is None:
        timeout = OPENAI_TIMEOUT
    
    return CachedEmbeddings(
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


async def stream_chat_completion(
    llm: ChatOpenAI,
    messages: list,
    **kwargs
) -> AsyncIterator[str]:
    """
    Stream chat completion tokens from OpenAI.
    
    Args:
        llm: ChatOpenAI instance (must support streaming)
        messages: List of message dicts (system, human, etc.) or LangChain messages
        **kwargs: Additional arguments for LLM invocation
        
    Yields:
        Token deltas (strings) as they arrive from the API
        
    Note:
        Falls back to non-streaming if streaming is not supported.
    """
    try:
        # Convert dict messages to LangChain format if needed
        from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
        
        langchain_messages = []
        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "system":
                    langchain_messages.append(SystemMessage(content=content))
                elif role == "assistant":
                    langchain_messages.append(AIMessage(content=content))
                else:
                    langchain_messages.append(HumanMessage(content=content))
            else:
                langchain_messages.append(msg)
        
        # Check if LLM supports streaming
        if not hasattr(llm, 'astream') and not hasattr(llm, 'stream'):
            # Fallback: use regular invoke and yield chunks
            logger.warning("LLM does not support streaming, using fallback")
            result = await llm.ainvoke(langchain_messages, **kwargs)
            content = result.content if hasattr(result, 'content') else str(result)
            # Yield in chunks for pseudo-streaming
            chunk_size = 10  # Small chunks for better UX
            for i in range(0, len(content), chunk_size):
                yield content[i:i + chunk_size]
                await asyncio.sleep(0.01)  # Small delay for UX
            return
        
        # Use async streaming if available
        if hasattr(llm, 'astream'):
            async for chunk in llm.astream(langchain_messages, **kwargs):
                if hasattr(chunk, 'content') and chunk.content:
                    yield chunk.content
                elif isinstance(chunk, dict) and 'content' in chunk:
                    yield chunk['content']
                elif isinstance(chunk, str):
                    yield chunk
        elif hasattr(llm, 'stream'):
            # Synchronous streaming (fallback)
            import asyncio
            for chunk in llm.stream(langchain_messages, **kwargs):
                if hasattr(chunk, 'content') and chunk.content:
                    yield chunk.content
                elif isinstance(chunk, dict) and 'content' in chunk:
                    yield chunk['content']
                elif isinstance(chunk, str):
                    yield chunk
                # Small delay to prevent blocking
                await asyncio.sleep(0.001)
    except Exception as e:
        logger.error(f"Error in stream_chat_completion: {e}", exc_info=True)
        # Fallback: return empty or error message
        yield f"[Error: {str(e)}]"


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

