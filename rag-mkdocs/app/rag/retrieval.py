"""
Retrieval module: retriever building logic.
"""

import logging
import os
from typing import Optional

try:
    from langchain.retrievers import ContextualCompressionRetriever
    from langchain.retrievers.document_compressors import LLMChainExtractor
    RERANKING_AVAILABLE = True
except ImportError:
    RERANKING_AVAILABLE = False

from app.infra.openai_utils import get_chat_llm

logger = logging.getLogger(__name__)

RERANKING_ENABLED = os.getenv("RERANKING_ENABLED", "0").lower() in ("1", "true", "yes")


def build_retriever(
    vectorstore,
    k: int,
    model: str = "gpt-4o",
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None
):
    """
    Build retriever from vectorstore with optional reranking.
    
    Args:
        vectorstore: FAISS vectorstore instance
        k: Number of relevant chunks to retrieve
        model: OpenAI model for reranking (if enabled)
        temperature: Generation temperature for reranking LLM
        max_tokens: Maximum tokens for reranking LLM
        
    Returns:
        Retriever instance (with or without reranking)
    """
    # Limit k range
    effective_k = max(1, min(10, k))
    
    # Create base retriever
    if RERANKING_ENABLED:
        raw_k = max(effective_k * 2, 8)
        logger.info("Creating base retriever with k=%s for reranking...", raw_k)
        base_retriever = vectorstore.as_retriever(search_kwargs={"k": raw_k})
    else:
        logger.info("Reranking disabled, using base retriever with k=%s", effective_k)
        base_retriever = vectorstore.as_retriever(search_kwargs={"k": effective_k})
    
    # Apply reranking if enabled
    if RERANKING_ENABLED and RERANKING_AVAILABLE:
        logger.info(
            "Initializing LLM for retriever: %s (temperature=%s, max_tokens=%s, reranking_enabled=%s)...",
            model,
            temperature,
            max_tokens,
            RERANKING_ENABLED,
        )
        llm = get_chat_llm(
            temperature=temperature,
            model=model,
            max_tokens=max_tokens,
        )
        
        try:
            compressor = LLMChainExtractor.from_llm(llm)
            retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=base_retriever,
            )
            logger.info("Reranking enabled, final k=%s", effective_k)
            return retriever
        except Exception as e:
            logger.warning("Error creating reranker: %s, using base retriever", e)
            return vectorstore.as_retriever(search_kwargs={"k": effective_k})
    else:
        if RERANKING_ENABLED and not RERANKING_AVAILABLE:
            logger.info("Reranking requested but not available in current LangChain version; using base retriever")
        return vectorstore.as_retriever(search_kwargs={"k": effective_k})

