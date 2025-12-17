"""
Common answering logic for v2 API endpoints.
"""

import asyncio
import hashlib
import logging
import os
from time import time
from typing import Dict, List, Optional, Tuple

from app.api.v2_models import (
    AnswerRequest,
    AnswerResponse,
    MetricsPayload,
    Source,
    generate_source_id,
    parse_history_to_text,
)
from app.core.markdown_utils import build_doc_url
from app.core.prompt_config import PromptSettings, detect_response_language, load_prompt_settings_from_env
from app.infra.cache import response_cache
from app.infra.conversations import append_message, get_or_create_conversation, load_history

logger = logging.getLogger(__name__)


def normalize_sources(
    result: Dict,
    prompt_settings: PromptSettings,
    max_snippet_length: int = 300
) -> List[Source]:
    """
    Normalize sources from RAG chain result to v2 Source format.
    
    Args:
        result: RAG chain result dictionary
        prompt_settings: Prompt settings for URL building
        max_snippet_length: Maximum snippet length (default 300)
        
    Returns:
        List of Source objects
    """
    sources = []
    seen_ids = set()
    
    # Extract sources from context
    context_docs = []
    if "context" in result:
        context_docs = result["context"]
        if not isinstance(context_docs, list):
            context_docs = [context_docs]
    
    # Fallback: try source_documents
    if not context_docs and "source_documents" in result:
        context_docs = result["source_documents"]
        if not isinstance(context_docs, list):
            context_docs = [context_docs]
    
    for idx, doc in enumerate(context_docs):
        if not hasattr(doc, "metadata") or not doc.metadata:
            continue
        
        source_path = doc.metadata.get("source", "unknown")
        section_anchor = doc.metadata.get("section_anchor")
        section_title = doc.metadata.get("section_title")
        filename = doc.metadata.get("filename", source_path.split("/")[-1])
        
        # Generate stable ID
        source_id = generate_source_id(source_path, section_anchor, idx)
        if source_id in seen_ids:
            continue
        seen_ids.add(source_id)
        
        # Build URL
        url = build_doc_url(
            prompt_settings.base_docs_url,
            source_path,
            section_anchor
        )
        
        # Extract snippet (first ~300 chars, clean whitespace)
        snippet = doc.page_content.strip()[:max_snippet_length]
        snippet = " ".join(snippet.split())  # Normalize whitespace
        
        # Get score if available
        score = doc.metadata.get("score")
        if score is not None:
            try:
                score = float(score)
            except (ValueError, TypeError):
                score = None
        
        # Build meta
        meta = {
            "source": source_path,
            "filename": filename,
        }
        if section_title:
            meta["section_title"] = section_title
        if section_anchor:
            meta["section_anchor"] = section_anchor
        if "page_url" in doc.metadata:
            meta["page_url"] = doc.metadata["page_url"]
        if "page_title" in doc.metadata:
            meta["page_title"] = doc.metadata["page_title"]
        
        # Title: prefer section_title, fallback to filename
        title = section_title or filename or "Unknown"
        
        sources.append(Source(
            id=source_id,
            title=title,
            url=url,
            snippet=snippet,
            score=score,
            meta=meta
        ))
    
    return sources


async def generate_answer(
    rag_chain,
    question: str,
    request_id: str,
    prompt_settings: PromptSettings,
    chat_history: str = "",
    response_language: Optional[str] = None,
    top_k_override: Optional[int] = None,
    context_hint: Optional[Dict] = None
) -> Tuple[str, List[Source], bool]:
    """
    Generate answer using RAG chain.
    
    Args:
        rag_chain: RAG chain instance
        question: User question
        request_id: Request ID for logging
        prompt_settings: Prompt settings
        chat_history: Formatted chat history text (empty string if none)
        response_language: Response language code (auto-detected if None)
        top_k_override: Override top_k (if None, uses default)
        context_hint: Context hint dict (page_url, page_title, language)
        
    Returns:
        Tuple (answer, sources, not_found)
    """
    # Detect language if not provided
    if response_language is None:
        if context_hint and context_hint.get("language"):
            response_language = context_hint["language"]
        else:
            response_language = detect_response_language(
                question,
                supported=set(prompt_settings.supported_languages),
                fallback=prompt_settings.fallback_language
            )
    
    # Build chain input
    chain_input = {
        "input": question,
        "response_language": response_language,
        "chat_history": chat_history or ""  # Empty string if None
    }
    
    # Call RAG chain
    try:
        result = await asyncio.to_thread(rag_chain.invoke, chain_input)
    except Exception as e:
        logger.error(f"[{request_id}] Error calling RAG chain: {e}", exc_info=True)
        raise
    
    # Extract answer
    answer = result.get("answer", "Failed to generate answer")
    
    # Normalize sources
    sources = normalize_sources(result, prompt_settings)
    
    # Determine not_found
    not_found_score_threshold = float(os.getenv("NOT_FOUND_SCORE_THRESHOLD", "0.20"))
    not_found = len(sources) == 0
    
    if sources:
        scores = [s.score for s in sources if s.score is not None]
        if scores:
            top_score = max(scores)
            if top_score < not_found_score_threshold:
                not_found = True
    
    return answer, sources, not_found


async def process_answer_request(
    rag_chain,
    request: AnswerRequest,
    request_id: str,
    prompt_settings: PromptSettings,
    db_sessionmaker,
    client_ip: str,
    user_agent: Optional[str] = None
) -> AnswerResponse:
    """
    Process answer request (common logic for /api/answer and /stream).
    
    Args:
        rag_chain: RAG chain instance
        request: Answer request
        request_id: Request ID
        prompt_settings: Prompt settings
        db_sessionmaker: Database sessionmaker (can be None)
        client_ip: Client IP address
        user_agent: User agent string
        
    Returns:
        AnswerResponse
    """
    start_time = time()
    
    # Get or create conversation ID
    conversation_id = await get_or_create_conversation(
        db_sessionmaker,
        request.conversation_id
    )
    
    # Load or parse history
    chat_history_text = ""
    if request.history:
        # Use provided history
        chat_history_text = parse_history_to_text(request.history)
    elif conversation_id and db_sessionmaker:
        # Load from DB
        history_list = await load_history(db_sessionmaker, conversation_id, limit=20)
        if history_list:
            chat_history_text = parse_history_to_text(history_list)
    
    # Build cache key (include history signature)
    history_signature = hashlib.md5(chat_history_text.encode()).hexdigest()[:8] if chat_history_text else "no_history"
    detector_version = "v1"
    reranking_enabled = os.getenv("RERANKING_ENABLED", "0").lower() in ("1", "true", "yes")
    top_k_value = prompt_settings.default_top_k
    if request.retrieval and request.retrieval.top_k:
        top_k_value = request.retrieval.top_k
    
    settings_signature = (
        f"mode={prompt_settings.mode}_"
        f"top_k={top_k_value}_"
        f"temp={prompt_settings.default_temperature}_"
        f"max_tokens={getattr(prompt_settings, 'default_max_tokens', None)}_"
        f"supported={','.join(sorted(prompt_settings.supported_languages))}_"
        f"fallback={prompt_settings.fallback_language}_"
        f"rerank={reranking_enabled}_"
        f"detector={detector_version}_"
        f"history={history_signature}"
    )
    
    cache_key = response_cache._generate_key(request.question, settings_signature)
    
    # Check cache (only if no history to avoid stale responses)
    cached_result: Optional[AnswerResponse] = None
    if not chat_history_text:
        cached_result = response_cache.get(cache_key)
    
    if cached_result:
        logger.debug(f"[{request_id}] Cache hit for query")
        latency_ms = int((time() - start_time) * 1000)
        
        # Update conversation_id in cached response
        cached_result.conversation_id = conversation_id
        
        # Save to conversation history if DB available
        if db_sessionmaker:
            await append_message(db_sessionmaker, conversation_id, "user", request.question)
            await append_message(db_sessionmaker, conversation_id, "assistant", cached_result.answer)
        
        return cached_result
    
    # Generate answer
    top_k_override = None
    if request.retrieval and request.retrieval.top_k:
        top_k_override = request.retrieval.top_k
    
    context_hint_dict = None
    if request.context_hint:
        context_hint_dict = request.context_hint.dict()
    
    answer, sources, not_found = await generate_answer(
        rag_chain,
        request.question,
        request_id,
        prompt_settings,
        chat_history=chat_history_text,
        top_k_override=top_k_override,
        context_hint=context_hint_dict
    )
    
    latency_sec = time() - start_time
    latency_ms = int(latency_sec * 1000)
    
    # Build response
    response = AnswerResponse(
        answer=answer,
        sources=sources,
        conversation_id=conversation_id,
        request_id=request_id,
        not_found=not_found,
        metrics=MetricsPayload(
            latency_ms=latency_ms,
            cache_hit=False,
            retrieved_chunks=len(sources),
            model=None  # TODO: extract from LLM response if available
        )
    )
    
    # Save to cache (only if no history)
    if not chat_history_text:
        response_cache.set(cache_key, response)
    
    # Save to conversation history if DB available
    if db_sessionmaker:
        await append_message(db_sessionmaker, conversation_id, "user", request.question)
        await append_message(db_sessionmaker, conversation_id, "assistant", answer)
    
    logger.info(
        f"[{request_id}] Answer generated: conversation_id={conversation_id}, "
        f"sources={len(sources)}, not_found={not_found}, latency_ms={latency_ms}"
    )
    
    return response

