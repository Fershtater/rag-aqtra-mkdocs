"""
Common answering logic for v2 API endpoints.

This module provides backward-compatible functions that use services internally.
For new code, prefer using services directly.
"""

import asyncio
import hashlib
import json
import logging
import os
import time
from time import time as time_func
from typing import Any, Dict, List, Optional, Tuple

from app.api.schemas.v2 import (
    AnswerRequest,
    AnswerResponse,
    MetricsPayload,
    Source,
    generate_source_id,
    parse_history_to_text,
)
from app.core.markdown_utils import build_doc_url
from app.core.prompt_config import (
    PromptSettings,
    detect_response_language,
    get_prompt_template_content,
    is_jinja_mode,
    build_system_prompt,
)
from app.infra.cache import response_cache
from app.infra.conversations import append_message, get_or_create_conversation, load_history
from app.services.answer_service import AnswerService
from app.services.conversation_service import ConversationService
from app.services.prompt_service import PromptService

logger = logging.getLogger(__name__)

# Create default service instances (will be initialized per-request with proper dependencies)
# These are module-level for backward compatibility
_default_prompt_service = PromptService()


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


def build_source_namespace(
    context_docs: List,
    prompt_settings: PromptSettings
) -> Dict[str, Any]:
    """
    Build source namespace for Jinja2 template.
    
    Args:
        context_docs: List of Document objects from retrieval
        prompt_settings: Prompt settings
        
    Returns:
        Dictionary with source namespace (content, count, documents)
    """
    # Extract content (concatenated page_content)
    content_parts = []
    documents = []
    
    for idx, doc in enumerate(context_docs):
        if not hasattr(doc, "page_content"):
            continue
        
        content_parts.append(doc.page_content)
        
        # Build document metadata
        source_path = doc.metadata.get("source", "unknown") if hasattr(doc, "metadata") and doc.metadata else "unknown"
        section_anchor = doc.metadata.get("section_anchor") if hasattr(doc, "metadata") and doc.metadata else None
        section_title = doc.metadata.get("section_title") if hasattr(doc, "metadata") and doc.metadata else None
        filename = doc.metadata.get("filename", source_path.split("/")[-1]) if hasattr(doc, "metadata") and doc.metadata else source_path.split("/")[-1]
        
        url = build_doc_url(prompt_settings.base_docs_url, source_path, section_anchor)
        
        # Extract snippet (first 300 chars)
        snippet = doc.page_content.strip()[:300]
        snippet = " ".join(snippet.split())
        
        score = None
        if hasattr(doc, "metadata") and doc.metadata:
            score_val = doc.metadata.get("score")
            if score_val is not None:
                try:
                    score = float(score_val)
                except (ValueError, TypeError):
                    pass
        
        doc_id = generate_source_id(source_path, section_anchor, idx)
        
        documents.append({
            "title": section_title or filename or "Unknown",
            "url": url,
            "snippet": snippet,
            "doc_id": doc_id,
            "path": source_path,
            "section": section_anchor,
            "heading": section_title,
            "chunk_id": doc_id,
            "score": score
        })
    
    content = "\n\n".join(content_parts)
    
    return {
        "content": content,
        "count": len(documents),
        "documents": documents
    }


def build_system_namespace(
    request_id: str,
    conversation_id: Optional[str] = None,
    mode: str = "strict",
    passthrough: Optional[Dict[str, Any]] = None,
    context_hint: Optional[Dict[str, Any]] = None,
    accept_language_header: Optional[str] = None
) -> Dict[str, Any]:
    """
    Build system namespace for Jinja2 template.
    
    Backward-compatible wrapper around PromptService.
    
    Args:
        request_id: Request ID
        conversation_id: Conversation ID (optional)
        mode: Prompt mode (strict/helpful)
        passthrough: Passthrough dictionary (for language selection)
        context_hint: Context hint dictionary (for language selection)
        accept_language_header: Accept-Language header (for language selection)
        
    Returns:
        Dictionary with system namespace (includes output_language and language_reason)
    """
    return _default_prompt_service.build_system_namespace(
        request_id,
        conversation_id,
        mode,
        passthrough,
        context_hint,
        accept_language_header
    )


def render_system_prompt(
    template_str: str,
    system: Dict[str, Any],
    source: Dict[str, Any],
    passthrough: Dict[str, Any],
    tools: Dict[str, Any],
    request_id: str
) -> str:
    """
    Render system prompt using Jinja2 or return legacy string.
    
    Backward-compatible wrapper around PromptService.
    
    Args:
        template_str: Template string (Jinja2 or legacy)
        system: System namespace
        source: Source namespace
        passthrough: Passthrough namespace
        tools: Tools namespace
        request_id: Request ID for logging
        
    Returns:
        Rendered prompt string
    """
    return _default_prompt_service.render_system_prompt(
        template_str,
        system,
        source,
        passthrough,
        tools,
        request_id
    )


async def generate_answer(
    rag_chain,
    question: str,
    request_id: str,
    prompt_settings: PromptSettings,
    chat_history: str = "",
    response_language: Optional[str] = None,
    top_k_override: Optional[int] = None,
    context_hint: Optional[Dict] = None,
    system_prompt: Optional[str] = None,
    vectorstore=None  # Optional: for short-circuit when retriever not extractable
) -> Tuple[str, List[Source], bool, Dict]:
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
        Tuple (answer, sources, not_found, context_docs_dict)
        context_docs_dict contains the raw context documents for namespace building
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
    
    # Build system prompt (use default if not provided)
    if not system_prompt:
        system_prompt = build_system_prompt(prompt_settings, response_language=response_language)
    
    # Short-circuit for strict mode + no sources
    # Try to retrieve documents first to check if we have any sources
    # This avoids unnecessary LLM call when strict mode and no documentation available
    if prompt_settings.mode == "strict":
        try:
            # Extract retriever from rag_chain
            # LangChain create_retrieval_chain stores retriever in chain
            retriever = None
            chain_attrs = []
            
            # Try multiple ways to get retriever
            if hasattr(rag_chain, 'retriever'):
                retriever = rag_chain.retriever
                chain_attrs.append("has retriever attr")
            elif hasattr(rag_chain, 'steps') and len(rag_chain.steps) > 0:
                # Try to get retriever from chain steps
                for step in rag_chain.steps:
                    if hasattr(step, 'retriever'):
                        retriever = step.retriever
                        chain_attrs.append(f"found in step: {type(step).__name__}")
                        break
            elif hasattr(rag_chain, 'first') and hasattr(rag_chain.first, 'retriever'):
                retriever = rag_chain.first.retriever
                chain_attrs.append("found in first")
            elif hasattr(rag_chain, 'bound') and hasattr(rag_chain.bound, 'retriever'):
                retriever = rag_chain.bound.retriever
                chain_attrs.append("found in bound")
            
            # #region agent log
            try:
                with open("/Users/ila/RagAqtraDocs/.cursor/debug.log", "a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "sessionId": "debug-session",
                        "runId": "short-circuit-check",
                        "location": "answering.py:generate_answer",
                        "message": "Attempting to extract retriever",
                        "data": {
                            "request_id": request_id,
                            "chain_type": type(rag_chain).__name__,
                            "chain_attrs": chain_attrs,
                            "retriever_found": retriever is not None,
                            "strict_mode": True
                        },
                        "timestamp": int(time.time() * 1000)
                    }) + "\n")
            except Exception:
                pass
            # #endregion
            
            if retriever:
                # Retrieve documents to check if we have any sources
                retrieved_docs = await asyncio.to_thread(retriever.invoke, question)
                # #region agent log
                try:
                    with open("/Users/ila/RagAqtraDocs/.cursor/debug.log", "a", encoding="utf-8") as f:
                        f.write(json.dumps({
                            "sessionId": "debug-session",
                            "runId": "short-circuit-check",
                            "location": "answering.py:generate_answer",
                            "message": "Retrieved docs for short-circuit check",
                            "data": {
                                "retrieved_count": len(retrieved_docs) if retrieved_docs else 0,
                                "strict_mode": True,
                                "request_id": request_id
                            },
                            "timestamp": int(time.time() * 1000)
                        }) + "\n")
                except Exception:
                    pass
                # #endregion
                # Check if we have any relevant sources (check scores if available)
                has_relevant_sources = False
                if retrieved_docs and len(retrieved_docs) > 0:
                    # Check scores if available in metadata
                    not_found_score_threshold = float(os.getenv("NOT_FOUND_SCORE_THRESHOLD", "0.20"))
                    scores = []
                    for doc in retrieved_docs:
                        if hasattr(doc, "metadata") and doc.metadata:
                            score = doc.metadata.get("score")
                            if score is not None:
                                try:
                                    scores.append(float(score))
                                except (ValueError, TypeError):
                                    pass
                    # If we have scores, check if any are above threshold
                    if scores:
                        max_score = max(scores)
                        has_relevant_sources = max_score >= not_found_score_threshold
                    else:
                        # No scores available - assume documents might be relevant
                        # Only short-circuit if truly no documents
                        has_relevant_sources = True
                
                if not has_relevant_sources:
                    # Strict mode + no relevant sources: return early without calling LLM
                    logger.info(f"[{request_id}] Strict mode + no relevant sources: short-circuiting LLM call")
                    # #region agent log
                    try:
                        with open("/Users/ila/RagAqtraDocs/.cursor/debug.log", "a", encoding="utf-8") as f:
                            f.write(json.dumps({
                                "sessionId": "debug-session",
                                "runId": "short-circuit-check",
                                "location": "answering.py:generate_answer",
                                "message": "Short-circuit triggered: strict mode + no relevant sources",
                                "data": {
                                    "request_id": request_id,
                                    "question": question[:100],
                                    "retrieved_count": len(retrieved_docs) if retrieved_docs else 0,
                                    "max_score": max(scores) if scores else None,
                                    "threshold": not_found_score_threshold
                                },
                                "timestamp": int(time.time() * 1000)
                            }) + "\n")
                    except Exception:
                        pass
                    # #endregion
                    not_found_message = "I don't have enough information in the documentation to answer this question."
                    return not_found_message, [], True, {"context_docs": []}
            else:
                # Alternative: try to use vectorstore directly if available
                if vectorstore:
                    try:
                        # Use similarity_search_with_score to get scores
                        effective_k = top_k_override if top_k_override else prompt_settings.default_top_k
                        effective_k = max(1, min(10, effective_k))
                        
                        # Search with scores (FAISS similarity_search_with_score takes query string)
                        docs_with_scores = await asyncio.to_thread(
                            vectorstore.similarity_search_with_score,
                            question,
                            k=effective_k
                        )
                        
                        # Extract docs and scores
                        retrieved_docs = [doc for doc, score in docs_with_scores]
                        scores = [score for doc, score in docs_with_scores]
                        # Check scores (FAISS returns distance, lower is better, convert to similarity)
                        not_found_score_threshold = float(os.getenv("NOT_FOUND_SCORE_THRESHOLD", "0.20"))
                        has_relevant_sources = False
                        
                        if retrieved_docs and len(retrieved_docs) > 0 and scores:
                            # FAISS returns distance (lower = more similar)
                            # Convert to similarity score (1 - normalized distance)
                            # For cosine similarity, distance is already normalized
                            # Check if any document has similarity above threshold
                            # Note: FAISS distance might need conversion depending on metric
                            # For now, check if min distance is low enough (indicating high similarity)
                            min_distance = min(scores)
                            # Rough conversion: if distance < 0.5, consider it relevant
                            # This is approximate and may need tuning
                            max_similarity = 1.0 - min_distance if min_distance <= 1.0 else 0.0
                            has_relevant_sources = max_similarity >= not_found_score_threshold
                        elif retrieved_docs and len(retrieved_docs) > 0:
                            # No scores available - assume documents might be relevant
                            has_relevant_sources = True
                        
                        # #region agent log
                        try:
                            with open("/Users/ila/RagAqtraDocs/.cursor/debug.log", "a", encoding="utf-8") as f:
                                f.write(json.dumps({
                                    "sessionId": "debug-session",
                                    "runId": "short-circuit-check",
                                    "location": "answering.py:generate_answer",
                                    "message": "Using vectorstore similarity_search_with_score for short-circuit",
                                    "data": {
                                        "retrieved_count": len(retrieved_docs) if retrieved_docs else 0,
                                        "scores": scores[:3] if scores else None,  # Log first 3 scores
                                        "min_distance": min(scores) if scores else None,
                                        "max_similarity": max_similarity if scores else None,
                                        "has_relevant_sources": has_relevant_sources,
                                        "threshold": not_found_score_threshold,
                                        "strict_mode": True,
                                        "request_id": request_id
                                    },
                                    "timestamp": int(time.time() * 1000)
                                }) + "\n")
                        except Exception:
                            pass
                        # #endregion
                        
                        if not has_relevant_sources:
                            logger.info(f"[{request_id}] Strict mode + no relevant sources: short-circuiting LLM call (via vectorstore)")
                            # #region agent log
                            try:
                                with open("/Users/ila/RagAqtraDocs/.cursor/debug.log", "a", encoding="utf-8") as f:
                                    f.write(json.dumps({
                                        "sessionId": "debug-session",
                                        "runId": "short-circuit-check",
                                        "location": "answering.py:generate_answer",
                                        "message": "Short-circuit triggered via vectorstore: strict mode + no relevant sources",
                                        "data": {
                                            "request_id": request_id,
                                            "question": question[:100],
                                            "retrieved_count": len(retrieved_docs) if retrieved_docs else 0,
                                            "max_score": max(scores) if scores else None,
                                            "threshold": not_found_score_threshold
                                        },
                                        "timestamp": int(time.time() * 1000)
                                    }) + "\n")
                            except Exception:
                                pass
                            # #endregion
                            not_found_message = "I don't have enough information in the documentation to answer this question."
                            return not_found_message, [], True, {"context_docs": []}
                    except Exception as e:
                        logger.debug(f"[{request_id}] Could not use vectorstore for short-circuit: {e}")
                else:
                    # #region agent log
                    try:
                        with open("/Users/ila/RagAqtraDocs/.cursor/debug.log", "a", encoding="utf-8") as f:
                            f.write(json.dumps({
                                "sessionId": "debug-session",
                                "runId": "short-circuit-check",
                                "location": "answering.py:generate_answer",
                                "message": "Retriever not found in chain and vectorstore not available, skipping short-circuit",
                                "data": {
                                    "request_id": request_id,
                                    "chain_type": type(rag_chain).__name__,
                                    "chain_attrs_checked": chain_attrs
                                },
                                "timestamp": int(time.time() * 1000)
                            }) + "\n")
                    except Exception:
                        pass
                    # #endregion
        except Exception as e:
            # If retrieval check fails, continue with normal flow
            logger.debug(f"[{request_id}] Could not check retrieval for short-circuit: {e}")
            # #region agent log
            try:
                with open("/Users/ila/RagAqtraDocs/.cursor/debug.log", "a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "sessionId": "debug-session",
                        "runId": "short-circuit-check",
                        "location": "answering.py:generate_answer",
                        "message": "Exception during short-circuit check",
                        "data": {"request_id": request_id, "error": str(e)},
                        "timestamp": int(time.time() * 1000)
                    }) + "\n")
            except Exception:
                pass
            # #endregion
    
    # Build chain input
    chain_input = {
        "input": question,
        "response_language": response_language,
        "chat_history": chat_history or "",  # Empty string if None
        "system_prompt": system_prompt
    }
    
    # Call RAG chain
    try:
        result = await asyncio.to_thread(rag_chain.invoke, chain_input)
    except Exception as e:
        logger.error(f"[{request_id}] Error calling RAG chain: {e}", exc_info=True)
        raise
    
    # Extract answer
    answer = result.get("answer", "Failed to generate answer")
    
    # Extract context documents for namespace building
    context_docs = []
    if "context" in result:
        context_docs = result["context"]
        if not isinstance(context_docs, list):
            context_docs = [context_docs]
    elif "source_documents" in result:
        context_docs = result["source_documents"]
        if not isinstance(context_docs, list):
            context_docs = [context_docs]
    
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
    
    return answer, sources, not_found, {"context_docs": context_docs}


async def process_answer_request(
    rag_chain,
    request: AnswerRequest,
    request_id: str,
    prompt_settings: PromptSettings,
    db_sessionmaker,
    client_ip: str,
    user_agent: Optional[str] = None,
    accept_language_header: Optional[str] = None,
    vectorstore=None  # Optional: for short-circuit when retriever not extractable
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
    start_time = time_func()
    
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
    
    # Build passthrough namespace early (needed for language selection and cache key)
    context_hint_dict = None
    if request.context_hint:
        context_hint_dict = request.context_hint.dict()
    
    passthrough_dict = {}
    if request.passthrough:
        passthrough_dict.update(request.passthrough)
    if context_hint_dict:
        passthrough_dict.update(context_hint_dict)
    
    # Select output language early (needed for cache key)
    system_namespace_preview = build_system_namespace(
        request_id,
        conversation_id,
        prompt_settings.mode,
        passthrough=passthrough_dict,
        context_hint=context_hint_dict,
        accept_language_header=accept_language_header
    )
    output_language = system_namespace_preview["output_language"]
    
    # Get template info for cache key
    from app.core.prompt_config import get_selected_template_info
    template_info = get_selected_template_info()
    template_identifier = template_info.get("selected_template", "legacy")
    
    # Build cache key with template and language
    top_k_value = prompt_settings.default_top_k
    if request.retrieval and request.retrieval.top_k:
        top_k_value = request.retrieval.top_k
    
    detector_version = "v1"
    reranking_enabled = os.getenv("RERANKING_ENABLED", "0").lower() in ("1", "true", "yes")
    settings_signature = (
        f"mode={prompt_settings.mode}_"
        f"template={template_identifier}_"
        f"lang={output_language}_"
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
        latency_ms = int((time_func() - start_time) * 1000)
        
        # Update conversation_id in cached response
        cached_result.conversation_id = conversation_id
        
        # Save to conversation history if DB available
        if db_sessionmaker:
            await append_message(db_sessionmaker, conversation_id, "user", request.question)
            await append_message(db_sessionmaker, conversation_id, "assistant", cached_result.answer)
        
        return cached_result
    
    # Get template string
    template_str = get_prompt_template_content(prompt_settings)
    
    # Generate answer (first call to get context_docs for namespace building)
    top_k_override = None
    if request.retrieval and request.retrieval.top_k:
        top_k_override = request.retrieval.top_k
    
    # For Jinja2 mode, we need context_docs to build source namespace
    # Call generate_answer once to get context_docs, then render prompt and call again
    # For legacy mode, just build default prompt
    if is_jinja_mode():
        # First call to get context_docs
        answer, sources, not_found, context_info = await generate_answer(
            rag_chain,
            request.question,
            request_id,
            prompt_settings,
            chat_history=chat_history_text,
            top_k_override=top_k_override,
            context_hint=context_hint_dict,
            system_prompt="",  # Temporary, will be replaced
            vectorstore=vectorstore
        )
        
        # Build namespaces from context_docs (system_namespace already built above for cache key)
        context_docs = context_info.get("context_docs", [])
        system_namespace = system_namespace_preview  # Reuse already built namespace
        source_namespace = build_source_namespace(context_docs, prompt_settings)
        
        tools_namespace = {}  # Empty for now
        
        # Render system prompt
        rendered_system_prompt = render_system_prompt(
            template_str,
            system_namespace,
            source_namespace,
            passthrough_dict,
            tools_namespace,
            request_id
        )
        
        # Regenerate answer with rendered prompt
        answer, sources, not_found, _ = await generate_answer(
            rag_chain,
            request.question,
            request_id,
            prompt_settings,
            chat_history=chat_history_text,
            top_k_override=top_k_override,
            context_hint=context_hint_dict,
            system_prompt=rendered_system_prompt,
            vectorstore=vectorstore
        )
    else:
        # Legacy mode: use default system prompt
        # system_namespace already built above for cache key
        response_language = system_namespace_preview["output_language"]  # Use selected language
        default_system_prompt = build_system_prompt(prompt_settings, response_language=response_language)
        
        answer, sources, not_found, _ = await generate_answer(
            rag_chain,
            request.question,
            request_id,
            prompt_settings,
            chat_history=chat_history_text,
            top_k_override=top_k_override,
            context_hint=context_hint_dict,
            system_prompt=default_system_prompt,
            vectorstore=vectorstore
        )
    
    latency_sec = time_func() - start_time
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

