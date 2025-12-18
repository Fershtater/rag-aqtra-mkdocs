"""
Common answering logic for v2 API endpoints.
"""

import asyncio
import hashlib
import logging
import os
from time import time
from typing import Any, Dict, List, Optional, Tuple

from app.api.v2_models import (
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
    load_prompt_settings_from_env,
    get_prompt_template_content,
    is_jinja_mode,
    build_system_prompt,
)
from app.core.prompt_renderer import PromptRenderer
from app.core.language_policy import select_output_language
from app.infra.cache import response_cache
from app.infra.conversations import append_message, get_or_create_conversation, load_history
from datetime import datetime

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
    # Select output language
    output_language, language_reason = select_output_language(
        passthrough=passthrough,
        context_hint=context_hint,
        accept_language_header=accept_language_header
    )
    
    return {
        "request_id": request_id,
        "conversation_id": conversation_id or "",
        "now_iso": datetime.utcnow().isoformat(),
        "timezone": "UTC",
        "app_version": os.getenv("APP_VERSION", ""),
        "mode": mode,
        "output_language": output_language,
        "language_reason": language_reason
    }


def render_system_prompt(
    template_str: str,
    system: Dict[str, any],
    source: Dict[str, any],
    passthrough: Dict[str, any],
    tools: Dict[str, any],
    request_id: str
) -> str:
    """
    Render system prompt using Jinja2 or return legacy string.
    
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
    if not is_jinja_mode():
        # Legacy mode: template_str is already the final prompt
        return template_str
    
    # Jinja2 mode: render template
    try:
        max_chars = int(os.getenv("PROMPT_MAX_CHARS", "40000"))
        strict_undefined = os.getenv("PROMPT_STRICT_UNDEFINED", "1").lower() in ("1", "true", "yes")
        
        renderer = PromptRenderer(max_chars=max_chars, strict_undefined=strict_undefined)
        rendered = renderer.render(
            template_str,
            system=system,
            source=source,
            passthrough=passthrough,
            tools=tools
        )
        
        # Log rendered prompt if enabled
        log_rendered = os.getenv("PROMPT_LOG_RENDERED", "0").lower() in ("1", "true", "yes")
        if log_rendered:
            log_length = min(len(rendered), 8000)
            logger.info(f"[{request_id}] Rendered prompt (first {log_length} chars):\n{rendered[:log_length]}")
        
        return rendered
        
    except Exception as e:
        logger.error(f"[{request_id}] Error rendering Jinja2 template: {e}, falling back to legacy", exc_info=True)
        # Fallback to legacy
        fail_hard = os.getenv("PROMPT_FAIL_HARD", "0").lower() in ("1", "true", "yes")
        if fail_hard:
            raise
        
        # Return legacy prompt
        prompt_settings = load_prompt_settings_from_env()
        return build_system_prompt(prompt_settings, response_language=system.get("mode", "strict"))


async def generate_answer(
    rag_chain,
    question: str,
    request_id: str,
    prompt_settings: PromptSettings,
    chat_history: str = "",
    response_language: Optional[str] = None,
    top_k_override: Optional[int] = None,
    context_hint: Optional[Dict] = None,
    system_prompt: Optional[str] = None
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
    
    # Get template string
    template_str = get_prompt_template_content(prompt_settings)
    
    # Generate answer (first call to get context_docs for namespace building)
    top_k_override = None
    if request.retrieval and request.retrieval.top_k:
        top_k_override = request.retrieval.top_k
    
    context_hint_dict = None
    if request.context_hint:
        context_hint_dict = request.context_hint.dict()
    
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
            system_prompt=""  # Temporary, will be replaced
        )
        
        # Build passthrough namespace (needed for language selection)
        passthrough_dict = {}
        if request.passthrough:
            passthrough_dict.update(request.passthrough)
        if context_hint_dict:
            passthrough_dict.update(context_hint_dict)
        
        # Build namespaces from context_docs
        context_docs = context_info.get("context_docs", [])
        system_namespace = build_system_namespace(
            request_id,
            conversation_id,
            prompt_settings.mode,
            passthrough=passthrough_dict,
            context_hint=context_hint_dict,
            accept_language_header=None  # TODO: pass from request if available
        )
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
            system_prompt=rendered_system_prompt
        )
    else:
        # Legacy mode: use default system prompt
        # Still select language for system namespace (for consistency)
        passthrough_dict = {}
        if request.passthrough:
            passthrough_dict.update(request.passthrough)
        if context_hint_dict:
            passthrough_dict.update(context_hint_dict)
        
        system_namespace = build_system_namespace(
            request_id,
            conversation_id,
            prompt_settings.mode,
            passthrough=passthrough_dict,
            context_hint=context_hint_dict,
            accept_language_header=None
        )
        response_language = system_namespace["output_language"]  # Use selected language
        default_system_prompt = build_system_prompt(prompt_settings, response_language=response_language)
        
        answer, sources, not_found, _ = await generate_answer(
            rag_chain,
            request.question,
            request_id,
            prompt_settings,
            chat_history=chat_history_text,
            top_k_override=top_k_override,
            context_hint=context_hint_dict,
            system_prompt=default_system_prompt
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

