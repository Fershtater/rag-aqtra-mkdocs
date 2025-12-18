"""
V1 query endpoint for backward compatibility.
"""

import asyncio
import logging
import os
from time import time
from typing import Optional

from fastapi import APIRouter, Request, JSONResponse

from app.api.schemas import Query, QueryResponse, ErrorResponse
from app.core.prompt_config import (
    load_prompt_settings_from_env,
    detect_response_language,
    get_prompt_template_content,
    is_jinja_mode,
    build_system_prompt,
    get_selected_template_info,
)
from app.services.prompt_service import PromptService
from app.core.markdown_utils import build_doc_url
from app.infra.rate_limit import query_limiter
from app.infra.cache import response_cache
from app.infra.metrics import (
    query_requests_total,
    rate_limit_hits_total,
    query_latency_seconds,
    PROMETHEUS_AVAILABLE,
)
from app.infra.analytics import hash_ip, log_query

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/query", response_model=QueryResponse, responses={429: {"model": ErrorResponse}, 503: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def query_documentation(query: Query, request: Request):
    """
    Main endpoint for documentation questions.
    
    Accepts user question and returns answer based on RAG system.
    Supports caching, rate limiting and metrics.
    """
    request_id = getattr(request.state, "request_id", "unknown")
    start_time = time()
    
    # Rate limiting
    client_ip = request.client.host if request.client else "unknown"
    allowed, error_msg = query_limiter.is_allowed(client_ip)
    if not allowed:
        if PROMETHEUS_AVAILABLE and rate_limit_hits_total is not None:
            rate_limit_hits_total.labels(endpoint="query").inc()
        logger.warning(f"[{request_id}] Rate limit exceeded for {client_ip}")
        return JSONResponse(
            status_code=429,
            content={"error": "Rate limit exceeded", "detail": error_msg}
        )
    
    # Get RAG chain from app.state
    rag_chain = getattr(request.app.state, "rag_chain", None)
    
    if rag_chain is None:
        logger.error(f"[{request_id}] RAG chain not initialized")
        if PROMETHEUS_AVAILABLE and query_requests_total is not None:
            query_requests_total.labels(status="error").inc()
        return JSONResponse(
            status_code=503,
            content={
                "error": "RAG chain not initialized",
                "detail": "Please try again later or check application logs"
            }
        )
    
    try:
        # Get prompt settings from app.state
        prompt_settings = getattr(request.app.state, "prompt_settings", None)
        if prompt_settings is None:
            prompt_settings = load_prompt_settings_from_env()

        # Detect response language from user question
        response_language = detect_response_language(
            query.question,
            supported=set(prompt_settings.supported_languages),
            fallback=prompt_settings.fallback_language
        )
        
        # Settings signature for cache (include template, language, index_version)
        template_info = get_selected_template_info()
        template_identifier = template_info.get("selected_template", "legacy")
        
        # Get index version
        index_version = getattr(request.app.state, "index_version", None) or ""
        
        detector_version = "v1"
        reranking_enabled = os.getenv("RERANKING_ENABLED", "0").lower() in ("1", "true", "yes")
        settings_signature = (
            f"mode={prompt_settings.mode}_"
            f"template={template_identifier}_"
            f"lang={response_language}_"
            f"top_k={prompt_settings.default_top_k}_"
            f"temp={prompt_settings.default_temperature}_"
            f"max_tokens={getattr(prompt_settings, 'default_max_tokens', None)}_"
            f"supported={','.join(sorted(prompt_settings.supported_languages))}_"
            f"fallback={prompt_settings.fallback_language}_"
            f"rerank={reranking_enabled}_"
            f"detector={detector_version}_"
            f"index_version={index_version}"
        )

        # Generate cache key
        cache_key = response_cache._generate_key(query.question, settings_signature)

        # Check cache
        cached_result: Optional[QueryResponse] = response_cache.get(cache_key)
        
        if cached_result:
            logger.debug(f"[{request_id}] Cache hit for query")
            latency = int((time() - start_time) * 1000)
            if PROMETHEUS_AVAILABLE and query_requests_total is not None:
                query_requests_total.labels(status="success").inc()
                query_latency_seconds.observe(latency / 1000.0)

            # Log request even on cache hit
            db_sessionmaker = getattr(request.app.state, "db_sessionmaker", None)
            sources = cached_result.sources or []
            not_found_flag = cached_result.not_found
            ip_hash_value = hash_ip(client_ip)
            user_agent = request.headers.get("User-Agent")

            await log_query(
                db_sessionmaker,
                request_id=request_id,
                ip_hash_value=ip_hash_value,
                user_agent=user_agent,
                page_url=query.page_url,
                page_title=query.page_title,
                question=query.question,
                answer=cached_result.answer,
                not_found=not_found_flag,
                cache_hit=True,
                latency_ms=latency,
                sources=sources,
                error=None,
            )

            return cached_result

        # Get template string for potential Jinja2 rendering
        template_str = get_prompt_template_content(prompt_settings)
        
        # Build system prompt
        if is_jinja_mode():
            # For Jinja2 mode, need to get context_docs first
            chain_input_preview = {
                "input": query.question,
                "response_language": response_language,
                "chat_history": "",
                "system_prompt": ""
            }
            preview_result = await asyncio.to_thread(rag_chain.invoke, chain_input_preview)
            
            # Extract context_docs
            context_docs = []
            if "context" in preview_result:
                context_docs = preview_result["context"]
                if not isinstance(context_docs, list):
                    context_docs = [context_docs]
            elif "source_documents" in preview_result:
                context_docs = preview_result["source_documents"]
                if not isinstance(context_docs, list):
                    context_docs = [context_docs]
            
            # Build passthrough namespace
            passthrough_ns = {}
            if query.page_url:
                passthrough_ns["page_url"] = query.page_url
            if query.page_title:
                passthrough_ns["page_title"] = query.page_title
            
            # Build namespaces
            prompt_service = PromptService()
            system_ns = prompt_service.build_system_namespace(
                request_id,
                None,
                prompt_settings.mode,
                passthrough=passthrough_ns,
                context_hint=None,
                accept_language_header=None
            )
            source_ns = prompt_service.build_source_namespace(context_docs, prompt_settings)
            tools_ns = {}
            
            # Render system prompt
            rendered_prompt = prompt_service.render_system_prompt(
                template_str,
                system_ns,
                source_ns,
                passthrough_ns,
                tools_ns,
                request_id
            )
            
            # Re-invoke with rendered prompt
            chain_input = {
                "input": query.question,
                "response_language": response_language,
                "chat_history": "",
                "system_prompt": rendered_prompt
            }
        else:
            # Legacy mode: use default system prompt
            passthrough_ns = {}
            if query.page_url:
                passthrough_ns["page_url"] = query.page_url
            if query.page_title:
                passthrough_ns["page_title"] = query.page_title
            
            prompt_service = PromptService()
            system_ns = prompt_service.build_system_namespace(
                request_id,
                None,
                prompt_settings.mode,
                passthrough=passthrough_ns,
                context_hint=None,
                accept_language_header=None
            )
            response_language = system_ns["output_language"]
            default_system_prompt = build_system_prompt(prompt_settings, response_language=response_language)
            chain_input = {
                "input": query.question,
                "response_language": response_language,
                "chat_history": "",
                "system_prompt": default_system_prompt
            }
        
        # Call RAG chain
        try:
            result = await asyncio.to_thread(rag_chain.invoke, chain_input)
        except Exception as e:
            logger.error(f"[{request_id}] Error calling LLM: {e}", exc_info=True)
            if PROMETHEUS_AVAILABLE and query_requests_total is not None:
                query_requests_total.labels(status="error").inc()
                query_latency_seconds.observe(time() - start_time)
            return JSONResponse(
                status_code=503,
                content={
                    "error": "Error processing request",
                    "detail": "Service temporarily unavailable. Please try again later."
                }
            )
        
        # Extract answer
        answer = result.get("answer", "Failed to generate answer")
        
        # Extract sources from metadata
        sources = []
        
        if "context" in result:
            context_docs = result["context"]
            if not isinstance(context_docs, list):
                context_docs = [context_docs]
            
            seen_sources = set()
            for doc in context_docs:
                if hasattr(doc, "metadata") and doc.metadata:
                    source = doc.metadata.get("source", "unknown")
                    section_anchor = doc.metadata.get("section_anchor")
                    source_key = (source, section_anchor)
                    if source_key in seen_sources:
                        continue
                    seen_sources.add(source_key)
                    
                    source_info = {
                        "source": source,
                        "filename": doc.metadata.get("filename", source.split("/")[-1])
                    }
                    
                    if "section_title" in doc.metadata:
                        source_info["section_title"] = doc.metadata["section_title"]
                    if "section_anchor" in doc.metadata:
                        source_info["section_anchor"] = doc.metadata["section_anchor"]
                        source_info["url"] = build_doc_url(
                            prompt_settings.base_docs_url,
                            source,
                            doc.metadata["section_anchor"]
                        )
                    elif source != "unknown":
                        source_info["url"] = build_doc_url(
                            prompt_settings.base_docs_url,
                            source,
                            None
                        )
                    
                    sources.append(source_info)
        
        # Fallback: try to extract from source_documents
        if not sources and "source_documents" in result:
            for doc in result["source_documents"]:
                if hasattr(doc, "metadata") and doc.metadata:
                    source = doc.metadata.get("source", "unknown")
                    source_info = {
                        "source": source,
                        "filename": doc.metadata.get("filename", source.split("/")[-1])
                    }
                    if "section_title" in doc.metadata:
                        source_info["section_title"] = doc.metadata["section_title"]
                    if "section_anchor" in doc.metadata:
                        source_info["section_anchor"] = doc.metadata["section_anchor"]
                        source_info["url"] = build_doc_url(
                            prompt_settings.base_docs_url,
                            source,
                            doc.metadata["section_anchor"]
                        )
                    elif source != "unknown":
                        source_info["url"] = build_doc_url(
                            prompt_settings.base_docs_url,
                            source,
                            None
                        )
                    sources.append(source_info)
        
        # Determine not_found by retrieval signals
        not_found_score_threshold = float(os.getenv("NOT_FOUND_SCORE_THRESHOLD", "0.20"))
        not_found_flag = len(sources) == 0
        
        if sources:
            scores = [s.get("score") for s in sources if isinstance(s, dict) and s.get("score") is not None]
            if scores:
                top_score = max(scores)
                if top_score < not_found_score_threshold:
                    not_found_flag = True

        latency_sec = time() - start_time
        latency_ms = int(latency_sec * 1000)

        # Form response
        response = QueryResponse(
            answer=answer,
            sources=sources,
            not_found=not_found_flag,
            request_id=request_id,
            latency_ms=latency_ms,
            cache_hit=False,
        )

        # Save to cache
        response_cache.set(cache_key, response)

        # Update metrics
        if PROMETHEUS_AVAILABLE and query_requests_total is not None:
            query_requests_total.labels(status="success").inc()
            query_latency_seconds.observe(latency_sec)

        # Log to database (if available)
        db_sessionmaker = getattr(request.app.state, "db_sessionmaker", None)
        ip_hash_value = hash_ip(client_ip)
        user_agent = request.headers.get("User-Agent")

        await log_query(
            db_sessionmaker,
            request_id=request_id,
            ip_hash_value=ip_hash_value,
            user_agent=user_agent,
            page_url=query.page_url,
            page_title=query.page_title,
            question=query.question,
            answer=answer,
            not_found=not_found_flag,
            cache_hit=False,
            latency_ms=latency_ms,
            sources=sources,
            error=None,
        )

        logger.info(
            "[%s] Request processed successfully, sources: %s, not_found=%s, cache_hit=%s, latency_ms=%s",
            request_id,
            len(sources),
            not_found_flag,
            False,
            latency_ms,
        )
        return response
        
    except Exception as e:
        logger.error(f"[{request_id}] Error processing request: {e}", exc_info=True)
        if PROMETHEUS_AVAILABLE and query_requests_total is not None:
            query_requests_total.labels(status="error").inc()
            query_latency_seconds.observe(time() - start_time)

        # Try to log request error
        try:
            db_sessionmaker = getattr(request.app.state, "db_sessionmaker", None)
            ip_hash_value = hash_ip(client_ip)
            user_agent = request.headers.get("User-Agent")
            latency_ms = int((time() - start_time) * 1000)

            await log_query(
                db_sessionmaker,
                request_id=request_id,
                ip_hash_value=ip_hash_value,
                user_agent=user_agent,
                page_url=query.page_url,
                page_title=query.page_title,
                question=query.question,
                answer=None,
                not_found=False,
                cache_hit=False,
                latency_ms=latency_ms,
                sources=[],
                error=str(e),
            )
        except Exception:
            pass
        return JSONResponse(
            status_code=500,
            content={
                "error": "Error processing request",
                "detail": "Internal server error"
            }
        )

