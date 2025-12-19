"""
Server-Sent Events (SSE) streaming endpoint.
"""

import logging
import os
from time import time as time_func
from typing import List

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from app.api.schemas.v2 import AnswerRequest, SSEEvent, ErrorPayload, MetricsPayload
from app.core.prompt_config import load_prompt_settings_from_env
from app.services.answer_service import AnswerService
from app.services.conversation_service import ConversationService
from app.services.prompt_service import PromptService
from app.infra.rate_limit import query_limiter
from app.infra.metrics import (
    rate_limit_hits_total,
    rag_ttft_seconds,
    PROMETHEUS_AVAILABLE,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def validate_api_key(api_key: str) -> bool:
    """
    Validate API key from RAG_API_KEYS environment variable.
    """
    import os
    rag_api_keys_str = os.getenv("RAG_API_KEYS", "")
    if not rag_api_keys_str:
        logger.warning("RAG_API_KEYS not set, allowing open access (consider setting it)")
        return True
    
    allowed_keys = [key.strip() for key in rag_api_keys_str.split(",") if key.strip()]
    return api_key in allowed_keys


def chunk_text_for_streaming(text: str, chunk_size: int = 80) -> List[str]:
    """
    Split text into chunks for pseudo-streaming (fallback only).
    """
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i + chunk_size])
    return chunks


@router.post("/stream")
async def stream_answer(request_data: AnswerRequest, request: Request):
    """
    Server-Sent Events (SSE) streaming endpoint.
    
    Returns answer in streaming format with events: id, answer (deltas), source, end.
    """
    request_id = getattr(request.state, "request_id", "unknown")
    
    # Validate API key
    if not validate_api_key(request_data.api_key):
        async def error_stream():
            error_event = SSEEvent(
                type="error",
                error=ErrorPayload(
                    code="INVALID_API_KEY",
                    message="Invalid API key",
                    request_id=request_id
                )
            )
            yield f"data: {error_event.json()}\n\n"
        
        return StreamingResponse(error_stream(), media_type="text/event-stream")
    
    # Rate limiting
    client_ip = request.client.host if request.client else "unknown"
    allowed, error_msg = query_limiter.is_allowed(client_ip)
    if not allowed:
        if PROMETHEUS_AVAILABLE and rate_limit_hits_total is not None:
            rate_limit_hits_total.labels(endpoint="stream").inc()
        
        async def rate_limit_stream():
            error_event = SSEEvent(
                type="error",
                error=ErrorPayload(
                    code="RATE_LIMIT_EXCEEDED",
                    message=error_msg,
                    request_id=request_id
                )
            )
            yield f"data: {error_event.json()}\n\n"
        
        return StreamingResponse(rate_limit_stream(), media_type="text/event-stream")
    
    # Get RAG chain
    rag_chain = getattr(request.app.state, "rag_chain", None)
    if rag_chain is None:
        async def unavailable_stream():
            error_event = SSEEvent(
                type="error",
                error=ErrorPayload(
                    code="SERVICE_UNAVAILABLE",
                    message="RAG chain not initialized",
                    request_id=request_id
                )
            )
            yield f"data: {error_event.json()}\n\n"
        
        return StreamingResponse(unavailable_stream(), media_type="text/event-stream")
    
    async def generate_stream():
        start_time = time_func()
        first_token_time = None
        ttft_ms = None
        
        try:
            # Get prompt settings
            prompt_settings = getattr(request.app.state, "prompt_settings", None)
            if prompt_settings is None:
                prompt_settings = load_prompt_settings_from_env()
            
            # Compute effective preset (request override > passthrough > server default)
            from app.settings import Settings
            settings = getattr(request.app.state, "settings", None)
            if settings is None:
                # Fallback: read from env
                server_preset = os.getenv("PROMPT_PRESET", "strict").lower()
            else:
                server_preset = settings.PROMPT_PRESET.lower()
            
            effective_preset = None
            if request_data.preset:
                effective_preset = request_data.preset.lower().strip()
            elif request_data.passthrough and isinstance(request_data.passthrough, dict):
                preset_from_passthrough = request_data.passthrough.get("preset")
                if preset_from_passthrough:
                    effective_preset = str(preset_from_passthrough).lower().strip()
            
            # Validate and normalize preset
            valid_presets = {"strict", "support", "developer"}
            if effective_preset not in valid_presets:
                effective_preset = server_preset
            
            # Update prompt_settings mode based on effective_preset
            # strict preset -> mode="strict", others -> mode="helpful"
            effective_mode = "strict" if effective_preset == "strict" else "helpful"
            if prompt_settings.mode != effective_mode:
                # Create new PromptSettings with updated mode
                from app.core.prompt_config import PromptSettings
                prompt_settings = PromptSettings(
                    supported_languages=prompt_settings.supported_languages,
                    fallback_language=prompt_settings.fallback_language,
                    base_docs_url=prompt_settings.base_docs_url,
                    not_found_message=prompt_settings.not_found_message,
                    include_sources_in_text=prompt_settings.include_sources_in_text,
                    mode=effective_mode,
                    default_temperature=prompt_settings.default_temperature,
                    default_top_k=prompt_settings.default_top_k,
                    default_max_tokens=prompt_settings.default_max_tokens
                )
            
            # Get database sessionmaker
            db_sessionmaker = getattr(request.app.state, "db_sessionmaker", None)
            
            # Get Accept-Language header
            accept_language_header = request.headers.get("Accept-Language")
            
            # Get vectorstore for short-circuit
            vectorstore = getattr(request.app.state, "vectorstore", None)
            
            # Get index version (not used in streaming, but kept for consistency)
            _ = getattr(request.app.state, "index_version", None)
            
            # Get services from app.state (created in lifespan)
            answer_service = getattr(request.app.state, "answer_service", None)
            if answer_service is None:
                # Fallback: create services if not in app.state
                conversation_service = ConversationService(db_sessionmaker)
                prompt_service = PromptService()
                answer_service = AnswerService(conversation_service, prompt_service)
            
            # Get or create conversation ID
            conversation_id = await answer_service.conversation_service.get_or_create_conversation(
                request_data.conversation_id
            )
            
            # Load or parse history
            from app.api.schemas.v2 import parse_history_to_text
            chat_history_text = ""
            if request_data.history:
                chat_history_text = parse_history_to_text(request_data.history)
            elif conversation_id:
                # Use streaming-specific history limit
                stream_history_max_messages = int(os.getenv("STREAM_HISTORY_MAX_MESSAGES", "10"))
                history_list = await answer_service.conversation_service.load_history(conversation_id, limit=stream_history_max_messages)
                if history_list:
                    chat_history_text = parse_history_to_text(history_list)
            
            # Limit history length for latency (use streaming-specific limit)
            stream_history_max_chars = int(os.getenv("STREAM_HISTORY_MAX_CHARS", "3000"))
            if len(chat_history_text) > stream_history_max_chars:
                chat_history_text = "..." + chat_history_text[-stream_history_max_chars + 3:]
            
            # Build context hint
            context_hint_dict = None
            if request_data.context_hint:
                context_hint_dict = request_data.context_hint.dict()
            
            # Build passthrough
            passthrough_dict = {}
            if request_data.passthrough:
                passthrough_dict.update(request_data.passthrough)
            if context_hint_dict:
                passthrough_dict.update(context_hint_dict)
            
            # Build system namespace for language detection
            system_namespace_preview = answer_service.prompt_service.build_system_namespace(
                request_id,
                conversation_id,
                prompt_settings.mode,
                passthrough=passthrough_dict,
                context_hint=context_hint_dict,
                accept_language_header=accept_language_header
            )
            output_language = system_namespace_preview["output_language"]
            
            # Get top_k override (prefer STREAM_TOP_K for streaming)
            top_k_override = None
            if request_data.retrieval and request_data.retrieval.top_k:
                top_k_override = request_data.retrieval.top_k
            else:
                # Use STREAM_TOP_K if set, otherwise use DEFAULT_TOP_K
                stream_top_k = os.getenv("STREAM_TOP_K")
                if stream_top_k:
                    try:
                        top_k_override = int(stream_top_k)
                    except ValueError:
                        pass
            
            # Stream answer using real streaming
            # Note: generate_answer_stream now handles Jinja2 template rendering internally
            stream_flush_chars = int(os.getenv("STREAM_FLUSH_EVERY_N_CHARS", "15"))
            sources_sent = False
            full_answer = ""
            final_sources = []
            final_not_found = False
            retrieved_chunks_from_context = None
            retrieval_ms = None
            embed_query_ms = None
            vector_search_ms = None
            format_sources_ms = None
            prompt_render_ms = None
            llm_connect_ms = None
            
            async for token_delta, sources, not_found, context_info in answer_service.generate_answer_stream(
                rag_chain,
                request_data.question,
                request_id,
                prompt_settings,
                chat_history=chat_history_text,
                response_language=output_language,
                top_k_override=top_k_override,
                context_hint=context_hint_dict,
                system_prompt=None,  # Let generate_answer_stream build it via Jinja2
                vectorstore=vectorstore,
                stream_flush_chars=stream_flush_chars,
                preset_override=effective_preset
            ):
                # Extract timing metrics and retrieved_chunks from context_info
                if context_info:
                    if "retrieval_ms" in context_info:
                        retrieval_ms = context_info["retrieval_ms"]
                    if "embed_query_ms" in context_info:
                        embed_query_ms = context_info["embed_query_ms"]
                    if "vector_search_ms" in context_info:
                        vector_search_ms = context_info["vector_search_ms"]
                    if "format_sources_ms" in context_info:
                        format_sources_ms = context_info["format_sources_ms"]
                    if "prompt_render_ms" in context_info:
                        prompt_render_ms = context_info["prompt_render_ms"]
                    if "llm_connect_ms" in context_info:
                        llm_connect_ms = context_info["llm_connect_ms"]
                    if "retrieved_chunks" in context_info:
                        retrieved_chunks_from_context = context_info["retrieved_chunks"]
                
                # Send ID event on first yield (before any tokens)
                if first_token_time is None:
                    id_event = SSEEvent(
                        type="id",
                        conversation_id=conversation_id,
                        request_id=request_id
                    )
                    yield f"data: {id_event.json()}\n\n"
                
                # Send sources as soon as they're available (before or with first token)
                # Note: sources are yielded from generate_answer_stream after retrieval
                if not sources_sent and sources and len(sources) > 0:
                    for source in sources:
                        source_event = SSEEvent(
                            type="source",
                            source=source
                        )
                        yield f"data: {source_event.json()}\n\n"
                    sources_sent = True
                    final_sources = sources
                    logger.debug(f"[{request_id}] Sent {len(sources)} sources in SSE stream")
                
                # Update final_sources if we get a new list (from final yield)
                if sources and len(sources) > 0:
                    final_sources = sources
                
                # Send answer delta and measure TTFT on first token
                if token_delta:
                    if first_token_time is None:
                        first_token_time = time_func()
                        ttft_ms = int((first_token_time - start_time) * 1000)
                        # Estimate prompt size (rough estimate: ~4 chars per token)
                        if prompt_render_ms is not None:
                            # We can't get exact prompt size here, but we can log it if available
                            pass
                    
                    full_answer += token_delta
                    answer_event = SSEEvent(
                        type="answer",
                        delta=token_delta
                    )
                    yield f"data: {answer_event.json()}\n\n"
                
                # Update final status and sources (from any yield)
                final_not_found = not_found
                if sources and len(sources) > 0:
                    final_sources = sources
                elif not_found and len(sources) == 0:
                    # Strict miss: ensure sources are empty
                    final_sources = []
            
            # Send end event with metrics
            total_latency_ms = int((time_func() - start_time) * 1000)
            
            # Ensure retrieved_chunks=0 for strict miss (when sources are empty)
            # Use retrieved_chunks from context_info if available (for strict miss), otherwise count sources
            if retrieved_chunks_from_context is not None:
                retrieved_chunks_count = retrieved_chunks_from_context
            else:
                retrieved_chunks_count = len(final_sources) if final_sources else 0
            
            metrics = MetricsPayload(
                latency_ms=total_latency_ms,
                cache_hit=False,  # Streaming doesn't use cache
                retrieved_chunks=retrieved_chunks_count,
                model=None,
                ttft_ms=ttft_ms,
                retrieval_ms=retrieval_ms,
                embed_query_ms=embed_query_ms,
                vector_search_ms=vector_search_ms,
                format_sources_ms=format_sources_ms,
                prompt_render_ms=prompt_render_ms,
                llm_connect_ms=llm_connect_ms
            )
            
            # Update Prometheus metrics
            if PROMETHEUS_AVAILABLE and rag_ttft_seconds is not None and ttft_ms is not None:
                rag_ttft_seconds.labels(endpoint="stream").observe(ttft_ms / 1000.0)
            
            # Log detailed metrics
            breakdown_parts = []
            if embed_query_ms is not None:
                breakdown_parts.append(f"embed_query={embed_query_ms}ms")
            if vector_search_ms is not None:
                breakdown_parts.append(f"vector_search={vector_search_ms}ms")
            if format_sources_ms is not None:
                breakdown_parts.append(f"format_sources={format_sources_ms}ms")
            if retrieval_ms is not None:
                breakdown_parts.append(f"retrieval_total={retrieval_ms}ms")
            if prompt_render_ms is not None:
                breakdown_parts.append(f"prompt_render={prompt_render_ms}ms")
            if llm_connect_ms is not None:
                breakdown_parts.append(f"llm_connect={llm_connect_ms}ms")
            breakdown_str = ", ".join(breakdown_parts) if breakdown_parts else "no breakdown"
            
            logger.info(
                f"[{request_id}] Stream completed: conversation_id={conversation_id}, "
                f"ttft_ms={ttft_ms}, total_ms={total_latency_ms}, "
                f"sources={len(final_sources)}, not_found={final_not_found}, "
                f"breakdown=[{breakdown_str}]"
            )
            
            # Save to conversation history
            await answer_service.conversation_service.append_message(conversation_id, "user", request_data.question)
            await answer_service.conversation_service.append_message(conversation_id, "assistant", full_answer)
            
            end_event = SSEEvent(
                type="end",
                metrics=metrics
            )
            yield f"data: {end_event.json()}\n\n"
            
        except Exception as e:
            logger.error(f"[{request_id}] Error in stream: {e}", exc_info=True)
            error_event = SSEEvent(
                type="error",
                error=ErrorPayload(
                    code="INTERNAL_ERROR",
                    message=str(e),
                    request_id=request_id
                )
            )
            yield f"data: {error_event.json()}\n\n"
            
            # Send end event after error
            end_event = SSEEvent(
                type="end",
                metrics=MetricsPayload(
                    latency_ms=int((time_func() - start_time) * 1000),
                    cache_hit=False,
                    retrieved_chunks=0,
                    model=None,
                    ttft_ms=ttft_ms
                )
            )
            yield f"data: {end_event.json()}\n\n"
    
    return StreamingResponse(generate_stream(), media_type="text/event-stream")

