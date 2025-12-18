"""
Server-Sent Events (SSE) streaming endpoint.
"""

import logging
from typing import List

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from app.api.schemas.v2 import AnswerRequest, SSEEvent, ErrorPayload
from app.core.prompt_config import load_prompt_settings_from_env
from app.services.answer_service import AnswerService
from app.services.conversation_service import ConversationService
from app.services.prompt_service import PromptService
from app.infra.rate_limit import query_limiter
from app.infra.metrics import (
    rate_limit_hits_total,
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
    Split text into chunks for pseudo-streaming.
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
        try:
            # Get prompt settings
            prompt_settings = getattr(request.app.state, "prompt_settings", None)
            if prompt_settings is None:
                prompt_settings = load_prompt_settings_from_env()
            
            # Get database sessionmaker
            db_sessionmaker = getattr(request.app.state, "db_sessionmaker", None)
            
            # Get Accept-Language header
            accept_language_header = request.headers.get("Accept-Language")
            
            # Get vectorstore for short-circuit
            vectorstore = getattr(request.app.state, "vectorstore", None)
            
            # Create services
            conversation_service = ConversationService(db_sessionmaker)
            prompt_service = PromptService()
            answer_service = AnswerService(conversation_service, prompt_service)
            
            # Process request (get full answer first)
            response = await answer_service.process_answer_request(
                rag_chain,
                request_data,
                request_id,
                prompt_settings,
                client_ip,
                request.headers.get("User-Agent"),
                accept_language_header,
                vectorstore
            )
            
            # Send ID event
            id_event = SSEEvent(
                type="id",
                conversation_id=response.conversation_id,
                request_id=response.request_id
            )
            yield f"data: {id_event.json()}\n\n"
            
            # Pseudo-stream answer (chunk into ~80 char pieces)
            answer_chunks = chunk_text_for_streaming(response.answer, chunk_size=80)
            for chunk in answer_chunks:
                answer_event = SSEEvent(
                    type="answer",
                    delta=chunk
                )
                yield f"data: {answer_event.json()}\n\n"
            
            # Send sources
            for source in response.sources:
                source_event = SSEEvent(
                    type="source",
                    source=source
                )
                yield f"data: {source_event.json()}\n\n"
            
            # Send end event with metrics
            end_event = SSEEvent(
                type="end",
                metrics=response.metrics
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
    
    return StreamingResponse(generate_stream(), media_type="text/event-stream")

