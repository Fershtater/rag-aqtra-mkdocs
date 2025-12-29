"""
V2 answer endpoint (DocsGPT-like).
"""

import logging
from typing import Optional
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.api.schemas.v2 import AnswerRequest, AnswerResponse, ErrorResponseV2
from app.core.prompt_config import load_prompt_settings_from_env
from app.services.answer_service import AnswerService
from app.services.conversation_service import ConversationService
from app.services.prompt_service import PromptService
from app.infra.rate_limit import query_limiter
from app.infra.metrics import (
    query_requests_total,
    rate_limit_hits_total,
    query_latency_seconds,
    PROMETHEUS_AVAILABLE,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def validate_api_key(api_key: Optional[str]) -> bool:
    """
    Validate API key from RAG_API_KEYS environment variable.
    
    Args:
        api_key: API key to validate (can be None in open mode)
        
    Returns:
        True if valid or if RAG_API_KEYS not set (open mode), False otherwise
    """
    import os
    rag_api_keys_str = os.getenv("RAG_API_KEYS", "")
    if not rag_api_keys_str:
        # Open mode: no API key required
        return True
    
    # Closed mode: API key is required
    if not api_key:
        return False
    
    allowed_keys = [key.strip() for key in rag_api_keys_str.split(",") if key.strip()]
    return api_key in allowed_keys


@router.post("/api/answer", response_model=AnswerResponse, responses={400: {"model": ErrorResponseV2}, 401: {"model": ErrorResponseV2}, 429: {"model": ErrorResponseV2}, 503: {"model": ErrorResponseV2}, 500: {"model": ErrorResponseV2}})
async def answer_question(request_data: AnswerRequest, request: Request):
    """
    DocsGPT-like answer endpoint.
    
    Accepts question with optional history and returns answer in v2 format.
    Supports conversation history and caching.
    """
    request_id = getattr(request.state, "request_id", "unknown")
    
    # Validate API key
    if not validate_api_key(request_data.api_key):
        logger.warning(f"[{request_id}] Invalid API key")
        return JSONResponse(
            status_code=401,
            content={
                "error": "Invalid API key",
                "detail": "API key is not authorized",
                "request_id": request_id,
                "code": "INVALID_API_KEY"
            }
        )
    
    # Rate limiting
    client_ip = request.client.host if request.client else "unknown"
    allowed, error_msg = query_limiter.is_allowed(client_ip)
    if not allowed:
        if PROMETHEUS_AVAILABLE and rate_limit_hits_total is not None:
            rate_limit_hits_total.labels(endpoint="api/answer").inc()
        logger.warning(f"[{request_id}] Rate limit exceeded for {client_ip}")
        return JSONResponse(
            status_code=429,
            content={
                "error": "Rate limit exceeded",
                "detail": error_msg,
                "request_id": request_id,
                "code": "RATE_LIMIT_EXCEEDED"
            }
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
                "detail": "Please try again later or check application logs",
                "request_id": request_id,
                "code": "SERVICE_UNAVAILABLE"
            }
        )
    
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
        
        # Get index version
        index_version = getattr(request.app.state, "index_version", None)
        
        # Get services from app.state (created in lifespan)
        answer_service = getattr(request.app.state, "answer_service", None)
        if answer_service is None:
            # Fallback: create services if not in app.state
            conversation_service = ConversationService(db_sessionmaker)
            prompt_service = PromptService()
            answer_service = AnswerService(conversation_service, prompt_service)
        
        # Process request
        response = await answer_service.process_answer_request(
            rag_chain,
            request_data,
            request_id,
            prompt_settings,
            client_ip,
            request.headers.get("User-Agent"),
            accept_language_header,
            vectorstore,
            index_version=index_version,
            endpoint_name="api/answer"
        )
        
        # Update metrics
        if PROMETHEUS_AVAILABLE and query_requests_total is not None:
            query_requests_total.labels(status="success").inc()
            query_latency_seconds.observe(response.metrics.latency_ms / 1000.0)
        
        return response
        
    except Exception as e:
        logger.error(f"[{request_id}] Error processing answer request: {e}", exc_info=True)
        if PROMETHEUS_AVAILABLE and query_requests_total is not None:
            query_requests_total.labels(status="error").inc()
        return JSONResponse(
            status_code=500,
            content={
                "error": "Error processing request",
                "detail": "Internal server error",
                "request_id": request_id,
                "code": "INTERNAL_ERROR"
            }
        )

