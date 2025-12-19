"""
Admin endpoint for updating vector index.
"""

import os
import logging
from typing import Dict, Optional
from time import time

from fastapi import APIRouter, Request, Header, HTTPException
from fastapi.responses import JSONResponse
from fastapi.responses import Response

from app.core.rag_chain import (
    build_or_load_vectorstore,
    build_rag_chain,
    chunk_documents,
    load_mkdocs_documents,
)
from app.core.prompt_config import load_prompt_settings_from_env
from app.rag.index_meta import get_index_version
from app.settings import get_settings
from app.infra.rate_limit import update_limiter
from app.infra.cache import response_cache
from app.infra.metrics import (
    update_index_metrics,
    update_index_requests_total,
    update_index_duration_seconds,
    PROMETHEUS_AVAILABLE,
    rate_limit_hits_total,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/update_index", responses={200: {"model": Dict}, 401: {"model": dict}, 429: {"model": dict}, 500: {"model": dict}})
async def update_index(
    request: Request,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key")
):
    """
    Endpoint for forced vector index update.
    
    Protected by API key. Recreates index once and updates app.state.
    Supports rate limiting and metrics.
    """
    request_id = getattr(request.state, "request_id", "unknown")
    start_time = time()
    
    # Rate limiting by API key or IP
    limiter_key = x_api_key or (request.client.host if request.client else "unknown")
    allowed, error_msg = update_limiter.is_allowed(limiter_key)
    if not allowed:
        if PROMETHEUS_AVAILABLE and rate_limit_hits_total is not None:
            rate_limit_hits_total.labels(endpoint="update_index").inc()
        logger.warning(f"[{request_id}] Rate limit exceeded for update_index")
        return JSONResponse(
            status_code=429,
            content={"error": "Rate limit exceeded", "detail": error_msg}
        )
    
    # Get settings
    settings = getattr(request.app.state, "settings", None)
    if not settings:
        settings = get_settings()
    
    # Check API key
    required_api_key = settings.UPDATE_API_KEY
    if not required_api_key:
        logger.warning(f"[{request_id}] UPDATE_API_KEY not set in .env")
        return JSONResponse(
            status_code=503,
            content={
                "error": "Endpoint unavailable",
                "detail": "UPDATE_API_KEY not configured"
            }
        )
    
    if not x_api_key or x_api_key != required_api_key:
        logger.warning(f"[{request_id}] Invalid API key for index update")
        if PROMETHEUS_AVAILABLE and update_index_requests_total is not None:
            update_index_requests_total.labels(status="error").inc()
        return JSONResponse(
            status_code=401,
            content={
                "error": "Invalid API key",
                "detail": "Specify correct X-API-Key in request header"
            }
        )
    
    try:
        logger.info(f"[{request_id}] Started index update...")
        
        # Load and chunk documents
        documents = load_mkdocs_documents(settings.DOCS_PATH)
        if not documents:
            return JSONResponse(
                status_code=500,
                content={
                    "error": "No documents found",
                    "detail": "Make sure that in data/mkdocs_docs has .md files"
                }
            )
        
        chunks = chunk_documents(
            documents,
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            min_chunk_size=settings.MIN_CHUNK_SIZE
        )
        logger.info("Loaded documents, created chunks")
        
        # Recreate index ONCE (with lock timeout from settings)
        try:
            vectorstore = build_or_load_vectorstore(
                chunks=chunks,
                force_rebuild=True,
                lock_timeout_seconds=settings.INDEX_LOCK_TIMEOUT_SECONDS,
                docs_path=settings.DOCS_PATH,
                index_path=settings.VECTORSTORE_DIR
            )
        except Exception as e:
            # Check if it's a lock-related error
            error_str = str(e).lower()
            if "lock" in error_str or "423" in error_str or "already in progress" in error_str:
                # Lock acquisition failed
                logger.warning(f"[{request_id}] Index rebuild lock failed: {e}")
                if PROMETHEUS_AVAILABLE and update_index_requests_total is not None:
                    update_index_requests_total.labels(status="error").inc()
                return JSONResponse(
                    status_code=409,
                    content={
                        "error": "Index rebuild is already in progress",
                        "detail": "Try again later"
                    }
                )
            # Re-raise other exceptions
            raise
        
        # Load prompt settings
        prompt_settings = load_prompt_settings_from_env()
        
        # Create new RAG chain from already rebuilt vectorstore
        rag_chain = build_rag_chain(
            vectorstore,
            prompt_settings=prompt_settings,
            k=prompt_settings.default_top_k,
            temperature=prompt_settings.default_temperature
        )
        
        # Get index version after rebuild
        index_version = get_index_version(settings.VECTORSTORE_DIR)
        
        # Update app.state
        request.app.state.vectorstore = vectorstore
        request.app.state.rag_chain = rag_chain
        request.app.state.prompt_settings = prompt_settings
        request.app.state.index_version = index_version

        # Cache invalidation of responses after index recreation
        response_cache.clear()
        logger.info("[%s] Cache cleared after update_index", request_id)
        
        # Update index metrics
        if PROMETHEUS_AVAILABLE:
            update_index_metrics(len(documents), len(chunks))
            if update_index_requests_total is not None:
                update_index_requests_total.labels(status="success").inc()
            if update_index_duration_seconds is not None:
                update_index_duration_seconds.observe(time() - start_time)
        
        logger.info(f"[{request_id}] Index successfully updated and RAG chain recreated")
        return {
            "status": "success",
            "message": "Index successfully updated",
            "documents_count": len(documents),
            "chunks_count": len(chunks),
            "index_size": vectorstore.index.ntotal
        }
        
    except Exception as e:
        logger.error(f"[{request_id}] Error updating index: {e}", exc_info=True)
        if PROMETHEUS_AVAILABLE:
            if update_index_requests_total is not None:
                update_index_requests_total.labels(status="error").inc()
            if update_index_duration_seconds is not None:
                update_index_duration_seconds.observe(time() - start_time)
        return JSONResponse(
            status_code=500,
            content={
                "error": "Error updating index",
                "detail": "Internal server error"
            }
        )

