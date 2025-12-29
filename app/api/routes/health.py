"""
Health check endpoint.
"""

import os
from fastapi import APIRouter, Request, Header
from typing import Optional

router = APIRouter()


def mask_secret(value: Optional[str]) -> str:
    """
    Mask secret values for safe logging/display.
    
    Args:
        value: Secret value to mask
        
    Returns:
        "***" if value exists, empty string otherwise
    """
    if not value:
        return ""
    return "***"


@router.get("/health")
async def health_check(
    request: Request,
    x_debug: Optional[str] = Header(None, alias="X-Debug")
):
    """
    Health check endpoint for monitoring.
    
    Returns:
        JSON with application status and RAG chain readiness.
        Includes diagnostics if ENV != "production" or X-Debug: 1 header is present.
    """
    rag_chain = getattr(request.app.state, "rag_chain", None)
    
    response = {
        "status": "ok" if rag_chain is not None else "degraded",
        "rag_chain_ready": rag_chain is not None
    }
    
    # Add diagnostics if in development or debug header is present
    settings = getattr(request.app.state, "settings", None)
    if settings:
        env = settings.ENV
        debug_header = x_debug == "1"
        
        if env != "production" or debug_header:
            diagnostics = {
                "env": env,
                "log_level": settings.LOG_LEVEL,
                "prompt": {
                    "template_mode": settings.PROMPT_TEMPLATE_MODE,
                    "preset": settings.PROMPT_PRESET,
                    "prompt_dir": settings.PROMPT_DIR,
                    "validate_on_startup": settings.PROMPT_VALIDATE_ON_STARTUP,
                    "strict_undefined": settings.PROMPT_STRICT_UNDEFINED,
                },
                "vectorstore_dir": settings.VECTORSTORE_DIR,
                "docs_path": settings.DOCS_PATH,
                "index_version": getattr(request.app.state, "index_version", None),
                "cache": {
                    "ttl_seconds": settings.CACHE_TTL_SECONDS,
                    "max_size": settings.CACHE_MAX_SIZE,
                },
                "rate_limit": {
                    "query": {
                        "limit": settings.QUERY_RATE_LIMIT,
                        "window_seconds": settings.QUERY_RATE_WINDOW_SECONDS,
                    },
                    "update": {
                        "limit": settings.UPDATE_RATE_LIMIT,
                        "window_seconds": settings.UPDATE_RATE_WINDOW_SECONDS,
                    },
                    "escalate": {
                        "limit": settings.ESCALATE_RATE_LIMIT,
                        "window_seconds": settings.ESCALATE_RATE_WINDOW_SECONDS,
                    },
                },
            }
            
            # Mask secrets (never include in diagnostics)
            # Note: OPENAI_API_KEY, UPDATE_API_KEY, ZOHO_* are not included in diagnostics
            
            response["diagnostics"] = diagnostics
    
    return response

