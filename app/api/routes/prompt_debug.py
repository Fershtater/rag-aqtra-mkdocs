"""
Debug endpoint for prompt rendering.
"""

import asyncio
import logging
import os
from typing import Optional
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.api.schemas.v2 import AnswerRequest, ErrorResponseV2
from app.core.prompt_config import (
    load_prompt_settings_from_env,
    detect_response_language,
    get_prompt_template_content,
    is_jinja_mode,
    get_selected_template_info,
)
from app.core.prompt_renderer import PromptRenderer
from app.services.prompt_service import PromptService
from app.infra.rate_limit import query_limiter
from app.infra.metrics import (
    rate_limit_hits_total,
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
    rag_api_keys_str = os.getenv("RAG_API_KEYS", "")
    if not rag_api_keys_str:
        # Open mode: no API key required
        return True
    
    # Closed mode: API key is required
    if not api_key:
        return False
    
    allowed_keys = [key.strip() for key in rag_api_keys_str.split(",") if key.strip()]
    return api_key in allowed_keys


@router.post("/api/prompt/render", responses={400: {"model": ErrorResponseV2}, 401: {"model": ErrorResponseV2}, 429: {"model": ErrorResponseV2}, 503: {"model": ErrorResponseV2}, 500: {"model": ErrorResponseV2}})
async def render_prompt(request_data: AnswerRequest, request: Request):
    """
    Debug endpoint to render prompt template without generating answer.
    
    Returns rendered prompt, namespaces, and validation status.
    """
    request_id = getattr(request.state, "request_id", "unknown")
    
    # Validate API key
    if not validate_api_key(request_data.api_key):
        logger.warning(f"[{request_id}] Invalid API key for /api/prompt/render")
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
            rate_limit_hits_total.labels(endpoint="api/prompt/render").inc()
        return JSONResponse(
            status_code=429,
            content={
                "error": "Rate limit exceeded",
                "detail": error_msg,
                "request_id": request_id,
                "code": "RATE_LIMIT_EXCEEDED"
            }
        )
    
    # Get RAG chain (needed for retrieval)
    rag_chain = getattr(request.app.state, "rag_chain", None)
    if rag_chain is None:
        return JSONResponse(
            status_code=503,
            content={
                "error": "RAG chain not initialized",
                "detail": "Please try again later",
                "request_id": request_id,
                "code": "SERVICE_UNAVAILABLE"
            }
        )
    
    try:
        # Get prompt settings
        prompt_settings = getattr(request.app.state, "prompt_settings", None)
        if prompt_settings is None:
            prompt_settings = load_prompt_settings_from_env()
        
        # Get template string
        template_str = get_prompt_template_content(prompt_settings)
        template_mode = "jinja" if is_jinja_mode() else "legacy"
        
        # Validate template if Jinja2 mode
        template_is_valid = True
        errors = []
        if is_jinja_mode():
            try:
                max_chars = int(os.getenv("PROMPT_MAX_CHARS", "40000"))
                strict_undefined = os.getenv("PROMPT_STRICT_UNDEFINED", "1").lower() in ("1", "true", "yes")
                renderer = PromptRenderer(max_chars=max_chars, strict_undefined=strict_undefined)
                renderer.validate_template(template_str)
            except Exception as e:
                template_is_valid = False
                errors.append(str(e))
        
        # Get context documents via retrieval (simulate)
        context_docs = []
        try:
            chain_input_preview = {
                "input": request_data.question,
                "response_language": detect_response_language(
                    request_data.question,
                    supported=set(prompt_settings.supported_languages),
                    fallback=prompt_settings.fallback_language
                ),
                "chat_history": "",
                "system_prompt": ""
            }
            preview_result = await asyncio.to_thread(rag_chain.invoke, chain_input_preview)
            
            if "context" in preview_result:
                context_docs = preview_result["context"]
                if not isinstance(context_docs, list):
                    context_docs = [context_docs]
            elif "source_documents" in preview_result:
                context_docs = preview_result["source_documents"]
                if not isinstance(context_docs, list):
                    context_docs = [context_docs]
        except Exception as e:
            logger.warning(f"[{request_id}] Error getting context docs for render: {e}")
        
        # Build passthrough namespace
        passthrough_ns = {}
        if request_data.passthrough:
            passthrough_ns.update(request_data.passthrough)
        if request_data.context_hint:
            passthrough_ns.update(request_data.context_hint.dict())
        
        # Get Accept-Language header if available
        accept_language_header = request.headers.get("Accept-Language")
        
        # Build namespaces
        conversation_id = None
        if request_data.conversation_id:
            conversation_id = request_data.conversation_id
        
        context_hint_dict = None
        if request_data.context_hint:
            context_hint_dict = request_data.context_hint.dict()
        
        # Create prompt service
        prompt_service = PromptService()
        
        system_ns = prompt_service.build_system_namespace(
            request_id,
            conversation_id,
            prompt_settings.mode,
            passthrough=passthrough_ns,
            context_hint=context_hint_dict,
            accept_language_header=accept_language_header
        )
        source_ns = prompt_service.build_source_namespace(context_docs, prompt_settings)
        
        tools_ns = {}
        
        # Get selected template info
        template_info = get_selected_template_info()
        
        # Render prompt
        rendered_prompt = ""
        if template_is_valid:
            try:
                rendered_prompt = prompt_service.render_system_prompt(
                    template_str,
                    system_ns,
                    source_ns,
                    passthrough_ns,
                    tools_ns,
                    request_id
                )
            except Exception as e:
                errors.append(f"Render error: {str(e)}")
                rendered_prompt = template_str
        else:
            rendered_prompt = template_str
        
        # Truncate rendered prompt for response (max 20k chars)
        rendered_prompt_display = rendered_prompt[:20000] if len(rendered_prompt) > 20000 else rendered_prompt
        
        # Build documents preview (max 3 items)
        documents_preview = source_ns.get("documents", [])[:3]
        for doc in documents_preview:
            if "snippet" in doc and len(doc["snippet"]) > 200:
                doc["snippet"] = doc["snippet"][:200] + "..."
        
        return {
            "template_mode": template_mode,
            "template_is_valid": template_is_valid,
            "rendered_prompt": rendered_prompt_display,
            "output_language": system_ns.get("output_language", "en"),
            "language_reason": system_ns.get("language_reason", "default"),
            "selected_template": template_info.get("selected_template", "legacy"),
            "selected_template_path": template_info.get("selected_template_path"),
            "namespaces": {
                "system": system_ns,
                "source_meta": {
                    "count": source_ns.get("count", 0),
                    "documents_preview": documents_preview
                },
                "passthrough": passthrough_ns,
                "tools": tools_ns
            },
            "errors": errors
        }
        
    except Exception as e:
        logger.error(f"[{request_id}] Error in /api/prompt/render: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": "Error rendering prompt",
                "detail": str(e),
                "request_id": request_id,
                "code": "INTERNAL_ERROR"
            }
        )

