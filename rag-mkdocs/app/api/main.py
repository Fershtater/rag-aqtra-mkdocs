"""
FastAPI application for RAG assistant MkDocs.

Rationale for choosing FastAPI for RAG API:

1. ASYNCHRONOUS:
   - Native async/await support for efficient LLM API work
   - Does not block event loop when waiting for OpenAI responses
   - Can handle multiple requests in parallel
   - Critical for RAG, where each request requires:
     * Vector search (can be slow)
     * OpenAI API calls (network delays)
     * Large context processing

2. PERFORMANCE:
   - One of the fastest Python web frameworks
   - Based on Starlette and Pydantic
   - Automatic data validation via Pydantic
   - Minimal overhead

3. TYPING AND VALIDATION:
   - Full type hints support
   - Automatic input data validation
   - Automatic OpenAPI/Swagger documentation generation
   - Improves reliability and debugging

4. AUTOMATIC DOCUMENTATION:
   - Swagger UI at /docs
   - ReDoc at /redoc
   - Allows testing API without additional tools
   - Simplifies integration for clients

5. LANGCHAIN INTEGRATION:
   - LangChain supports async calls
   - FastAPI easily integrates with async chains
   - Can use background tasks for long operations

6. SCALABILITY:
   - Easy to add middleware (logging, CORS, authentication)
   - WebSocket support for streaming responses
   - Can easily add rate limiting
   - Ready for production deployment
"""

import asyncio
import logging
import os
import uuid
from contextlib import asynccontextmanager
from typing import Dict, List, Optional
from time import time

from dotenv import load_dotenv
from fastapi import FastAPI, Header, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr, Field, validator

from app.core.rag_chain import (
    build_or_load_vectorstore,
    build_rag_chain,
    build_rag_chain_and_settings,
    chunk_documents,
    load_mkdocs_documents,
)
from app.core.prompt_config import (
    load_prompt_settings_from_env,
    detect_response_language,
    get_prompt_template_content,
    is_jinja_mode,
    build_system_prompt,
    get_selected_template_info,
)
from app.core.prompt_renderer import PromptRenderer
from app.api.answering import (
    build_system_namespace,
    build_source_namespace,
    render_system_prompt,
)
from app.core.markdown_utils import build_doc_url
from app.infra.rate_limit import query_limiter, update_limiter, escalate_limiter
from app.infra.cache import response_cache
from app.infra.metrics import (
    get_metrics_response,
    update_index_metrics,
    query_requests_total,
    update_index_requests_total,
    rate_limit_hits_total,
    query_latency_seconds,
    update_index_duration_seconds,
    PROMETHEUS_AVAILABLE,
)
from app.infra.db import init_db
from app.infra.analytics import hash_ip, log_query, log_escalation
from app.infra.zoho_desk import create_ticket
from app.api.v2_models import AnswerRequest, AnswerResponse, ErrorResponseV2, SSEEvent, ErrorPayload, MetricsPayload
from app.api.answering import process_answer_request
from fastapi.responses import StreamingResponse

# Load environment variables from .env file
load_dotenv()

# Configure logging with LOG_LEVEL support
env = os.getenv("ENV", "production").lower()
log_level_str = os.getenv("LOG_LEVEL", "DEBUG" if env == "development" else "INFO")
log_level = getattr(logging, log_level_str.upper(), logging.INFO)

logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

logger.info(f"Logging configured: level={log_level_str}, mode={env}")

# Check for required environment variables on startup
api_key = os.getenv("OPENAI_API_KEY")
if api_key is None:
    logger.error("OPENAI_API_KEY not found in environment variables")
    raise ValueError(
        "OPENAI_API_KEY not set in .env\n"
        "Create .env file from .env.example and add OPENAI_API_KEY=your-key"
    )
logger.info("✓ OPENAI_API_KEY found in environment variables")


# Pydantic models for data validation
class Query(BaseModel):
    """User request model."""
    question: str = Field(..., max_length=2000, description="User question (max 2000 characters)")
    page_url: Optional[str] = Field(None, description="URL of the page from which the question was asked")
    page_title: Optional[str] = Field(None, description="Title of the page from which the question was asked")
    
    @validator('question')
    def validate_question_length(cls, v):
        if len(v.strip()) > 2000:
            raise ValueError("Question is too long (maximum 2000 characters)")
        return v.strip()
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "How do I use this function?",
                "page_url": "https://your-app.example.com/docs",
                "page_title": "Documentation page"
            }
        }


class QueryResponse(BaseModel):
    """RAG system response model."""
    answer: str
    sources: List[Dict[str, str]]
    not_found: bool
    request_id: str
    latency_ms: int
    cache_hit: bool
    
    class Config:
        json_schema_extra = {
            "example": {
                "answer": "To use this function you need...",
                "sources": [
                    {
                        "source": "data/mkdocs_docs/docs/example.md",
                        "filename": "example.md"
                    }
                ]
            }
        }


class EscalateRequest(BaseModel):
    """Support escalation request."""

    email: EmailStr
    request_id: str
    comment: Optional[str] = None

def calculate_effective_top_k(question: str, base_top_k: int, mode: str) -> int:
    """
    Simple top_k adaptation based on question length.
    
    Args:
        question: User question text
        base_top_k: Base top_k value (from settings or request)
        mode: Prompt mode (strict|helpful) — reserved for future extensions
    
    Returns:
        Adapted top_k value within reasonable limits.
    """
    words = question.split()
    length = len(words)

    k = base_top_k
    if length > 20:
        k = max(base_top_k, 8)
    elif length > 10:
        k = max(base_top_k, 6)

    # Limit from above to avoid overloading retriever
    return min(max(1, k), 10)


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str
    detail: Optional[str] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context for initialization and resource cleanup.
    
    Executed on startup:
    - Loads vectorstore
    - Creates RAG chain
    - Saves to app.state
    
    Executed on shutdown:
    - Cleans up resources
    """
    # Startup: RAG chain initialization
    logger.info("=" * 60)
    logger.info("STARTING FASTAPI APPLICATION")
    logger.info("=" * 60)
    logger.info("Initializing RAG chain...")
    
    try:
        # Load vectorstore and create RAG chain with settings
        rag_chain, vectorstore, prompt_settings = build_rag_chain_and_settings()
        
        # Save to app.state for access from endpoints
        app.state.vectorstore = vectorstore
        app.state.rag_chain = rag_chain
        app.state.prompt_settings = prompt_settings
        
        logger.info("RAG chain ready for use")
        logger.info("=" * 60)
    except Exception as e:
        logger.error(f"Error initializing RAG chain: {e}", exc_info=True)
        logger.error("Application started but RAG is unavailable")
        app.state.vectorstore = None
        app.state.rag_chain = None
        app.state.prompt_settings = None

    # Initialize database for logging if DATABASE_URL is configured
    db_url = os.getenv("DATABASE_URL")
    if db_url:
        try:
            logger.info("Initializing database connection for logging...")
            db_sessionmaker = await init_db(db_url)
            app.state.db_sessionmaker = db_sessionmaker
            logger.info("Database connection successfully initialized")
        except Exception as e:
            logger.error("Failed to initialize database: %s", e, exc_info=True)
            app.state.db_sessionmaker = None
    else:
        logger.info("DATABASE_URL not set, database logging disabled")
        app.state.db_sessionmaker = None
    
    yield
    
    # Shutdown: resource cleanup
    logger.info("Stopping application...")
    app.state.vectorstore = None
    app.state.rag_chain = None
    app.state.prompt_settings = None
    app.state.db_sessionmaker = None


# Create FastAPI application with lifespan
app = FastAPI(
    title="RAG MkDocs Assistant API",
    description="API for documentation questions using RAG",
    version="1.0.0",
    lifespan=lifespan
)

# Middleware for correlation IDs
@app.middleware("http")
async def correlation_id_middleware(request: Request, call_next):
    """Adds correlation ID to each request."""
    # Read or generate request ID
    request_id = request.headers.get("X-Request-Id", str(uuid.uuid4()))
    request.state.request_id = request_id
    
    # Update logging format to include request_id
    old_factory = logging.getLogRecordFactory()
    def record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        record.request_id = request_id
        return record
    logging.setLogRecordFactory(record_factory)
    
    response = await call_next(request)
    response.headers["X-Request-Id"] = request_id
    return response

# Add CORS middleware for frontend integration
# CORS origins configured via environment variable CORS_ORIGINS (comma-separated list)
# If not specified, "*" is used only in development mode
cors_origins_str = os.getenv("CORS_ORIGINS", "")
if cors_origins_str:
    cors_origins = [origin.strip() for origin in cors_origins_str.split(",") if origin.strip()]
else:
    # Fallback: allow all only in development
    cors_origins = ["*"] if env == "development" else []

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],  # Limit methods
    allow_headers=["Content-Type", "X-API-Key", "X-Request-Id"],  # Limit headers
)


@app.get("/")
async def root():
    """Root endpoint to check API status."""
    return {
        "message": "RAG MkDocs Assistant API",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/metrics")
async def metrics_endpoint():
    """Metrics endpoint Prometheus."""
    content, content_type = get_metrics_response()
    return Response(content=content, media_type=content_type)


@app.get("/health")
async def health_check(request: Request):
    """
    Health check endpoint for monitoring.
    
    Returns:
        JSON with application status and RAG chain readiness
    """
    rag_chain = getattr(request.app.state, "rag_chain", None)
    return {
        "status": "ok" if rag_chain is not None else "degraded",
        "rag_chain_ready": rag_chain is not None
    }


@app.get("/config/prompt")
async def get_prompt_config(request: Request):
    """
    Returns current prompt settings.
    
    Returns:
        JSON with prompt settings from app.state.prompt_settings
    """
    prompt_settings = getattr(request.app.state, "prompt_settings", None)
    if prompt_settings is None:
        return JSONResponse(
            status_code=503,
            content={
                "error": "Prompt settings not loaded",
                "detail": "RAG chain not initialized"
            }
        )
    
    return {
        "supported_languages": list(prompt_settings.supported_languages),
        "fallback_language": prompt_settings.fallback_language,
        "base_docs_url": prompt_settings.base_docs_url,
        "not_found_message": prompt_settings.not_found_message,
        "include_sources_in_text": prompt_settings.include_sources_in_text,
        "mode": prompt_settings.mode,
        "default_temperature": prompt_settings.default_temperature,
        "default_top_k": prompt_settings.default_top_k,
        "default_max_tokens": getattr(prompt_settings, "default_max_tokens", None),
    }


@app.post("/query", response_model=QueryResponse, responses={429: {"model": ErrorResponse}, 503: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def query_documentation(query: Query, request: Request):
    """
    Main endpoint for documentation questions.
    
    Accepts user question and returns answer based on RAG system.
    Supports caching, rate limiting and metrics.
    
    Args:
        query: Query object with user question
        request: FastAPI Request for accessing app.state
        
    Returns:
        QueryResponse with answer and list of sources
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
        
        # Settings signature for cache: include language configuration for cache stability
        # Include detector_version to invalidate cache if detection rules change
        detector_version = "v1"
        reranking_enabled = os.getenv("RERANKING_ENABLED", "0").lower() in ("1", "true", "yes")
        settings_signature = (
            f"mode={prompt_settings.mode}_"
            f"top_k={prompt_settings.default_top_k}_"
            f"temp={prompt_settings.default_temperature}_"
            f"max_tokens={getattr(prompt_settings, 'default_max_tokens', None)}_"
            f"supported={','.join(sorted(prompt_settings.supported_languages))}_"
            f"fallback={prompt_settings.fallback_language}_"
            f"rerank={reranking_enabled}_"
            f"detector={detector_version}"
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
            # Do a preliminary call to get docs
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
            system_ns = build_system_namespace(
                request_id,
                None,
                prompt_settings.mode,
                passthrough=passthrough_ns,
                context_hint=None,
                accept_language_header=None  # TODO: pass from request if available
            )
            source_ns = build_source_namespace(context_docs, prompt_settings)
            tools_ns = {}
            
            # Render system prompt
            rendered_prompt = render_system_prompt(
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
            # Still select language for consistency
            passthrough_ns = {}
            if query.page_url:
                passthrough_ns["page_url"] = query.page_url
            if query.page_title:
                passthrough_ns["page_title"] = query.page_title
            
            system_ns = build_system_namespace(
                request_id,
                None,
                prompt_settings.mode,
                passthrough=passthrough_ns,
                context_hint=None,
                accept_language_header=None
            )
            response_language = system_ns["output_language"]  # Use selected language
            default_system_prompt = build_system_prompt(prompt_settings, response_language=response_language)
            chain_input = {
                "input": query.question,
                "response_language": response_language,
                "chat_history": "",
                "system_prompt": default_system_prompt
            }
        
        # Call RAG chain (or re-invoke for Jinja2 mode)
        try:
            if is_jinja_mode():
                # Re-invoke with rendered prompt
                result = await asyncio.to_thread(rag_chain.invoke, chain_input)
            else:
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
        
        # Extract sources from metadata of found documents
        # In LangChain 0.2.0 create_retrieval_chain returns:
        # - "answer": generated answer
        # - "context": list of Document objects used for generation
        sources = []
        
        # Extract sources from context
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
                    
                    # Add new fields if available
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
        
        # Determine not_found by retrieval signals (sources count and score threshold)
        # Do not rely on exact string match to support multilingual output
        not_found_score_threshold = float(os.getenv("NOT_FOUND_SCORE_THRESHOLD", "0.20"))
        not_found_flag = len(sources) == 0
        
        # If sources have scores, also check threshold
        if sources:
            # Extract scores from sources if available
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
            # Skip logging errors to avoid masking the main error
            pass
        return JSONResponse(
            status_code=500,
            content={
                "error": "Error processing request",
                "detail": "Internal server error"
            }
        )


@app.post("/escalate", responses={400: {"model": ErrorResponse}, 404: {"model": ErrorResponse}, 429: {"model": ErrorResponse}, 503: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def escalate_issue(payload: EscalateRequest, request: Request):
    """
    Escalation endpoint to support service (Zoho Desk).
    
    Requires that original request had not_found=true.
    """
    request_id = getattr(request.state, "request_id", "unknown")
    client_ip = request.client.host if request.client else "unknown"

    # Rate limiting by IP
    allowed, error_msg = escalate_limiter.is_allowed(client_ip)
    if not allowed:
        if PROMETHEUS_AVAILABLE and rate_limit_hits_total is not None:
            rate_limit_hits_total.labels(endpoint="escalate").inc()
        logger.warning("[%s] Escalate rate limit exceeded for %s", request_id, client_ip)
        return JSONResponse(
            status_code=429,
            content={"error": "Rate limit exceeded", "detail": error_msg},
        )

    db_sessionmaker = getattr(request.app.state, "db_sessionmaker", None)
    if db_sessionmaker is None:
        logger.error("[%s] DATABASE_URL not configured, escalation unavailable", request_id)
        return JSONResponse(
            status_code=503,
            content={
                "error": "Escalation is not configured",
                "detail": "DATABASE_URL is not set on the server",
            },
        )

    # Search for corresponding QueryLog record
    from sqlalchemy import select
    from app.infra.models import QueryLog
    from sqlalchemy.exc import SQLAlchemyError

    try:
        async with db_sessionmaker() as session:
            result = await session.execute(
                select(QueryLog).where(QueryLog.request_id == payload.request_id)
            )
            query_log = result.scalar_one_or_none()
    except SQLAlchemyError as e:
        logger.error("[%s] Error searching QueryLog for escalation: %s", request_id, e, exc_info=True)
        await log_escalation(
            db_sessionmaker,
            request_id=payload.request_id,
            email=payload.email,
            status="db_error",
            error=str(e),
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": "Database error",
                "detail": "Failed to load query log for escalation",
            },
        )

    if query_log is None:
        logger.warning("[%s] QueryLog not found for request_id=%s", request_id, payload.request_id)
        await log_escalation(
            db_sessionmaker,
            request_id=payload.request_id,
            email=payload.email,
            status="not_found",
            error="QueryLog not found",
        )
        return JSONResponse(
            status_code=404,
            content={
                "error": "Query not found",
                "detail": "Cannot escalate request that is not logged",
            },
        )

    if not query_log.not_found:
        logger.warning("[%s] Escalation requested for non-not-found query_id=%s", request_id, payload.request_id)
        await log_escalation(
            db_sessionmaker,
            request_id=payload.request_id,
            email=payload.email,
            status="rejected",
            error="Escalation allowed only for not_found queries",
        )
        return JSONResponse(
            status_code=400,
            content={
                "error": "Escalation not allowed",
                "detail": "Escalation is allowed only when the original answer was not_found",
            },
        )

    # Constants for limiting text length in ticket
    TICKET_TEXT_LIMIT = 8000  # Maximum length answer in ticket
    QUESTION_LIMIT = 2000  # Maximum length question in ticket
    COMMENT_LIMIT = 2000  # Maximum length comment in ticket

    # Form subject and description for ticket
    question = (query_log.question or "")[:QUESTION_LIMIT]
    answer_text = (query_log.answer or "[no answer captured]")[:TICKET_TEXT_LIMIT]
    subject = f"Aqtra Docs: {question[:80] or 'User escalation'}"

    parts = [
        f"User email: {payload.email}",
        f"Request ID: {payload.request_id}",
        "",
        f"Question: {question}",
        f"Answer: {answer_text}",
        f"Not found: {query_log.not_found}",
        f"Page URL: {query_log.page_url or '-'}",
        f"Page Title: {query_log.page_title or '-'}",
        "",
    ]

    # Format sources in readable form
    sources = query_log.sources
    if not sources:
        parts.append("Sources: -")
    else:
        parts.append("Sources:")
        try:
            for source in sources:
                title = source.get("section_title") or source.get("title") or source.get("source", "Unknown")
                url = source.get("url") or "-"
                score = source.get("score")
                if score is not None:
                    parts.append(f"- {title} — {url} (score={score:.3f})")
                else:
                    parts.append(f"- {title} — {url}")
        except (TypeError, AttributeError, KeyError):
            # Fallback if sources in unexpected format
            parts.append(f"Sources: {sources}")

    if payload.comment:
        comment_text = (payload.comment or "")[:COMMENT_LIMIT]
        parts.append("")
        parts.append(f"User comment: {comment_text}")

    description = "\n".join(parts)

    # Create ticket in Zoho Desk
    zoho_ticket_id = None
    zoho_ticket_number = None

    try:
        ticket_response = await create_ticket(
            email=payload.email,
            subject=subject,
            description=description,
        )
        zoho_ticket_id = str(ticket_response.get("id") or "")
        zoho_ticket_number = str(ticket_response.get("ticketNumber") or "")

        await log_escalation(
            db_sessionmaker,
            request_id=payload.request_id,
            email=payload.email,
            status="success",
            zoho_ticket_id=zoho_ticket_id or None,
            zoho_ticket_number=zoho_ticket_number or None,
            error=None,
        )

        return {
            "status": "success",
            "ticket_id": zoho_ticket_id,
            "ticket_number": zoho_ticket_number,
        }
    except Exception as e:
        logger.error("[%s] Error creating ticket Zoho Desk: %s", request_id, e, exc_info=True)
        await log_escalation(
            db_sessionmaker,
            request_id=payload.request_id,
            email=payload.email,
            status="error",
            error=str(e),
        )
        return JSONResponse(
            status_code=503,
            content={
                "error": "Escalation failed",
                "detail": "Failed to create ticket in Zoho Desk",
            },
        )


@app.post("/update_index", responses={200: {"model": Dict}, 401: {"model": ErrorResponse}, 429: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def update_index(
    request: Request,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key")
):
    """
    Endpoint for forced vector index update.
    
    Protected by API key. Recreates index once and updates app.state.
    Supports rate limiting and metrics.
    
    Args:
        request: FastAPI Request for accessing app.state
        x_api_key: API key in X-API-Key header
        
    Returns:
        JSON with index update result
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
    
    # Check API key
    required_api_key = os.getenv("UPDATE_API_KEY")
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
        documents = load_mkdocs_documents()
        if not documents:
            return JSONResponse(
                status_code=500,
                content={
                    "error": "No documents found",
                    "detail": "Make sure that in data/mkdocs_docs has .md files"
                }
            )
        
        chunks = chunk_documents(documents)
        logger.info("Loaded documents, created chunks")
        
        # Recreate index ONCE
        vectorstore = build_or_load_vectorstore(
            chunks=chunks,
            force_rebuild=True
        )
        
        # Load prompt settings
        prompt_settings = load_prompt_settings_from_env()
        
        # Create new RAG chain from already rebuilt vectorstore
        # (do not call get_rag_chain, to avoid reloading)
        rag_chain = build_rag_chain(
            vectorstore,
            prompt_settings=prompt_settings,
            k=prompt_settings.default_top_k,
            temperature=prompt_settings.default_temperature
        )
        
        # Update app.state
        request.app.state.vectorstore = vectorstore
        request.app.state.rag_chain = rag_chain
        request.app.state.prompt_settings = prompt_settings

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


def validate_api_key(api_key: str) -> bool:
    """
    Validate API key from RAG_API_KEYS environment variable.
    
    Args:
        api_key: API key to validate
        
    Returns:
        True if valid or if RAG_API_KEYS is not set (open access with warning)
    """
    rag_api_keys_str = os.getenv("RAG_API_KEYS", "")
    if not rag_api_keys_str:
        logger.warning("RAG_API_KEYS not set, allowing open access (consider setting it)")
        return True
    
    allowed_keys = [key.strip() for key in rag_api_keys_str.split(",") if key.strip()]
    return api_key in allowed_keys


@app.post("/api/answer", response_model=AnswerResponse, responses={400: {"model": ErrorResponseV2}, 401: {"model": ErrorResponseV2}, 429: {"model": ErrorResponseV2}, 503: {"model": ErrorResponseV2}, 500: {"model": ErrorResponseV2}})
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
        
        # Get user agent
        user_agent = request.headers.get("User-Agent")
        
        # Process request
        response = await process_answer_request(
            rag_chain,
            request_data,
            request_id,
            prompt_settings,
            db_sessionmaker,
            client_ip,
            user_agent
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


def chunk_text_for_streaming(text: str, chunk_size: int = 80) -> List[str]:
    """
    Split text into chunks for pseudo-streaming.
    
    Args:
        text: Text to chunk
        chunk_size: Target chunk size in characters
        
    Returns:
        List of text chunks
    """
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i + chunk_size])
    return chunks


@app.post("/stream")
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
            
            # Process request (get full answer first)
            response = await process_answer_request(
                rag_chain,
                request_data,
                request_id,
                prompt_settings,
                db_sessionmaker,
                client_ip,
                request.headers.get("User-Agent")
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


@app.post("/api/prompt/render", responses={400: {"model": ErrorResponseV2}, 401: {"model": ErrorResponseV2}, 429: {"model": ErrorResponseV2}, 503: {"model": ErrorResponseV2}, 500: {"model": ErrorResponseV2}})
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
        # We'll do a quick retrieval to get documents for namespace
        context_docs = []
        try:
            # Access retriever from rag_chain if possible
            # For now, do a minimal invoke to get context
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
            # Continue with empty context_docs
        
        # Build passthrough namespace (needed for language selection)
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
        
        system_ns = build_system_namespace(
            request_id,
            conversation_id,
            prompt_settings.mode,
            passthrough=passthrough_ns,
            context_hint=context_hint_dict,
            accept_language_header=accept_language_header
        )
        source_ns = build_source_namespace(context_docs, prompt_settings)
        
        tools_ns = {}
        
        # Get selected template info
        template_info = get_selected_template_info()
        
        # Render prompt
        rendered_prompt = ""
        if template_is_valid:
            try:
                rendered_prompt = render_system_prompt(
                    template_str,
                    system_ns,
                    source_ns,
                    passthrough_ns,
                    tools_ns,
                    request_id
                )
            except Exception as e:
                errors.append(f"Render error: {str(e)}")
                rendered_prompt = template_str  # Fallback
        else:
            rendered_prompt = template_str  # Use template as-is if invalid
        
        # Truncate rendered prompt for response (max 20k chars)
        rendered_prompt_display = rendered_prompt[:20000] if len(rendered_prompt) > 20000 else rendered_prompt
        
        # Build documents preview (max 3 items)
        documents_preview = source_ns.get("documents", [])[:3]
        for doc in documents_preview:
            # Truncate snippet
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


if __name__ == "__main__":
    """
    Running application via uvicorn.
    
    Usage:
        python -m app.api.main
        
    Or via uvicorn directly:
        uvicorn app.api.main:app --reload --port 8000
    """
    import uvicorn
    
    # Start server
    uvicorn.run(
        "app.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )

