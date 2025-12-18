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

import logging
import os
import uuid
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.rag_chain import build_rag_chain_and_settings
from app.infra.db import init_db
from app.api.routes import (
    health,
    metrics,
    query_v1,
    answer_v2,
    stream,
    prompt_debug,
    admin_index,
    escalate,
)

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


# Pydantic models moved to app.api.schemas


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
        # Validate prompt template on startup if enabled
        validate_on_startup = os.getenv("PROMPT_VALIDATE_ON_STARTUP", "1").lower() in ("1", "true", "yes")
        if validate_on_startup:
            try:
                from app.core.prompt_config import get_prompt_template_content, is_jinja_mode
                from app.core.prompt_renderer import PromptRenderer
                
                if is_jinja_mode():
                    template_str = get_prompt_template_content()
                    max_chars = int(os.getenv("PROMPT_MAX_CHARS", "40000"))
                    strict_undefined = os.getenv("PROMPT_STRICT_UNDEFINED", "1").lower() in ("1", "true", "yes")
                    renderer = PromptRenderer(max_chars=max_chars, strict_undefined=strict_undefined)
                    renderer.validate_template(template_str)
                    logger.info("✓ Prompt template validated successfully")
                else:
                    logger.info("Prompt template validation skipped (legacy mode)")
            except Exception as e:
                logger.error(f"Prompt template validation failed: {e}", exc_info=True)
                fail_hard = os.getenv("PROMPT_FAIL_HARD", "0").lower() in ("1", "true", "yes")
                if fail_hard:
                    raise
        
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


# Include routers
app.include_router(health.router)
app.include_router(metrics.router)
app.include_router(query_v1.router)
app.include_router(answer_v2.router)
app.include_router(stream.router)
app.include_router(prompt_debug.router)
app.include_router(admin_index.router)
app.include_router(escalate.router)


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

