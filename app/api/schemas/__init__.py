"""
API schemas module.

Contains Pydantic models for request/response validation.
"""

# Re-export v1 schemas
from app.api.schemas.v1 import (
    Query,
    QueryResponse,
    EscalateRequest,
    ErrorResponse,
)

# Re-export v2 schemas for convenience
from app.api.schemas.v2 import (
    AnswerRequest,
    AnswerResponse,
    Source,
    SSEEvent,
    ErrorResponseV2,
    ErrorPayload,
    MetricsPayload,
    ContextHint,
    RetrievalConfig,
    DebugConfig,
    generate_source_id,
    parse_history_to_text,
)

__all__ = [
    # V1 schemas
    "Query",
    "QueryResponse",
    "EscalateRequest",
    "ErrorResponse",
    # V2 schemas
    "AnswerRequest",
    "AnswerResponse",
    "Source",
    "SSEEvent",
    "ErrorResponseV2",
    "ErrorPayload",
    "MetricsPayload",
    "ContextHint",
    "RetrievalConfig",
    "DebugConfig",
    "generate_source_id",
    "parse_history_to_text",
]
