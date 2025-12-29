"""
Pydantic models for v2 API (DocsGPT-like interface).
"""

import hashlib
import json
import logging
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class Source(BaseModel):
    """Unified source format for v2 API."""
    
    id: str = Field(..., description="Stable source ID")
    title: str = Field(..., description="Source title (section_title or filename)")
    url: str = Field(..., description="Full URL to source")
    snippet: str = Field(..., description="First ~240-360 characters of content")
    score: Optional[float] = Field(None, description="Relevance score if available")
    meta: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "abc123",
                "title": "Button Component",
                "url": "https://docs.aqtra.io/app-development/ui-components/button.html",
                "snippet": "The Button component is used to...",
                "score": 0.85,
                "meta": {
                    "source": "docs/app-development/ui-components/button.md",
                    "filename": "button.md",
                    "section_anchor": "primary-button"
                }
            }
        }


class ContextHint(BaseModel):
    """Context hint for better retrieval."""
    
    page_url: Optional[str] = None
    page_title: Optional[str] = None
    language: Optional[str] = None


class RetrievalConfig(BaseModel):
    """Retrieval configuration override."""
    
    top_k: Optional[int] = Field(None, ge=1, le=10, description="Override top_k")
    min_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Minimum score threshold")
    hybrid: Optional[bool] = Field(None, description="Hybrid search (reserved for future)")


class DebugConfig(BaseModel):
    """Debug configuration."""
    
    return_prompt: Optional[bool] = False
    return_chunks: Optional[bool] = False


class AnswerRequest(BaseModel):
    """Request model for /api/answer endpoint."""
    
    question: str = Field(..., max_length=2000, description="User question")
    api_key: Optional[str] = Field(None, description="API key for authentication (optional in open mode)")
    history: Optional[Union[str, List[Union[Dict[str, str], Dict[str, Any]]]]] = Field(
        None,
        description="Conversation history as JSON string or array of {prompt,answer} or {role,content}"
    )
    conversation_id: Optional[str] = Field(None, description="Conversation ID for history persistence")
    preset: Optional[Literal["strict", "support", "developer"]] = Field(
        None,
        description="Override prompt preset for this request (strict/support/developer). If not set, uses server default."
    )
    passthrough: Optional[Dict[str, Any]] = Field(None, description="Additional passthrough data")
    context_hint: Optional[ContextHint] = Field(None, description="Context hints")
    retrieval: Optional[RetrievalConfig] = Field(None, description="Retrieval configuration")
    response_format: Literal["markdown", "text", "json"] = Field("markdown", description="Response format")
    debug: Optional[DebugConfig] = Field(None, description="Debug options")
    
    @validator('question')
    def validate_question(cls, v):
        if len(v.strip()) > 2000:
            raise ValueError("Question is too long (maximum 2000 characters)")
        return v.strip()
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "How do I use the Button component?",
                "api_key": "your-api-key",
                "history": '[{"prompt": "What is a button?", "answer": "A button is..."}]',
                "conversation_id": "c_abc123",
                "response_format": "markdown"
            }
        }


class MetricsPayload(BaseModel):
    """Metrics in response."""
    
    latency_ms: int
    cache_hit: bool
    retrieved_chunks: int
    model: Optional[str] = None
    ttft_ms: Optional[int] = Field(None, description="Time to first token in milliseconds (for streaming)")
    retrieval_ms: Optional[int] = Field(None, description="Time to retrieve documents in milliseconds")
    prompt_render_ms: Optional[int] = Field(None, description="Time to render prompt in milliseconds")
    llm_connect_ms: Optional[int] = Field(None, description="Time to connect to LLM and start streaming in milliseconds")
    embed_query_ms: Optional[int] = Field(None, description="Time to compute query embedding in milliseconds")
    vector_search_ms: Optional[int] = Field(None, description="Time for vector similarity search in milliseconds")
    format_sources_ms: Optional[int] = Field(None, description="Time to format sources in milliseconds")


class AnswerResponse(BaseModel):
    """Response model for /api/answer endpoint."""
    
    answer: str = Field(..., description="Generated answer")
    sources: List[Source] = Field(default_factory=list, description="List of sources")
    conversation_id: str = Field(..., description="Conversation ID")
    request_id: str = Field(..., description="Request ID")
    not_found: bool = Field(False, description="Whether answer was not found")
    metrics: MetricsPayload = Field(..., description="Request metrics")
    retrieved_chunks: Optional[int] = Field(None, description="Number of retrieved chunks (alias for metrics.retrieved_chunks)")
    usage: Optional[Dict[str, Any]] = Field(None, description="Token usage (reserved)")
    debug: Optional[Dict[str, Any]] = Field(None, description="Debug information")
    
    class Config:
        json_schema_extra = {
            "example": {
                "answer": "The Button component is used to...",
                "sources": [],
                "conversation_id": "c_abc123",
                "request_id": "req_xyz789",
                "not_found": False,
                "metrics": {
                    "latency_ms": 1234,
                    "cache_hit": False,
                    "retrieved_chunks": 4,
                    "model": "gpt-4o"
                }
            }
        }


class ErrorPayload(BaseModel):
    """Error payload for SSE events."""
    
    code: str
    message: str
    request_id: str


class SSEEvent(BaseModel):
    """Server-Sent Event model."""
    
    type: Literal["id", "answer", "source", "error", "end"]
    conversation_id: Optional[str] = None
    request_id: Optional[str] = None
    delta: Optional[str] = None
    source: Optional[Source] = None
    error: Optional[ErrorPayload] = None
    metrics: Optional[MetricsPayload] = None


class ErrorResponseV2(BaseModel):
    """Unified error response for v2 API."""
    
    error: str
    detail: Optional[str] = None
    request_id: Optional[str] = None
    code: Optional[str] = None


def parse_history_to_text(history: Optional[Union[str, List[Union[Dict[str, str], Dict[str, Any]]]]], max_length: int = 6000) -> str:
    """
    Parse history into compact text format for prompt.
    
    Args:
        history: History as JSON string or array of objects
        max_length: Maximum length of resulting text (default 6000)
        
    Returns:
        Formatted history text or empty string if parsing fails
    """
    if not history:
        return ""
    
    try:
        # Parse if string
        if isinstance(history, str):
            try:
                parsed = json.loads(history)
            except json.JSONDecodeError:
                logger.warning("Failed to parse history as JSON string, ignoring")
                return ""
            history = parsed
        
        # Ensure it's a list
        if not isinstance(history, list):
            logger.warning("History is not a list, ignoring")
            return ""
        
        # Build text
        lines = []
        for item in history:
            if not isinstance(item, dict):
                continue
            
            # Support both formats: {prompt, answer} and {role, content}
            if "prompt" in item and "answer" in item:
                user_text = str(item.get("prompt", "")).strip()
                assistant_text = str(item.get("answer", "")).strip()
                if user_text:
                    lines.append(f"User: {user_text}")
                if assistant_text:
                    lines.append(f"Assistant: {assistant_text}")
            elif "role" in item and "content" in item:
                role = str(item.get("role", "")).strip().lower()
                content = str(item.get("content", "")).strip()
                if content:
                    # Normalize role names
                    if role in ("user", "human"):
                        lines.append(f"User: {content}")
                    elif role in ("assistant", "ai", "system"):
                        lines.append(f"Assistant: {content}")
        
        result = "\n".join(lines)
        
        # Truncate if too long (keep last part)
        if len(result) > max_length:
            result = "..." + result[-max_length + 3:]
            logger.debug(f"History truncated to {max_length} characters")
        
        return result
        
    except Exception as e:
        logger.warning(f"Error parsing history: {e}, ignoring history")
        return ""


def generate_source_id(source: str, section_anchor: Optional[str], index: int) -> str:
    """
    Generate stable source ID.
    
    Args:
        source: Source path
        section_anchor: Section anchor if available
        index: Index in results
        
    Returns:
        Stable source ID (sha1 hash)
    """
    key = f"{source}#{section_anchor or ''}|{index}"
    return hashlib.sha1(key.encode()).hexdigest()[:16]

