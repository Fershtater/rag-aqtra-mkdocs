"""
Pydantic models for API requests and responses.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, EmailStr, Field, validator


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


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str
    detail: Optional[str] = None

