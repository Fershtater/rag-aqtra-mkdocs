"""
Health check endpoint.
"""

from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/health")
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

