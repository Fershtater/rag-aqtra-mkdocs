"""
Prometheus metrics endpoint.
"""

from fastapi import APIRouter, Response
from app.infra.metrics import get_metrics_response

router = APIRouter()


@router.get("/metrics")
async def metrics_endpoint():
    """Metrics endpoint Prometheus."""
    content, content_type = get_metrics_response()
    return Response(content=content, media_type=content_type)

