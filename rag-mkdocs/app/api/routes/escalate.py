"""
Escalation endpoint to support service (Zoho Desk).
"""

import logging
from fastapi import APIRouter, Request, JSONResponse

from app.api.schemas import EscalateRequest, ErrorResponse
from app.infra.rate_limit import escalate_limiter
from app.infra.metrics import (
    rate_limit_hits_total,
    PROMETHEUS_AVAILABLE,
)
from app.infra.analytics import log_escalation
from app.infra.zoho_desk import create_ticket

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/escalate", responses={400: {"model": ErrorResponse}, 404: {"model": ErrorResponse}, 429: {"model": ErrorResponse}, 503: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
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
    TICKET_TEXT_LIMIT = 8000
    QUESTION_LIMIT = 2000
    COMMENT_LIMIT = 2000

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

