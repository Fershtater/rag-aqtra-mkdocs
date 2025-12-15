"""
Helpers for anonymized analytics logging (queries and escalations).
"""

from __future__ import annotations

import hashlib
import logging
import os
from typing import Any, Dict, Iterable, Optional

from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession

from app.infra.models import QueryLog, EscalationLog

logger = logging.getLogger(__name__)


IP_HASH_SALT = os.getenv("IP_HASH_SALT", "CHANGE_ME_SALT")


def hash_ip(ip: Optional[str]) -> Optional[str]:
    """
    Return a salted SHA-256 hash of the IP address, or None if ip is falsy.
    """
    if not ip:
        return None
    data = (IP_HASH_SALT + ip).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


async def log_query(
    session_maker: Optional[async_sessionmaker[AsyncSession]],
    *,
    request_id: str,
    ip_hash_value: Optional[str],
    user_agent: Optional[str],
    page_url: Optional[str],
    page_title: Optional[str],
    question: str,
    answer: Optional[str],
    not_found: bool,
    cache_hit: bool,
    latency_ms: int,
    sources: Iterable[Dict[str, Any]],
    error: Optional[str] = None,
) -> None:
    """
    Persist a query log entry if DB logging is enabled.
    """
    if session_maker is None:
        return

    # Prepare sources as JSON-serializable list
    try:
        sources_payload = list(sources)
    except Exception:
        sources_payload = []

    try:
        async with session_maker() as session:
            session: AsyncSession
            entry = QueryLog(
                request_id=request_id,
                ip_hash=ip_hash_value,
                user_agent=user_agent,
                page_url=page_url,
                page_title=page_title,
                question=question,
                answer=(answer[:16000] if isinstance(answer, str) else None),
                not_found=not_found,
                cache_hit=cache_hit,
                latency_ms=int(latency_ms),
                sources=sources_payload or None,
                error=error,
            )
            session.add(entry)
            await session.commit()
    except Exception as e:  # pragma: no cover - best-effort logging
        logger.warning("Failed to log query analytics: %s", e, exc_info=True)


async def log_escalation(
    session_maker: Optional[async_sessionmaker[AsyncSession]],
    *,
    request_id: str,
    email: str,
    status: str,
    zoho_ticket_id: Optional[str] = None,
    zoho_ticket_number: Optional[str] = None,
    error: Optional[str] = None,
) -> None:
    """
    Persist an escalation log entry if DB logging is enabled.
    """
    if session_maker is None:
        return

    try:
        async with session_maker() as session:
            session: AsyncSession
            entry = EscalationLog(
                request_id=request_id,
                email=email,
                status=status,
                zoho_ticket_id=zoho_ticket_id,
                zoho_ticket_number=zoho_ticket_number,
                error=error,
            )
            session.add(entry)
            await session.commit()
    except Exception as e:  # pragma: no cover
        logger.warning("Failed to log escalation analytics: %s", e, exc_info=True)


