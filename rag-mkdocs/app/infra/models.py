"""
ORM models for analytics and escalation logging.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import BigInteger, Boolean, Integer, JSON, Text
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from app.infra.db import Base


class QueryLog(Base):
    __tablename__ = "query_logs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(
        nullable=False, server_default=func.now()
    )

    request_id: Mapped[str] = mapped_column(Text, index=True, nullable=False)
    ip_hash: Mapped[Optional[str]] = mapped_column(Text, index=True)
    user_agent: Mapped[Optional[str]] = mapped_column(Text)

    page_url: Mapped[Optional[str]] = mapped_column(Text)
    page_title: Mapped[Optional[str]] = mapped_column(Text)

    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[Optional[str]] = mapped_column(Text)
    not_found: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    cache_hit: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    sources: Mapped[Optional[dict]] = mapped_column(JSON)

    error: Mapped[Optional[str]] = mapped_column(Text)


class EscalationLog(Base):
    __tablename__ = "escalation_logs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(
        nullable=False, server_default=func.now()
    )

    request_id: Mapped[str] = mapped_column(Text, index=True, nullable=False)
    email: Mapped[str] = mapped_column(Text, index=True, nullable=False)

    zoho_ticket_id: Mapped[Optional[str]] = mapped_column(Text)
    zoho_ticket_number: Mapped[Optional[str]] = mapped_column(Text)

    status: Mapped[str] = mapped_column(Text, nullable=False, default="pending")
    error: Mapped[Optional[str]] = mapped_column(Text)


class Conversation(Base):
    __tablename__ = "conversations"

    conversation_id: Mapped[str] = mapped_column(Text, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(
        nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        nullable=False, server_default=func.now(), onupdate=func.now()
    )


class ConversationMessage(Base):
    __tablename__ = "conversation_messages"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    conversation_id: Mapped[str] = mapped_column(Text, index=True, nullable=False)
    role: Mapped[str] = mapped_column(Text, nullable=False)  # "user" or "assistant"
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        nullable=False, server_default=func.now()
    )


