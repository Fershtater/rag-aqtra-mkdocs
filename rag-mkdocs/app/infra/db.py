"""
Async SQLAlchemy database setup for logging and analytics.
"""

from __future__ import annotations

import logging
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """Base class for all ORM models."""


_engine: Optional[AsyncEngine] = None
_sessionmaker: Optional[async_sessionmaker[AsyncSession]] = None


async def init_db(database_url: str) -> async_sessionmaker[AsyncSession]:
    """
    Initialize async engine, sessionmaker and create tables.

    Args:
        database_url: DATABASE_URL from environment (postgresql+asyncpg://...)

    Returns:
        async_sessionmaker instance bound to the engine.
    """
    global _engine, _sessionmaker

    if not database_url:
        raise ValueError("DATABASE_URL is empty")

    if _engine is None:
        logger.info("Initializing async DB engine...")
        _engine = create_async_engine(
            database_url,
            echo=False,
            pool_pre_ping=True,
        )
        _sessionmaker = async_sessionmaker(_engine, expire_on_commit=False)

        # Import models here to ensure they are registered with Base.metadata
        from app.infra import models  # noqa: F401

        async with _engine.begin() as conn:
            logger.info("Creating database tables (if not exist)...")
            await conn.run_sync(Base.metadata.create_all)
            logger.info("Database tables are up to date")
    else:
        logger.debug("DB engine already initialized, reusing existing instance")

    assert _sessionmaker is not None  # for type checkers
    return _sessionmaker


def get_sessionmaker() -> Optional[async_sessionmaker[AsyncSession]]:
    """Return initialized sessionmaker if DB is configured, else None."""
    return _sessionmaker


