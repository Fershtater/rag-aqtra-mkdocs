"""
Async SQLAlchemy database setup for logging and analytics.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Optional
from urllib.parse import urlparse, urlunparse, parse_qs, urlencode

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """Base class for all ORM models."""


_engine: Optional[AsyncEngine] = None
_sessionmaker: Optional[async_sessionmaker[AsyncSession]] = None
_init_lock = asyncio.Lock()


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

    # Convert postgresql:// to postgresql+asyncpg:// (Neon DB approach)
    database_url = re.sub(r'^postgresql:', 'postgresql+asyncpg:', database_url)
    
    # Remove unsupported parameters for asyncpg (sslmode, channel_binding, etc.)
    try:
        parsed = urlparse(database_url)
        query_params = parse_qs(parsed.query, keep_blank_values=True)
        unsupported_params = ['sslmode', 'channel_binding']
        removed = []
        for param in unsupported_params:
            if param in query_params:
                query_params.pop(param)
                removed.append(param)
        if removed:
            new_query = urlencode(query_params, doseq=True)
            database_url = urlunparse((
                parsed.scheme,
                parsed.netloc,
                parsed.path,
                parsed.params,
                new_query,
                parsed.fragment
            ))
            logger.info(f"Removed unsupported parameters from DATABASE_URL: {', '.join(removed)}")
    except Exception as e:
        logger.warning(f"Failed to clean DATABASE_URL parameters: {e}")

    # Idempotent initialization protected by a lock to avoid race conditions
    global _engine, _sessionmaker
    if _engine is not None and _sessionmaker is not None:
        logger.debug("DB engine already initialized, reusing existing instance")
        return _sessionmaker

    async with _init_lock:
        # Double-check under the lock
        if _engine is None or _sessionmaker is None:
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
            logger.debug("DB engine already initialized (inside lock), reusing existing instance")

    assert _sessionmaker is not None  # for type checkers
    return _sessionmaker


def get_sessionmaker() -> Optional[async_sessionmaker[AsyncSession]]:
    """Return initialized sessionmaker if DB is configured, else None."""
    return _sessionmaker


