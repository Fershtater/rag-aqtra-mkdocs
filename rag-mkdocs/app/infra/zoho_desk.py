"""
Minimal Zoho Desk integration for escalation tickets.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, Dict, Optional

import httpx

logger = logging.getLogger(__name__)


ZOHO_ACCOUNTS_BASE_URL = os.getenv("ZOHO_ACCOUNTS_BASE_URL", "https://accounts.zoho.eu").rstrip("/")
ZOHO_DESK_BASE_URL = os.getenv("ZOHO_DESK_BASE_URL", "https://desk.zoho.eu").rstrip("/")
ZOHO_CLIENT_ID = os.getenv("ZOHO_CLIENT_ID")
ZOHO_CLIENT_SECRET = os.getenv("ZOHO_CLIENT_SECRET")
ZOHO_REFRESH_TOKEN = os.getenv("ZOHO_REFRESH_TOKEN")
ZOHO_DESK_ORG_ID = os.getenv("ZOHO_DESK_ORG_ID")
ZOHO_DESK_DEPARTMENT_ID = os.getenv("ZOHO_DESK_DEPARTMENT_ID")

_access_token: Optional[str] = None
_access_token_expires_at: float = 0.0
_token_lock = asyncio.Lock()


async def _fetch_access_token() -> str:
    """
    Request a new access token from Zoho using the refresh token flow.
    Includes simple retry with exponential backoff.
    """
    if not (ZOHO_CLIENT_ID and ZOHO_CLIENT_SECRET and ZOHO_REFRESH_TOKEN):
        raise RuntimeError("Zoho Desk OAuth env vars are not fully configured")

    token_url = f"{ZOHO_ACCOUNTS_BASE_URL}/oauth/v2/token"
    params = {
        "grant_type": "refresh_token",
        "client_id": ZOHO_CLIENT_ID,
        "client_secret": ZOHO_CLIENT_SECRET,
        "refresh_token": ZOHO_REFRESH_TOKEN,
    }

    timeout = httpx.Timeout(connect=5.0, read=10.0, write=10.0, pool=None)
    last_exc: Optional[Exception] = None

    for attempt in range(2):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(token_url, params=params)
                resp.raise_for_status()
                data = resp.json()
            access_token = data.get("access_token")
            expires_in = data.get("expires_in", 3600)
            if not access_token:
                raise RuntimeError("Zoho OAuth token response missing access_token")

            global _access_token, _access_token_expires_at
            _access_token = access_token
            # refresh a bit earlier than actual expiry
            _access_token_expires_at = time.time() + int(expires_in) * 0.9

            logger.info("Obtained new Zoho Desk access token")
            return access_token
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            wait = 0.5 if attempt == 0 else 1.5
            logger.warning("Failed to refresh Zoho token (attempt %s): %s", attempt + 1, exc)
            await asyncio.sleep(wait)

    assert last_exc is not None
    raise RuntimeError(f"Failed to refresh Zoho access token: {last_exc}") from last_exc


async def get_access_token() -> str:
    """
    Get a cached access token or refresh if expired.
    Protected by an async lock to avoid concurrent refreshes.
    """
    global _access_token, _access_token_expires_at
    now = time.time()
    if _access_token and now < _access_token_expires_at - 30:
        return _access_token

    async with _token_lock:
        # Double-check under the lock
        now = time.time()
        if _access_token and now < _access_token_expires_at - 30:
            return _access_token
        return await _fetch_access_token()


async def create_ticket(
    email: str,
    subject: str,
    description: str,
) -> Dict[str, Any]:
    """
    Create a ticket in Zoho Desk.

    Returns:
        Parsed JSON response from Zoho Desk.
    """
    if not (ZOHO_DESK_ORG_ID and ZOHO_DESK_DEPARTMENT_ID):
        raise RuntimeError("Zoho Desk ORG/DEPARTMENT env vars are not configured")

    token = await get_access_token()
    url = f"{ZOHO_DESK_BASE_URL}/api/v1/tickets"

    headers = {
        "Authorization": f"Zoho-oauthtoken {token}",
        "orgId": ZOHO_DESK_ORG_ID,
        "Content-Type": "application/json",
    }

    payload: Dict[str, Any] = {
        "subject": subject,
        "departmentId": ZOHO_DESK_DEPARTMENT_ID,
        "contact": {
            "email": email,
            "lastName": (email.split("@")[0] or "User") if "@" in email else "User",
        },
        "description": description,
    }

    timeout = httpx.Timeout(connect=5.0, read=10.0, write=10.0, pool=None)
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        try:
            data = resp.json()
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Zoho Desk ticket response is not valid JSON: {exc}") from exc

    logger.info("Zoho Desk ticket created with status %s", resp.status_code)
    return data


