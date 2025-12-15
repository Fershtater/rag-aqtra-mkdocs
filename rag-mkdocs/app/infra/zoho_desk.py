"""
Minimal Zoho Desk integration for escalation tickets.
"""

from __future__ import annotations

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


async def _fetch_access_token() -> str:
    """
    Request a new access token from Zoho using the refresh token flow.
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

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(token_url, params=params)
        resp.raise_for_status()
        data = resp.json()

    access_token = data.get("access_token")
    expires_in = data.get("expires_in", 3600)
    if not access_token:
        raise RuntimeError(f"Zoho OAuth token response missing access_token: {data}")

    global _access_token, _access_token_expires_at
    _access_token = access_token
    # refresh a bit earlier than actual expiry
    _access_token_expires_at = time.time() + int(expires_in) * 0.9

    logger.info("Obtained new Zoho Desk access token")
    return access_token


async def get_access_token() -> str:
    """
    Get a cached access token or refresh if expired.
    """
    global _access_token, _access_token_expires_at
    now = time.time()
    if _access_token and now < _access_token_expires_at:
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
            "lastName": email.split("@")[0] or "User",
        },
        "description": description,
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()

    logger.info("Zoho Desk ticket created: %s", data)
    return data


