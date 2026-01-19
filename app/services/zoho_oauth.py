"""
Zoho OAuth service for authorization code flow.
Handles token exchange and refresh operations.
"""

import logging
import secrets
from typing import Optional, Dict, Any
import httpx

logger = logging.getLogger(__name__)


def generate_state() -> str:
    """Generate a secure random state token (32+ characters)."""
    return secrets.token_urlsafe(32)


def get_accounts_base_url(location: Optional[str] = None, default: Optional[str] = None) -> str:
    """
    Get Zoho Accounts base URL based on location (DC).
    
    Args:
        location: Location/DC returned by Zoho (e.g., 'eu', 'com', 'in', 'au', 'jp')
        default: Default base URL from env (e.g., 'https://accounts.zoho.com')
    
    Returns:
        Accounts base URL (e.g., 'https://accounts.zoho.eu')
    """
    if location:
        # Map location to accounts domain
        location_lower = location.lower()
        location_map = {
            'eu': 'https://accounts.zoho.eu',
            'com': 'https://accounts.zoho.com',
            'in': 'https://accounts.zoho.in',
            'au': 'https://accounts.zoho.com.au',
            'jp': 'https://accounts.zoho.jp',
            'cn': 'https://accounts.zoho.com.cn',
        }
        
        base_url = location_map.get(location_lower)
        if base_url:
            logger.debug(f"Using Zoho Accounts URL for location '{location}': {base_url}")
            return base_url
    
    # Fallback to default
    if default:
        logger.debug(f"Using default Zoho Accounts URL: {default}")
        return default.rstrip('/')
    
    # Last resort: default to .com
    logger.warning("No location or default provided, using accounts.zoho.com as fallback")
    return 'https://accounts.zoho.com'


async def exchange_code_for_tokens(
    code: str,
    client_id: str,
    client_secret: str,
    redirect_uri: str,
    accounts_base_url: str,
) -> Dict[str, Any]:
    """
    Exchange authorization code for access and refresh tokens.
    
    Args:
        code: Authorization code from callback
        client_id: Zoho client ID
        client_secret: Zoho client secret
        redirect_uri: Redirect URI (must match registered URI)
        accounts_base_url: Zoho Accounts base URL (e.g., 'https://accounts.zoho.eu')
    
    Returns:
        Token response dict with 'access_token', 'refresh_token', 'expires_in', etc.
    
    Raises:
        httpx.HTTPStatusError: If token exchange fails
        RuntimeError: If response is invalid
    """
    token_url = f"{accounts_base_url}/oauth/v2/token"
    
    data = {
        'grant_type': 'authorization_code',
        'code': code,
        'client_id': client_id,
        'client_secret': client_secret,
        'redirect_uri': redirect_uri,
    }
    
    timeout = httpx.Timeout(connect=5.0, read=10.0, write=10.0, pool=None)
    
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            # Use application/x-www-form-urlencoded
            response = await client.post(
                token_url,
                data=data,  # httpx will encode as form-urlencoded
                headers={'Content-Type': 'application/x-www-form-urlencoded'}
            )
            response.raise_for_status()
            
            token_data = response.json()
            
            # Validate required fields
            if 'access_token' not in token_data:
                raise RuntimeError("Token response missing access_token")
            
            # Log success (without secrets)
            logger.info(
                "Successfully exchanged authorization code for tokens",
                extra={
                    'has_access_token': True,
                    'has_refresh_token': 'refresh_token' in token_data,
                    'expires_in': token_data.get('expires_in'),
                }
            )
            
            return token_data
            
    except httpx.HTTPStatusError as e:
        # Log error without secrets
        logger.error(
            f"Token exchange failed: HTTP {e.response.status_code}",
            extra={
                'status_code': e.response.status_code,
                'error_detail': e.response.text[:200] if e.response.text else None,
            }
        )
        raise
    except Exception as e:
        logger.error(f"Token exchange failed with unexpected error: {type(e).__name__}")
        raise RuntimeError(f"Token exchange failed: {e}") from e


async def refresh_access_token(
    refresh_token: str,
    client_id: str,
    client_secret: str,
    accounts_base_url: str,
) -> Dict[str, Any]:
    """
    Refresh access token using refresh token.
    
    Args:
        refresh_token: Refresh token from previous authorization
        client_id: Zoho client ID
        client_secret: Zoho client secret
        accounts_base_url: Zoho Accounts base URL
    
    Returns:
        Token response dict with new 'access_token', 'expires_in', etc.
    
    Raises:
        httpx.HTTPStatusError: If token refresh fails
        RuntimeError: If response is invalid
    """
    token_url = f"{accounts_base_url}/oauth/v2/token"
    
    data = {
        'grant_type': 'refresh_token',
        'refresh_token': refresh_token,
        'client_id': client_id,
        'client_secret': client_secret,
    }
    
    timeout = httpx.Timeout(connect=5.0, read=10.0, write=10.0, pool=None)
    
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                token_url,
                data=data,
                headers={'Content-Type': 'application/x-www-form-urlencoded'}
            )
            response.raise_for_status()
            
            token_data = response.json()
            
            if 'access_token' not in token_data:
                raise RuntimeError("Token refresh response missing access_token")
            
            logger.info("Successfully refreshed access token")
            
            return token_data
            
    except httpx.HTTPStatusError as e:
        logger.error(
            f"Token refresh failed: HTTP {e.response.status_code}",
            extra={
                'status_code': e.response.status_code,
            }
        )
        raise
    except Exception as e:
        logger.error(f"Token refresh failed with unexpected error: {type(e).__name__}")
        raise RuntimeError(f"Token refresh failed: {e}") from e
