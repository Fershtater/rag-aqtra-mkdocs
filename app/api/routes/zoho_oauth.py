"""
Zoho OAuth routes for authorization code flow.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Request, Query, HTTPException
from fastapi.responses import HTMLResponse

from app.settings import get_settings
from app.services.zoho_oauth import (
    generate_state,
    get_accounts_base_url,
    exchange_code_for_tokens,
)
from app.infra.oauth_state import oauth_state_cache
from app.infra.oauth_tokens import oauth_token_cache

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/oauth/start")
async def oauth_start(request: Request):
    """
    Start OAuth flow by generating a state token.
    
    Returns:
        JSON with state token and authorization URL
    """
    settings = get_settings()
    
    if not settings.ZOHO_CLIENT_ID:
        raise HTTPException(
            status_code=503,
            detail="Zoho OAuth not configured (ZOHO_CLIENT_ID missing)"
        )
    
    # Generate state token
    state = generate_state()
    
    # Store state in cache
    oauth_state_cache.store(state)
    
    # Build authorization URL
    redirect_uri = settings.ZOHO_REDIRECT_URI or "https://agent.aqtra.io/oauth/callback"
    scopes = settings.ZOHO_SCOPES or "SalesIQ.tickets.READ,SalesIQ.tickets.WRITE"
    
    # Use default accounts URL for authorization (user will redirect to correct DC)
    accounts_base_url = settings.ZOHO_ACCOUNTS_BASE_URL or "https://accounts.zoho.com"
    auth_url = (
        f"{accounts_base_url}/oauth/v2/auth?"
        f"client_id={settings.ZOHO_CLIENT_ID}&"
        f"response_type=code&"
        f"scope={scopes}&"
        f"access_type=offline&"
        f"redirect_uri={redirect_uri}&"
        f"state={state}"
    )
    
    request_id = getattr(request.state, "request_id", "unknown")
    logger.info(f"[{request_id}] Generated OAuth state and authorization URL")
    
    return {
        "state": state,
        "authorization_url": auth_url,
    }


@router.get("/oauth/callback", response_class=HTMLResponse)
async def oauth_callback(
    request: Request,
    code: Optional[str] = Query(None, description="Authorization code from Zoho"),
    state: Optional[str] = Query(None, description="State token for CSRF protection"),
    location: Optional[str] = Query(None, description="Zoho location/DC (e.g., 'eu', 'com')"),
):
    """
    OAuth callback endpoint for Zoho SalesIQ.
    
    Accepts authorization code, exchanges it for tokens, and stores them.
    
    Returns:
        HTML success page or error response
    """
    request_id = getattr(request.state, "request_id", "unknown")
    
    # Validate required parameters
    if not code:
        logger.warning(f"[{request_id}] OAuth callback missing 'code' parameter")
        raise HTTPException(
            status_code=400,
            detail="Missing required parameter: code"
        )
    
    if not state:
        logger.warning(f"[{request_id}] OAuth callback missing 'state' parameter")
        raise HTTPException(
            status_code=400,
            detail="Missing required parameter: state"
        )
    
    # Validate state (CSRF protection)
    if not oauth_state_cache.validate_and_consume(state):
        logger.warning(f"[{request_id}] OAuth callback invalid or expired state token")
        raise HTTPException(
            status_code=400,
            detail="Invalid or expired state token"
        )
    
    # Get settings
    settings = get_settings()
    
    if not settings.ZOHO_CLIENT_ID or not settings.ZOHO_CLIENT_SECRET:
        logger.error(f"[{request_id}] Zoho OAuth credentials not configured")
        raise HTTPException(
            status_code=503,
            detail="Zoho OAuth not configured"
        )
    
    # Determine accounts base URL from location or default
    accounts_base_url = get_accounts_base_url(
        location=location,
        default=settings.ZOHO_ACCOUNTS_BASE_URL
    )
    
    redirect_uri = settings.ZOHO_REDIRECT_URI or "https://agent.aqtra.io/oauth/callback"
    
    # Exchange code for tokens
    try:
        token_data = await exchange_code_for_tokens(
            code=code,
            client_id=settings.ZOHO_CLIENT_ID,
            client_secret=settings.ZOHO_CLIENT_SECRET,  # Never logged
            redirect_uri=redirect_uri,
            accounts_base_url=accounts_base_url,
        )
        
        access_token = token_data.get("access_token")
        refresh_token = token_data.get("refresh_token")  # Optional, for offline access
        expires_in = token_data.get("expires_in", 3600)  # Default 1 hour
        
        if not access_token:
            raise HTTPException(
                status_code=502,
                detail="Token exchange succeeded but no access_token in response"
            )
        
        # Store tokens (keyed by state for now; can be migrated to installation_id later)
        # Note: state is already consumed, but we use it as a temporary key
        # In production, you'd want to use a more permanent identifier
        token_key = f"oauth_state_{state}"
        
        oauth_token_cache.store(
            key=token_key,
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=expires_in,
            accounts_base_url=accounts_base_url,
        )
        
        logger.info(
            f"[{request_id}] Successfully exchanged code for tokens and stored them",
            extra={
                'has_refresh_token': refresh_token is not None,
                'expires_in': expires_in,
            }
        )
        
        # Return success page
        return HTMLResponse(
            content="""
            <!DOCTYPE html>
            <html>
            <head>
                <title>OAuth Success</title>
                <style>
                    body {
                        font-family: Arial, sans-serif;
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        height: 100vh;
                        margin: 0;
                        background-color: #f5f5f5;
                    }
                    .container {
                        text-align: center;
                        padding: 2rem;
                        background: white;
                        border-radius: 8px;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    }
                    h1 {
                        color: #4CAF50;
                        margin-bottom: 1rem;
                    }
                    p {
                        color: #666;
                    }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>✓ Success</h1>
                    <p>Authorization successful! You can close this tab.</p>
                </div>
            </body>
            </html>
            """,
            status_code=200
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(
            f"[{request_id}] Error exchanging authorization code: {type(e).__name__}",
            exc_info=True,
            extra={
                'error_type': type(e).__name__,
            }
        )
        raise HTTPException(
            status_code=502,
            detail="Failed to exchange authorization code for tokens"
        )
