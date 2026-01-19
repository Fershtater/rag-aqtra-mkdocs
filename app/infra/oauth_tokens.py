"""
OAuth token storage for access and refresh tokens.
Uses in-memory LRU cache with TTL for access tokens.
"""

import logging
import time
from typing import Optional, Dict, Any
from collections import OrderedDict

logger = logging.getLogger(__name__)

# Token cache settings
TOKEN_CACHE_MAX_SIZE = 100  # Maximum number of token entries
ACCESS_TOKEN_BUFFER_SECONDS = 60  # Refresh access token 60s before expiry


class TokenEntry:
    """Entry for stored tokens."""
    
    def __init__(
        self,
        access_token: str,
        refresh_token: Optional[str],
        expires_at: float,
        accounts_base_url: str,
    ):
        self.access_token = access_token
        self.refresh_token = refresh_token
        self.expires_at = expires_at
        self.accounts_base_url = accounts_base_url


class OAuthTokenCache:
    """LRU cache for OAuth tokens with TTL."""
    
    def __init__(self, max_size: int = TOKEN_CACHE_MAX_SIZE):
        """
        Args:
            max_size: Maximum number of token entries
        """
        self.max_size = max_size
        self.cache: OrderedDict[str, TokenEntry] = OrderedDict()
    
    def store(
        self,
        key: str,
        access_token: str,
        refresh_token: Optional[str],
        expires_in: int,
        accounts_base_url: str,
    ) -> None:
        """
        Store tokens for a given key (e.g., installation_id or state).
        
        Args:
            key: Unique key for this token set
            access_token: Access token
            refresh_token: Refresh token (optional)
            expires_in: Token expiry in seconds from now
            accounts_base_url: Zoho Accounts base URL used
        """
        # Remove old entries if limit reached
        while len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)  # Remove oldest
        
        expires_at = time.time() + expires_in
        self.cache[key] = TokenEntry(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_at=expires_at,
            accounts_base_url=accounts_base_url,
        )
        
        # Move to end (LRU)
        self.cache.move_to_end(key)
        
        logger.debug(
            f"Stored OAuth tokens for key '{key[:8]}...' "
            f"(expires in {expires_in}s, cache size: {len(self.cache)})"
        )
    
    def get_access_token(self, key: str) -> Optional[str]:
        """
        Get access token if valid, otherwise None.
        
        Args:
            key: Key to retrieve tokens for
        
        Returns:
            Access token if valid, None otherwise
        """
        if key not in self.cache:
            return None
        
        entry = self.cache[key]
        
        # Check if expired
        if time.time() >= entry.expires_at - ACCESS_TOKEN_BUFFER_SECONDS:
            logger.debug(f"Access token for key '{key[:8]}...' expired or near expiry")
            return None
        
        # Move to end (LRU)
        self.cache.move_to_end(key)
        
        return entry.access_token
    
    def get_refresh_token(self, key: str) -> Optional[str]:
        """
        Get refresh token for a key.
        
        Args:
            key: Key to retrieve refresh token for
        
        Returns:
            Refresh token if available, None otherwise
        """
        if key not in self.cache:
            return None
        
        entry = self.cache[key]
        return entry.refresh_token
    
    def get_accounts_base_url(self, key: str) -> Optional[str]:
        """
        Get accounts base URL for a key.
        
        Args:
            key: Key to retrieve accounts URL for
        
        Returns:
            Accounts base URL if available, None otherwise
        """
        if key not in self.cache:
            return None
        
        entry = self.cache[key]
        return entry.accounts_base_url
    
    def update_access_token(
        self,
        key: str,
        access_token: str,
        expires_in: int,
    ) -> None:
        """
        Update access token for existing entry.
        
        Args:
            key: Key to update
            access_token: New access token
            expires_in: Token expiry in seconds from now
        """
        if key not in self.cache:
            logger.warning(f"Cannot update access token: key '{key[:8]}...' not found")
            return
        
        entry = self.cache[key]
        entry.access_token = access_token
        entry.expires_at = time.time() + expires_in
        
        # Move to end (LRU)
        self.cache.move_to_end(key)
        
        logger.debug(f"Updated access token for key '{key[:8]}...'")
    
    def remove(self, key: str) -> None:
        """
        Remove tokens for a key.
        
        Args:
            key: Key to remove
        """
        if key in self.cache:
            del self.cache[key]
            logger.debug(f"Removed tokens for key '{key[:8]}...'")
    
    def clear(self) -> None:
        """Clear all stored tokens."""
        self.cache.clear()
        logger.debug("Cleared OAuth token cache")
    
    def size(self) -> int:
        """Returns current cache size."""
        return len(self.cache)


# Global token cache instance
oauth_token_cache = OAuthTokenCache()
