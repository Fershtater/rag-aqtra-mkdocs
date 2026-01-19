"""
OAuth state storage for CSRF protection.
Uses in-memory LRU cache with TTL.
"""

import logging
import time
from typing import Optional
from collections import OrderedDict

logger = logging.getLogger(__name__)

# State cache settings
STATE_TTL_SECONDS = 600  # 10 minutes
STATE_MAX_SIZE = 1000  # Maximum number of state entries


class OAuthStateCache:
    """LRU cache for OAuth state tokens with TTL."""
    
    def __init__(self, max_size: int = STATE_MAX_SIZE, ttl_seconds: int = STATE_TTL_SECONDS):
        """
        Args:
            max_size: Maximum number of state entries
            ttl_seconds: State token lifetime in seconds
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: OrderedDict[str, float] = OrderedDict()
    
    def store(self, state: str) -> None:
        """
        Store a state token with current timestamp.
        
        Args:
            state: State token to store
        """
        # Remove old entries if limit reached
        while len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)  # Remove oldest
        
        self.cache[state] = time.time()
        logger.debug(f"Stored OAuth state token (cache size: {len(self.cache)})")
    
    def validate_and_consume(self, state: str) -> bool:
        """
        Validate and consume a state token (one-time use).
        
        Args:
            state: State token to validate
        
        Returns:
            True if state is valid and consumed, False otherwise
        """
        if state not in self.cache:
            logger.warning(f"OAuth state token not found in cache")
            return False
        
        timestamp = self.cache[state]
        
        # Check TTL
        if time.time() - timestamp > self.ttl_seconds:
            del self.cache[state]
            logger.warning(f"OAuth state token expired")
            return False
        
        # Consume (remove) the state token (one-time use)
        del self.cache[state]
        logger.debug(f"Validated and consumed OAuth state token (cache size: {len(self.cache)})")
        return True
    
    def clear(self) -> None:
        """Clear all stored state tokens."""
        self.cache.clear()
        logger.debug("Cleared OAuth state cache")
    
    def size(self) -> int:
        """Returns current cache size."""
        return len(self.cache)


# Global state cache instance
oauth_state_cache = OAuthStateCache()
