"""
In-memory cache for RAG responses.
"""

import hashlib
import logging
import os
import time
from typing import Dict, Optional, Tuple, Any
from collections import OrderedDict

logger = logging.getLogger(__name__)

# Cache settings
CACHE_MAX_SIZE = int(os.getenv("CACHE_MAX_SIZE", "500"))
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "600"))  # 10 minutes by default


class LRUCache:
    """LRU cache with TTL."""
    
    def __init__(self, max_size: int = CACHE_MAX_SIZE, ttl_seconds: int = CACHE_TTL_SECONDS):
        """
        Args:
            max_size: Maximum number of elements
            ttl_seconds: Entry lifetime in seconds
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: OrderedDict[str, Tuple[Any, float]] = OrderedDict()
    
    def _generate_key(self, question: str, settings_signature: str) -> str:
        """Generates cache key.
        
        Key is based only on question text and prompt settings signature,
        to avoid explosive cache growth due to per-request parameters.
        """
        normalized_question = question.strip().lower()[:500]  # Limit length
        key_data = f"{normalized_question}|{settings_signature}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """
        Gets value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Value or None if not found or expired
        """
        # Log cache access (hash only, no sensitive data)
        key_hash = hashlib.sha256(key.encode()).hexdigest()[:12]
        cache_size = len(self.cache)
        
        if key not in self.cache:
            logger.debug(f"CACHE_GET hash={key_hash} hit=False size={cache_size}")
            return None
        
        value, timestamp = self.cache[key]
        
        # Check TTL
        if time.time() - timestamp > self.ttl_seconds:
            del self.cache[key]
            logger.debug(f"CACHE_GET hash={key_hash} hit=False (expired) size={len(self.cache)}")
            return None
        
        # Move to end (LRU)
        self.cache.move_to_end(key)
        logger.info(f"CACHE_GET hash={key_hash} hit=True size={cache_size}")
        return value
    
    def set(self, key: str, value: Any):
        """
        Saves value to cache.
        
        Args:
            key: Cache key
            value: Value to save
        """
        # Log cache write (hash only, no sensitive data)
        key_hash = hashlib.sha256(key.encode()).hexdigest()[:12]
        size_before = len(self.cache)
        
        # Remove old entries if limit reached
        while len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)  # Remove oldest
        
        self.cache[key] = (value, time.time())
        size_after = len(self.cache)
        logger.info(f"CACHE_SET hash={key_hash} size_before={size_before} size_after={size_after}")
    
    def clear(self):
        """Clears cache."""
        self.cache.clear()
    
    def size(self) -> int:
        """Returns current cache size."""
        return len(self.cache)


# Global cache instance
response_cache = LRUCache()

