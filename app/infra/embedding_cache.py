"""
In-memory LRU cache for query embeddings.

Caches embeddings to avoid redundant API calls for repeated questions.
"""

import hashlib
import logging
import os
import threading
import time
from typing import List, Optional, Tuple
from collections import OrderedDict

logger = logging.getLogger(__name__)

# Cache settings
EMBEDDING_CACHE_MAX_SIZE = int(os.getenv("EMBEDDING_CACHE_MAX_SIZE", "2000"))
EMBEDDING_CACHE_TTL_SECONDS = int(os.getenv("EMBEDDING_CACHE_TTL_SECONDS", "3600"))


def normalize_question(question: str) -> str:
    """
    Normalize question for cache key.
    
    Args:
        question: Raw question string
        
    Returns:
        Normalized question (lowercase, stripped, collapsed whitespace)
    """
    # Strip, lowercase, and collapse whitespace
    normalized = " ".join(question.strip().lower().split())
    return normalized


def generate_embedding_cache_key(question: str, embedding_model: str = "default") -> str:
    """
    Generate cache key for embedding.
    
    Args:
        question: User question
        embedding_model: Embedding model name (for cache separation)
        
    Returns:
        Cache key (MD5 hash)
    """
    normalized = normalize_question(question)
    key_data = f"{embedding_model}|{normalized}"
    return hashlib.md5(key_data.encode()).hexdigest()


class EmbeddingCache:
    """Thread-safe LRU cache for query embeddings."""
    
    def __init__(self, max_size: int = EMBEDDING_CACHE_MAX_SIZE, ttl_seconds: int = EMBEDDING_CACHE_TTL_SECONDS):
        """
        Args:
            max_size: Maximum number of cached embeddings
            ttl_seconds: Entry lifetime in seconds (0 = no expiration)
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: OrderedDict[str, Tuple[List[float], float]] = OrderedDict()
        self.lock = threading.RLock()
        self.hits = 0
        self.misses = 0
    
    def get(self, question: str, embedding_model: str = "default") -> Optional[List[float]]:
        """
        Get embedding from cache.
        
        Args:
            question: User question
            embedding_model: Embedding model name
            
        Returns:
            Cached embedding vector or None if not found/expired
        """
        key = generate_embedding_cache_key(question, embedding_model)
        
        with self.lock:
            if key not in self.cache:
                self.misses += 1
                # Update Prometheus metrics
                try:
                    from app.infra.metrics import rag_embedding_cache_misses_total
                    if rag_embedding_cache_misses_total is not None:
                        rag_embedding_cache_misses_total.inc()
                except Exception:
                    pass
                logger.debug(f"Embedding cache MISS: key={key[:8]}...")
                return None
            
            embedding, timestamp = self.cache[key]
            
            # Check TTL
            if self.ttl_seconds > 0 and (time.time() - timestamp) > self.ttl_seconds:
                del self.cache[key]
                self.misses += 1
                # Update Prometheus metrics
                try:
                    from app.infra.metrics import rag_embedding_cache_misses_total
                    if rag_embedding_cache_misses_total is not None:
                        rag_embedding_cache_misses_total.inc()
                except Exception:
                    pass
                logger.debug(f"Embedding cache EXPIRED: key={key[:8]}...")
                return None
            
            # Move to end (LRU)
            self.cache.move_to_end(key)
            self.hits += 1
            # Update Prometheus metrics
            try:
                from app.infra.metrics import rag_embedding_cache_hits_total
                if rag_embedding_cache_hits_total is not None:
                    rag_embedding_cache_hits_total.inc()
            except Exception:
                pass
            logger.debug(f"Embedding cache HIT: key={key[:8]}...")
            return embedding
    
    def set(self, question: str, embedding: List[float], embedding_model: str = "default"):
        """
        Store embedding in cache.
        
        Args:
            question: User question
            embedding: Embedding vector
            embedding_model: Embedding model name
        """
        key = generate_embedding_cache_key(question, embedding_model)
        
        with self.lock:
            # Remove old entries if limit reached
            while len(self.cache) >= self.max_size:
                self.cache.popitem(last=False)  # Remove oldest
            
            self.cache[key] = (embedding, time.time())
            logger.debug(f"Embedding cache SET: key={key[:8]}..., size={len(self.cache)}")
    
    def clear(self):
        """Clear all cached embeddings."""
        with self.lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0
    
    def size(self) -> int:
        """Get current cache size."""
        with self.lock:
            return len(self.cache)
    
    def stats(self) -> dict:
        """Get cache statistics."""
        with self.lock:
            total = self.hits + self.misses
            hit_rate = (self.hits / total * 100) if total > 0 else 0.0
            return {
                "size": len(self.cache),
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": hit_rate
            }


# Global cache instance
embedding_cache = EmbeddingCache()

