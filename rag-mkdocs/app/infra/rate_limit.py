"""
Простой in-memory rate limiter для API endpoints.
"""

import logging
import os
import time
from collections import defaultdict
from typing import Dict, Tuple, Optional

logger = logging.getLogger(__name__)

# Настройки по умолчанию
QUERY_RATE_LIMIT = int(os.getenv("QUERY_RATE_LIMIT", "30"))
QUERY_RATE_WINDOW_SECONDS = int(os.getenv("QUERY_RATE_WINDOW_SECONDS", "60"))
UPDATE_RATE_LIMIT = int(os.getenv("UPDATE_RATE_LIMIT", "3"))
UPDATE_RATE_WINDOW_SECONDS = int(os.getenv("UPDATE_RATE_WINDOW_SECONDS", "3600"))
ESCALATE_RATE_LIMIT = int(os.getenv("ESCALATE_RATE_LIMIT", "5"))
ESCALATE_RATE_WINDOW_SECONDS = int(os.getenv("ESCALATE_RATE_WINDOW_SECONDS", "3600"))


class RateLimiter:
    """Простой in-memory rate limiter с sliding window."""
    
    def __init__(self, limit: int, window_seconds: int):
        """
        Args:
            limit: Максимальное количество запросов
            window_seconds: Окно времени в секундах
        """
        self.limit = limit
        self.window_seconds = window_seconds
        self.requests: Dict[str, list] = defaultdict(list)
        self._last_cleanup = time.time()
        self._cleanup_interval = 300  # Очистка каждые 5 минут
    
    def _cleanup_old_entries(self):
        """Удаляет старые записи из словаря."""
        current_time = time.time()
        if current_time - self._last_cleanup < self._cleanup_interval:
            return
        
        cutoff_time = current_time - self.window_seconds * 2
        
        keys_to_remove = []
        for key, timestamps in self.requests.items():
            self.requests[key] = [ts for ts in timestamps if ts > cutoff_time]
            if not self.requests[key]:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.requests[key]
        
        self._last_cleanup = current_time
    
    def is_allowed(self, key: str) -> Tuple[bool, Optional[str]]:
        """
        Проверяет, разрешен ли запрос.
        
        Args:
            key: Идентификатор клиента (IP, API key и т.п.)
            
        Returns:
            Кортеж (разрешен, сообщение_об_ошибке)
        """
        self._cleanup_old_entries()
        
        current_time = time.time()
        cutoff_time = current_time - self.window_seconds
        
        # Фильтруем старые запросы
        timestamps = self.requests[key]
        timestamps = [ts for ts in timestamps if ts > cutoff_time]
        self.requests[key] = timestamps
        
        # Проверяем лимит
        if len(timestamps) >= self.limit:
            oldest_request = min(timestamps)
            retry_after = int(self.window_seconds - (current_time - oldest_request))
            return False, f"Rate limit exceeded. Try again after {retry_after} seconds."
        
        # Добавляем текущий запрос
        timestamps.append(current_time)
        self.requests[key] = timestamps
        
        return True, None


# Глобальные экземпляры лимитеров
query_limiter = RateLimiter(QUERY_RATE_LIMIT, QUERY_RATE_WINDOW_SECONDS)
update_limiter = RateLimiter(UPDATE_RATE_LIMIT, UPDATE_RATE_WINDOW_SECONDS)
escalate_limiter = RateLimiter(ESCALATE_RATE_LIMIT, ESCALATE_RATE_WINDOW_SECONDS)

