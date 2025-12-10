"""
In-memory кэш для ответов RAG.
"""

import hashlib
import logging
import os
import time
from typing import Dict, Optional, Tuple, Any
from collections import OrderedDict

logger = logging.getLogger(__name__)

# Настройки кэша
CACHE_MAX_SIZE = int(os.getenv("CACHE_MAX_SIZE", "500"))
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "600"))  # 10 минут по умолчанию


class LRUCache:
    """LRU кэш с TTL."""
    
    def __init__(self, max_size: int = CACHE_MAX_SIZE, ttl_seconds: int = CACHE_TTL_SECONDS):
        """
        Args:
            max_size: Максимальное количество элементов
            ttl_seconds: Время жизни записи в секундах
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: OrderedDict[str, Tuple[Any, float]] = OrderedDict()
    
    def _generate_key(self, question: str, top_k: int, temperature: float, settings_signature: str) -> str:
        """Генерирует ключ кэша."""
        normalized_question = question.strip().lower()[:500]  # Ограничиваем длину
        key_data = f"{normalized_question}|{top_k}|{round(temperature, 2)}|{settings_signature}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """
        Получает значение из кэша.
        
        Args:
            key: Ключ кэша
            
        Returns:
            Значение или None если не найдено или истекло
        """
        if key not in self.cache:
            return None
        
        value, timestamp = self.cache[key]
        
        # Проверяем TTL
        if time.time() - timestamp > self.ttl_seconds:
            del self.cache[key]
            return None
        
        # Перемещаем в конец (LRU)
        self.cache.move_to_end(key)
        return value
    
    def set(self, key: str, value: Any):
        """
        Сохраняет значение в кэш.
        
        Args:
            key: Ключ кэша
            value: Значение для сохранения
        """
        # Удаляем старые записи если достигли лимита
        while len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)  # Удаляем самый старый
        
        self.cache[key] = (value, time.time())
    
    def clear(self):
        """Очищает кэш."""
        self.cache.clear()
    
    def size(self) -> int:
        """Возвращает текущий размер кэша."""
        return len(self.cache)


# Глобальный экземпляр кэша
response_cache = LRUCache()

