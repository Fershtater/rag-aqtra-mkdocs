"""
Утилиты для работы с OpenAI API.

Централизованный error handling, таймауты и retry логика.
"""

import logging
import os
import time
from typing import Optional, Callable, Any

from langchain_openai import OpenAIEmbeddings, ChatOpenAI

logger = logging.getLogger(__name__)

# Настройки таймаутов (в секундах)
OPENAI_TIMEOUT = int(os.getenv("OPENAI_TIMEOUT", "30"))
# Таймаут для LLM запросов (Chat API) - по умолчанию 2 минуты для больших контекстов
OPENAI_LLM_TIMEOUT = int(os.getenv("OPENAI_LLM_TIMEOUT", "120"))
# Таймаут для batch операций (создание индекса) - по умолчанию 5 минут
OPENAI_BATCH_TIMEOUT = int(os.getenv("OPENAI_BATCH_TIMEOUT", "300"))
OPENAI_MAX_RETRIES = int(os.getenv("OPENAI_MAX_RETRIES", "3"))
OPENAI_RETRY_BACKOFF_BASE = float(os.getenv("OPENAI_RETRY_BACKOFF_BASE", "1.5"))


def get_embeddings_client(timeout: Optional[int] = None) -> OpenAIEmbeddings:
    """
    Создает клиент для OpenAI embeddings с настройками таймаутов.
    
    Args:
        timeout: Таймаут в секундах. Если не указан, используется OPENAI_TIMEOUT.
                 Для batch операций используйте OPENAI_BATCH_TIMEOUT или передайте явно.
    
    Returns:
        OpenAIEmbeddings клиент
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set in environment")
    
    if timeout is None:
        timeout = OPENAI_TIMEOUT
    
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=api_key,
        timeout=timeout,
        max_retries=OPENAI_MAX_RETRIES
    )


def get_chat_llm(
    temperature: float = 0.0,
    model: str = "gpt-4o-mini",
    max_tokens: Optional[int] = None,
    timeout: Optional[int] = None
) -> ChatOpenAI:
    """
    Создает клиент для OpenAI Chat API с настройками таймаутов.
    
    Args:
        temperature: Температура генерации
        model: Модель OpenAI
        max_tokens: Максимальное количество токенов (опционально)
        timeout: Таймаут в секундах. Если не указан, используется OPENAI_LLM_TIMEOUT.
        
    Returns:
        ChatOpenAI клиент
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set in environment")
    
    if timeout is None:
        timeout = OPENAI_LLM_TIMEOUT
    
    kwargs = {
        "model": model,
        "temperature": temperature,
        "api_key": api_key,
        "timeout": timeout,
        "max_retries": OPENAI_MAX_RETRIES
    }
    
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    
    return ChatOpenAI(**kwargs)


def with_retries(
    fn: Callable,
    *args,
    max_retries: int = OPENAI_MAX_RETRIES,
    backoff_base: float = OPENAI_RETRY_BACKOFF_BASE,
    **kwargs
) -> Any:
    """
    Выполняет функцию с экспоненциальным backoff при ошибках.
    
    Args:
        fn: Функция для выполнения
        *args: Позиционные аргументы
        max_retries: Максимальное количество попыток
        backoff_base: База для экспоненциального backoff
        **kwargs: Именованные аргументы
        
    Returns:
        Результат выполнения функции
        
    Raises:
        Последнее исключение, если все попытки исчерпаны
    """
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_exception = e
            if attempt < max_retries - 1:
                wait_time = backoff_base ** attempt
                logger.warning(
                    f"Ошибка при вызове {fn.__name__} (попытка {attempt + 1}/{max_retries}): {e}. "
                    f"Повтор через {wait_time:.2f}с"
                )
                time.sleep(wait_time)
            else:
                logger.error(
                    f"Все попытки исчерпаны для {fn.__name__}: {e}",
                    exc_info=True
                )
    
    raise last_exception

