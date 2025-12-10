"""
FastAPI приложение для RAG-ассистента MkDocs.

Обоснование выбора FastAPI для RAG API:

1. АСИНХРОННОСТЬ:
   - Нативная поддержка async/await для эффективной работы с LLM API
   - Не блокирует event loop при ожидании ответов от OpenAI
   - Может обрабатывать несколько запросов параллельно
   - Критично для RAG, где каждый запрос требует:
     * Векторный поиск (может быть медленным)
     * API вызовы к OpenAI (сетевые задержки)
     * Обработка больших контекстов

2. ПРОИЗВОДИТЕЛЬНОСТЬ:
   - Один из самых быстрых Python веб-фреймворков
   - Основан на Starlette и Pydantic
   - Автоматическая валидация данных через Pydantic
   - Минимальные накладные расходы

3. ТИПИЗАЦИЯ И ВАЛИДАЦИЯ:
   - Полная поддержка type hints
   - Автоматическая валидация входных данных
   - Автоматическая генерация OpenAPI/Swagger документации
   - Улучшает надежность и отладку

4. АВТОМАТИЧЕСКАЯ ДОКУМЕНТАЦИЯ:
   - Swagger UI на /docs
   - ReDoc на /redoc
   - Позволяет тестировать API без дополнительных инструментов
   - Упрощает интеграцию для клиентов

5. ИНТЕГРАЦИЯ С LANGCHAIN:
   - LangChain поддерживает async вызовы
   - FastAPI легко интегрируется с async цепочками
   - Можно использовать background tasks для долгих операций

6. МАСШТАБИРУЕМОСТЬ:
   - Легко добавить middleware (логирование, CORS, аутентификация)
   - Поддержка WebSocket для streaming ответов
   - Можно легко добавить rate limiting
   - Готов к production deployment
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.rag_chain import (
    build_or_load_vectorstore,
    build_rag_chain,
    build_rag_chain_and_settings,
    chunk_documents,
    load_mkdocs_documents
)
from app.prompt_config import load_prompt_settings_from_env

# Загружаем переменные окружения из .env файла
load_dotenv()

# Настройка логирования с поддержкой LOG_LEVEL
env = os.getenv("ENV", "production").lower()
log_level_str = os.getenv("LOG_LEVEL", "DEBUG" if env == "development" else "INFO")
log_level = getattr(logging, log_level_str.upper(), logging.INFO)

logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

logger.info(f"Логирование настроено: уровень={log_level_str}, режим={env}")

# Проверяем наличие обязательных переменных окружения при старте
api_key = os.getenv("OPENAI_API_KEY")
if api_key is None:
    logger.error("OPENAI_API_KEY не найден в переменных окружения")
    raise ValueError(
        "OPENAI_API_KEY not set in .env\n"
        "Создайте файл .env из .env.example и добавьте OPENAI_API_KEY=your-key"
    )
logger.info("✓ OPENAI_API_KEY найден в переменных окружения")


# Pydantic модели для валидации данных
class Query(BaseModel):
    """Модель запроса пользователя."""
    question: str
    top_k: Optional[int] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "Как использовать эту функцию?",
                "top_k": 4,
                "temperature": 0.0
            }
        }


class QueryResponse(BaseModel):
    """Модель ответа RAG-системы."""
    answer: str
    sources: List[Dict[str, str]]
    
    class Config:
        json_schema_extra = {
            "example": {
                "answer": "Для использования функции нужно...",
                "sources": [
                    {
                        "source": "data/mkdocs_docs/docs/example.md",
                        "filename": "example.md"
                    }
                ]
            }
        }


class ErrorResponse(BaseModel):
    """Модель ответа с ошибкой."""
    error: str
    detail: Optional[str] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan контекст для инициализации и очистки ресурсов.
    
    Выполняется при старте:
    - Загружает vectorstore
    - Создает RAG цепочку
    - Сохраняет в app.state
    
    Выполняется при остановке:
    - Очищает ресурсы
    """
    # Startup: инициализация RAG цепочки
    logger.info("=" * 60)
    logger.info("ЗАПУСК FASTAPI ПРИЛОЖЕНИЯ")
    logger.info("=" * 60)
    logger.info("Инициализация RAG цепочки...")
    
    try:
        # Загружаем vectorstore и создаем RAG цепочку с настройками
        rag_chain, vectorstore, prompt_settings = build_rag_chain_and_settings()
        
        # Сохраняем в app.state для доступа из endpoints
        app.state.vectorstore = vectorstore
        app.state.rag_chain = rag_chain
        app.state.prompt_settings = prompt_settings
        
        logger.info("RAG цепочка готова к использованию")
        logger.info("=" * 60)
    except Exception as e:
        logger.error(f"Ошибка при инициализации RAG цепочки: {e}", exc_info=True)
        logger.error("Приложение запущено, но RAG недоступен")
        app.state.vectorstore = None
        app.state.rag_chain = None
        app.state.prompt_settings = None
    
    yield
    
    # Shutdown: очистка ресурсов
    logger.info("Остановка приложения...")
    app.state.vectorstore = None
    app.state.rag_chain = None
    app.state.prompt_settings = None


# Создаем FastAPI приложение с lifespan
app = FastAPI(
    title="RAG MkDocs Assistant API",
    description="API для вопросов по документации MkDocs с использованием RAG",
    version="1.0.0",
    lifespan=lifespan
)

# Добавляем CORS middleware для работы с фронтендом
# В production настройте origins более строго
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В production укажите конкретные домены
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Корневой endpoint для проверки работы API."""
    return {
        "message": "RAG MkDocs Assistant API",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check(request: Request):
    """
    Health check endpoint для мониторинга.
    
    Returns:
        JSON с статусом приложения и готовностью RAG цепочки
    """
    rag_chain = getattr(request.app.state, "rag_chain", None)
    return {
        "status": "ok" if rag_chain is not None else "degraded",
        "rag_chain_ready": rag_chain is not None
    }


@app.get("/config/prompt")
async def get_prompt_config(request: Request):
    """
    Возвращает текущие настройки промпта.
    
    Returns:
        JSON с настройками промпта из app.state.prompt_settings
    """
    prompt_settings = getattr(request.app.state, "prompt_settings", None)
    if prompt_settings is None:
        return JSONResponse(
            status_code=503,
            content={
                "error": "Настройки промпта не загружены",
                "detail": "RAG цепочка не инициализирована"
            }
        )
    
    return {
        "language": prompt_settings.language,
        "base_docs_url": prompt_settings.base_docs_url,
        "not_found_message": prompt_settings.not_found_message,
        "include_sources_in_text": prompt_settings.include_sources_in_text,
        "mode": prompt_settings.mode,
        "default_temperature": prompt_settings.default_temperature,
        "default_top_k": prompt_settings.default_top_k
    }


@app.post("/query", response_model=QueryResponse, responses={503: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def query_documentation(query: Query, request: Request):
    """
    Основной endpoint для вопросов по документации.
    
    Принимает вопрос пользователя и возвращает ответ на основе RAG-системы.
    
    Args:
        query: Объект Query с вопросом пользователя
        request: FastAPI Request для доступа к app.state
        
    Returns:
        QueryResponse с ответом и списком источников
    """
    # Получаем RAG цепочку из app.state
    rag_chain = getattr(request.app.state, "rag_chain", None)
    
    if rag_chain is None:
        logger.error("Попытка использовать RAG цепочку, но она не инициализирована")
        return JSONResponse(
            status_code=503,
            content={
                "error": "RAG цепочка не инициализирована",
                "detail": "Попробуйте позже или проверьте логи приложения"
            }
        )
    
    try:
        # Получаем настройки промпта из app.state
        prompt_settings = getattr(request.app.state, "prompt_settings", None)
        if prompt_settings is None:
            prompt_settings = load_prompt_settings_from_env()
        
        # Вычисляем эффективные значения из запроса или настроек
        effective_top_k = query.top_k if query.top_k is not None else prompt_settings.default_top_k
        effective_temperature = query.temperature if query.temperature is not None else prompt_settings.default_temperature
        
        # Безопасная валидация диапазонов
        effective_top_k = max(1, min(10, effective_top_k))
        effective_temperature = max(0.0, min(1.0, effective_temperature))
        
        # Если параметры отличаются от дефолтных, создаем новую цепочку с этими параметрами
        vectorstore = getattr(request.app.state, "vectorstore", None)
        if vectorstore is None:
            logger.error("Vectorstore не найден в app.state")
            return JSONResponse(
                status_code=503,
                content={
                    "error": "Vectorstore не инициализирован",
                    "detail": "Попробуйте позже или проверьте логи приложения"
                }
            )
        
        # Если параметры отличаются от дефолтных, создаем временную цепочку
        if effective_top_k != prompt_settings.default_top_k or effective_temperature != prompt_settings.default_temperature:
            logger.info(f"Создаю временную цепочку с top_k={effective_top_k}, temperature={effective_temperature}")
            temp_rag_chain = await asyncio.to_thread(
                build_rag_chain,
                vectorstore,
                prompt_settings=prompt_settings,
                k=effective_top_k,
                temperature=effective_temperature
            )
            rag_chain = temp_rag_chain
        # Иначе используем существующую цепочку из app.state
        
        # Вызываем RAG цепочку с вопросом пользователя асинхронно
        # Используем asyncio.to_thread() для выполнения синхронного invoke() в отдельном потоке
        # Это не блокирует event loop FastAPI
        logger.info(f"Обработка запроса: {query.question[:50]}...")
        result = await asyncio.to_thread(rag_chain.invoke, {"input": query.question})
        
        # Извлекаем ответ
        answer = result.get("answer", "Не удалось сгенерировать ответ")
        
        # Извлекаем источники из metadata найденных документов
        # В LangChain 0.2.0 create_retrieval_chain возвращает:
        # - "answer": сгенерированный ответ
        # - "context": список Document объектов, использованных для генерации
        sources = []
        
        # Пытаемся извлечь источники из context (основной способ в LangChain 0.2.0)
        if "context" in result:
            context_docs = result["context"]
            # context может быть списком или одним объектом
            if not isinstance(context_docs, list):
                context_docs = [context_docs]
            
            for doc in context_docs:
                if hasattr(doc, "metadata") and doc.metadata:
                    source_info = {
                        "source": doc.metadata.get("source", "unknown"),
                        "filename": doc.metadata.get("filename", doc.metadata.get("source", "unknown").split("/")[-1])
                    }
                    # Избегаем дубликатов
                    if source_info not in sources:
                        sources.append(source_info)
        
        # Fallback: пытаемся извлечь из source_documents (для совместимости)
        if not sources and "source_documents" in result:
            for doc in result["source_documents"]:
                if hasattr(doc, "metadata") and doc.metadata:
                    source_info = {
                        "source": doc.metadata.get("source", "unknown"),
                        "filename": doc.metadata.get("filename", doc.metadata.get("source", "unknown").split("/")[-1])
                    }
                    if source_info not in sources:
                        sources.append(source_info)
        
        logger.info(f"Успешно обработан запрос, найдено {len(sources)} источников")
        return QueryResponse(
            answer=answer,
            sources=sources
        )
        
    except Exception as e:
        # Логируем ошибку для отладки
        logger.error(f"Ошибка при обработке запроса: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": "Ошибка при обработке запроса",
                "detail": str(e)
            }
        )


@app.post("/update_index", responses={200: {"model": Dict}, 401: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def update_index(
    request: Request,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key")
):
    """
    Endpoint для принудительного обновления векторного индекса.
    
    Защищен API ключом. Пересоздает индекс один раз и обновляет app.state.
    
    Args:
        request: FastAPI Request для доступа к app.state
        x_api_key: API ключ в заголовке X-API-Key
        
    Returns:
        JSON с результатом обновления индекса
    """
    # Проверяем API ключ
    required_api_key = os.getenv("UPDATE_API_KEY")
    if not required_api_key:
        logger.warning("UPDATE_API_KEY не установлен в .env, endpoint /update_index недоступен")
        return JSONResponse(
            status_code=503,
            content={
                "error": "Endpoint недоступен",
                "detail": "UPDATE_API_KEY не настроен"
            }
        )
    
    if not x_api_key or x_api_key != required_api_key:
        logger.warning("Неверный API ключ для обновления индекса")
        return JSONResponse(
            status_code=401,
            content={
                "error": "Неверный API ключ",
                "detail": "Укажите правильный X-API-Key в заголовке запроса"
            }
        )
    
    try:
        logger.info("Начато обновление индекса...")
        
        # Загружаем и чанкируем документы
        documents = load_mkdocs_documents()
        if not documents:
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Не найдено документов",
                    "detail": "Убедитесь, что в data/mkdocs_docs есть .md файлы"
                }
            )
        
        chunks = chunk_documents(documents)
        logger.info(f"Загружено {len(documents)} документов, создано {len(chunks)} чанков")
        
        # Пересоздаем индекс ОДИН РАЗ
        vectorstore = build_or_load_vectorstore(
            chunks=chunks,
            force_rebuild=True
        )
        
        # Загружаем настройки промпта
        prompt_settings = load_prompt_settings_from_env()
        
        # Создаем новую RAG цепочку из уже пересобранного vectorstore
        # (не вызываем get_rag_chain, чтобы избежать повторной загрузки)
        rag_chain = build_rag_chain(
            vectorstore,
            prompt_settings=prompt_settings,
            k=prompt_settings.default_top_k,
            temperature=prompt_settings.default_temperature
        )
        
        # Обновляем app.state
        request.app.state.vectorstore = vectorstore
        request.app.state.rag_chain = rag_chain
        request.app.state.prompt_settings = prompt_settings
        
        logger.info("Индекс успешно обновлен и RAG цепочка пересоздана")
        return {
            "status": "success",
            "message": "Индекс успешно обновлен",
            "documents_count": len(documents),
            "chunks_count": len(chunks),
            "index_size": vectorstore.index.ntotal
        }
        
    except Exception as e:
        logger.error(f"Ошибка при обновлении индекса: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": "Ошибка при обновлении индекса",
                "detail": str(e)
            }
        )


if __name__ == "__main__":
    """
    Запуск приложения через uvicorn.
    
    Использование:
        python app/main.py
        
    Или через uvicorn напрямую:
        uvicorn app.main:app --reload
        
    Параметры:
        --reload: автоматическая перезагрузка при изменении кода (для разработки)
        --host: хост для прослушивания (по умолчанию 127.0.0.1)
        --port: порт (по умолчанию 8000)
    """
    import uvicorn
    
    # Запускаем сервер
    # Используем строку "app.main:app" для правильной работы с reload
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True  # Автоматическая перезагрузка при изменении кода
    )

