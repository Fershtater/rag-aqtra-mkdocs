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
import uuid
from contextlib import asynccontextmanager
from typing import Dict, List, Optional
from time import time

from dotenv import load_dotenv
from fastapi import FastAPI, Header, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr, Field, validator

from app.core.rag_chain import (
    build_or_load_vectorstore,
    build_rag_chain,
    build_rag_chain_and_settings,
    chunk_documents,
    load_mkdocs_documents,
)
from app.core.prompt_config import load_prompt_settings_from_env
from app.core.markdown_utils import build_doc_url
from app.infra.rate_limit import query_limiter, update_limiter, escalate_limiter
from app.infra.cache import response_cache
from app.infra.metrics import (
    get_metrics_response,
    update_index_metrics,
    query_requests_total,
    update_index_requests_total,
    rate_limit_hits_total,
    query_latency_seconds,
    update_index_duration_seconds,
    PROMETHEUS_AVAILABLE,
)
from app.infra.db import init_db, get_sessionmaker
from app.infra.analytics import hash_ip, log_query, log_escalation
from app.infra.zoho_desk import create_ticket

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
    question: str = Field(..., max_length=2000, description="Вопрос пользователя (макс. 2000 символов)")
    page_url: Optional[str] = Field(None, description="URL страницы, с которой задан вопрос")
    page_title: Optional[str] = Field(None, description="Заголовок страницы, с которой задан вопрос")
    
    @validator('question')
    def validate_question_length(cls, v):
        if len(v.strip()) > 2000:
            raise ValueError("Вопрос слишком длинный (максимум 2000 символов)")
        return v.strip()
    
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
    not_found: bool
    request_id: str
    latency_ms: int
    cache_hit: bool
    
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


def calculate_effective_top_k(question: str, base_top_k: int, mode: str) -> int:
    """
    Простая адаптация top_k по длине вопроса.
    
    Args:
        question: Текст вопроса пользователя
        base_top_k: Базовое значение top_k (из настроек или запроса)
        mode: Режим промпта (strict|helpful) — оставлен для будущих расширений
    
    Returns:
        Адаптированное значение top_k в разумных пределах.
    """
    words = question.split()
    length = len(words)

    k = base_top_k
    if length > 20:
        k = max(base_top_k, 8)
    elif length > 10:
        k = max(base_top_k, 6)

    # Ограничим сверху, чтобы не перегружать retriever
    return min(max(1, k), 10)


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

    # Инициализируем БД для логирования, если настроен DATABASE_URL
    db_url = os.getenv("DATABASE_URL")
    if db_url:
        try:
            logger.info("Инициализирую подключение к БД для логирования...")
            db_sessionmaker = await init_db(db_url)
            app.state.db_sessionmaker = db_sessionmaker
            logger.info("Подключение к БД успешно инициализировано")
        except Exception as e:
            logger.error("Не удалось инициализировать БД: %s", e, exc_info=True)
            app.state.db_sessionmaker = None
    else:
        logger.info("DATABASE_URL не задан, логирование в БД отключено")
        app.state.db_sessionmaker = None
    
    yield
    
    # Shutdown: очистка ресурсов
    logger.info("Остановка приложения...")
    app.state.vectorstore = None
    app.state.rag_chain = None
    app.state.prompt_settings = None
    app.state.db_sessionmaker = None


# Создаем FastAPI приложение с lifespan
app = FastAPI(
    title="RAG MkDocs Assistant API",
    description="API для вопросов по документации MkDocs с использованием RAG",
    version="1.0.0",
    lifespan=lifespan
)

# Middleware для correlation IDs
@app.middleware("http")
async def correlation_id_middleware(request: Request, call_next):
    """Добавляет correlation ID к каждому запросу."""
    # Читаем или генерируем request ID
    request_id = request.headers.get("X-Request-Id", str(uuid.uuid4()))
    request.state.request_id = request_id
    
    # Обновляем формат логирования для включения request_id
    old_factory = logging.getLogRecordFactory()
    def record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        record.request_id = request_id
        return record
    logging.setLogRecordFactory(record_factory)
    
    response = await call_next(request)
    response.headers["X-Request-Id"] = request_id
    return response

# Добавляем CORS middleware для работы с фронтендом
# CORS origins настраиваются через переменную окружения CORS_ORIGINS (список через запятую)
# Если не указано, используется "*" только для development режима
cors_origins_str = os.getenv("CORS_ORIGINS", "")
if cors_origins_str:
    cors_origins = [origin.strip() for origin in cors_origins_str.split(",") if origin.strip()]
else:
    # Fallback: разрешаем все только в development
    cors_origins = ["*"] if env == "development" else []

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],  # Ограничиваем методы
    allow_headers=["Content-Type", "X-API-Key", "X-Request-Id"],  # Ограничиваем заголовки
)


@app.get("/")
async def root():
    """Корневой endpoint для проверки работы API."""
    return {
        "message": "RAG MkDocs Assistant API",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/metrics")
async def metrics_endpoint():
    """Endpoint для метрик Prometheus."""
    content, content_type = get_metrics_response()
    return Response(content=content, media_type=content_type)


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
        "default_top_k": prompt_settings.default_top_k,
        "default_max_tokens": getattr(prompt_settings, "default_max_tokens", None),
    }


@app.post("/query", response_model=QueryResponse, responses={429: {"model": ErrorResponse}, 503: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def query_documentation(query: Query, request: Request):
    """
    Основной endpoint для вопросов по документации.
    
    Принимает вопрос пользователя и возвращает ответ на основе RAG-системы.
    Поддерживает кэширование, rate limiting и метрики.
    
    Args:
        query: Объект Query с вопросом пользователя
        request: FastAPI Request для доступа к app.state
        
    Returns:
        QueryResponse с ответом и списком источников
    """
    request_id = getattr(request.state, "request_id", "unknown")
    start_time = time()
    
    # Rate limiting
    client_ip = request.client.host if request.client else "unknown"
    allowed, error_msg = query_limiter.is_allowed(client_ip)
    if not allowed:
        if PROMETHEUS_AVAILABLE and rate_limit_hits_total is not None:
            rate_limit_hits_total.labels(endpoint="query").inc()
        logger.warning(f"[{request_id}] Rate limit exceeded for {client_ip}")
        return JSONResponse(
            status_code=429,
            content={"error": "Rate limit exceeded", "detail": error_msg}
        )
    
    # Получаем RAG цепочку из app.state
    rag_chain = getattr(request.app.state, "rag_chain", None)
    
    if rag_chain is None:
        logger.error(f"[{request_id}] RAG цепочка не инициализирована")
        if PROMETHEUS_AVAILABLE and query_requests_total is not None:
            query_requests_total.labels(status="error").inc()
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

        # Сигнатура настроек для кэша: фиксируем только стабильные параметры
        settings_signature = (
            f"{prompt_settings.language}_"
            f"{prompt_settings.mode}_"
            f"{prompt_settings.default_top_k}_"
            f"{prompt_settings.default_temperature}_"
            f"{getattr(prompt_settings, 'default_max_tokens', None)}"
        )

        # Генерируем ключ кэша
        cache_key = response_cache._generate_key(query.question, settings_signature)

        # Проверяем кэш
        cached_result: Optional[QueryResponse] = response_cache.get(cache_key)
        cache_hit = cached_result is not None

        if cached_result:
            logger.debug(f"[{request_id}] Cache hit for query")
            latency = int((time() - start_time) * 1000)
            if PROMETHEUS_AVAILABLE and query_requests_total is not None:
                query_requests_total.labels(status="success").inc()
                query_latency_seconds.observe(latency / 1000.0)

            # Логируем запрос даже при cache hit
            db_sessionmaker = getattr(request.app.state, "db_sessionmaker", None)
            sources = cached_result.sources or []
            not_found_flag = cached_result.not_found
            ip_hash_value = hash_ip(client_ip)
            user_agent = request.headers.get("User-Agent")

            await log_query(
                db_sessionmaker,
                request_id=request_id,
                ip_hash_value=ip_hash_value,
                user_agent=user_agent,
                page_url=query.page_url,
                page_title=query.page_title,
                question=query.question,
                not_found=not_found_flag,
                cache_hit=True,
                latency_ms=latency,
                sources=sources,
                error=None,
            )

            return cached_result

        # Вызываем RAG цепочку с вопросом пользователя асинхронно
        try:
            result = await asyncio.to_thread(rag_chain.invoke, {"input": query.question})
        except Exception as e:
            logger.error(f"[{request_id}] Ошибка при вызове LLM: {e}", exc_info=True)
            if PROMETHEUS_AVAILABLE and query_requests_total is not None:
                query_requests_total.labels(status="error").inc()
                query_latency_seconds.observe(time() - start_time)
            return JSONResponse(
                status_code=503,
                content={
                    "error": "Ошибка при обработке запроса",
                    "detail": "Сервис временно недоступен. Попробуйте позже."
                }
            )
        
        # Извлекаем ответ
        answer = result.get("answer", "Не удалось сгенерировать ответ")
        
        # Извлекаем источники из metadata найденных документов
        # В LangChain 0.2.0 create_retrieval_chain возвращает:
        # - "answer": сгенерированный ответ
        # - "context": список Document объектов, использованных для генерации
        sources = []
        
        # Извлекаем источники из context
        if "context" in result:
            context_docs = result["context"]
            if not isinstance(context_docs, list):
                context_docs = [context_docs]
            
            seen_sources = set()
            for doc in context_docs:
                if hasattr(doc, "metadata") and doc.metadata:
                    source = doc.metadata.get("source", "unknown")
                    section_anchor = doc.metadata.get("section_anchor")
                    source_key = (source, section_anchor)
                    if source_key in seen_sources:
                        continue
                    seen_sources.add(source_key)
                    
                    source_info = {
                        "source": source,
                        "filename": doc.metadata.get("filename", source.split("/")[-1])
                    }
                    
                    # Добавляем новые поля если есть
                    if "section_title" in doc.metadata:
                        source_info["section_title"] = doc.metadata["section_title"]
                    if "section_anchor" in doc.metadata:
                        source_info["section_anchor"] = doc.metadata["section_anchor"]
                        source_info["url"] = build_doc_url(
                            prompt_settings.base_docs_url,
                            source,
                            doc.metadata["section_anchor"]
                        )
                    elif source != "unknown":
                        source_info["url"] = build_doc_url(
                            prompt_settings.base_docs_url,
                            source,
                            None
                        )
                    
                    sources.append(source_info)
        
        # Fallback: пытаемся извлечь из source_documents
        if not sources and "source_documents" in result:
            for doc in result["source_documents"]:
                if hasattr(doc, "metadata") and doc.metadata:
                    source = doc.metadata.get("source", "unknown")
                    source_info = {
                        "source": source,
                        "filename": doc.metadata.get("filename", source.split("/")[-1])
                    }
                    if "section_title" in doc.metadata:
                        source_info["section_title"] = doc.metadata["section_title"]
                    if "section_anchor" in doc.metadata:
                        source_info["section_anchor"] = doc.metadata["section_anchor"]
                        source_info["url"] = build_doc_url(
                            prompt_settings.base_docs_url,
                            source,
                            doc.metadata["section_anchor"]
                        )
                    elif source != "unknown":
                        source_info["url"] = build_doc_url(
                            prompt_settings.base_docs_url,
                            source,
                            None
                        )
                    sources.append(source_info)
        
        # Определяем not_found по сообщению и/или отсутствию источников
        not_found_flag = (
            answer.strip() == prompt_settings.not_found_message.strip()
            or len(sources) == 0
        )

        latency_sec = time() - start_time
        latency_ms = int(latency_sec * 1000)

        # Формируем ответ
        response = QueryResponse(
            answer=answer,
            sources=sources,
            not_found=not_found_flag,
            request_id=request_id,
            latency_ms=latency_ms,
            cache_hit=False,
        )

        # Сохраняем в кэш
        response_cache.set(cache_key, response)

        # Обновляем метрики
        if PROMETHEUS_AVAILABLE and query_requests_total is not None:
            query_requests_total.labels(status="success").inc()
            query_latency_seconds.observe(latency_sec)

        # Логируем в БД (если доступно)
        db_sessionmaker = getattr(request.app.state, "db_sessionmaker", None)
        ip_hash_value = hash_ip(client_ip)
        user_agent = request.headers.get("User-Agent")

        await log_query(
            db_sessionmaker,
            request_id=request_id,
            ip_hash_value=ip_hash_value,
            user_agent=user_agent,
            page_url=query.page_url,
            page_title=query.page_title,
            question=query.question,
            not_found=not_found_flag,
            cache_hit=False,
            latency_ms=latency_ms,
            sources=sources,
            error=None,
        )

        logger.info(
            "[%s] Запрос обработан успешно, источников: %s, not_found=%s, cache_hit=%s, latency_ms=%s",
            request_id,
            len(sources),
            not_found_flag,
            False,
            latency_ms,
        )
        return response
        
    except Exception as e:
        logger.error(f"[{request_id}] Ошибка при обработке запроса: {e}", exc_info=True)
        if PROMETHEUS_AVAILABLE and query_requests_total is not None:
            query_requests_total.labels(status="error").inc()
            query_latency_seconds.observe(time() - start_time)

        # Пытаемся залогировать ошибку запроса
        try:
            db_sessionmaker = getattr(request.app.state, "db_sessionmaker", None)
            ip_hash_value = hash_ip(client_ip)
            user_agent = request.headers.get("User-Agent")
            latency_ms = int((time() - start_time) * 1000)

            await log_query(
                db_sessionmaker,
                request_id=request_id,
                ip_hash_value=ip_hash_value,
                user_agent=user_agent,
                page_url=query.page_url,
                page_title=query.page_title,
                question=query.question,
                not_found=False,
                cache_hit=False,
                latency_ms=latency_ms,
                sources=[],
                error=str(e),
            )
        except Exception:
            # Логирование ошибок логирования опускаем, чтобы не маскировать основную ошибку
            pass
        return JSONResponse(
            status_code=500,
            content={
                "error": "Ошибка при обработке запроса",
                "detail": "Внутренняя ошибка сервера"
            }
        )


@app.post("/escalate", responses={400: {"model": ErrorResponse}, 404: {"model": ErrorResponse}, 429: {"model": ErrorResponse}, 503: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def escalate_issue(payload: EscalateRequest, request: Request):
    """
    Endpoint для эскалации в службу поддержки (Zoho Desk).
    
    Требует, чтобы исходный запрос имел not_found=true.
    """
    request_id = getattr(request.state, "request_id", "unknown")
    client_ip = request.client.host if request.client else "unknown"

    # Rate limiting по IP
    allowed, error_msg = escalate_limiter.is_allowed(client_ip)
    if not allowed:
        if PROMETHEUS_AVAILABLE and rate_limit_hits_total is not None:
            rate_limit_hits_total.labels(endpoint="escalate").inc()
        logger.warning("[%s] Escalate rate limit exceeded for %s", request_id, client_ip)
        return JSONResponse(
            status_code=429,
            content={"error": "Rate limit exceeded", "detail": error_msg},
        )

    db_sessionmaker = getattr(request.app.state, "db_sessionmaker", None)
    if db_sessionmaker is None:
        logger.error("[%s] DATABASE_URL не настроен, эскалация недоступна", request_id)
        return JSONResponse(
            status_code=503,
            content={
                "error": "Escalation is not configured",
                "detail": "DATABASE_URL is not set on the server",
            },
        )

    # Ищем соответствующую запись QueryLog
    from sqlalchemy import select
    from app.infra.models import QueryLog
    from sqlalchemy.exc import SQLAlchemyError

    try:
        async with db_sessionmaker() as session:
            result = await session.execute(
                select(QueryLog).where(QueryLog.request_id == payload.request_id)
            )
            query_log = result.scalar_one_or_none()
    except SQLAlchemyError as e:
        logger.error("[%s] Ошибка при поиске QueryLog для эскалации: %s", request_id, e, exc_info=True)
        await log_escalation(
            db_sessionmaker,
            request_id=payload.request_id,
            email=payload.email,
            status="db_error",
            error=str(e),
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": "Database error",
                "detail": "Failed to load query log for escalation",
            },
        )

    if query_log is None:
        logger.warning("[%s] QueryLog not found for request_id=%s", request_id, payload.request_id)
        await log_escalation(
            db_sessionmaker,
            request_id=payload.request_id,
            email=payload.email,
            status="not_found",
            error="QueryLog not found",
        )
        return JSONResponse(
            status_code=404,
            content={
                "error": "Query not found",
                "detail": "Cannot escalate request that is not logged",
            },
        )

    if not query_log.not_found:
        logger.warning("[%s] Escalation requested for non-not-found query_id=%s", request_id, payload.request_id)
        await log_escalation(
            db_sessionmaker,
            request_id=payload.request_id,
            email=payload.email,
            status="rejected",
            error="Escalation allowed only for not_found queries",
        )
        return JSONResponse(
            status_code=400,
            content={
                "error": "Escalation not allowed",
                "detail": "Escalation is allowed only when the original answer was not_found",
            },
        )

    # Формируем subject и description для тикета
    question = query_log.question or ""
    subject = f"Aqtra Docs: {question[:80] or 'User escalation'}"

    parts = [
        f"User email: {payload.email}",
        f"Request ID: {payload.request_id}",
        "",
        f"Question: {question}",
        f"Not found: {query_log.not_found}",
        f"Page URL: {query_log.page_url or '-'}",
        f"Page Title: {query_log.page_title or '-'}",
        "",
        f"Sources: {query_log.sources or '[]'}",
    ]
    if payload.comment:
        parts.append("")
        parts.append(f"User comment: {payload.comment}")

    description = "\n".join(parts)

    # Создаем тикет в Zoho Desk
    zoho_ticket_id = None
    zoho_ticket_number = None

    try:
        ticket_response = await create_ticket(
            email=payload.email,
            subject=subject,
            description=description,
        )
        zoho_ticket_id = str(ticket_response.get("id") or "")
        zoho_ticket_number = str(ticket_response.get("ticketNumber") or "")

        await log_escalation(
            db_sessionmaker,
            request_id=payload.request_id,
            email=payload.email,
            status="success",
            zoho_ticket_id=zoho_ticket_id or None,
            zoho_ticket_number=zoho_ticket_number or None,
            error=None,
        )

        return {
            "status": "success",
            "ticket_id": zoho_ticket_id,
            "ticket_number": zoho_ticket_number,
        }
    except Exception as e:
        logger.error("[%s] Ошибка при создании тикета Zoho Desk: %s", request_id, e, exc_info=True)
        await log_escalation(
            db_sessionmaker,
            request_id=payload.request_id,
            email=payload.email,
            status="error",
            error=str(e),
        )
        return JSONResponse(
            status_code=503,
            content={
                "error": "Escalation failed",
                "detail": "Failed to create ticket in Zoho Desk",
            },
        )


@app.post("/update_index", responses={200: {"model": Dict}, 401: {"model": ErrorResponse}, 429: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def update_index(
    request: Request,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key")
):
    """
    Endpoint для принудительного обновления векторного индекса.
    
    Защищен API ключом. Пересоздает индекс один раз и обновляет app.state.
    Поддерживает rate limiting и метрики.
    
    Args:
        request: FastAPI Request для доступа к app.state
        x_api_key: API ключ в заголовке X-API-Key
        
    Returns:
        JSON с результатом обновления индекса
    """
    request_id = getattr(request.state, "request_id", "unknown")
    start_time = time()
    
    # Rate limiting по API ключу или IP
    limiter_key = x_api_key or (request.client.host if request.client else "unknown")
    allowed, error_msg = update_limiter.is_allowed(limiter_key)
    if not allowed:
        if PROMETHEUS_AVAILABLE and rate_limit_hits_total is not None:
            rate_limit_hits_total.labels(endpoint="update_index").inc()
        logger.warning(f"[{request_id}] Rate limit exceeded for update_index")
        return JSONResponse(
            status_code=429,
            content={"error": "Rate limit exceeded", "detail": error_msg}
        )
    
    # Проверяем API ключ
    required_api_key = os.getenv("UPDATE_API_KEY")
    if not required_api_key:
        logger.warning(f"[{request_id}] UPDATE_API_KEY не установлен в .env")
        return JSONResponse(
            status_code=503,
            content={
                "error": "Endpoint недоступен",
                "detail": "UPDATE_API_KEY не настроен"
            }
        )
    
    if not x_api_key or x_api_key != required_api_key:
        logger.warning(f"[{request_id}] Неверный API ключ для обновления индекса")
        if PROMETHEUS_AVAILABLE and update_index_requests_total is not None:
            update_index_requests_total.labels(status="error").inc()
        return JSONResponse(
            status_code=401,
            content={
                "error": "Неверный API ключ",
                "detail": "Укажите правильный X-API-Key в заголовке запроса"
            }
        )
    
    try:
        logger.info(f"[{request_id}] Начато обновление индекса...")
        
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
        
        # Обновляем метрики индекса
        if PROMETHEUS_AVAILABLE:
            update_index_metrics(len(documents), len(chunks))
            if update_index_requests_total is not None:
                update_index_requests_total.labels(status="success").inc()
            if update_index_duration_seconds is not None:
                update_index_duration_seconds.observe(time() - start_time)
        
        logger.info(f"[{request_id}] Индекс успешно обновлен и RAG цепочка пересоздана")
        return {
            "status": "success",
            "message": "Индекс успешно обновлен",
            "documents_count": len(documents),
            "chunks_count": len(chunks),
            "index_size": vectorstore.index.ntotal
        }
        
    except Exception as e:
        logger.error(f"[{request_id}] Ошибка при обновлении индекса: {e}", exc_info=True)
        if PROMETHEUS_AVAILABLE:
            if update_index_requests_total is not None:
                update_index_requests_total.labels(status="error").inc()
            if update_index_duration_seconds is not None:
                update_index_duration_seconds.observe(time() - start_time)
        return JSONResponse(
            status_code=500,
            content={
                "error": "Ошибка при обновлении индекса",
                "detail": "Внутренняя ошибка сервера"
            }
        )


if __name__ == "__main__":
    """
    Запуск приложения через uvicorn.
    
    Использование:
        python -m app.api.main
        
    Или через uvicorn напрямую:
        uvicorn app.api.main:app --reload --port 8000
    """
    import uvicorn
    
    # Запускаем сервер
    uvicorn.run(
        "app.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )

