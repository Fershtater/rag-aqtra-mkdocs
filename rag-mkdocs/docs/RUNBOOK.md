# RAG Service Runbook

Краткое руководство по эксплуатации RAG сервиса.

## Запуск сервиса

### Установка зависимостей

```bash
poetry install
# или
pip install -r requirements.txt
```

### Настройка окружения

Создайте `.env` файл из `.env.example` и настройте минимум:
- `OPENAI_API_KEY` - обязательный
- `UPDATE_API_KEY` - для `/update_index` endpoint
- `RAG_API_KEYS` - для `/api/answer` и `/stream` (опционально, если не задан - открытый доступ)

### Запуск сервера

```bash
poetry run uvicorn app.api.main:app --reload --port 8000
# или
uvicorn app.api.main:app --reload --port 8000
```

Сервер будет доступен на `http://localhost:8000`

## Управление индексом

### Где лежит индекс

Индекс хранится в `var/vectorstore/faiss_index/` (настраивается через `VECTORSTORE_DIR`).

### Пересборка индекса

#### Через HTTP endpoint (рекомендуется)

```bash
curl -X POST "http://localhost:8000/update_index" \
  -H "X-API-Key: your-update-api-key"
```

#### Через CLI скрипт

```bash
poetry run python scripts/update_index.py
```

### index.meta.json и index_version

При каждой пересборке создается файл `index.meta.json` с метаданными:
- `index_version` - уникальная версия индекса (timestamp-uuid)
- `created_at` - время создания
- `docs_hash` - хеш документов
- `chunks_count` - количество чанков
- и другие параметры

`index_version` используется для автоматической инвалидации кэша при смене индекса.

### Что делать если rebuild "завис"

Если `/update_index` возвращает ошибку `423 Locked`:

1. **Проверьте lock файл:**
   ```bash
   ls -la var/vectorstore/faiss_index.lock
   ```

2. **Проверьте возраст lock:**
   - Если lock старше `INDEX_LOCK_TIMEOUT_SECONDS * 2` (по умолчанию 600 секунд), он считается stale
   - Stale lock автоматически удаляется при следующей попытке

3. **Принудительное удаление lock (только если уверены):**
   ```bash
   rm var/vectorstore/faiss_index.lock
   ```
   ⚠️ **Внимание:** Удаляйте lock только если уверены, что другой процесс rebuild не выполняется.

4. **Проверьте логи:**
   - Lock информация логируется с PID и возрастом
   - Ищите сообщения типа "Failed to acquire lock" или "Stale lock detected"

## Debug и диагностика

### Включение debug.performance в /api/answer

Добавьте в запрос поле `debug`:

```json
{
  "question": "test",
  "api_key": "your-key",
  "debug": {
    "return_prompt": false,
    "return_chunks": true
  }
}
```

В ответе появится поле `debug.performance` с таймингами:
- `retrieval_ms` - время retrieval
- `prompt_render_ms` - время рендеринга промпта
- `llm_ms` - время генерации LLM
- `total_ms` - общее время

### Health check с диагностикой

#### Базовый health check

```bash
curl http://localhost:8000/health
```

#### Health check с диагностикой (только в development или с заголовком)

```bash
# В development режиме
ENV=development uvicorn app.api.main:app

# Или с заголовком X-Debug
curl -H "X-Debug: 1" http://localhost:8000/health
```

В ответе появится поле `diagnostics` с конфигурацией (без секретов).

### Проверка языковой политики

Язык ответа определяется приоритетом:
1. `passthrough.language` в запросе
2. `context_hint.language` в запросе
3. `Accept-Language` HTTP заголовок
4. Default: English

Проверка через `Accept-Language`:

```bash
curl -X POST "http://localhost:8000/api/answer" \
  -H "Content-Type: application/json" \
  -H "Accept-Language: fr-FR,fr;q=0.9" \
  -d '{"question": "test", "api_key": "your-key"}'
```

Поддерживаемые языки: `en`, `fr`, `de`, `es`, `pt` (настраивается через `PROMPT_SUPPORTED_LANGUAGES`).

## Мониторинг

### Prometheus метрики

```bash
curl http://localhost:8000/metrics
```

Доступные метрики:
- `rag_query_requests_total{status}` - количество запросов
- `rag_query_latency_seconds` - латентность запросов
- `rag_retrieval_latency_seconds{endpoint}` - латентность retrieval
- `rag_prompt_render_latency_seconds{endpoint}` - латентность рендеринга промпта
- `rag_llm_latency_seconds{endpoint}` - латентность LLM
- `rag_update_index_requests_total{status}` - количество пересборок индекса
- `rag_documents_in_index` - количество документов в индексе

### Логи

Логи включают:
- `request_id` для трейсинга запросов
- Stage timings в debug режиме
- Lock информация при rebuild
- Index version при загрузке

## Конфигурация

Основные настройки в `.env`:

- `VECTORSTORE_DIR` - путь к индексу (default: `var/vectorstore/faiss_index`)
- `DOCS_PATH` - путь к документам (default: `data/mkdocs_docs`)
- `PROMPT_TEMPLATE_MODE` - режим промпта: `legacy` или `jinja`
- `PROMPT_PRESET` - пресет промпта: `strict`, `support`, `developer`
- `PROMPT_VALIDATE_ON_STARTUP` - валидация промпта при старте (default: `true`)
- `INDEX_LOCK_TIMEOUT_SECONDS` - timeout для lock (default: `300`)
- `CACHE_TTL_SECONDS` - TTL кэша (default: `600`)
- `CACHE_MAX_SIZE` - размер кэша (default: `500`)

Полный список настроек см. в `app/settings.py`.

## Troubleshooting

### Индекс не загружается

1. Проверьте наличие индекса: `ls -la var/vectorstore/faiss_index/`
2. Проверьте логи при старте
3. Пересоберите индекс: `/update_index`

### Кэш не инвалидируется

1. Проверьте `index_version` в health diagnostics
2. Убедитесь, что `index_version` включен в cache key (проверьте логи)
3. Очистите кэш вручную (перезапустите сервер)

### Strict mode не работает

1. Проверьте `PROMPT_MODE=strict` в настройках
2. Проверьте `STRICT_SHORT_CIRCUIT=true`
3. Проверьте `NOT_FOUND_SCORE_THRESHOLD` (default: `0.20`)

### Язык ответа не соответствует ожидаемому

1. Проверьте `PROMPT_SUPPORTED_LANGUAGES` (должен включать нужный язык)
2. Проверьте приоритет: passthrough > context_hint > Accept-Language > default
3. Проверьте логи для `language_reason`

