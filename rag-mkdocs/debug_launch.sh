#!/bin/bash

# Скрипт автоматизации запуска и тестирования RAG MkDocs приложения
# 
# Использование:
#   chmod +x debug_launch.sh
#   ./debug_launch.sh
#
# Скрипт выполняет:
# 1. Проверку окружения и установку зависимостей
# 2. Обновление векторного индекса
# 3. Запуск сервера в фоне
# 4. Запуск автоматических тестов
# 5. Остановку сервера

set -e  # Остановка при ошибке

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Функции для вывода
info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Переменные
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_PID=""
PORT=8000
BASE_URL="http://localhost:${PORT}"

# Функция очистки при выходе
cleanup() {
    if [ ! -z "$SERVER_PID" ]; then
        info "Остановка сервера (PID: $SERVER_PID)..."
        kill $SERVER_PID 2>/dev/null || true
        wait $SERVER_PID 2>/dev/null || true
        success "Сервер остановлен"
    fi
}

# Регистрация обработчика выхода
trap cleanup EXIT INT TERM

# Шаг 1: Проверка окружения
echo "=========================================="
echo "ШАГ 1: Проверка окружения"
echo "=========================================="

# Инициализация pyenv (если установлен)
if command -v pyenv &> /dev/null; then
    eval "$(pyenv init -)" 2>/dev/null || true
fi

# Проверка Poetry
if ! command -v poetry &> /dev/null; then
    error "Poetry не установлен. Установите: curl -sSL https://install.python-poetry.org | python3 -"
    exit 1
fi
info "Poetry найден: $(poetry --version)"

# Проверка Python 3.12
PYTHON_CMD=""
if command -v python3.12 &> /dev/null; then
    PYTHON_CMD="python3.12"
    info "Python 3.12 найден в PATH: $(python3.12 --version)"
elif command -v pyenv &> /dev/null; then
    # Проверяем, установлен ли Python 3.12 через pyenv
    if pyenv versions --bare | grep -q "^3.12"; then
        PYTHON_CMD=$(pyenv which python3.12 2>/dev/null || echo "")
        if [ ! -z "$PYTHON_CMD" ]; then
            info "Python 3.12 найден через pyenv: $($PYTHON_CMD --version 2>&1)"
        fi
    fi
fi

# Если Python 3.12 не найден, проверяем существующее окружение Poetry
if [ -z "$PYTHON_CMD" ]; then
    info "Python 3.12 не найден в PATH, проверяю окружение Poetry..."
    POETRY_ENV_PYTHON=$(poetry env info -p 2>/dev/null)/bin/python
    if [ -f "$POETRY_ENV_PYTHON" ]; then
        PYTHON_VERSION=$($POETRY_ENV_PYTHON --version 2>&1 | grep -o "3\.12" || echo "")
        if [ ! -z "$PYTHON_VERSION" ]; then
            info "Найдено окружение Poetry с Python 3.12"
            PYTHON_CMD="$POETRY_ENV_PYTHON"
        fi
    fi
fi

# Если всё ещё не найден, предлагаем установить
if [ -z "$PYTHON_CMD" ]; then
    error "Python 3.12 не найден"
    error "ВАЖНО: LangChain 0.2.0 требует Python 3.12 (не 3.14!)"
    error ""
    error "Варианты решения:"
    error "1. Установить через pyenv:"
    error "   pyenv install 3.12.0"
    error "   pyenv local 3.12.0"
    error ""
    error "2. Или Poetry создаст окружение автоматически при установке:"
    error "   poetry env use python3.12  # после установки Python 3.12"
    error "   poetry install"
    exit 1
fi

# Проверка .env файла
if [ ! -f "${SCRIPT_DIR}/.env" ]; then
    warning "Файл .env не найден"
    if [ -f "${SCRIPT_DIR}/.env.example" ]; then
        info "Создание .env из .env.example..."
        cp "${SCRIPT_DIR}/.env.example" "${SCRIPT_DIR}/.env"
        warning "⚠ ВАЖНО: Отредактируйте .env и добавьте OPENAI_API_KEY"
        warning "Нажмите Enter после добавления ключа..."
        read
    else
        error "Файл .env.example не найден"
        exit 1
    fi
fi

# Проверка OPENAI_API_KEY
if ! grep -q "OPENAI_API_KEY=" "${SCRIPT_DIR}/.env" || grep -q "OPENAI_API_KEY=your-key" "${SCRIPT_DIR}/.env"; then
    error "OPENAI_API_KEY не настроен в .env"
    error "Отредактируйте .env и добавьте ваш OpenAI API ключ"
    exit 1
fi
success "✓ .env файл настроен"

# Шаг 2: Настройка окружения Poetry
echo ""
echo "=========================================="
echo "ШАГ 2: Настройка окружения Poetry"
echo "=========================================="

cd "${SCRIPT_DIR}"

# Создание окружения с Python 3.12
info "Создание виртуального окружения..."
if poetry env use ${PYTHON_CMD} 2>&1 | grep -q "Using virtualenv"; then
    success "✓ Виртуальное окружение создано"
else
    warning "Окружение уже существует или используется существующее"
fi

# Установка зависимостей
info "Установка зависимостей..."
if poetry install --no-interaction; then
    success "✓ Зависимости установлены"
else
    error "Ошибка при установке зависимостей"
    exit 1
fi

# Шаг 3: Обновление индекса
echo ""
echo "=========================================="
echo "ШАГ 3: Обновление векторного индекса"
echo "=========================================="

info "Запуск update_index.py..."
if poetry run python update_index.py; then
    success "✓ Индекс успешно обновлен"
else
    error "Ошибка при обновлении индекса"
    exit 1
fi

# Шаг 4: Запуск сервера
echo ""
echo "=========================================="
echo "ШАГ 4: Запуск FastAPI сервера"
echo "=========================================="

info "Запуск сервера на порту ${PORT}..."

# Запуск сервера в фоне
poetry run uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --reload > /tmp/rag_server.log 2>&1 &
SERVER_PID=$!

# Ожидание запуска сервера
info "Ожидание запуска сервера..."
for i in {1..30}; do
    if curl -s "${BASE_URL}/health" > /dev/null 2>&1; then
        success "✓ Сервер запущен (PID: $SERVER_PID)"
        success "✓ Сервер доступен на ${BASE_URL}"
        break
    fi
    if [ $i -eq 30 ]; then
        error "Сервер не запустился за 30 секунд"
        error "Проверьте логи: tail -f /tmp/rag_server.log"
        exit 1
    fi
    sleep 1
done

# Проверка health
info "Проверка health check..."
HEALTH_RESPONSE=$(curl -s "${BASE_URL}/health")
if echo "$HEALTH_RESPONSE" | grep -q '"rag_chain_ready":true'; then
    success "✓ RAG цепочка готова к работе"
else
    warning "⚠ RAG цепочка может быть не готова"
    echo "Response: $HEALTH_RESPONSE"
fi

# Шаг 5: Автоматическое тестирование
echo ""
echo "=========================================="
echo "ШАГ 5: Автоматическое тестирование"
echo "=========================================="

info "Запуск test_api.py..."
if poetry run python test_api.py; then
    success "✓ Все тесты пройдены"
else
    warning "⚠ Некоторые тесты не пройдены"
    warning "Проверьте логи сервера: tail -f /tmp/rag_server.log"
fi

# Шаг 6: Интерактивное тестирование
echo ""
echo "=========================================="
echo "ШАГ 6: Интерактивное тестирование"
echo "=========================================="

info "Сервер запущен и готов к тестированию"
info "Вы можете:"
info "  1. Открыть Swagger UI: ${BASE_URL}/docs"
info "  2. Отправить запрос: curl -X POST ${BASE_URL}/query -H 'Content-Type: application/json' -d '{\"question\": \"Ваш вопрос\"}'"
info "  3. Просмотреть логи сервера: tail -f /tmp/rag_server.log"
echo ""
warning "Нажмите Enter для остановки сервера..."
read

# Очистка выполнится автоматически через trap

success "Готово! Все шаги выполнены."

