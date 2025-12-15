#!/usr/bin/env bash

# Скрипт для локального запуска dev-сервера RAG-ассистента
# Использование: ./debug_launch.sh [дополнительные параметры uvicorn]

set -euo pipefail

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

info() {
    echo -e "${BLUE}[debug_launch]${NC} $1" >&2
}

error() {
    echo -e "${RED}[debug_launch] ERROR:${NC} $1" >&2
}

# Определяем директорию скрипта и корень проекта
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"
cd "${PROJECT_ROOT}"

# Загрузка .env если существует
if [ -f "${PROJECT_ROOT}/.env" ]; then
    set -o allexport
    # Загружаем .env, но не выводим значения переменных
    source "${PROJECT_ROOT}/.env" 2>/dev/null || true
    set +o allexport
    info ".env файл загружен"
else
    info ".env файл не найден, используем системные переменные окружения"
fi

# Установка дефолтов для окружения
ENV="${ENV:-development}"
LOG_LEVEL="${LOG_LEVEL:-DEBUG}"
PORT="${PORT:-8000}"

# Проверка Poetry
if ! command -v poetry &> /dev/null; then
    error "Poetry is required to run debug server"
    error "Install Poetry: curl -sSL https://install.python-poetry.org | python3 -"
    exit 1
fi

# Вывод информации о запуске
info "ENV=${ENV}, LOG_LEVEL=${LOG_LEVEL}, PORT=${PORT}"
info "Starting uvicorn app.api.main:app ..."

# Запуск сервера через Poetry
# Передаем дополнительные параметры, если они есть
exec poetry run uvicorn app.api.main:app --reload --port "${PORT}" "$@"
