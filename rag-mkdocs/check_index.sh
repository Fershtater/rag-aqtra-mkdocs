#!/bin/bash

# Скрипт для проверки состояния векторного индекса
# Использование: ./check_index.sh

set -e

# Цвета
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INDEX_PATH="${SCRIPT_DIR}/vectorstore/faiss_index"

echo "=========================================="
echo "ПРОВЕРКА СОСТОЯНИЯ ВЕКТОРНОГО ИНДЕКСА"
echo "=========================================="
echo ""

# 1. Проверка существования директории
if [ ! -d "$INDEX_PATH" ]; then
    echo -e "${RED}✗ Директория индекса не найдена: $INDEX_PATH${NC}"
    echo "  Индекс еще не создан. Запустите: poetry run python update_index.py"
    exit 1
fi
echo -e "${GREEN}✓ Директория индекса существует${NC}"

# 2. Проверка файлов индекса
echo ""
echo "Файлы индекса:"
if [ -f "${INDEX_PATH}/index.faiss" ]; then
    SIZE=$(du -h "${INDEX_PATH}/index.faiss" | cut -f1)
    echo -e "  ${GREEN}✓ index.faiss${NC} (размер: $SIZE)"
else
    echo -e "  ${RED}✗ index.faiss не найден${NC}"
fi

if [ -f "${INDEX_PATH}/index.pkl" ]; then
    SIZE=$(du -h "${INDEX_PATH}/index.pkl" | cut -f1)
    echo -e "  ${GREEN}✓ index.pkl${NC} (размер: $SIZE)"
else
    echo -e "  ${RED}✗ index.pkl не найден${NC}"
fi

if [ -f "${INDEX_PATH}/.docs_hash" ]; then
    HASH=$(cat "${INDEX_PATH}/.docs_hash" | head -1)
    echo -e "  ${GREEN}✓ .docs_hash${NC} (hash: ${HASH:0:16}...)"
else
    echo -e "  ${YELLOW}⚠ .docs_hash не найден${NC} (индекс может быть устаревшим)"
fi

# 3. Размер индекса
echo ""
TOTAL_SIZE=$(du -sh "${INDEX_PATH}" 2>/dev/null | cut -f1)
echo "Общий размер индекса: $TOTAL_SIZE"

# 4. Проверка через Python (если возможно)
echo ""
echo "Детальная информация:"
cd "$SCRIPT_DIR"

if command -v poetry &> /dev/null; then
    # Инициализация pyenv если нужно
    if command -v pyenv &> /dev/null; then
        eval "$(pyenv init -)" 2>/dev/null || true
    fi
    
    # Попытка загрузить индекс и получить информацию
    python_code='
import sys
from pathlib import Path

index_path = Path("vectorstore/faiss_index")
if not index_path.exists():
    print("  ✗ Индекс не найден")
    sys.exit(1)

try:
    from langchain_community.vectorstores import FAISS
    from langchain_openai import OpenAIEmbeddings
    from dotenv import load_dotenv
    import os
    
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("  ⚠ OPENAI_API_KEY не найден, пропускаю проверку загрузки")
        sys.exit(0)
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key)
    is_dev = os.getenv("ENV", "development").lower() == "development"
    
    vectorstore = FAISS.load_local(
        str(index_path),
        embeddings,
        allow_dangerous_deserialization=is_dev
    )
    
    print(f"  ✓ Индекс успешно загружен")
    print(f"  ✓ Количество документов: {vectorstore.index.ntotal}")
    print(f"  ✓ Размерность векторов: {vectorstore.index.d}")
    print(f"  ✓ Индекс валиден и готов к использованию")
except Exception as e:
    print(f"  ⚠ Не удалось загрузить индекс для проверки: {e}")
    print("  (Это нормально, если API ключ не настроен)")
'

    poetry run python -c "$python_code" 2>&1 || echo "  ⚠ Не удалось проверить индекс через Python"
else
    echo "  ⚠ Poetry не найден, пропускаю детальную проверку"
fi

# 5. Проверка процесса создания
echo ""
echo "Процессы создания индекса:"
if ps aux | grep -E "(update_index|python.*update)" | grep -v grep > /dev/null; then
    echo -e "  ${YELLOW}⚠ Процесс создания индекса все еще запущен${NC}"
    ps aux | grep -E "(update_index|python.*update)" | grep -v grep | head -2
else
    echo -e "  ${GREEN}✓ Процесс создания индекса не запущен (завершен)${NC}"
fi

# 6. Итоговый статус
echo ""
echo "=========================================="
if [ -f "${INDEX_PATH}/index.faiss" ] && [ -f "${INDEX_PATH}/index.pkl" ]; then
    echo -e "${GREEN}✓ ИНДЕКС СОЗДАН И ГОТОВ К ИСПОЛЬЗОВАНИЮ${NC}"
    echo ""
    echo "Следующие шаги:"
    echo "  1. Запустите сервер: poetry run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload"
    echo "  2. Или используйте автоматический скрипт: ./debug_launch.sh"
else
    echo -e "${RED}✗ ИНДЕКС НЕ СОЗДАН ИЛИ НЕПОЛНЫЙ${NC}"
    echo ""
    echo "Запустите создание индекса:"
    echo "  poetry run python update_index.py"
fi
echo "=========================================="

