"""
Опциональный скрипт для обновления векторного индекса.

Использование:
    python update_index.py

Этот скрипт:
1. Загружает все .md файлы из data/mkdocs_docs
2. Разбивает их на чанки
3. Пересоздает векторный индекс FAISS
4. Сохраняет индекс в vectorstore/faiss_index

Альтернатива: использовать endpoint /update_index через API.
"""

import logging
import sys
from pathlib import Path

# Добавляем корневую директорию проекта в путь
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
from app.core.rag_chain import build_or_load_vectorstore, chunk_documents, load_mkdocs_documents

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Основная функция для обновления индекса."""
    # Загружаем переменные окружения
    load_dotenv()
    
    logger.info("=" * 60)
    logger.info("ОБНОВЛЕНИЕ ВЕКТОРНОГО ИНДЕКСА")
    logger.info("=" * 60)
    
    try:
        # Загружаем документы
        logger.info("Загрузка документов из data/mkdocs_docs...")
        documents = load_mkdocs_documents()
        
        if not documents:
            logger.error("Не найдено документов для индексации")
            logger.error("Убедитесь, что в data/mkdocs_docs есть .md файлы")
            sys.exit(1)
        
        logger.info(f"Загружено {len(documents)} документов")
        
        # Разбиваем на чанки
        logger.info("Разбиение документов на чанки...")
        chunks = chunk_documents(documents)
        logger.info(f"Создано {len(chunks)} чанков")
        
        # Пересоздаем индекс
        logger.info("Создание векторного индекса...")
        vectorstore = build_or_load_vectorstore(
            chunks=chunks,
            force_rebuild=True
        )
        
        logger.info("=" * 60)
        logger.info("Индекс успешно обновлен!")
        logger.info(f"Количество документов в индексе: {vectorstore.index.ntotal}")
        logger.info(f"Размерность векторов: {vectorstore.index.d}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Ошибка при обновлении индекса: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

