"""
RAG Chain для работы с документацией MkDocs.

Этот модуль содержит функции для загрузки и обработки Markdown документов
для использования в RAG (Retrieval-Augmented Generation) системе.
"""

import hashlib
import logging
import os
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from app.prompt_config import (
    PromptSettings,
    load_prompt_settings_from_env,
    build_system_prompt
)
try:
    # Для LangChain >= 1.0
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    # Для LangChain < 1.0
    from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS

from app.markdown_utils import extract_sections, slugify
from app.openai_utils import get_embeddings_client, get_chat_llm

try:
    from langchain.retrievers import ContextualCompressionRetriever
    from langchain.retrievers.document_compressors import LLMChainExtractor
    RERANKING_AVAILABLE = True
except ImportError:
    RERANKING_AVAILABLE = False
try:
    # Для LangChain >= 1.0
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain.chains import create_retrieval_chain
except ImportError:
    # Для LangChain < 1.0 или альтернативные импорты
    try:
        # Используем альтернативный подход для LangChain 1.x
        def create_stuff_documents_chain(llm, prompt):
            def chain(inputs):
                context = "\n\n".join([doc.page_content for doc in inputs.get("context", [])])
                return llm.invoke(prompt.format_messages(context=context, input=inputs.get("input", "")))
            return chain
        
        def create_retrieval_chain(retriever, combine_docs_chain):
            def chain(inputs):
                docs = retriever.invoke(inputs.get("input", ""))
                return combine_docs_chain({"context": docs, "input": inputs.get("input", "")})
            return chain
    except ImportError:
        raise ImportError("Не удалось импортировать необходимые модули LangChain")

# Настройка логирования в начале модуля для отладки
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Загружаем переменные окружения из .env файла в начале модуля
# Это гарантирует, что все функции имеют доступ к переменным окружения
logger.info("Загрузка переменных окружения из .env файла...")
env_loaded = load_dotenv()
if env_loaded:
    logger.info("✓ Переменные окружения успешно загружены из .env")
else:
    logger.warning("⚠ Файл .env не найден или пуст")


def load_mkdocs_documents(docs_path: str = "data/mkdocs_docs") -> List[Document]:
    """
    Загружает все .md файлы из указанной директории.
    
    Использует TextLoader для сохранения исходного формата. Добавляет metadata
    с source (относительно docs/) для отслеживания источников в RAG.
    
    Args:
        docs_path: Путь к директории с Markdown документами
        
    Returns:
        Список Document объектов с загруженным контентом и metadata
    """
    # Преобразуем относительный путь в абсолютный относительно корня проекта
    project_root = Path(__file__).parent.parent
    full_docs_path = project_root / docs_path
    
    if not full_docs_path.exists():
        raise ValueError(f"Директория {full_docs_path} не существует")
    
    documents = []
    
    # Рекурсивно находим все .md файлы
    md_files = list(full_docs_path.rglob("*.md"))
    
    if not md_files:
        logger.warning(f"Не найдено .md файлов в {full_docs_path}")
        return documents
    
    logger.info(f"Найдено {len(md_files)} Markdown файлов для загрузки...")
    
    for md_file in md_files:
        try:
            # Используем TextLoader для загрузки файла
            loader = TextLoader(str(md_file), encoding='utf-8')
            loaded_docs = loader.load()
            
            # Добавляем metadata с source для каждого документа
            # source должен быть относительно корня документации (docs/...)
            # для правильного формирования URL в system prompt
            for doc in loaded_docs:
                # Вычисляем путь относительно корня документации (data/mkdocs_docs)
                # чтобы получить формат docs/.../file.md
                relative_path = md_file.relative_to(full_docs_path)
                # Нормализуем путь (заменяем обратные слеши на прямые для кроссплатформенности)
                source_path = str(relative_path).replace("\\", "/")
                doc.metadata["source"] = source_path
                # Добавляем имя файла для удобства
                doc.metadata["filename"] = md_file.name
                # Добавляем полный путь для отладки
                doc.metadata["full_path"] = str(md_file)
                # Сохраняем исходный текст для markdown-aware чанкинга
                doc.metadata["_original_text"] = doc.page_content
            
            documents.extend(loaded_docs)
            logger.debug(f"Загружен: {md_file.relative_to(project_root)}")
            
        except Exception as e:
            logger.error(f"Ошибка при загрузке {md_file}: {e}")
            continue
    
    logger.info(f"Всего загружено документов: {len(documents)}")
    return documents


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[Document]:
    """
    Разбивает документы на чанки с учетом структуры Markdown.
    
    Использует markdown-aware подход: сначала разбивает по секциям,
    затем применяет RecursiveCharacterTextSplitter внутри секций.
    Добавляет metadata о секциях и якорях.
    
    Args:
        documents: Список Document объектов для разбиения
        chunk_size: Максимальный размер чанка в символах (по умолчанию 1000)
        chunk_overlap: Количество перекрывающихся символов между чанками (по умолчанию 200)
        
    Returns:
        Список Document объектов, разбитых на чанки с сохранением metadata
    """
    if not documents:
        logger.warning("Получен пустой список документов")
        return []
    
    logger.info(f"Начинаю markdown-aware разбиение {len(documents)} документов на чанки...")
    logger.info(f"Параметры: chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
    
    all_chunks = []
    
    # Создаем RecursiveCharacterTextSplitter для разбиения внутри секций
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n## ",      # Заголовки уровня 2
            "\n\n### ",     # Заголовки уровня 3
            "\n\n",         # Параграфы
            "\n",           # Строки
            " ",            # Слова
            ""              # Символы (последний резерв)
        ],
        length_function=len,
    )
    
    for doc in documents:
        text = doc.page_content
        
        # Извлекаем секции из Markdown
        sections = extract_sections(text)
        
        if not sections:
            # Если секций нет, обрабатываем как обычно
            chunks = text_splitter.split_text(text)
            for chunk_text in chunks:
                chunk = Document(
                    page_content=chunk_text,
                    metadata=doc.metadata.copy()
                )
                all_chunks.append(chunk)
            continue
        
        # Обрабатываем каждую секцию
        for section_level, section_title, section_content in sections:
            # Разбиваем секцию на чанки
            section_chunks = text_splitter.split_text(section_content)
            
            for chunk_text in section_chunks:
                chunk = Document(
                    page_content=chunk_text,
                    metadata=doc.metadata.copy()
                )
                
                # Добавляем информацию о секции
                chunk.metadata["section_title"] = section_title
                chunk.metadata["section_level"] = section_level
                chunk.metadata["section_anchor"] = slugify(section_title)
                
                # Удаляем служебное поле
                chunk.metadata.pop("_original_text", None)
                
                all_chunks.append(chunk)
    
    logger.info(f"Всего создано чанков: {len(all_chunks)}")
    logger.info(f"Чанков с секциями: {sum(1 for c in all_chunks if 'section_title' in c.metadata)}")
    return all_chunks


def _compute_docs_hash(docs_path: str) -> str:
    """
    Вычисляет hash всех .md файлов в директории для проверки устаревания индекса.
    
    Args:
        docs_path: Путь к директории с документами
        
    Returns:
        SHA256 hash всех файлов в виде строки
    """
    project_root = Path(__file__).parent.parent
    full_docs_path = project_root / docs_path
    
    if not full_docs_path.exists():
        return ""
    
    md_files = sorted(full_docs_path.rglob("*.md"))
    hasher = hashlib.sha256()
    
    for md_file in md_files:
        try:
            with open(md_file, 'rb') as f:
                hasher.update(f.read())
                # Также добавляем путь и время модификации
                hasher.update(str(md_file.relative_to(project_root)).encode())
                hasher.update(str(md_file.stat().st_mtime).encode())
        except Exception as e:
            logger.warning(f"Не удалось прочитать файл {md_file}: {e}")
    
    return hasher.hexdigest()


def _save_index_hash(index_path: str, docs_hash: str) -> None:
    """
    Сохраняет hash документов в файл рядом с индексом.
    
    Args:
        index_path: Путь к директории с индексом
        docs_hash: Hash документов для сохранения
    """
    project_root = Path(__file__).parent.parent
    full_index_path = project_root / index_path
    hash_file = full_index_path / ".docs_hash"
    
    try:
        hash_file.parent.mkdir(parents=True, exist_ok=True)
        hash_file.write_text(docs_hash)
    except Exception as e:
        logger.warning(f"Не удалось сохранить hash индекса: {e}")


def _load_index_hash(index_path: str) -> Optional[str]:
    """
    Загружает сохраненный hash документов.
    
    Args:
        index_path: Путь к директории с индексом
        
    Returns:
        Hash документов или None если файл не найден
    """
    project_root = Path(__file__).parent.parent
    full_index_path = project_root / index_path
    hash_file = full_index_path / ".docs_hash"
    
    if hash_file.exists():
        try:
            return hash_file.read_text().strip()
        except Exception as e:
            logger.warning(f"Не удалось прочитать hash индекса: {e}")
    
    return None


def build_or_load_vectorstore(
    chunks: Optional[List[Document]] = None,
    index_path: str = "vectorstore/faiss_index",
    docs_path: str = "data/mkdocs_docs",
    force_rebuild: bool = False
):
    """
    Создает или загружает векторное хранилище FAISS.
    
    FAISS выбран для локального хранения: бесплатный, быстрый, не требует
    внешних сервисов. Если индекс не существует и chunks=None, автоматически
    загружает и чанкирует документы. Проверяет устаревание по hash документов.
    
    Args:
        chunks: Список Document объектов для создания индекса.
                Если None и индекс не существует, автоматически загружает и чанкирует.
        index_path: Путь к директории с FAISS индексом
        docs_path: Путь к директории с исходными документами
        force_rebuild: Если True, пересоздает индекс даже если он существует
        
    Returns:
        FAISS векторное хранилище, готовое к использованию для поиска
    """
    # Загружаем переменные окружения из .env файла
    logger.info("Загрузка переменных окружения для vectorstore...")
    env_loaded = load_dotenv()
    if env_loaded:
        logger.debug("✓ .env файл найден и загружен")
    else:
        logger.warning("⚠ .env файл не найден, используем системные переменные окружения")
    
    # Получаем API ключ из переменных окружения
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY не найден в переменных окружения")
        raise ValueError(
            "OPENAI_API_KEY не найден в переменных окружения. "
            "Создайте файл .env с OPENAI_API_KEY=your-key"
        )
    logger.info("✓ OPENAI_API_KEY найден в переменных окружения")
    
    # Определяем режим разработки (для allow_dangerous_deserialization)
    # По умолчанию production (безопаснее)
    env = os.getenv("ENV", "production").lower()
    is_dev = env == "development"
    
    logger.info("=" * 60)
    logger.info("ВЕКТОРНОЕ ХРАНИЛИЩЕ FAISS")
    logger.info("=" * 60)
    
    # Определяем абсолютный путь к индексу
    project_root = Path(__file__).parent.parent
    full_index_path = project_root / index_path
    
    # Проверяем существование индекса
    index_exists = full_index_path.exists() and any(full_index_path.iterdir())
    
    # Проверяем устаревание индекса по hash документов
    index_stale = False
    if index_exists and not force_rebuild:
        current_hash = _compute_docs_hash(docs_path)
        saved_hash = _load_index_hash(index_path)
        
        if current_hash and saved_hash:
            if current_hash != saved_hash:
                logger.info("Индекс устарел: документы были изменены")
                index_stale = True
        elif current_hash and not saved_hash:
            # Hash не был сохранен ранее, считаем индекс устаревшим
            logger.info("Hash индекса не найден, пересоздаю индекс")
            index_stale = True
    
    if index_exists and not force_rebuild and not index_stale:
        logger.info(f"Найден существующий индекс в {index_path}")
        logger.info("Загружаю индекс из файловой системы...")
        
        try:
            # Инициализируем embeddings (нужны для загрузки индекса)
            embeddings = get_embeddings_client()
            
            # Загружаем существующий индекс
            # allow_dangerous_deserialization только в dev режиме
            vectorstore = FAISS.load_local(
                str(full_index_path),
                embeddings,
                allow_dangerous_deserialization=is_dev
            )
            
            logger.info("Индекс успешно загружен")
            logger.info(f"Количество документов в индексе: {vectorstore.index.ntotal}")
            logger.info("=" * 60)
            
            return vectorstore
            
        except Exception as e:
            logger.warning(f"Ошибка при загрузке индекса: {e}")
            logger.info("Будет создан новый индекс...")
            index_exists = False
    
    # Создаем новый индекс
    if not index_exists or force_rebuild or index_stale:
        # Если чанки не предоставлены, автоматически загружаем и чанкируем документы
        if chunks is None or len(chunks) == 0:
            logger.info("Чанки не предоставлены, автоматически загружаю документы...")
            documents = load_mkdocs_documents(docs_path)
            if not documents:
                raise ValueError(
                    f"Не найдено документов в {docs_path}. "
                    "Убедитесь, что директория содержит .md файлы."
                )
            chunks = chunk_documents(documents)
            logger.info(f"Автоматически загружено и разбито на {len(chunks)} чанков")
        
        if force_rebuild:
            logger.info("Режим force_rebuild: пересоздаю индекс...")
        elif index_stale:
            logger.info("Индекс устарел: пересоздаю индекс...")
        else:
            logger.info(f"Индекс не найден в {index_path}")
        
        logger.info(f"Создаю новый индекс из {len(chunks)} чанков...")
        
        # Инициализируем OpenAI Embeddings с увеличенным таймаутом для batch операций
        logger.info("Инициализирую OpenAI Embeddings (text-embedding-3-small)...")
        from app.openai_utils import OPENAI_BATCH_TIMEOUT
        embeddings = get_embeddings_client(timeout=OPENAI_BATCH_TIMEOUT)
        logger.info(f"Использую таймаут {OPENAI_BATCH_TIMEOUT}с для batch операций создания embeddings")
        
        # Создаем FAISS векторное хранилище из чанков
        logger.info("Генерирую embeddings и создаю индекс...")
        logger.info("(Это может занять несколько минут для большого количества чанков)")
        
        vectorstore = FAISS.from_documents(
            documents=chunks,
            embedding=embeddings
        )
        
        # Создаем директорию для индекса, если её нет
        full_index_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Сохраняем индекс на диск
        logger.info(f"Сохраняю индекс в {index_path}...")
        vectorstore.save_local(str(full_index_path))
        
        # Сохраняем hash документов для проверки устаревания
        docs_hash = _compute_docs_hash(docs_path)
        if docs_hash:
            _save_index_hash(index_path, docs_hash)
        
        logger.info("Индекс успешно создан и сохранен")
        logger.info(f"Количество документов в индексе: {vectorstore.index.ntotal}")
        logger.info(f"Размерность векторов: {vectorstore.index.d}")
        logger.info("=" * 60)
        
        return vectorstore


def build_rag_chain(
    vectorstore,
    prompt_settings: Optional[PromptSettings] = None,
    k: Optional[int] = None,
    model: str = "gpt-4o-mini",
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None
):
    """
    Создает RAG цепочку из готового vectorstore.
    
    Использует настройки из PromptSettings для system prompt и параметров LLM.
    
    Args:
        vectorstore: Готовый FAISS vectorstore
        prompt_settings: Настройки промпта (если None, загружаются из окружения)
        k: Количество релевантных чанков (если None, используется из prompt_settings)
        model: Модель OpenAI (по умолчанию "gpt-4o-mini")
        temperature: Температура генерации (если None, используется из prompt_settings)
        max_tokens: Максимальное количество токенов (опционально)
        
    Returns:
        RAG цепочка для ответов на вопросы
    """
    # Загружаем настройки, если не переданы
    if prompt_settings is None:
        prompt_settings = load_prompt_settings_from_env()
    
    # Определяем эффективные значения
    effective_k = k if k is not None else prompt_settings.default_top_k
    effective_temperature = temperature if temperature is not None else prompt_settings.default_temperature
    
    # Ограничиваем диапазоны
    effective_k = max(1, min(10, effective_k))
    effective_temperature = max(0.0, min(1.0, effective_temperature))
    
    # Создаем базовый retriever с увеличенным k для reranking
    raw_k = max(effective_k * 2, 8)
    logger.info(f"Создаю базовый retriever с k={raw_k} для reranking...")
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": raw_k})
    
    # Применяем reranking через ContextualCompressionRetriever
    logger.info(f"Инициализирую LLM для reranking: {model} (temperature={effective_temperature})...")
    llm = get_chat_llm(temperature=effective_temperature, model=model, max_tokens=max_tokens)
    
    # Используем LLMChainExtractor для фильтрации менее релевантных чанков
    if RERANKING_AVAILABLE:
        try:
            compressor = LLMChainExtractor.from_llm(llm)
            retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=base_retriever
            )
            logger.info(f"Reranking включен, финальный k={effective_k}")
        except Exception as e:
            logger.warning(f"Ошибка при создании reranker: {e}, используем базовый retriever")
            retriever = vectorstore.as_retriever(search_kwargs={"k": effective_k})
    else:
        # Fallback если ContextualCompressionRetriever недоступен
        logger.info("Reranking недоступен, используем базовый retriever")
        retriever = vectorstore.as_retriever(search_kwargs={"k": effective_k})
    
    # Собираем system prompt из настроек
    system_prompt = build_system_prompt(prompt_settings)
    
    logger.info("Создаю prompt template...")
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Контекст из документации:\n\n{context}\n\nВопрос: {input}")
    ])
    
    logger.info("Создаю Stuff Documents Chain...")
    document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    
    logger.info("Создаю Retrieval Chain...")
    rag_chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=document_chain
    )
    
    logger.info("RAG цепочка успешно создана")
    return rag_chain


def get_rag_chain(
    index_path: str = "vectorstore/faiss_index",
    k: Optional[int] = None,
    model: str = "gpt-4o-mini",
    temperature: Optional[float] = None
):
    """
    Создает RAG цепочку, загружая vectorstore и строя chain.
    
    Helper функция для обратной совместимости. Использует build_rag_chain().
    
    Args:
        index_path: Путь к директории с FAISS индексом
        k: Количество релевантных чанков (если None, используется из настроек)
        model: Модель OpenAI (по умолчанию "gpt-4o-mini")
        temperature: Температура генерации (если None, используется из настроек)
        
    Returns:
        RAG цепочка для ответов на вопросы
    """
    logger.info("=" * 60)
    logger.info("ИНИЦИАЛИЗАЦИЯ RAG ЦЕПОЧКИ")
    logger.info("=" * 60)
    
    logger.info("Загружаю векторное хранилище...")
    vectorstore = build_or_load_vectorstore(chunks=None, index_path=index_path)
    
    rag_chain = build_rag_chain(vectorstore, k=k, model=model, temperature=temperature)
    
    logger.info("=" * 60)
    return rag_chain


def build_rag_chain_and_settings(
    index_path: str = "vectorstore/faiss_index"
):
    """
    Создает RAG цепочку и возвращает настройки промпта.
    
    Используется при инициализации приложения для сохранения настроек в app.state.
    
    Args:
        index_path: Путь к директории с FAISS индексом
        
    Returns:
        Кортеж (rag_chain, vectorstore, prompt_settings)
    """
    logger.info("=" * 60)
    logger.info("ИНИЦИАЛИЗАЦИЯ RAG ЦЕПОЧКИ С НАСТРОЙКАМИ")
    logger.info("=" * 60)
    
    logger.info("Загружаю векторное хранилище...")
    vectorstore = build_or_load_vectorstore(chunks=None, index_path=index_path)
    
    logger.info("Загружаю настройки промпта...")
    prompt_settings = load_prompt_settings_from_env()
    
    logger.info(f"Настройки: language={prompt_settings.language}, mode={prompt_settings.mode}, "
                f"temperature={prompt_settings.default_temperature}, top_k={prompt_settings.default_top_k}")
    
    rag_chain = build_rag_chain(
        vectorstore,
        prompt_settings=prompt_settings,
        k=prompt_settings.default_top_k,
        temperature=prompt_settings.default_temperature
    )
    
    logger.info("=" * 60)
    return rag_chain, vectorstore, prompt_settings

