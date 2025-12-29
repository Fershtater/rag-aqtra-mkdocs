"""
Indexing module: document loading, chunking, and vectorstore management.
"""

import hashlib
import logging
import os
import uuid
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

from app.core.markdown_utils import extract_sections, slugify
from app.infra.openai_utils import get_embeddings_client, OPENAI_BATCH_TIMEOUT
from app.rag.index_meta import generate_index_version, save_index_meta, get_index_version
from app.rag.index_lock import IndexLock

logger = logging.getLogger(__name__)

# Chunking parameters with configuration support via environment variables
DEFAULT_CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1500"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "300"))
DEFAULT_MIN_CHUNK_SIZE = int(os.getenv("MIN_CHUNK_SIZE", "200"))


def get_vectorstore_dir() -> str:
    """
    Get vectorstore directory from environment variable or default.
    
    Returns:
        Path to vectorstore directory (default: var/vectorstore/faiss_index)
    """
    return os.getenv("VECTORSTORE_DIR", "var/vectorstore/faiss_index")


def load_mkdocs_documents(docs_path: str = "data/mkdocs_docs") -> List[Document]:
    """
    Loads all .md files from the specified directory.
    
    Uses TextLoader to preserve original format. Adds metadata
    with source (relative to docs/) for tracking sources in RAG.
    
    Args:
        docs_path: Path to directory with Markdown documents
        
    Returns:
        List of Document objects with loaded content and metadata
    """
    # Convert relative path to absolute path relative to project root
    project_root = Path(__file__).parent.parent.parent
    full_docs_path = project_root / docs_path
    
    if not full_docs_path.exists():
        raise ValueError(f"Directory {full_docs_path} does not exist")
    
    documents = []
    
    # Recursively find all .md files
    md_files = list(full_docs_path.rglob("*.md"))
    
    if not md_files:
        logger.warning(f"No .md files found in {full_docs_path}")
        return documents
    
    logger.info(f"Found {len(md_files)} Markdown files to load...")
    
    for md_file in md_files:
        try:
            loader = TextLoader(str(md_file), encoding='utf-8')
            loaded_docs = loader.load()
            
            for doc in loaded_docs:
                # Calculate relative path to docs_root (safe fallback if not in subpath)
                try:
                    relative_path = md_file.relative_to(full_docs_path)
                except ValueError:
                    # If file is not in docs_path subpath, use filename only
                    relative_path = Path(md_file.name)
                source_path = str(relative_path).replace("\\", "/")
                doc.metadata["source"] = source_path
                doc.metadata["filename"] = md_file.name
                doc.metadata["full_path"] = str(md_file)
                doc.metadata["_original_text"] = doc.page_content
            
            documents.extend(loaded_docs)
            # Safe logging: use relative to docs_path or filename
            try:
                log_path = md_file.relative_to(full_docs_path)
            except ValueError:
                log_path = Path(md_file.name)
            logger.debug(f"Loaded: {log_path}")
            
        except Exception as e:
            logger.error(f"Error loading {md_file}: {e}")
            continue
    
    logger.info(f"Total documents loaded: {len(documents)}")
    return documents


def chunk_documents(
    documents: List[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    min_chunk_size: int = DEFAULT_MIN_CHUNK_SIZE,
) -> List[Document]:
    """
    Splits documents into chunks considering Markdown structure.
    
    Uses markdown-aware approach: first splits by sections,
    then applies RecursiveCharacterTextSplitter within sections.
    Adds metadata about sections and anchors.
    
    Args:
        documents: List of Document objects to split
        chunk_size: Maximum chunk size in characters
        chunk_overlap: Number of overlapping characters between chunks
        min_chunk_size: Minimum allowed chunk size; smaller chunks are discarded
        
    Returns:
        List of Document objects split into chunks with preserved metadata
    """
    if not documents:
        logger.warning("Received empty document list")
        return []
    
    logger.info(f"Starting markdown-aware splitting of {len(documents)} documents into chunks...")
    logger.info(
        "Chunking parameters: chunk_size=%s, chunk_overlap=%s, min_chunk_size=%s",
        chunk_size,
        chunk_overlap,
        min_chunk_size,
    )
    
    all_chunks = []
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n```",
            "\n\n## ",
            "\n\n### ",
            "\n\n#### ",
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
        length_function=len,
        keep_separator=True,
    )
    
    for doc in documents:
        text = doc.page_content
        sections = extract_sections(text)
        
        if not sections:
            chunks = text_splitter.split_text(text)
            for chunk_text in chunks:
                if len(chunk_text) < min_chunk_size:
                    continue
                chunk = Document(
                    page_content=chunk_text,
                    metadata=doc.metadata.copy()
                )
                chunk.metadata.pop("_original_text", None)
                all_chunks.append(chunk)
            continue
        
        for section_level, section_title, section_content in sections:
            section_chunks = text_splitter.split_text(section_content)
            
            for chunk_text in section_chunks:
                if len(chunk_text) < min_chunk_size:
                    continue
                chunk = Document(
                    page_content=chunk_text,
                    metadata=doc.metadata.copy()
                )
                
                chunk.metadata["section_title"] = section_title
                chunk.metadata["section_level"] = section_level
                chunk.metadata["section_anchor"] = slugify(section_title)
                chunk.metadata.pop("_original_text", None)
                
                all_chunks.append(chunk)
    
    logger.info(f"Total chunks created: {len(all_chunks)}")
    logger.info(f"Chunks with sections: {sum(1 for c in all_chunks if 'section_title' in c.metadata)}")
    return all_chunks


def _compute_docs_hash(docs_path: str) -> str:
    """
    Computes hash of all .md files in directory to check index staleness.
    
    Args:
        docs_path: Path to directory with documents
        
    Returns:
        SHA256 hash of all files as string
    """
    project_root = Path(__file__).parent.parent.parent
    # Resolve docs_path: if relative, resolve against project_root; if absolute, use as-is
    docs_path_resolved = Path(docs_path)
    if not docs_path_resolved.is_absolute():
        full_docs_path = (project_root / docs_path).resolve()
    else:
        full_docs_path = docs_path_resolved.resolve()
    
    if not full_docs_path.exists():
        return ""
    
    md_files = sorted(full_docs_path.rglob("*.md"))
    hasher = hashlib.sha256()
    
    for md_file in md_files:
        try:
            # Read file content for hashing
            with open(md_file, 'rb') as f:
                content = f.read()
                hasher.update(content)
            
            # Use deterministic path: relative to docs_path if possible, otherwise filename only
            # This ensures hash is stable across different tmp_path locations
            try:
                rel_path = md_file.relative_to(full_docs_path)
                path_key = str(rel_path).replace("\\", "/")  # Normalize path separators
            except ValueError:
                # If not in subpath, use filename only (deterministic)
                path_key = md_file.name
            
            hasher.update(path_key.encode('utf-8'))
            # Include modification time for change detection
            hasher.update(str(md_file.stat().st_mtime).encode())
        except Exception as e:
            logger.warning(f"Error hashing {md_file}: {e}")
            continue
    
    return hasher.hexdigest()


def _save_index_hash(index_path: str, docs_hash: str) -> None:
    """
    Saves document hash to file for staleness checking.
    
    Args:
        index_path: Path to index directory (can be absolute or relative)
        docs_hash: SHA256 hash of documents
    """
    index_path_resolved = Path(index_path)
    if not index_path_resolved.is_absolute():
        project_root = Path(__file__).parent.parent.parent
        full_index_path = (project_root / index_path).resolve()
    else:
        full_index_path = index_path_resolved.resolve()
    hash_file = full_index_path / ".docs_hash"
    
    try:
        full_index_path.mkdir(parents=True, exist_ok=True)
        with open(hash_file, 'w') as f:
            f.write(docs_hash)
        logger.debug(f"Saved document hash to {hash_file}")
    except Exception as e:
        logger.warning(f"Error saving hash: {e}")


def _load_index_hash(index_path: str) -> Optional[str]:
    """
    Loads document hash from file.
    
    Args:
        index_path: Path to index directory (can be absolute or relative)
        
    Returns:
        SHA256 hash string or None if not found
    """
    index_path_resolved = Path(index_path)
    if not index_path_resolved.is_absolute():
        project_root = Path(__file__).parent.parent.parent
        full_index_path = (project_root / index_path).resolve()
    else:
        full_index_path = index_path_resolved.resolve()
    hash_file = full_index_path / ".docs_hash"
    
    if not hash_file.exists():
        return None
    
    try:
        with open(hash_file, 'r') as f:
            return f.read().strip()
    except Exception as e:
        logger.warning(f"Error loading hash: {e}")
        return None


def build_or_load_vectorstore(
    chunks: Optional[List[Document]] = None,
    index_path: Optional[str] = None,
    docs_path: str = "data/mkdocs_docs",
    force_rebuild: bool = False,
    lock_timeout_seconds: int = 300
):
    """
    Creates or loads FAISS vector store.
    
    FAISS chosen for local storage: free, fast, no external services required.
    If index doesn't exist and chunks=None, automatically loads and chunks documents.
    Checks staleness by document hash.
    
    Args:
        chunks: Optional list of Document chunks to index (if None, loads existing or auto-loads)
        index_path: Optional path to index directory (if None, uses VECTORSTORE_DIR env or default var/vectorstore/faiss_index)
        docs_path: Path to directory with source documents
        force_rebuild: If True, recreates index even if it exists
        
    Returns:
        FAISS vector store ready for search
    """
    logger.info("Loading environment variables for vectorstore...")
    env_loaded = load_dotenv()
    if env_loaded:
        logger.debug("✓ .env file found and loaded")
    else:
        logger.warning("⚠ .env file not found, using system environment variables")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY not found in environment variables")
        raise ValueError(
            "OPENAI_API_KEY not found in environment variables. "
            "Create .env file with OPENAI_API_KEY=your-key"
        )
    logger.info("✓ OPENAI_API_KEY found in environment variables")
    
    env = os.getenv("ENV", "production").lower()
    is_dev = env == "development"
    
    logger.info("=" * 60)
    logger.info("FAISS VECTOR STORE")
    logger.info("=" * 60)
    
    # Use configured index path or default
    if index_path is None:
        index_path = get_vectorstore_dir()
    
    # Resolve index_path: if relative, resolve against project_root; if absolute, use as-is
    index_path_resolved = Path(index_path)
    if not index_path_resolved.is_absolute():
        project_root = Path(__file__).parent.parent.parent
        full_index_path = (project_root / index_path).resolve()
    else:
        full_index_path = index_path_resolved.resolve()
    
    index_exists = full_index_path.exists() and any(full_index_path.iterdir())
    
    index_stale = False
    if index_exists and not force_rebuild:
        current_hash = _compute_docs_hash(docs_path)
        saved_hash = _load_index_hash(index_path)
        
        if current_hash and saved_hash:
            if current_hash != saved_hash:
                logger.info("Index is stale: documents were changed")
                index_stale = True
        elif current_hash and not saved_hash:
            logger.info("Index hash not found, recreating index")
            index_stale = True
    
    if index_exists and not force_rebuild and not index_stale:
        logger.info(f"Found existing index in {index_path}")
        logger.info("Loading index from filesystem...")
        
        try:
            embeddings = get_embeddings_client()
            
            vectorstore = FAISS.load_local(
                str(full_index_path),
                embeddings,
                allow_dangerous_deserialization=is_dev
            )
            
            # Load and log index version
            index_version = get_index_version(index_path)
            if index_version:
                logger.info(f"Index version: {index_version}")
            
            logger.info("Index successfully loaded")
            logger.info(f"Number of documents in index: {vectorstore.index.ntotal}")
            logger.info("=" * 60)
            
            return vectorstore
            
        except Exception as e:
            logger.warning(f"Error loading index: {e}")
            logger.info("Will create new index...")
            index_exists = False
    
    if not index_exists or force_rebuild or index_stale:
        # Acquire lock for atomic rebuild
        lock = IndexLock(index_path, timeout_seconds=lock_timeout_seconds)
        
        try:
            if not lock.acquire():
                lock_info = lock._get_lock_info()
                lock_file_path = str(lock.lock_file)
                age_info = f"{lock_info.get('age_seconds', 'unknown')}s" if lock_info else "unknown"
                pid_info = lock_info.get('pid', 'unknown') if lock_info else 'unknown'
                
                logger.warning(
                    f"Index rebuild lock acquisition failed. "
                    f"Lock file: {lock_file_path}, Age: {age_info}, PID: {pid_info}"
                )
                
                # Raise RuntimeError for lock failure (will be caught by caller)
                raise RuntimeError(
                    f"Index rebuild is already in progress. Lock file: {lock_file_path}, "
                    f"Age: {age_info}, PID: {pid_info}. Try again later."
                )
            
            # Double-check index existence after acquiring lock
            # (another process might have created it)
            if not force_rebuild and not index_stale:
                index_exists_after_lock = full_index_path.exists() and any(full_index_path.iterdir())
                if index_exists_after_lock:
                    # Check staleness again
                    current_hash = _compute_docs_hash(docs_path)
                    saved_hash = _load_index_hash(index_path)
                    if current_hash and saved_hash and current_hash == saved_hash:
                        logger.info("Index was created by another process, loading it...")
                        lock.release()
                        embeddings = get_embeddings_client()
                        vectorstore = FAISS.load_local(
                            str(full_index_path),
                            embeddings,
                            allow_dangerous_deserialization=is_dev
                        )
                        index_version = get_index_version(index_path)
                        if index_version:
                            logger.info(f"Index version: {index_version}")
                        return vectorstore
            
            if chunks is None or len(chunks) == 0:
                logger.info("Chunks not provided, automatically loading documents...")
                documents = load_mkdocs_documents(docs_path)
                if not documents:
                    raise ValueError(
                        f"No documents found in {docs_path}. "
                        "Make sure directory contains .md files."
                    )
                chunks = chunk_documents(documents)
                logger.info(f"Automatically loaded and split into {len(chunks)} chunks")
            
            if force_rebuild:
                logger.info("force_rebuild mode: recreating index...")
            elif index_stale:
                logger.info("Index is stale: recreating index...")
            else:
                logger.info(f"Index not found in {index_path}")
            
            # Generate index version
            index_version = generate_index_version()
            logger.info(f"Building index version: {index_version}")
            
            # Create temporary directory for atomic rebuild
            # Use parent directory of index_path for tmp (to keep them together)
            index_path_resolved = Path(index_path)
            if index_path_resolved.is_absolute():
                tmp_index_path = str(index_path_resolved.parent / f"{index_path_resolved.name}.tmp-{uuid.uuid4().hex[:8]}")
            else:
                project_root = Path(__file__).parent.parent.parent
                tmp_index_path = f"{index_path}.tmp-{uuid.uuid4().hex[:8]}"
            tmp_index_path_resolved = Path(tmp_index_path)
            if tmp_index_path_resolved.is_absolute():
                tmp_full_index_path = tmp_index_path_resolved
            else:
                project_root = Path(__file__).parent.parent.parent
                tmp_full_index_path = (project_root / tmp_index_path).resolve()
            
            logger.info("Creating temporary index in %s...", tmp_index_path)
            
            logger.info(f"Creating new index from {len(chunks)} chunks...")
            
            logger.info("Initializing OpenAI Embeddings (text-embedding-3-small)...")
            embeddings = get_embeddings_client(timeout=OPENAI_BATCH_TIMEOUT)
            logger.info(f"Using timeout {OPENAI_BATCH_TIMEOUT}s for batch embedding operations")
            
            logger.info("Generating embeddings and creating index...")
            logger.info("(This may take several minutes for large number of chunks)")
            
            vectorstore = FAISS.from_documents(
                documents=chunks,
                embedding=embeddings
            )
            
            tmp_full_index_path.parent.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Saving index to temporary location {tmp_index_path}...")
            vectorstore.save_local(str(tmp_full_index_path))
            
            # Save metadata
            docs_hash = _compute_docs_hash(docs_path)
            save_index_meta(
                tmp_index_path,
                index_version=index_version,
                docs_hash=docs_hash or "",
                docs_path=docs_path,
                embedding_model="text-embedding-3-small",
                chunk_size=DEFAULT_CHUNK_SIZE,
                chunk_overlap=DEFAULT_CHUNK_OVERLAP,
                chunks_count=len(chunks)
            )
            
            if docs_hash:
                _save_index_hash(tmp_index_path, docs_hash)
            
            # Atomic swap: tmp -> index
            logger.info("Performing atomic swap...")
            
            # Remove old backup if exists (keep only one backup)
            if full_index_path.is_absolute():
                backup_path = full_index_path.parent / f"{full_index_path.name}.bak"
            else:
                project_root = Path(__file__).parent.parent.parent
                backup_path = project_root / f"{index_path}.bak"
            if backup_path.exists():
                import shutil
                try:
                    shutil.rmtree(backup_path)
                    logger.debug(f"Removed old backup: {backup_path}")
                except Exception as e:
                    logger.warning(f"Failed to remove old backup: {e}")
            
            # Backup existing index if it exists
            if full_index_path.exists():
                try:
                    full_index_path.rename(backup_path)
                    logger.info(f"Backed up existing index to {backup_path}")
                except Exception as e:
                    logger.warning(f"Failed to backup existing index: {e}")
            
            # Atomic swap: tmp -> index
            try:
                tmp_full_index_path.rename(full_index_path)
                logger.info(f"Index successfully swapped: {tmp_index_path} -> {index_path}")
            except Exception as e:
                logger.error(f"Failed to swap index: {e}")
                # Try to restore backup
                if backup_path.exists():
                    try:
                        backup_path.rename(full_index_path)
                        logger.info("Restored backup index")
                    except Exception as restore_error:
                        logger.error("Failed to restore backup: %s", restore_error)
                raise
            
            logger.info("Index successfully created and saved")
            logger.info(f"Index version: {index_version}")
            logger.info(f"Number of documents in index: {vectorstore.index.ntotal}")
            logger.info(f"Vector dimension: {vectorstore.index.d}")
            logger.info("=" * 60)
            
            return vectorstore
            
        finally:
            lock.release()

