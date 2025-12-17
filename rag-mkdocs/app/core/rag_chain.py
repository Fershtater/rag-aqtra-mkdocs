"""
RAG Chain for working with MkDocs documentation.

This module contains functions for loading and processing Markdown documents
for use in a RAG (Retrieval-Augmented Generation) system.
"""

import hashlib
import logging
import os
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from app.core.prompt_config import (
    PromptSettings,
    load_prompt_settings_from_env,
    build_system_prompt,
    detect_response_language,
)
try:
    # For LangChain >= 1.0
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    # For LangChain < 1.0
    from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS

from app.core.markdown_utils import extract_sections, slugify
from app.infra.openai_utils import get_embeddings_client, get_chat_llm

try:
    from langchain.retrievers import ContextualCompressionRetriever
    from langchain.retrievers.document_compressors import LLMChainExtractor

    RERANKING_AVAILABLE = True
except ImportError:
    RERANKING_AVAILABLE = False
try:
    # For LangChain >= 1.0
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain.chains import create_retrieval_chain
except ImportError:
    # For LangChain < 1.0 or alternative imports
    try:
        # Use alternative approach for LangChain 1.x
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
        raise ImportError("Failed to import required LangChain modules")

# Configure logging at the beginning of the module for debugging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file at the beginning of the module
# This ensures all functions have access to environment variables
logger.info("Loading environment variables from .env file...")
env_loaded = load_dotenv()
if env_loaded:
    logger.info("✓ Environment variables successfully loaded from .env")
else:
    logger.warning("⚠ .env file not found or empty")

# Chunking parameters with configuration support via environment variables.
# Balanced defaults optimized for technical documentation.
DEFAULT_CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1500"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "300"))
DEFAULT_MIN_CHUNK_SIZE = int(os.getenv("MIN_CHUNK_SIZE", "200"))

# LLM-based reranking control.
RERANKING_ENABLED = os.getenv("RERANKING_ENABLED", "0").lower() in ("1", "true", "yes")


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
    # __file__ is in app/core/rag_chain.py, so need to go up 3 levels
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
            # Use TextLoader to load file
            loader = TextLoader(str(md_file), encoding='utf-8')
            loaded_docs = loader.load()
            
            # Add metadata with source for each document
            # source should be relative to documentation root (docs/...)
            # for proper URL formation in system prompt
            for doc in loaded_docs:
                # Calculate path relative to documentation root (data/mkdocs_docs)
                # to get format docs/.../file.md
                relative_path = md_file.relative_to(full_docs_path)
                # Normalize path (replace backslashes with forward slashes for cross-platform compatibility)
                source_path = str(relative_path).replace("\\", "/")
                doc.metadata["source"] = source_path
                # Add filename for convenience
                doc.metadata["filename"] = md_file.name
                # Add full path for debugging
                doc.metadata["full_path"] = str(md_file)
                # Save original text for markdown-aware chunking
                doc.metadata["_original_text"] = doc.page_content
            
            documents.extend(loaded_docs)
            logger.debug(f"Loaded: {md_file.relative_to(project_root)}")
            
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
    
    Chunking parameters can be configured via environment variables:
    - CHUNK_SIZE (default 1500)
    - CHUNK_OVERLAP (default 300)
    - MIN_CHUNK_SIZE (default 200)
    
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
    
    # Create RecursiveCharacterTextSplitter for splitting within sections.
    # List of separators is chosen to maximize structure preservation:
    # - first code blocks and large headers,
    # - then paragraphs, lines and sentences.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n```",      # Code blocks
            "\n\n## ",      # Level 2 headers
            "\n\n### ",     # Level 3 headers
            "\n\n#### ",    # Level 4 headers
            "\n\n",         # Paragraphs
            "\n",           # Lines
            ". ",           # Sentences
            " ",            # Words
            "",             # Characters (last resort)
        ],
        length_function=len,
        keep_separator=True,
    )
    
    for doc in documents:
        text = doc.page_content
        
        # Extract sections from Markdown
        sections = extract_sections(text)
        
        if not sections:
            # If no sections, process as usual
            chunks = text_splitter.split_text(text)
            for chunk_text in chunks:
                if len(chunk_text) < min_chunk_size:
                    continue
                chunk = Document(
                    page_content=chunk_text,
                    metadata=doc.metadata.copy()
                )
                # Remove service field if it remained after loading
                chunk.metadata.pop("_original_text", None)
                all_chunks.append(chunk)
            continue
        
        # Process each section
        for section_level, section_title, section_content in sections:
            # Split section into chunks
            section_chunks = text_splitter.split_text(section_content)
            
            for chunk_text in section_chunks:
                if len(chunk_text) < min_chunk_size:
                    continue
                chunk = Document(
                    page_content=chunk_text,
                    metadata=doc.metadata.copy()
                )
                
                # Add section information
                chunk.metadata["section_title"] = section_title
                chunk.metadata["section_level"] = section_level
                chunk.metadata["section_anchor"] = slugify(section_title)
                
                # Remove service field
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
    full_docs_path = project_root / docs_path
    
    if not full_docs_path.exists():
        return ""
    
    md_files = sorted(full_docs_path.rglob("*.md"))
    hasher = hashlib.sha256()
    
    for md_file in md_files:
        try:
            with open(md_file, 'rb') as f:
                hasher.update(f.read())
                # Also add path and modification time
                hasher.update(str(md_file.relative_to(project_root)).encode())
                hasher.update(str(md_file.stat().st_mtime).encode())
        except Exception as e:
            logger.warning(f"Failed to read file {md_file}: {e}")
    
    return hasher.hexdigest()


def _save_index_hash(index_path: str, docs_hash: str) -> None:
    """
    Saves document hash to file next to index.
    
    Args:
        index_path: Path to directory with index
        docs_hash: Document hash to save
    """
    project_root = Path(__file__).parent.parent.parent
    full_index_path = project_root / index_path
    hash_file = full_index_path / ".docs_hash"
    
    try:
        hash_file.parent.mkdir(parents=True, exist_ok=True)
        hash_file.write_text(docs_hash)
    except Exception as e:
        logger.warning(f"Failed to save index hash: {e}")


def _load_index_hash(index_path: str) -> Optional[str]:
    """
    Loads saved document hash.
    
    Args:
        index_path: Path to directory with index
        
    Returns:
        Document hash or None if file not found
    """
    project_root = Path(__file__).parent.parent.parent
    full_index_path = project_root / index_path
    hash_file = full_index_path / ".docs_hash"
    
    if hash_file.exists():
        try:
            return hash_file.read_text().strip()
        except Exception as e:
            logger.warning(f"Failed to read index hash: {e}")
    
    return None


def build_or_load_vectorstore(
    chunks: Optional[List[Document]] = None,
    index_path: str = "vectorstore/faiss_index",
    docs_path: str = "data/mkdocs_docs",
    force_rebuild: bool = False
):
    """
    Creates or loads FAISS vector store.
    
    FAISS chosen for local storage: free, fast, no external services required.
    If index doesn't exist and chunks=None, automatically loads and chunks documents.
    Checks staleness by document hash.
    
    Args:
        chunks: List of Document objects for index creation.
                If None and index doesn't exist, automatically loads and chunks.
        index_path: Path to directory with FAISS index
        docs_path: Path to directory with source documents
        force_rebuild: If True, recreates index even if it exists
        
    Returns:
        FAISS vector store ready for search
    """
    # Load environment variables from .env file
    logger.info("Loading environment variables for vectorstore...")
    env_loaded = load_dotenv()
    if env_loaded:
        logger.debug("✓ .env file found and loaded")
    else:
        logger.warning("⚠ .env file not found, using system environment variables")
    
    # Get API key from environment variables
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY not found in environment variables")
        raise ValueError(
            "OPENAI_API_KEY not found in environment variables. "
            "Create .env file with OPENAI_API_KEY=your-key"
        )
    logger.info("✓ OPENAI_API_KEY found in environment variables")
    
    # Determine development mode (for allow_dangerous_deserialization)
    # Default is production (safer)
    env = os.getenv("ENV", "production").lower()
    is_dev = env == "development"
    
    logger.info("=" * 60)
    logger.info("FAISS VECTOR STORE")
    logger.info("=" * 60)
    
    # Determine absolute path to index
    project_root = Path(__file__).parent.parent.parent
    full_index_path = project_root / index_path
    
    # Check index existence
    index_exists = full_index_path.exists() and any(full_index_path.iterdir())
    
    # Check index staleness by document hash
    index_stale = False
    if index_exists and not force_rebuild:
        current_hash = _compute_docs_hash(docs_path)
        saved_hash = _load_index_hash(index_path)
        
        if current_hash and saved_hash:
            if current_hash != saved_hash:
                logger.info("Index is stale: documents were changed")
                index_stale = True
        elif current_hash and not saved_hash:
            # Hash was not saved before, consider index stale
            logger.info("Index hash not found, recreating index")
            index_stale = True
    
    if index_exists and not force_rebuild and not index_stale:
        logger.info(f"Found existing index in {index_path}")
        logger.info("Loading index from filesystem...")
        
        try:
            # Initialize embeddings (needed for loading index)
            embeddings = get_embeddings_client()
            
            # Load existing index
            # allow_dangerous_deserialization only in dev mode
            vectorstore = FAISS.load_local(
                str(full_index_path),
                embeddings,
                allow_dangerous_deserialization=is_dev
            )
            
            logger.info("Index successfully loaded")
            logger.info(f"Number of documents in index: {vectorstore.index.ntotal}")
            logger.info("=" * 60)
            
            return vectorstore
            
        except Exception as e:
            logger.warning(f"Error loading index: {e}")
            logger.info("Will create new index...")
            index_exists = False
    
    # Create new index
    if not index_exists or force_rebuild or index_stale:
        # If chunks not provided, automatically load and chunk documents
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
        
        logger.info(f"Creating new index from {len(chunks)} chunks...")
        
        # Initialize OpenAI Embeddings with increased timeout for batch operations
        logger.info("Initializing OpenAI Embeddings (text-embedding-3-small)...")
        from app.infra.openai_utils import OPENAI_BATCH_TIMEOUT
        embeddings = get_embeddings_client(timeout=OPENAI_BATCH_TIMEOUT)
        logger.info(f"Using timeout {OPENAI_BATCH_TIMEOUT}s for batch embedding operations")
        
        # Create FAISS vector store from chunks
        logger.info("Generating embeddings and creating index...")
        logger.info("(This may take several minutes for large number of chunks)")
        
        vectorstore = FAISS.from_documents(
            documents=chunks,
            embedding=embeddings
        )
        
        # Create directory for index if it doesn't exist
        full_index_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save index to disk
        logger.info(f"Saving index to {index_path}...")
        vectorstore.save_local(str(full_index_path))
        
        # Save document hash for staleness checking
        docs_hash = _compute_docs_hash(docs_path)
        if docs_hash:
            _save_index_hash(index_path, docs_hash)
        
        logger.info("Index successfully created and saved")
        logger.info(f"Number of documents in index: {vectorstore.index.ntotal}")
        logger.info(f"Vector dimension: {vectorstore.index.d}")
        logger.info("=" * 60)
        
        return vectorstore


def build_rag_chain(
    vectorstore,
    prompt_settings: Optional[PromptSettings] = None,
    k: Optional[int] = None,
    model: str = "gpt-4o",
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None
):
    """
    Creates RAG chain from ready vectorstore.
    
    Uses settings from PromptSettings for system prompt and LLM parameters.
    
    Args:
        vectorstore: Ready FAISS vectorstore
        prompt_settings: Prompt settings (if None, loaded from environment)
        k: Number of relevant chunks (if None, used from prompt_settings)
        model: OpenAI model (default "gpt-4o-mini")
        temperature: Generation temperature (if None, used from prompt_settings)
        max_tokens: Maximum number of tokens (optional)
        
    Returns:
        RAG chain for answering questions
    """
    # Load settings if not provided
    if prompt_settings is None:
        prompt_settings = load_prompt_settings_from_env()
    
    # Determine effective values
    effective_k = k if k is not None else prompt_settings.default_top_k
    effective_temperature = temperature if temperature is not None else prompt_settings.default_temperature
    # If max_tokens not explicitly passed, use default setting from PromptSettings
    effective_max_tokens = max_tokens if max_tokens is not None else prompt_settings.default_max_tokens

    # Limit ranges
    effective_k = max(1, min(10, effective_k))
    effective_temperature = max(0.0, min(1.0, effective_temperature))
    if effective_max_tokens is not None:
        effective_max_tokens = max(128, min(4096, effective_max_tokens))

    # Create base retriever
    if RERANKING_ENABLED:
        raw_k = max(effective_k * 2, 8)
        logger.info("Creating base retriever with k=%s for reranking...", raw_k)
        base_retriever = vectorstore.as_retriever(search_kwargs={"k": raw_k})
    else:
        logger.info("Reranking disabled, using base retriever with k=%s", effective_k)
        base_retriever = vectorstore.as_retriever(search_kwargs={"k": effective_k})

    # Apply reranking via ContextualCompressionRetriever (optional)
    logger.info(
        "Initializing LLM for retriever: %s (temperature=%s, max_tokens=%s, reranking_enabled=%s)...",
        model,
        effective_temperature,
        effective_max_tokens,
        RERANKING_ENABLED,
    )
    llm = get_chat_llm(
        temperature=effective_temperature,
        model=model,
        max_tokens=effective_max_tokens,
    )

    if RERANKING_ENABLED and RERANKING_AVAILABLE:
        try:
            compressor = LLMChainExtractor.from_llm(llm)
            retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=base_retriever,
            )
            logger.info("Reranking enabled, final k=%s", effective_k)
        except Exception as e:
            logger.warning("Error creating reranker: %s, using base retriever", e)
            retriever = vectorstore.as_retriever(search_kwargs={"k": effective_k})
    else:
        if RERANKING_ENABLED and not RERANKING_AVAILABLE:
            logger.info("Reranking requested but not available in current LangChain version; using base retriever")
        retriever = vectorstore.as_retriever(search_kwargs={"k": effective_k})
    
    # Build system prompt template with {response_language} placeholder
    # Language will be determined per request and injected at runtime
    system_prompt_template = build_system_prompt(prompt_settings, response_language="{response_language}")
    
    # Human prompt with explicit structure: context first, then question and instructions.
    # Include chat_history if provided (empty string if not)
    # If chat_history is empty, the "Conversation history:" line will still appear but with empty content
    human_template = (
        "Documentation context (relevant fragments):\n\n"
        "{context}\n\n"
        "---\n\n"
        "Conversation history:\n{chat_history}\n\n"
        "User question: {input}\n\n"
        "Instructions:\n"
        "- Use ONLY the information from the context\n"
        "- Answer as clearly and structurally as possible\n"
        "- Provide examples and step-by-step instructions when helpful\n"
        "- If the context is insufficient, explain what exactly is missing"
    )

    logger.info("Creating prompt template...")
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt_template),
        ("human", human_template),
    ])
    
    logger.info("Creating Stuff Documents Chain...")
    document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    
    logger.info("Creating Retrieval Chain...")
    rag_chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=document_chain
    )
    
    logger.info("RAG chain successfully created")
    return rag_chain


def get_rag_chain(
    index_path: str = "vectorstore/faiss_index",
    k: Optional[int] = None,
    model: str = "gpt-4o-mini",
    temperature: Optional[float] = None
):
    """
    Creates RAG chain by loading vectorstore and building chain.
    
    Helper function for backward compatibility. Uses build_rag_chain().
    
    Args:
        index_path: Path to directory with FAISS index
        k: Number of relevant chunks (if None, used from settings)
        model: OpenAI model (default "gpt-4o-mini")
        temperature: Generation temperature (if None, used from settings)
        
    Returns:
        RAG chain for answering questions
    """
    logger.info("=" * 60)
    logger.info("RAG CHAIN INITIALIZATION")
    logger.info("=" * 60)
    
    logger.info("Loading vector store...")
    vectorstore = build_or_load_vectorstore(chunks=None, index_path=index_path)
    
    rag_chain = build_rag_chain(vectorstore, k=k, model=model, temperature=temperature)
    
    logger.info("=" * 60)
    return rag_chain


def build_rag_chain_and_settings(
    index_path: str = "vectorstore/faiss_index"
):
    """
    Creates RAG chain and returns prompt settings.
    
    Used during application initialization to save settings in app.state.
    
    Args:
        index_path: Path to directory with FAISS index
        
    Returns:
        Tuple (rag_chain, vectorstore, prompt_settings)
    """
    logger.info("=" * 60)
    logger.info("RAG CHAIN INITIALIZATION WITH SETTINGS")
    logger.info("=" * 60)
    
    logger.info("Loading vector store...")
    vectorstore = build_or_load_vectorstore(chunks=None, index_path=index_path)
    
    logger.info("Loading prompt settings...")
    prompt_settings = load_prompt_settings_from_env()
    
    logger.info(f"Settings: supported_languages={prompt_settings.supported_languages}, fallback={prompt_settings.fallback_language}, "
                f"mode={prompt_settings.mode}, temperature={prompt_settings.default_temperature}, top_k={prompt_settings.default_top_k}")
    
    rag_chain = build_rag_chain(
        vectorstore,
        prompt_settings=prompt_settings,
        k=prompt_settings.default_top_k,
        temperature=prompt_settings.default_temperature
    )
    
    logger.info("=" * 60)
    return rag_chain, vectorstore, prompt_settings

