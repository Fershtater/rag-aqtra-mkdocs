"""
RAG modules for indexing, retrieval, and chain building.
"""

# Re-export for convenience
from app.rag.indexing import (
    load_mkdocs_documents,
    chunk_documents,
    build_or_load_vectorstore,
    get_vectorstore_dir,
)
from app.rag.retrieval import build_retriever
from app.rag.chain import (
    build_rag_chain,
    get_rag_chain,
    build_rag_chain_and_settings,
)

__all__ = [
    "load_mkdocs_documents",
    "chunk_documents",
    "build_or_load_vectorstore",
    "get_vectorstore_dir",
    "build_retriever",
    "build_rag_chain",
    "get_rag_chain",
    "build_rag_chain_and_settings",
]

