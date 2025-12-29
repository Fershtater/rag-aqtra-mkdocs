"""
RAG Chain for working with MkDocs documentation.

This module provides backward-compatible facade for rag_chain functions.
All functions are re-exported from app.rag modules.
"""

# Re-export all functions for backward compatibility
from app.rag.indexing import (
    load_mkdocs_documents,
    chunk_documents,
    build_or_load_vectorstore,
    get_vectorstore_dir,
)
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
    "build_rag_chain",
    "get_rag_chain",
    "build_rag_chain_and_settings",
]
