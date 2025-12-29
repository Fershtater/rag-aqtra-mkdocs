"""
Index metadata and versioning utilities.
"""

import json
import logging
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


def generate_index_version() -> str:
    """
    Generate a unique index version identifier.
    
    Returns:
        Version string (timestamp-based UUID)
    """
    return f"{int(time.time())}-{uuid.uuid4().hex[:8]}"


def save_index_meta(
    index_path: str,
    index_version: str,
    docs_hash: str,
    docs_path: str,
    embedding_model: str = "text-embedding-3-small",
    chunk_size: int = 1500,
    chunk_overlap: int = 300,
    chunks_count: int = 0,
    notes: Optional[str] = None
) -> None:
    """
    Save index metadata to index.meta.json.
    
    Args:
        index_path: Path to index directory
        index_version: Index version identifier
        docs_hash: Document hash
        docs_path: Path to source documents
        embedding_model: Embedding model name
        chunk_size: Chunk size used
        chunk_overlap: Chunk overlap used
        chunks_count: Number of chunks in index
        notes: Optional notes
    """
    project_root = Path(__file__).parent.parent.parent
    full_index_path = project_root / index_path
    meta_file = full_index_path / "index.meta.json"
    
    meta = {
        "index_version": index_version,
        "created_at": datetime.utcnow().isoformat(),
        "docs_hash": docs_hash,
        "docs_path": docs_path,
        "embedding_model": embedding_model,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "chunks_count": chunks_count,
        "notes": notes or ""
    }
    
    try:
        full_index_path.mkdir(parents=True, exist_ok=True)
        with open(meta_file, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2)
        logger.debug(f"Saved index metadata to {meta_file}")
    except Exception as e:
        logger.warning(f"Error saving index metadata: {e}")


def load_index_meta(index_path: str) -> Optional[Dict]:
    """
    Load index metadata from index.meta.json.
    
    Args:
        index_path: Path to index directory
        
    Returns:
        Metadata dictionary or None if not found
    """
    project_root = Path(__file__).parent.parent.parent
    full_index_path = project_root / index_path
    meta_file = full_index_path / "index.meta.json"
    
    if not meta_file.exists():
        return None
    
    try:
        with open(meta_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Error loading index metadata: {e}")
        return None


def get_index_version(index_path: str) -> Optional[str]:
    """
    Get index version from metadata.
    
    Args:
        index_path: Path to index directory
        
    Returns:
        Index version string or None if not found
    """
    meta = load_index_meta(index_path)
    if meta:
        return meta.get("index_version")
    return None

