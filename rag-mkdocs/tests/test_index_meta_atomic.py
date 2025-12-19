"""
Unit tests for index metadata and atomic rebuild.
"""

import json
import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.rag.index_meta import (
    generate_index_version,
    get_index_version,
    load_index_meta,
    save_index_meta,
)
from app.rag.index_lock import IndexLock


def test_generate_index_version():
    """Test that index version is generated correctly."""
    version = generate_index_version()
    assert isinstance(version, str)
    assert "-" in version
    # Should be timestamp-uuid format
    parts = version.split("-")
    assert len(parts) >= 2


def test_save_and_load_index_meta(tmp_path):
    """Test saving and loading index metadata."""
    index_path = str(tmp_path / "test_index")
    
    version = generate_index_version()
    save_index_meta(
        index_path,
        index_version=version,
        docs_hash="abc123",
        docs_path="data/docs",
        chunks_count=100
    )
    
    meta = load_index_meta(index_path)
    assert meta is not None
    assert meta["index_version"] == version
    assert meta["docs_hash"] == "abc123"
    assert meta["docs_path"] == "data/docs"
    assert meta["chunks_count"] == 100
    assert "created_at" in meta


def test_get_index_version(tmp_path):
    """Test getting index version from metadata."""
    index_path = str(tmp_path / "test_index")
    
    version = generate_index_version()
    save_index_meta(
        index_path,
        index_version=version,
        docs_hash="abc123",
        docs_path="data/docs"
    )
    
    loaded_version = get_index_version(index_path)
    assert loaded_version == version


def test_index_lock_acquire_release(tmp_path):
    """Test that lock can be acquired and released."""
    index_path = str(tmp_path / "test_index")
    lock = IndexLock(index_path, timeout_seconds=5)
    
    assert lock.acquire()
    assert lock.lock_file.exists()
    
    lock.release()
    assert not lock.lock_file.exists()


def test_index_lock_context_manager(tmp_path):
    """Test lock as context manager."""
    index_path = str(tmp_path / "test_index")
    lock = IndexLock(index_path, timeout_seconds=5)
    
    with lock:
        assert lock.lock_file.exists()
    
    assert not lock.lock_file.exists()


def test_index_lock_stale_detection(tmp_path):
    """Test that stale locks are detected."""
    index_path = str(tmp_path / "test_index")
    lock_file = tmp_path / f"{Path(index_path).name}.lock"
    
    # Create a stale lock (old timestamp)
    import time
    lock_file.write_text(f"12345\n{time.time() - 1000}\n")
    
    lock = IndexLock(index_path, timeout_seconds=5)
    assert lock._is_lock_stale()


def test_atomic_rebuild_simulation(tmp_path):
    """Simulate atomic rebuild process."""
    index_path = str(tmp_path / "index")
    tmp_index_path = f"{index_path}.tmp-abc123"
    
    # Create temporary index
    tmp_full = tmp_path / tmp_index_path
    tmp_full.mkdir(parents=True)
    
    # Save metadata to tmp
    version = generate_index_version()
    save_index_meta(
        tmp_index_path,
        index_version=version,
        docs_hash="test_hash",
        docs_path="data/docs",
        chunks_count=50
    )
    
    # Simulate atomic swap
    if (tmp_path / index_path).exists():
        (tmp_path / index_path).rename(tmp_path / f"{index_path}.bak")
    
    tmp_full.rename(tmp_path / index_path)
    
    # Verify metadata exists after swap
    meta = load_index_meta(index_path)
    assert meta is not None
    assert meta["index_version"] == version


def test_index_lock_timeout(tmp_path):
    """Test that lock timeout returns False without hanging."""
    index_path = str(tmp_path / "index")
    lock = IndexLock(index_path, timeout_seconds=0.1)  # Very short timeout for test
    
    # Create a lock file manually to simulate existing lock
    lock_file = tmp_path / f"{tmp_path.name}.lock"
    import time
    lock_file.write_text(f"12345\n{time.time()}\n")
    lock.lock_file = lock_file
    
    # Try to acquire - should timeout quickly
    result = lock.acquire()
    assert result is False
    
    # Verify lock info is available
    lock_info = lock._get_lock_info()
    assert "pid" in lock_info or lock_info == {}  # May be empty if file was removed

