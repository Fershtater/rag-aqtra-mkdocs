"""
File-based lock for index rebuild operations.
"""

import logging
import os
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class IndexLock:
    """File-based lock for atomic index operations."""
    
    def __init__(self, index_path: str, timeout_seconds: int = 300):
        """
        Initialize lock.
        
        Args:
            index_path: Path to index directory
            timeout_seconds: Lock timeout in seconds
        """
        project_root = Path(__file__).parent.parent.parent
        full_index_path = project_root / index_path
        self.lock_file = full_index_path.parent / f"{full_index_path.name}.lock"
        self.timeout_seconds = timeout_seconds
        self.lock_fd: Optional[int] = None
    
    def acquire(self) -> bool:
        """
        Acquire lock (spin-wait until timeout).
        
        Returns:
            True if lock acquired, False if timeout
        """
        start_time = time.time()
        spin_interval = 0.5  # Check every 500ms
        
        while time.time() - start_time < self.timeout_seconds:
            try:
                # Try to create lock file exclusively (O_CREAT | O_EXCL)
                # On Unix, this is atomic
                self.lock_fd = os.open(
                    str(self.lock_file),
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY
                )
                
                # Write PID and timestamp
                pid = os.getpid()
                timestamp = time.time()
                lock_content = f"{pid}\n{timestamp}\n"
                os.write(self.lock_fd, lock_content.encode())
                os.fsync(self.lock_fd)
                
                logger.info(f"Lock acquired: {self.lock_file}")
                return True
                
            except FileExistsError:
                # Lock exists, check if it's stale
                if self._is_lock_stale():
                    lock_info = self._get_lock_info()
                    age_info = f"{lock_info.get('age_seconds', 'unknown')}s" if lock_info else "unknown"
                    pid_info = lock_info.get('pid', 'unknown') if lock_info else 'unknown'
                    logger.warning(
                        f"Stale lock detected, removing: {self.lock_file}. "
                        f"Age: {age_info}, PID: {pid_info}"
                    )
                    try:
                        self.lock_file.unlink()
                    except Exception as e:
                        logger.warning(f"Failed to remove stale lock: {e}")
                else:
                    # Lock is active, wait and retry
                    time.sleep(spin_interval)
            except Exception as e:
                logger.error(f"Error acquiring lock: {e}")
                return False
        
        # Get lock info for better error message
        lock_info = self._get_lock_info()
        logger.error(
            f"Failed to acquire lock within {self.timeout_seconds} seconds. "
            f"Lock file: {self.lock_file}, Age: {lock_info.get('age_seconds', 'unknown')}s, "
            f"PID: {lock_info.get('pid', 'unknown')}"
        )
        return False
    
    def _get_lock_info(self) -> dict:
        """
        Get information about existing lock file.
        
        Returns:
            Dictionary with lock info (pid, timestamp, age_seconds)
        """
        if not self.lock_file.exists():
            return {}
        
        try:
            with open(self.lock_file, 'r') as f:
                lines = f.readlines()
                if len(lines) >= 2:
                    pid = lines[0].strip()
                    timestamp = float(lines[1].strip())
                    age_seconds = time.time() - timestamp
                    return {
                        "pid": pid,
                        "timestamp": timestamp,
                        "age_seconds": round(age_seconds, 2)
                    }
        except Exception as e:
            logger.warning(f"Error reading lock info: {e}")
        
        return {}
    
    def release(self) -> None:
        """Release lock."""
        if self.lock_fd is not None:
            try:
                os.close(self.lock_fd)
                self.lock_fd = None
            except Exception as e:
                logger.warning(f"Error closing lock file: {e}")
        
        try:
            if self.lock_file.exists():
                self.lock_file.unlink()
                logger.info(f"Lock released: {self.lock_file}")
        except Exception as e:
            logger.warning(f"Error removing lock file: {e}")
    
    def _is_lock_stale(self) -> bool:
        """
        Check if lock is stale (older than timeout * 2).
        
        Returns:
            True if lock is stale
        """
        if not self.lock_file.exists():
            return False
        
        try:
            with open(self.lock_file, 'r') as f:
                lines = f.readlines()
                if len(lines) >= 2:
                    timestamp = float(lines[1].strip())
                    age = time.time() - timestamp
                    # Consider stale if older than timeout * 2
                    return age > (self.timeout_seconds * 2)
        except Exception as e:
            logger.warning(f"Error checking lock staleness: {e}")
        
        return False
    
    def __enter__(self):
        """Context manager entry."""
        if not self.acquire():
            raise RuntimeError(f"Failed to acquire lock: {self.lock_file}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()

