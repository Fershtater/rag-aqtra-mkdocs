"""
Conversation service for managing conversation history.
"""

import logging
from typing import List, Optional

from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession

from app.infra.conversations import (
    get_or_create_conversation as _get_or_create_conversation,
    load_history as _load_history,
    append_message as _append_message,
)

logger = logging.getLogger(__name__)


class ConversationService:
    """Service for managing conversation history."""
    
    def __init__(self, db_sessionmaker: Optional[async_sessionmaker[AsyncSession]]):
        """
        Initialize conversation service.
        
        Args:
            db_sessionmaker: Database sessionmaker (can be None if DB not configured)
        """
        self.db_sessionmaker = db_sessionmaker
    
    async def get_or_create_conversation(
        self,
        conversation_id: Optional[str]
    ) -> str:
        """
        Get or create conversation ID.
        
        Args:
            conversation_id: Existing conversation ID or None
            
        Returns:
            Conversation ID string
        """
        return await _get_or_create_conversation(self.db_sessionmaker, conversation_id)
    
    async def load_history(
        self,
        conversation_id: str,
        limit: int = 20
    ) -> List[dict]:
        """
        Load conversation history from database.
        
        Args:
            conversation_id: Conversation ID
            limit: Maximum number of messages to load
            
        Returns:
            List of {"role": "user"|"assistant", "content": "..."} dictionaries
        """
        return await _load_history(self.db_sessionmaker, conversation_id, limit)
    
    async def append_message(
        self,
        conversation_id: str,
        role: str,
        content: str
    ):
        """
        Append message to conversation history.
        
        Args:
            conversation_id: Conversation ID
            role: Message role ("user" or "assistant")
            content: Message content
        """
        await _append_message(self.db_sessionmaker, conversation_id, role, content)

