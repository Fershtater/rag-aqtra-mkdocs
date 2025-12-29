"""
Conversation history management for v2 API.
"""

import logging
import uuid
from datetime import datetime
from typing import List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.infra.models import Conversation, ConversationMessage

logger = logging.getLogger(__name__)


async def get_or_create_conversation(
    sessionmaker: Optional[async_sessionmaker[AsyncSession]],
    conversation_id: Optional[str]
) -> str:
    """
    Get or create conversation ID.
    
    If conversation_id is None, generates new UUID-like ID (c_<uuid>).
    If sessionmaker is None (DB not configured), still returns/generates ID but doesn't persist.
    
    Args:
        sessionmaker: Database sessionmaker (can be None)
        conversation_id: Existing conversation ID or None
        
    Returns:
        Conversation ID string
    """
    if conversation_id:
        # If DB is configured, verify conversation exists (or create)
        if sessionmaker:
            try:
                async with sessionmaker() as session:
                    result = await session.execute(
                        select(Conversation).where(Conversation.conversation_id == conversation_id)
                    )
                    conv = result.scalar_one_or_none()
                    if not conv:
                        # Create new conversation
                        conv = Conversation(conversation_id=conversation_id)
                        session.add(conv)
                        await session.commit()
                        logger.debug(f"Created new conversation: {conversation_id}")
                    return conversation_id
            except Exception as e:
                logger.warning(f"Error checking conversation in DB: {e}, using ID anyway")
                return conversation_id
        else:
            # No DB, just return the ID
            return conversation_id
    else:
        # Generate new ID
        new_id = f"c_{uuid.uuid4().hex[:16]}"
        if sessionmaker:
            try:
                async with sessionmaker() as session:
                    conv = Conversation(conversation_id=new_id)
                    session.add(conv)
                    await session.commit()
                    logger.debug(f"Created new conversation: {new_id}")
            except Exception as e:
                logger.warning(f"Error creating conversation in DB: {e}, using ID anyway")
        return new_id


async def load_history(
    sessionmaker: Optional[async_sessionmaker[AsyncSession]],
    conversation_id: str,
    limit: int = 20
) -> List[dict]:
    """
    Load conversation history from database.
    
    Args:
        sessionmaker: Database sessionmaker (can be None)
        conversation_id: Conversation ID
        limit: Maximum number of messages to load
        
    Returns:
        List of {"role": "user"|"assistant", "content": "..."} dictionaries
    """
    if not sessionmaker:
        return []
    
    try:
        async with sessionmaker() as session:
            result = await session.execute(
                select(ConversationMessage)
                .where(ConversationMessage.conversation_id == conversation_id)
                .order_by(ConversationMessage.created_at.desc())
                .limit(limit)
            )
            messages = result.scalars().all()
            
            # Reverse to get chronological order
            messages = list(reversed(messages))
            
            return [
                {"role": msg.role, "content": msg.content}
                for msg in messages
            ]
    except Exception as e:
        logger.warning(f"Error loading history for {conversation_id}: {e}")
        return []


async def append_message(
    sessionmaker: Optional[async_sessionmaker[AsyncSession]],
    conversation_id: str,
    role: str,
    content: str
):
    """
    Append message to conversation history.
    
    Args:
        sessionmaker: Database sessionmaker (can be None)
        conversation_id: Conversation ID
        role: Message role ("user" or "assistant")
        content: Message content
    """
    if not sessionmaker:
        return
    
    if role not in ("user", "assistant"):
        logger.warning(f"Invalid role: {role}, skipping message save")
        return
    
    try:
        async with sessionmaker() as session:
            # Update conversation updated_at
            result = await session.execute(
                select(Conversation).where(Conversation.conversation_id == conversation_id)
            )
            conv = result.scalar_one_or_none()
            if conv:
                conv.updated_at = datetime.utcnow()
            
            # Add message
            msg = ConversationMessage(
                conversation_id=conversation_id,
                role=role,
                content=content
            )
            session.add(msg)
            await session.commit()
            logger.debug(f"Appended {role} message to conversation {conversation_id}")
    except Exception as e:
        logger.warning(f"Error appending message to {conversation_id}: {e}")

