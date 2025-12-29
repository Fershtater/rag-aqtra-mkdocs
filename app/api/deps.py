"""
FastAPI dependency injection for services and settings.
"""

from fastapi import Request

from app.settings import Settings
from app.services.answer_service import AnswerService
from app.services.conversation_service import ConversationService
from app.services.prompt_service import PromptService


def get_settings_dep(request: Request) -> Settings:
    """
    Get settings from app.state.
    
    Args:
        request: FastAPI request
        
    Returns:
        Settings instance
    """
    return request.app.state.settings


def get_prompt_service(request: Request) -> PromptService:
    """
    Get prompt service from app.state.
    
    Args:
        request: FastAPI request
        
    Returns:
        PromptService instance
    """
    return request.app.state.prompt_service


def get_conversation_service(request: Request) -> ConversationService:
    """
    Get conversation service from app.state.
    
    Args:
        request: FastAPI request
        
    Returns:
        ConversationService instance
    """
    return request.app.state.conversation_service


def get_answer_service(request: Request) -> AnswerService:
    """
    Get answer service from app.state.
    
    Args:
        request: FastAPI request
        
    Returns:
        AnswerService instance
    """
    return request.app.state.answer_service

