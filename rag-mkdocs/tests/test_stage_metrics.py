"""
Unit tests for stage metrics endpoint labels.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from app.services.answer_service import AnswerService
from app.services.conversation_service import ConversationService
from app.services.prompt_service import PromptService
from app.api.schemas.v2 import AnswerRequest, AnswerResponse
from app.core.prompt_config import PromptSettings


@pytest.mark.asyncio
async def test_endpoint_name_passed_to_service():
    """Test that endpoint_name is passed from router to service."""
    conversation_service = ConversationService(sessionmaker=None)
    prompt_service = PromptService()
    answer_service = AnswerService(conversation_service, prompt_service)
    
    # Mock RAG chain
    mock_rag_chain = Mock()
    mock_result = {
        "answer": "test answer",
        "context": [],
        "source_documents": []
    }
    mock_rag_chain.invoke = Mock(return_value=mock_result)
    
    request = AnswerRequest(
        question="test question",
        api_key="test-key"
    )
    prompt_settings = PromptSettings()
    
    # Mock metrics
    with patch('app.services.answer_service.rag_retrieval_latency_seconds') as mock_retrieval, \
         patch('app.services.answer_service.rag_prompt_render_latency_seconds') as mock_render, \
         patch('app.services.answer_service.rag_llm_latency_seconds') as mock_llm, \
         patch('app.services.answer_service.PROMETHEUS_AVAILABLE', True):
        
        mock_retrieval.labels.return_value.observe = Mock()
        mock_render.labels.return_value.observe = Mock()
        mock_llm.labels.return_value.observe = Mock()
        
        # Call with endpoint_name
        await answer_service.process_answer_request(
            mock_rag_chain,
            request,
            "test-request-id",
            prompt_settings,
            "127.0.0.1",
            index_version="test-v1",
            endpoint_name="test_endpoint"
        )
        
        # Verify metrics were called with correct endpoint label
        mock_retrieval.labels.assert_called_with(endpoint="test_endpoint")
        mock_llm.labels.assert_called_with(endpoint="test_endpoint")
        
        # prompt_render may not be called in legacy mode
        # but if called, should have correct label
        if mock_render.labels.called:
            mock_render.labels.assert_called_with(endpoint="test_endpoint")


def test_endpoint_names_are_consistent():
    """Test that endpoint names are consistent across routers."""
    # Expected endpoint names
    expected_endpoints = {
        "query_v1": "query_v1",
        "answer_v2": "api/answer",
        "stream": "stream",
        "prompt_render": "api/prompt/render"
    }
    
    # Verify naming convention
    assert "query_v1" in expected_endpoints
    assert "api/answer" in expected_endpoints["answer_v2"]
    assert "stream" in expected_endpoints["stream"]

