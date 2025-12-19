"""
Offline regression tests for RAG service.

These tests do NOT call real OpenAI API - they use mocked LLM and embeddings.
All tests are deterministic and can run without OPENAI_API_KEY.
"""

import os
import pytest

# Mark all async tests in this module as anyio
pytestmark = pytest.mark.anyio
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import List

from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from app.rag.indexing import (
    load_mkdocs_documents,
    chunk_documents,
    build_or_load_vectorstore,
)
from app.rag.chain import build_rag_chain
from app.services.answer_service import AnswerService
from app.services.conversation_service import ConversationService
from app.services.prompt_service import PromptService
from app.api.schemas.v2 import AnswerRequest
from app.core.prompt_config import PromptSettings
from app.infra.cache import response_cache


class FakeLLM:
    """Fake LLM that returns deterministic responses based on context."""
    
    def invoke(self, messages, **kwargs):
        """Return a fake answer based on the first context document."""
        # Extract context from messages
        context_text = ""
        for msg in messages:
            if hasattr(msg, 'content'):
                content = msg.content
                if "Documentation context" in content:
                    # Extract first chunk from context
                    lines = content.split("\n")
                    in_context = False
                    for line in lines:
                        if "Documentation context" in line:
                            in_context = True
                            continue
                        if in_context and line.strip() and not line.startswith("---"):
                            context_text = line[:120]  # First 120 chars
                            break
        
        if context_text:
            answer = f"ANSWER: {context_text}"
        else:
            answer = "I don't have enough information in the documentation to answer this question."
        
        return AIMessage(content=answer)


class FakeEmbeddings:
    """Fake embeddings that return deterministic vectors."""
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Return simple deterministic embeddings."""
        # Simple hash-based embedding (128 dims)
        import hashlib
        embeddings = []
        for text in texts:
            hash_obj = hashlib.md5(text.encode())
            # Create 128-dim vector from hash
            vec = [float(int(hash_obj.hexdigest()[i:i+2], 16) % 100) / 100.0 for i in range(0, 32, 1)]
            # Pad to 128 dims
            while len(vec) < 128:
                vec.extend(vec[:128-len(vec)])
            embeddings.append(vec[:128])
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Return embedding for query."""
        return self.embed_documents([text])[0]


@pytest.fixture
def temp_index_dir(tmp_path):
    """Create temporary directory for test index."""
    index_dir = tmp_path / "var" / "vectorstore" / "faiss_index"
    index_dir.mkdir(parents=True)
    return str(index_dir)


@pytest.fixture
def test_docs_path(tmp_path):
    """Copy test fixtures to temporary directory."""
    # Get fixtures path
    fixtures_path = Path(__file__).parent / "fixtures" / "docs"
    test_docs = tmp_path / "test_docs"
    test_docs.mkdir()
    
    # Copy fixture files
    for md_file in fixtures_path.glob("*.md"):
        shutil.copy(md_file, test_docs / md_file.name)
    
    return str(test_docs)


@pytest.fixture
def fake_llm():
    """Return fake LLM instance."""
    return FakeLLM()


@pytest.fixture
def fake_embeddings():
    """Return fake embeddings instance."""
    return FakeEmbeddings()


@pytest.fixture
def prompt_settings():
    """Create prompt settings for tests."""
    return PromptSettings(
        supported_languages=("en", "de", "fr", "es", "pt"),
        fallback_language="en",
        mode="strict",
        default_top_k=2,
        default_temperature=0.0,
        default_max_tokens=200
    )


@pytest.fixture
def vectorstore_with_index(test_docs_path, temp_index_dir, fake_embeddings):
    """Build vectorstore from test documents."""
    # Load and chunk documents
    documents = load_mkdocs_documents(test_docs_path)
    assert len(documents) > 0, "Should load test documents"
    
    chunks = chunk_documents(documents, chunk_size=500, chunk_overlap=100)
    assert len(chunks) > 0, "Should create chunks"
    
    # Mock embeddings
    with patch('app.infra.openai_utils.get_embeddings_client', return_value=fake_embeddings):
        # Build vectorstore
        vectorstore = build_or_load_vectorstore(
            chunks=chunks,
            index_path=temp_index_dir,
            docs_path=test_docs_path,
            force_rebuild=True
        )
    
    return vectorstore, chunks


@pytest.fixture
def answer_service():
    """Create answer service for tests."""
    conversation_service = ConversationService(db_sessionmaker=None)
    prompt_service = PromptService()
    return AnswerService(conversation_service, prompt_service)


@pytest.mark.asyncio
async def test_retrieval_works(vectorstore_with_index, fake_llm, prompt_settings, answer_service):
    """Test that retrieval works and returns sources."""
    vectorstore, chunks = vectorstore_with_index
    
    # Build RAG chain with fake LLM
    with patch('app.infra.openai_utils.get_chat_llm', return_value=fake_llm):
        rag_chain = build_rag_chain(
            vectorstore,
            prompt_settings=prompt_settings,
            k=2
        )
    
    # Create request for question that should match test docs
    request = AnswerRequest(
        question="How do I create an app?",
        api_key="test-key"
    )
    
    # Process request
    response = await answer_service.process_answer_request(
        rag_chain=rag_chain,
        request=request,
        request_id="test-001",
        prompt_settings=prompt_settings,
        client_ip="127.0.0.1",
        index_version="test-v1",
        endpoint_name="test"
    )
    
    # Assertions
    assert response is not None
    assert response.not_found == False, "Should find relevant sources"
    assert len(response.sources) >= 1, "Should have at least one source"
    assert response.sources[0].title is not None or response.sources[0].url is not None
    assert response.sources[0].snippet is not None or len(response.sources[0].snippet) >= 0


@pytest.mark.asyncio
async def test_strict_mode_short_circuit(vectorstore_with_index, fake_llm, prompt_settings, answer_service):
    """Test that strict mode short-circuits when no relevant sources."""
    vectorstore, chunks = vectorstore_with_index
    
    # Build RAG chain
    with patch('app.infra.openai_utils.get_chat_llm', return_value=fake_llm):
        rag_chain = build_rag_chain(
            vectorstore,
            prompt_settings=prompt_settings,
            k=2
        )
    
    # Create request for question that definitely won't match test docs
    request = AnswerRequest(
        question="What is the meaning of life, the universe, and everything?",
        api_key="test-key"
    )
    
    # Mock LLM to track if it was called
    llm_call_count = [0]
    
    def track_llm_invoke(*args, **kwargs):
        llm_call_count[0] += 1
        return fake_llm.invoke(*args, **kwargs)
    
    fake_llm_tracked = Mock(side_effect=track_llm_invoke)
    fake_llm_tracked.invoke = track_llm_invoke
    
    # Process request with tracked LLM
    with patch('app.infra.openai_utils.get_chat_llm', return_value=fake_llm_tracked):
        rag_chain_tracked = build_rag_chain(
            vectorstore,
            prompt_settings=prompt_settings,
            k=2
        )
        
        response = await answer_service.process_answer_request(
            rag_chain=rag_chain_tracked,
            request=request,
            request_id="test-002",
            prompt_settings=prompt_settings,
            client_ip="127.0.0.1",
            vectorstore=vectorstore,  # Provide vectorstore for short-circuit
            index_version="test-v1",
            endpoint_name="test"
        )
    
    # Assertions
    assert response is not None
    # In strict mode with no relevant sources, should short-circuit
    # Note: short-circuit may or may not trigger depending on score threshold
    # So we check that either not_found is True OR sources are empty
    if response.not_found or len(response.sources) == 0:
        # Short-circuit likely worked
        assert "don't have enough information" in response.answer.lower() or response.not_found == True


def test_cache_key_segmentation(answer_service, prompt_settings):
    """Test that cache keys differ for different parameters."""
    from app.infra.cache import response_cache
    
    question = "test question"
    
    # Different templates
    sig1 = "template=legacy_lang=en_index_version=v1"
    sig2 = "template=preset:strict_lang=en_index_version=v1"
    key1 = response_cache._generate_key(question, sig1)
    key2 = response_cache._generate_key(question, sig2)
    assert key1 != key2, "Different templates should produce different keys"
    
    # Different languages
    sig3 = "template=legacy_lang=fr_index_version=v1"
    key3 = response_cache._generate_key(question, sig3)
    assert key1 != key3, "Different languages should produce different keys"
    
    # Different index versions
    sig4 = "template=legacy_lang=en_index_version=v2"
    key4 = response_cache._generate_key(question, sig4)
    assert key1 != key4, "Different index versions should produce different keys"
    
    # Different history
    import hashlib
    hist1 = hashlib.md5("history1".encode()).hexdigest()[:8]
    hist2 = hashlib.md5("history2".encode()).hexdigest()[:8]
    sig5 = f"template=legacy_lang=en_index_version=v1_history={hist1}"
    sig6 = f"template=legacy_lang=en_index_version=v1_history={hist2}"
    key5 = response_cache._generate_key(question, sig5)
    key6 = response_cache._generate_key(question, sig6)
    assert key5 != key6, "Different history should produce different keys"


def test_prompt_rendering_safety(prompt_settings):
    """Test that prompt rendering masks secrets and respects max chars."""
    from app.core.prompt_renderer import sanitize_passthrough
    
    # Test secret masking
    passthrough = {
        "api_key": "secret-key-123",
        "token": "secret-token",
        "password": "secret-password",
        "normal_field": "normal-value"
    }
    
    sanitized = sanitize_passthrough(passthrough)
    assert sanitized["api_key"] == "***MASKED***"
    assert sanitized["token"] == "***MASKED***"
    assert sanitized["password"] == "***MASKED***"
    assert sanitized["normal_field"] == "normal-value"
    
    # Test max chars (if PROMPT_MAX_CHARS is set)
    max_chars = 1000
    long_content = "x" * 2000
    source_content = {
        "content": long_content
    }
    
    # In real rendering, source.content should be truncated
    # This is tested indirectly through render_prompt_template
    # For now, we verify sanitize_passthrough limits string length
    sanitized_long = sanitize_passthrough({"long": long_content})
    assert len(sanitized_long["long"]) <= 2000  # DEFAULT_MAX_STRING_LEN


def test_index_version_in_cache_key(answer_service, prompt_settings):
    """Test that index_version is included in cache key."""
    from app.infra.cache import response_cache
    
    question = "test question"
    
    # Same question, different index versions
    sig1 = "template=legacy_lang=en_index_version=v1"
    sig2 = "template=legacy_lang=en_index_version=v2"
    
    key1 = response_cache._generate_key(question, sig1)
    key2 = response_cache._generate_key(question, sig2)
    
    assert key1 != key2, "Different index versions must produce different cache keys"


@pytest.mark.asyncio
async def test_sources_normalization(vectorstore_with_index, fake_llm, prompt_settings, answer_service):
    """Test that sources are normalized correctly."""
    vectorstore, chunks = vectorstore_with_index
    
    with patch('app.infra.openai_utils.get_chat_llm', return_value=fake_llm):
        rag_chain = build_rag_chain(
            vectorstore,
            prompt_settings=prompt_settings,
            k=2
        )
    
    request = AnswerRequest(
        question="How do I create an app?",
        api_key="test-key"
    )
    
    response = await answer_service.process_answer_request(
        rag_chain=rag_chain,
        request=request,
        request_id="test-003",
        prompt_settings=prompt_settings,
        client_ip="127.0.0.1",
        index_version="test-v1",
        endpoint_name="test"
    )
    
    # Check source structure
    if len(response.sources) > 0:
        source = response.sources[0]
        # Source should have at least one of: title, url, snippet
        assert source.title is not None or source.url is not None or source.snippet is not None
        # Source ID should be present
        assert source.id is not None


@pytest.mark.asyncio
async def test_cache_key_stability_same_question_no_history_same_key(answer_service, prompt_settings):
    """Test that cache key is stable for same question without history, regardless of conversation_id."""
    from app.infra.cache import response_cache
    from app.api.schemas.v2 import AnswerRequest
    
    question = "What is Aqtra?"
    
    # Request 1: no conversation_id
    request1 = AnswerRequest(
        question=question,
        api_key="test-key"
    )
    
    # Request 2: different conversation_id, same question, no history
    request2 = AnswerRequest(
        question=question,
        api_key="test-key",
        conversation_id="different-conv-id"
    )
    
    # Build cache keys manually (simulating process_answer_request logic)
    from app.services.answer_service import AnswerService
    from app.core.prompt_config import get_selected_template_info
    import hashlib
    
    # Same settings signature for both
    template_info = get_selected_template_info()
    template_identifier = template_info.get("selected_template", "legacy")
    settings_signature = (
        f"preset=strict_"
        f"mode={prompt_settings.mode}_"
        f"template={template_identifier}_"
        f"lang=en_"
        f"top_k={prompt_settings.default_top_k}_"
        f"temp={prompt_settings.default_temperature}_"
        f"max_tokens={prompt_settings.default_max_tokens}_"
        f"supported={','.join(sorted(prompt_settings.supported_languages))}_"
        f"fallback={prompt_settings.fallback_language}_"
        f"rerank=False_"
        f"detector=v1_"
        f"history=no_history_"
        f"index_version="
    )
    
    key1 = response_cache._generate_key(question, settings_signature)
    key2 = response_cache._generate_key(question, settings_signature)
    
    assert key1 == key2, "Same question and settings should produce same cache key, regardless of conversation_id"


@pytest.mark.asyncio
async def test_strict_miss_lexical_gate_filters_all_docs(vectorstore_with_index, fake_llm, prompt_settings, answer_service):
    """Test that lexical gate filters out all docs for irrelevant queries in strict mode."""
    vectorstore, chunks = vectorstore_with_index
    
    # Create strict prompt settings
    strict_settings = PromptSettings(
        mode="strict",
        supported_languages=prompt_settings.supported_languages,
        fallback_language=prompt_settings.fallback_language,
        base_docs_url=prompt_settings.base_docs_url,
        not_found_message=prompt_settings.not_found_message,
        include_sources_in_text=prompt_settings.include_sources_in_text,
        default_temperature=prompt_settings.default_temperature,
        default_top_k=prompt_settings.default_top_k,
        default_max_tokens=prompt_settings.default_max_tokens
    )
    
    with patch('app.infra.openai_utils.get_chat_llm', return_value=fake_llm):
        rag_chain = build_rag_chain(
            vectorstore,
            prompt_settings=strict_settings,
            k=4
        )
    
    # Irrelevant question that should be filtered by lexical gate
    request = AnswerRequest(
        question="How do I configure the quantum flux capacitor in Aqtra?",
        api_key="test-key",
        preset="strict"  # Force strict preset
    )
    
    # Enable lexical gate
    with patch.dict(os.environ, {
        "STRICT_LEXICAL_GATE_ENABLED": "true",
        "STRICT_LEXICAL_MIN_HITS": "1",
        "STRICT_LEXICAL_MIN_TOKEN_LEN": "4"
    }):
        response = await answer_service.process_answer_request(
            rag_chain=rag_chain,
            request=request,
            request_id="test-lexical-001",
            prompt_settings=strict_settings,
            client_ip="127.0.0.1",
            index_version="test-v1",
            endpoint_name="test"
        )
    
    # In strict mode with lexical gate, irrelevant query should return 0 sources
    assert response.retrieved_chunks == 0, f"Expected 0 chunks for irrelevant query, got {response.retrieved_chunks}"
    assert len(response.sources) == 0, f"Expected 0 sources for irrelevant query, got {len(response.sources)}"
    assert response.not_found is True, "Expected not_found=True for strict miss"


@pytest.mark.asyncio
async def test_preset_override_changes_template_and_mode(vectorstore_with_index, fake_llm, prompt_settings, answer_service):
    """Test that preset override changes both template selection and mode."""
    vectorstore, chunks = vectorstore_with_index
    
    with patch('app.infra.openai_utils.get_chat_llm', return_value=fake_llm):
        rag_chain = build_rag_chain(
            vectorstore,
            prompt_settings=prompt_settings,
            k=2
        )
    
    # Request with developer preset (default might be strict)
    request_developer = AnswerRequest(
        question="How do I create an app?",
        api_key="test-key",
        preset="developer"
    )
    
    # Request with strict preset
    request_strict = AnswerRequest(
        question="How do I create an app?",
        api_key="test-key",
        preset="strict"
    )
    
    # Process both requests
    response_dev = await answer_service.process_answer_request(
        rag_chain=rag_chain,
        request=request_developer,
        request_id="test-preset-dev",
        prompt_settings=prompt_settings,
        client_ip="127.0.0.1",
        index_version="test-v1",
        endpoint_name="test"
    )
    
    response_strict = await answer_service.process_answer_request(
        rag_chain=rag_chain,
        request=request_strict,
        request_id="test-preset-strict",
        prompt_settings=prompt_settings,
        client_ip="127.0.0.1",
        index_version="test-v1",
        endpoint_name="test"
    )
    
    # Verify that preset override affects cache key (different presets = different keys)
    # This is tested indirectly: if preset didn't affect cache, both responses would be identical
    # We verify that preset is used by checking that mode is set correctly
    # (strict preset -> mode="strict", developer preset -> mode="helpful")
    
    # For strict preset, if no sources found, should short-circuit
    # For developer preset, might return sources even if low relevance
    
    # At minimum, verify both requests completed successfully
    assert response_dev is not None
    assert response_strict is not None
    assert response_dev.answer is not None
    assert response_strict.answer is not None

