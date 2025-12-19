"""
Unit tests for prompt_renderer module.
"""

import os
import pytest

# Skip tests if jinja2 is not available
try:
    from jinja2 import TemplateSyntaxError, UndefinedError
    from app.core.prompt_renderer import (
        PromptRenderer,
        sanitize_passthrough,
        convert_legacy_summaries,
    )
    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False
    pytestmark = pytest.mark.skip("jinja2 not available")


@pytest.fixture
def renderer():
    """Create a PromptRenderer instance for testing."""
    return PromptRenderer(max_chars=10000, strict_undefined=True)


@pytest.fixture
def renderer_lenient():
    """Create a lenient PromptRenderer (no strict undefined)."""
    return PromptRenderer(max_chars=10000, strict_undefined=False)


def test_render_basic_namespaces(renderer):
    """Test rendering with basic namespaces."""
    template = """
You are an assistant.
Request ID: {{ system.request_id }}
Mode: {{ system.mode }}
Sources count: {{ source.count }}
"""
    
    system = {"request_id": "test123", "mode": "strict", "now_iso": "2024-01-01T00:00:00"}
    source = {"content": "Test content", "count": 2, "documents": []}
    passthrough = {}
    tools = {}
    
    result = renderer.render(template, system=system, source=source, passthrough=passthrough, tools=tools)
    
    assert "test123" in result
    assert "strict" in result
    assert "2" in result


def test_sanitize_passthrough_blocks_objects():
    """Test that sanitize_passthrough blocks dangerous objects."""
    # Test with a class instance
    class TestClass:
        def __init__(self):
            self.secret = "secret"
    
    obj = TestClass()
    sanitized = sanitize_passthrough(obj)
    
    # Should be converted to string
    assert isinstance(sanitized, str)
    
    # Test with dict containing non-str keys
    bad_dict = {123: "value", "key": "value"}
    sanitized_dict = sanitize_passthrough(bad_dict)
    
    # Keys should be strings
    assert all(isinstance(k, str) for k in sanitized_dict.keys())
    
    # Test with nested structure
    nested = {
        "level1": {
            "level2": {
                "level3": "deep"
            }
        }
    }
    sanitized_nested = sanitize_passthrough(nested, max_depth=2)
    
    # Should be truncated at max_depth
    assert isinstance(sanitized_nested, (str, dict))


def test_sanitize_passthrough_masks_secrets():
    """Test that sensitive keys are masked."""
    passthrough = {
        "api_key": "secret123",
        "token": "token456",
        "password": "pass789",
        "normal_field": "value"
    }
    
    sanitized = sanitize_passthrough(passthrough)
    
    assert sanitized["api_key"] == "***MASKED***"
    assert sanitized["token"] == "***MASKED***"
    assert sanitized["password"] == "***MASKED***"
    assert sanitized["normal_field"] == "value"


def test_legacy_summaries_mapping():
    """Test conversion of legacy {summaries} placeholder."""
    template = "Context: {summaries}\nQuestion: {input}"
    
    converted = convert_legacy_summaries(template)
    
    assert "{{ source.content }}" in converted
    assert "{summaries}" not in converted
    
    # Should not convert if already Jinja2
    jinja_template = "Context: {{ source.content }}\nQuestion: {{ input }}"
    converted_jinja = convert_legacy_summaries(jinja_template)
    assert converted_jinja == jinja_template


def test_strict_undefined_raises(renderer):
    """Test that strict undefined mode raises error on missing variables."""
    template = "Hello {{ system.missing_var }}"
    
    system = {"request_id": "test"}
    source = {"content": "", "count": 0, "documents": []}
    passthrough = {}
    tools = {}
    
    with pytest.raises((UndefinedError, Exception)):
        renderer.render(template, system=system, source=source, passthrough=passthrough, tools=tools)


def test_lenient_undefined_allows_missing(renderer_lenient):
    """Test that lenient mode allows missing variables."""
    template = "Hello {{ system.missing_var }}"
    
    system = {"request_id": "test"}
    source = {"content": "", "count": 0, "documents": []}
    passthrough = {}
    tools = {}
    
    # Should not raise
    result = renderer_lenient.render(template, system=system, source=source, passthrough=passthrough, tools=tools)
    assert isinstance(result, str)


def test_size_limit_trims_source_content(renderer):
    """Test that size limit trims source.content."""
    # Create a renderer with small max_chars
    small_renderer = PromptRenderer(max_chars=100, strict_undefined=False)
    
    template = "Sources: {{ source.content }}"
    
    system = {"request_id": "test", "mode": "strict"}
    # Create very long content
    long_content = "A" * 1000
    source = {"content": long_content, "count": 1, "documents": []}
    passthrough = {}
    tools = {}
    
    result = small_renderer.render(template, system=system, source=source, passthrough=passthrough, tools=tools)
    
    # Should be trimmed
    assert len(result) <= 100 + 50  # Allow some overhead
    assert "[... prompt truncated ...]" in result or "[... content truncated ...]" in result


def test_validate_template_syntax_error(renderer):
    """Test that validate_template catches syntax errors."""
    bad_template = "{{ system.request_id }"
    
    with pytest.raises(TemplateSyntaxError):
        renderer.validate_template(bad_template)


def test_filters_truncate_chars(renderer):
    """Test truncate_chars filter."""
    template = "{{ 'very long string' | truncate_chars(5) }}"
    
    system = {"request_id": "test"}
    source = {"content": "", "count": 0, "documents": []}
    passthrough = {}
    tools = {}
    
    result = renderer.render(template, system=system, source=source, passthrough=passthrough, tools=tools)
    
    assert len(result.strip()) <= 8  # "very..." is 8 chars


def test_filters_safe_newlines(renderer):
    """Test safe_newlines filter."""
    template = "{{ 'line1\n\n\n\nline2' | safe_newlines }}"
    
    system = {"request_id": "test"}
    source = {"content": "", "count": 0, "documents": []}
    passthrough = {}
    tools = {}
    
    result = renderer.render(template, system=system, source=source, passthrough=passthrough, tools=tools)
    
    # Should not have more than 2 consecutive newlines
    assert "\n\n\n" not in result



