"""
Integration tests for prompt preset selection.
"""

import os
import pytest
from pathlib import Path

from app.core.prompt_config import get_prompt_template_content, get_selected_template_info, is_jinja_mode


def test_preset_strict_selection(monkeypatch):
    """Test that PROMPT_PRESET=strict selects correct template."""
    monkeypatch.setenv("PROMPT_TEMPLATE_MODE", "jinja")
    monkeypatch.setenv("PROMPT_PRESET", "strict")
    monkeypatch.delenv("PROMPT_TEMPLATE", raising=False)
    monkeypatch.delenv("PROMPT_TEMPLATE_PATH", raising=False)
    
    # Check that template is loaded
    template = get_prompt_template_content()
    
    # Should contain strict template content
    assert "STRICT MODE" in template or "strict" in template.lower()
    assert len(template) > 0
    
    # Check template info
    info = get_selected_template_info()
    assert info["selected_template"] == "preset:strict"
    assert "aqtra_strict_en.j2" in info["selected_template_path"]


def test_preset_support_selection(monkeypatch):
    """Test that PROMPT_PRESET=support selects correct template."""
    monkeypatch.setenv("PROMPT_TEMPLATE_MODE", "jinja")
    monkeypatch.setenv("PROMPT_PRESET", "support")
    monkeypatch.delenv("PROMPT_TEMPLATE", raising=False)
    monkeypatch.delenv("PROMPT_TEMPLATE_PATH", raising=False)
    
    template = get_prompt_template_content()
    
    # Should contain support template content
    assert "SUPPORT" in template or "support" in template.lower() or "friendly" in template.lower()
    assert len(template) > 0
    
    info = get_selected_template_info()
    assert info["selected_template"] == "preset:support"
    assert "aqtra_support_en.j2" in info["selected_template_path"]


def test_preset_developer_selection(monkeypatch):
    """Test that PROMPT_PRESET=developer selects correct template."""
    monkeypatch.setenv("PROMPT_TEMPLATE_MODE", "jinja")
    monkeypatch.setenv("PROMPT_PRESET", "developer")
    monkeypatch.delenv("PROMPT_TEMPLATE", raising=False)
    monkeypatch.delenv("PROMPT_TEMPLATE_PATH", raising=False)
    
    template = get_prompt_template_content()
    
    # Should contain developer template content
    assert "DEVELOPER" in template or "developer" in template.lower() or "technical" in template.lower()
    assert len(template) > 0
    
    info = get_selected_template_info()
    assert info["selected_template"] == "preset:developer"
    assert "aqtra_developer_en.j2" in info["selected_template_path"]


def test_preset_default_strict(monkeypatch):
    """Test that default preset is strict."""
    monkeypatch.setenv("PROMPT_TEMPLATE_MODE", "jinja")
    monkeypatch.delenv("PROMPT_PRESET", raising=False)
    monkeypatch.delenv("PROMPT_TEMPLATE", raising=False)
    monkeypatch.delenv("PROMPT_TEMPLATE_PATH", raising=False)
    
    info = get_selected_template_info()
    assert info["selected_template"] == "preset:strict"
    assert info["preset"] == "strict"


def test_preset_priority_over_template_path(monkeypatch):
    """Test that PROMPT_TEMPLATE has priority over preset."""
    monkeypatch.setenv("PROMPT_TEMPLATE_MODE", "jinja")
    monkeypatch.setenv("PROMPT_TEMPLATE", "Custom template {{ system.request_id }}")
    monkeypatch.setenv("PROMPT_PRESET", "support")
    
    template = get_prompt_template_content()
    
    assert "Custom template" in template
    assert "{{ system.request_id }}" in template
    
    info = get_selected_template_info()
    assert info["selected_template"] == "inline"


def test_preset_priority_template_path_over_preset(monkeypatch, tmp_path):
    """Test that PROMPT_TEMPLATE_PATH has priority over preset."""
    # Create a test template file
    test_template = tmp_path / "test_template.j2"
    test_template.write_text("Test template from file")
    
    monkeypatch.setenv("PROMPT_TEMPLATE_MODE", "jinja")
    monkeypatch.setenv("PROMPT_TEMPLATE_PATH", str(test_template))
    monkeypatch.setenv("PROMPT_PRESET", "developer")
    monkeypatch.delenv("PROMPT_TEMPLATE", raising=False)
    
    template = get_prompt_template_content()
    
    assert "Test template from file" in template
    
    info = get_selected_template_info()
    assert info["selected_template"] == "path"
    assert info["selected_template_path"] == str(test_template)


def test_legacy_mode_ignores_preset(monkeypatch):
    """Test that legacy mode ignores preset."""
    monkeypatch.setenv("PROMPT_TEMPLATE_MODE", "legacy")
    monkeypatch.setenv("PROMPT_PRESET", "support")
    
    template = get_prompt_template_content()
    
    # Should be legacy prompt (contains CRITICAL RULES or similar)
    assert "CRITICAL RULES" in template or "RESPOND ONLY" in template
    
    info = get_selected_template_info()
    assert info["selected_template"] == "legacy"
    assert info["preset"] is None



