"""
Safe Jinja2 prompt renderer for RAG assistant.

Provides sandboxed template rendering with namespaces (system/source/passthrough/tools)
and size limits.
"""

import logging
import os
import re
from typing import Any, Dict, List, Optional

try:
    from jinja2 import (
        Environment,
        StrictUndefined,
        Undefined,
        TemplateSyntaxError,
        UndefinedError,
        sandbox,
    )
    from jinja2.sandbox import SandboxedEnvironment
    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False
    # Create dummy classes for type hints
    class SandboxedEnvironment:
        pass
    class StrictUndefined:
        pass
    class Undefined:
        pass
    TemplateSyntaxError = Exception
    UndefinedError = Exception

logger = logging.getLogger(__name__)

# Default limits
DEFAULT_MAX_CHARS = int(os.getenv("PROMPT_MAX_CHARS", "40000"))
DEFAULT_STRICT_UNDEFINED = os.getenv("PROMPT_STRICT_UNDEFINED", "1").lower() in ("1", "true", "yes")
DEFAULT_MAX_DEPTH = 6
DEFAULT_MAX_ITEMS = 200
DEFAULT_MAX_STRING_LEN = 2000


def sanitize_passthrough(obj: Any, depth: int = 0, max_depth: int = DEFAULT_MAX_DEPTH) -> Any:
    """
    Sanitize passthrough data to prevent injection and limit size.
    
    Allowed types:
    - str, int, float, bool, None
    - dict (with str keys only)
    - list, tuple
    
    Args:
        obj: Object to sanitize
        depth: Current recursion depth
        max_depth: Maximum recursion depth
        
    Returns:
        Sanitized object (primitives or safe containers)
    """
    if depth > max_depth:
        logger.warning(f"Passthrough sanitization: max depth {max_depth} exceeded, converting to string")
        return str(obj)[:DEFAULT_MAX_STRING_LEN]
    
    # Primitives
    if obj is None or isinstance(obj, (str, int, float, bool)):
        if isinstance(obj, str) and len(obj) > DEFAULT_MAX_STRING_LEN:
            return obj[:DEFAULT_MAX_STRING_LEN]
        return obj
    
    # Dict: only str keys, recursive sanitize values
    if isinstance(obj, dict):
        if len(obj) > DEFAULT_MAX_ITEMS:
            logger.warning(f"Passthrough dict has {len(obj)} items, truncating to {DEFAULT_MAX_ITEMS}")
            items = list(obj.items())[:DEFAULT_MAX_ITEMS]
        else:
            items = obj.items()
        
        sanitized = {}
        for key, value in items:
            if not isinstance(key, str):
                key = str(key)
            # Mask sensitive keys
            if any(sensitive in key.lower() for sensitive in ["token", "key", "secret", "password", "api_key"]):
                sanitized[key] = "***MASKED***"
            else:
                sanitized[key] = sanitize_passthrough(value, depth + 1, max_depth)
        return sanitized
    
    # List/tuple: recursive sanitize items
    if isinstance(obj, (list, tuple)):
        if len(obj) > DEFAULT_MAX_ITEMS:
            logger.warning(f"Passthrough list has {len(obj)} items, truncating to {DEFAULT_MAX_ITEMS}")
            items = obj[:DEFAULT_MAX_ITEMS]
        else:
            items = obj
        
        sanitized = [sanitize_passthrough(item, depth + 1, max_depth) for item in items]
        return tuple(sanitized) if isinstance(obj, tuple) else sanitized
    
    # Everything else: convert to string
    result = str(obj)
    if len(result) > DEFAULT_MAX_STRING_LEN:
        result = result[:DEFAULT_MAX_STRING_LEN]
    return result


def convert_legacy_summaries(template_str: str) -> str:
    """
    Convert legacy {summaries} placeholder to Jinja2 format.
    
    If template doesn't look like Jinja2 (no {{ or {%), but contains {summaries}:
    - Replace {summaries} with {{ source.content }}
    
    Args:
        template_str: Template string
        
    Returns:
        Converted template string
    """
    # Check if it's already Jinja2-like
    if "{{" in template_str or "{%" in template_str:
        return template_str
    
    # Check for legacy {summaries}
    if "{summaries}" in template_str:
        return template_str.replace("{summaries}", "{{ source.content }}")
    
    return template_str


def safe_newlines(value: str) -> str:
    """Normalize multiple empty lines to max 2 consecutive."""
    if not isinstance(value, str):
        return str(value)
    # Replace 3+ newlines with 2
    return re.sub(r'\n{3,}', '\n\n', value)


def truncate_chars(value: Any, n: int = 400) -> str:
    """Truncate string to n characters."""
    s = str(value)
    if len(s) <= n:
        return s
    return s[:n] + "..."


class PromptRenderer:
    """
    Safe Jinja2 prompt renderer with namespaces and size limits.
    """
    
    def __init__(
        self,
        max_chars: int = DEFAULT_MAX_CHARS,
        strict_undefined: bool = DEFAULT_STRICT_UNDEFINED
    ):
        """
        Initialize renderer.
        
        Args:
            max_chars: Maximum characters in rendered prompt
            strict_undefined: If True, raise error on undefined variables
        """
        if not JINJA2_AVAILABLE:
            raise ImportError("jinja2 is not installed. Install with: pip install jinja2")
        
        self.max_chars = max_chars
        self.strict_undefined = strict_undefined
        
        # Create sandboxed environment
        undefined_class = StrictUndefined if strict_undefined else Undefined
        self.env = SandboxedEnvironment(
            autoescape=True,
            undefined=undefined_class,
            trim_blocks=True,
            lstrip_blocks=True
        )
        
        # Add custom filters
        self.env.filters['truncate_chars'] = truncate_chars
        self.env.filters['safe_newlines'] = safe_newlines
        try:
            import json
            self.env.filters['tojson'] = lambda x: json.dumps(x, ensure_ascii=False)
        except ImportError:
            pass
    
    def validate_template(self, template_str: str) -> None:
        """
        Validate template syntax.
        
        Args:
            template_str: Template string to validate
            
        Raises:
            TemplateSyntaxError: If template has syntax errors
            UndefinedError: If strict_undefined=True and variables are undefined
        """
        try:
            # Convert legacy placeholders
            template_str = convert_legacy_summaries(template_str)
            
            # Compile template
            self.env.parse(template_str)
        except TemplateSyntaxError as e:
            raise TemplateSyntaxError(f"Template syntax error: {e}") from e
        except UndefinedError as e:
            raise UndefinedError(f"Undefined variable: {e}") from e
    
    def render(
        self,
        template_str: str,
        *,
        system: Dict[str, Any],
        source: Dict[str, Any],
        passthrough: Dict[str, Any],
        tools: Dict[str, Any]
    ) -> str:
        """
        Render template with namespaces.
        
        Args:
            template_str: Jinja2 template string
            system: System namespace (request_id, conversation_id, now_iso, etc.)
            source: Source namespace (content, count, documents)
            passthrough: Passthrough namespace (sanitized user data)
            tools: Tools namespace (memory, etc.)
            
        Returns:
            Rendered prompt string (trimmed to max_chars if needed)
        """
        # Convert legacy placeholders
        template_str = convert_legacy_summaries(template_str)
        
        # Sanitize passthrough
        sanitized_passthrough = sanitize_passthrough(passthrough)
        
        # Build context
        context = {
            "system": system,
            "source": source,
            "passthrough": sanitized_passthrough,
            "tools": tools
        }
        
        # Compile and render
        try:
            template = self.env.from_string(template_str)
            rendered = template.render(**context)
        except (TemplateSyntaxError, UndefinedError) as e:
            logger.error(f"Template render error: {e}")
            raise
        
        # Apply size limit: trim source.content if needed
        trim_strategy = os.getenv("PROMPT_TRIM_STRATEGY", "trim_source").lower()
        if len(rendered) > self.max_chars and trim_strategy == "trim_source":
            # Try to trim source.content
            original_content = source.get("content", "")
            if original_content:
                # Calculate how much to trim
                excess = len(rendered) - self.max_chars
                # Trim more than excess to account for template structure
                trim_amount = excess + 1000
                
                if len(original_content) > trim_amount:
                    trimmed_content = original_content[:-trim_amount] + "\n[... content truncated ...]"
                    source_trimmed = source.copy()
                    source_trimmed["content"] = trimmed_content
                    
                    # Re-render with trimmed content
                    context["source"] = source_trimmed
                    try:
                        rendered = template.render(**context)
                    except Exception:
                        pass  # Fall through to final truncation
                
                # If still too long, hard truncate
                if len(rendered) > self.max_chars:
                    rendered = rendered[:self.max_chars] + "\n[... prompt truncated ...]"
        elif len(rendered) > self.max_chars:
            # Hard truncate
            rendered = rendered[:self.max_chars] + "\n[... prompt truncated ...]"
        
        return rendered



