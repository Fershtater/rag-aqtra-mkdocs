"""
Prompt service for template rendering and namespace building.
"""

import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from app.core.language_policy import select_output_language
from app.core.markdown_utils import build_doc_url
from app.core.prompt_config import PromptSettings
from app.core.prompt_renderer import PromptRenderer
from app.api.schemas.v2 import generate_source_id

logger = logging.getLogger(__name__)


class PromptService:
    """Service for prompt template rendering and namespace building."""
    
    def __init__(self, max_chars: int = 40000, strict_undefined: bool = True):
        """
        Initialize prompt service.
        
        Args:
            max_chars: Maximum characters for rendered prompt
            strict_undefined: Whether to use strict undefined in Jinja2
        """
        self.max_chars = max_chars
        self.strict_undefined = strict_undefined
        self.renderer = PromptRenderer(max_chars=max_chars, strict_undefined=strict_undefined)
    
    def build_system_namespace(
        self,
        request_id: str,
        conversation_id: Optional[str] = None,
        mode: str = "strict",
        passthrough: Optional[Dict[str, Any]] = None,
        context_hint: Optional[Dict[str, Any]] = None,
        accept_language_header: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Build system namespace for Jinja2 template.
        
        Args:
            request_id: Request ID
            conversation_id: Conversation ID (optional)
            mode: Prompt mode (strict/helpful)
            passthrough: Passthrough dictionary (for language selection)
            context_hint: Context hint dictionary (for language selection)
            accept_language_header: Accept-Language header (for language selection)
            
        Returns:
            Dictionary with system namespace (includes output_language and language_reason)
        """
        # Select output language
        output_language, language_reason = select_output_language(
            passthrough=passthrough,
            context_hint=context_hint,
            accept_language_header=accept_language_header
        )
        
        return {
            "request_id": request_id,
            "conversation_id": conversation_id or "",
            "now_iso": datetime.utcnow().isoformat(),
            "timezone": "UTC",
            "app_version": os.getenv("APP_VERSION", ""),
            "mode": mode,
            "output_language": output_language,
            "language_reason": language_reason
        }
    
    def build_source_namespace(
        self,
        context_docs: List,
        prompt_settings: PromptSettings
    ) -> Dict[str, Any]:
        """
        Build source namespace for Jinja2 template.
        
        Args:
            context_docs: List of Document objects from retrieval
            prompt_settings: Prompt settings
            
        Returns:
            Dictionary with source namespace (content, count, documents)
        """
        # Extract content (concatenated page_content)
        content_parts = []
        documents = []
        
        for idx, doc in enumerate(context_docs):
            if not hasattr(doc, "page_content"):
                continue
            
            content_parts.append(doc.page_content)
            
            # Build document metadata
            source_path = doc.metadata.get("source", "unknown") if hasattr(doc, "metadata") and doc.metadata else "unknown"
            section_anchor = doc.metadata.get("section_anchor") if hasattr(doc, "metadata") and doc.metadata else None
            section_title = doc.metadata.get("section_title") if hasattr(doc, "metadata") and doc.metadata else None
            filename = doc.metadata.get("filename", source_path.split("/")[-1]) if hasattr(doc, "metadata") and doc.metadata else source_path.split("/")[-1]
            
            url = build_doc_url(prompt_settings.base_docs_url, source_path, section_anchor)
            
            # Extract snippet (first 300 chars)
            snippet = doc.page_content.strip()[:300]
            snippet = " ".join(snippet.split())
            
            score = None
            if hasattr(doc, "metadata") and doc.metadata:
                score_val = doc.metadata.get("score")
                if score_val is not None:
                    try:
                        score = float(score_val)
                    except (ValueError, TypeError):
                        pass
            
            doc_id = generate_source_id(source_path, section_anchor, idx)
            
            documents.append({
                "title": section_title or filename or "Unknown",
                "url": url,
                "snippet": snippet,
                "doc_id": doc_id,
                "path": source_path,
                "section": section_anchor,
                "heading": section_title,
                "chunk_id": doc_id,
                "score": score
            })
        
        content = "\n\n".join(content_parts)
        
        return {
            "content": content,
            "count": len(documents),
            "documents": documents
        }
    
    def render_system_prompt(
        self,
        template_str: str,
        system: Dict[str, Any],
        source: Dict[str, Any],
        passthrough: Dict[str, Any],
        tools: Dict[str, Any],
        request_id: str
    ) -> str:
        """
        Render system prompt using Jinja2 or return legacy string.
        
        Args:
            template_str: Template string (Jinja2 or legacy)
            system: System namespace
            source: Source namespace
            passthrough: Passthrough namespace
            tools: Tools namespace
            request_id: Request ID for logging
            
        Returns:
            Rendered prompt string
        """
        from app.core.prompt_config import is_jinja_mode
        
        # Normalize passthrough to avoid StrictUndefined errors
        passthrough = dict(passthrough or {})
        passthrough.setdefault("page_url", None)
        passthrough.setdefault("language", None)
        passthrough.setdefault("lang", None)
        
        if is_jinja_mode():
            try:
                return self.renderer.render(
                    template_str,
                    system=system,
                    source=source,
                    passthrough=passthrough,
                    tools=tools
                )
            except Exception as e:
                logger.error(f"[{request_id}] Error rendering prompt: {e}", exc_info=True)
                # Fallback to legacy
                from app.core.prompt_config import build_system_prompt, load_prompt_settings_from_env
                prompt_settings = load_prompt_settings_from_env()
                return build_system_prompt(prompt_settings, response_language=system.get("output_language", "en"))
        else:
            # Legacy mode: return as-is (should be pre-built)
            return template_str

