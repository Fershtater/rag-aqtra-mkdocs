"""
Answer service for generating RAG answers.
"""

import asyncio
import hashlib
import json
import logging
import os
import time
from time import time as time_func
from typing import Any, Dict, List, Optional, Tuple

from app.api.schemas.v2 import (
    AnswerRequest,
    AnswerResponse,
    MetricsPayload,
    Source,
    generate_source_id,
    parse_history_to_text,
)
from app.core.markdown_utils import build_doc_url
from app.core.prompt_config import (
    PromptSettings,
    detect_response_language,
    get_prompt_template_content,
    is_jinja_mode,
    build_system_prompt,
    get_selected_template_info,
)
from app.infra.cache import response_cache
from app.services.conversation_service import ConversationService
from app.services.prompt_service import PromptService

logger = logging.getLogger(__name__)


class AnswerService:
    """Service for generating RAG answers."""
    
    def __init__(
        self,
        conversation_service: ConversationService,
        prompt_service: PromptService
    ):
        """
        Initialize answer service.
        
        Args:
            conversation_service: Conversation service instance
            prompt_service: Prompt service instance
        """
        self.conversation_service = conversation_service
        self.prompt_service = prompt_service
    
    def normalize_sources(
        self,
        result: Dict,
        prompt_settings: PromptSettings,
        max_snippet_length: int = 300
    ) -> List[Source]:
        """
        Normalize sources from RAG chain result to v2 Source format.
        
        Args:
            result: RAG chain result dictionary
            prompt_settings: Prompt settings for URL building
            max_snippet_length: Maximum snippet length (default 300)
            
        Returns:
            List of Source objects
        """
        sources = []
        seen_ids = set()
        
        # Extract sources from context
        context_docs = []
        if "context" in result:
            context_docs = result["context"]
            if not isinstance(context_docs, list):
                context_docs = [context_docs]
        
        # Fallback: try source_documents
        if not context_docs and "source_documents" in result:
            context_docs = result["source_documents"]
            if not isinstance(context_docs, list):
                context_docs = [context_docs]
        
        for idx, doc in enumerate(context_docs):
            if not hasattr(doc, "metadata") or not doc.metadata:
                continue
            
            source_path = doc.metadata.get("source", "unknown")
            section_anchor = doc.metadata.get("section_anchor")
            section_title = doc.metadata.get("section_title")
            filename = doc.metadata.get("filename", source_path.split("/")[-1])
            
            # Generate stable ID
            source_id = generate_source_id(source_path, section_anchor, idx)
            if source_id in seen_ids:
                continue
            seen_ids.add(source_id)
            
            # Build URL
            url = build_doc_url(
                prompt_settings.base_docs_url,
                source_path,
                section_anchor
            )
            
            # Extract snippet (first ~300 chars, clean whitespace)
            snippet = doc.page_content.strip()[:max_snippet_length]
            snippet = " ".join(snippet.split())  # Normalize whitespace
            
            # Get score if available
            score = doc.metadata.get("score")
            if score is not None:
                try:
                    score = float(score)
                except (ValueError, TypeError):
                    score = None
            
            sources.append(Source(
                id=source_id,
                title=section_title or filename or "Unknown",
                url=url,
                snippet=snippet,
                score=score,
                meta={
                    "source": source_path,
                    "filename": filename,
                    "section_anchor": section_anchor,
                    "section_title": section_title,
                }
            ))
        
        return sources
    
    async def generate_answer(
        self,
        rag_chain,
        question: str,
        request_id: str,
        prompt_settings: PromptSettings,
        chat_history: str = "",
        response_language: Optional[str] = None,
        top_k_override: Optional[int] = None,
        context_hint: Optional[Dict] = None,
        system_prompt: Optional[str] = None,
        vectorstore=None  # Optional: for short-circuit when retriever not extractable
    ) -> Tuple[str, List[Source], bool, Dict]:
        """
        Generate answer using RAG chain.
        
        Args:
            rag_chain: RAG chain instance
            question: User question
            request_id: Request ID for logging
            prompt_settings: Prompt settings
            chat_history: Formatted chat history text (empty string if none)
            response_language: Response language code (auto-detected if None)
            top_k_override: Override top_k (if None, uses default)
            context_hint: Context hint dict (page_url, page_title, language)
            system_prompt: System prompt (uses default if None)
            vectorstore: Vectorstore instance (for short-circuit)
            
        Returns:
            Tuple (answer, sources, not_found, context_docs_dict)
        """
        # Detect language if not provided
        if response_language is None:
            if context_hint and context_hint.get("language"):
                response_language = context_hint["language"]
            else:
                response_language = detect_response_language(
                    question,
                    supported=set(prompt_settings.supported_languages),
                    fallback=prompt_settings.fallback_language
                )
        
        # Build system prompt (use default if not provided)
        if not system_prompt:
            system_prompt = build_system_prompt(prompt_settings, response_language=response_language)
        
        # Short-circuit for strict mode + no sources
        if prompt_settings.mode == "strict":
            try:
                # Extract retriever from rag_chain
                retriever = None
                chain_attrs = []
                
                if hasattr(rag_chain, 'retriever'):
                    retriever = rag_chain.retriever
                    chain_attrs.append("has retriever attr")
                elif hasattr(rag_chain, 'steps') and len(rag_chain.steps) > 0:
                    for step in rag_chain.steps:
                        if hasattr(step, 'retriever'):
                            retriever = step.retriever
                            chain_attrs.append(f"found in step: {type(step).__name__}")
                            break
                elif hasattr(rag_chain, 'first') and hasattr(rag_chain.first, 'retriever'):
                    retriever = rag_chain.first.retriever
                    chain_attrs.append("found in first")
                elif hasattr(rag_chain, 'bound') and hasattr(rag_chain.bound, 'retriever'):
                    retriever = rag_chain.bound.retriever
                    chain_attrs.append("found in bound")
                
                if retriever:
                    # Retrieve documents to check if we have any sources
                    retrieved_docs = await asyncio.to_thread(retriever.invoke, question)
                    
                    # Check if we have any relevant sources (check scores if available)
                    has_relevant_sources = False
                    if retrieved_docs and len(retrieved_docs) > 0:
                        not_found_score_threshold = float(os.getenv("NOT_FOUND_SCORE_THRESHOLD", "0.20"))
                        scores = []
                        for doc in retrieved_docs:
                            if hasattr(doc, "metadata") and doc.metadata:
                                score = doc.metadata.get("score")
                                if score is not None:
                                    try:
                                        scores.append(float(score))
                                    except (ValueError, TypeError):
                                        pass
                        if scores:
                            max_score = max(scores)
                            has_relevant_sources = max_score >= not_found_score_threshold
                        else:
                            has_relevant_sources = True
                    
                    if not has_relevant_sources:
                        logger.info(f"[{request_id}] Strict mode + no relevant sources: short-circuiting LLM call")
                        not_found_message = "I don't have enough information in the documentation to answer this question."
                        return not_found_message, [], True, {"context_docs": []}
                elif vectorstore:
                    # Use similarity_search_with_score to get scores
                    effective_k = top_k_override if top_k_override else prompt_settings.default_top_k
                    effective_k = max(1, min(10, effective_k))
                    
                    docs_with_scores = await asyncio.to_thread(
                        vectorstore.similarity_search_with_score,
                        question,
                        k=effective_k
                    )
                    
                    retrieved_docs = [doc for doc, score in docs_with_scores]
                    scores = [score for doc, score in docs_with_scores]
                    
                    not_found_score_threshold = float(os.getenv("NOT_FOUND_SCORE_THRESHOLD", "0.20"))
                    has_relevant_sources = False
                    
                    if retrieved_docs and len(retrieved_docs) > 0 and scores:
                        min_distance = min(scores)
                        max_similarity = 1.0 - min_distance if min_distance <= 1.0 else 0.0
                        has_relevant_sources = max_similarity >= not_found_score_threshold
                    elif retrieved_docs and len(retrieved_docs) > 0:
                        has_relevant_sources = True
                    
                    if not has_relevant_sources:
                        logger.info(f"[{request_id}] Strict mode + no relevant sources: short-circuiting LLM call (via vectorstore)")
                        not_found_message = "I don't have enough information in the documentation to answer this question."
                        return not_found_message, [], True, {"context_docs": []}
            except Exception as e:
                logger.debug(f"[{request_id}] Could not check retrieval for short-circuit: {e}")
        
        # Build chain input
        chain_input = {
            "input": question,
            "response_language": response_language,
            "chat_history": chat_history or "",
            "system_prompt": system_prompt
        }
        
        # Call RAG chain
        try:
            result = await asyncio.to_thread(rag_chain.invoke, chain_input)
        except Exception as e:
            logger.error(f"[{request_id}] Error calling RAG chain: {e}", exc_info=True)
            raise
        
        # Extract answer
        answer = result.get("answer", "Failed to generate answer")
        
        # Extract context documents for namespace building
        context_docs = []
        if "context" in result:
            context_docs = result["context"]
            if not isinstance(context_docs, list):
                context_docs = [context_docs]
        elif "source_documents" in result:
            context_docs = result["source_documents"]
            if not isinstance(context_docs, list):
                context_docs = [context_docs]
        
        # Normalize sources
        sources = self.normalize_sources(result, prompt_settings)
        
        # Determine not_found
        not_found_score_threshold = float(os.getenv("NOT_FOUND_SCORE_THRESHOLD", "0.20"))
        not_found = len(sources) == 0
        
        if sources:
            scores = [s.score for s in sources if s.score is not None]
            if scores:
                top_score = max(scores)
                if top_score < not_found_score_threshold:
                    not_found = True
        
        return answer, sources, not_found, {"context_docs": context_docs}
    
    async def process_answer_request(
        self,
        rag_chain,
        request: AnswerRequest,
        request_id: str,
        prompt_settings: PromptSettings,
        client_ip: str,
        user_agent: Optional[str] = None,
        accept_language_header: Optional[str] = None,
        vectorstore=None,
        index_version: Optional[str] = None,
        endpoint_name: str = "unknown"
    ) -> AnswerResponse:
        """
        Process answer request (common logic for /api/answer and /stream).
        
        Args:
            rag_chain: RAG chain instance
            request: Answer request
            request_id: Request ID
            prompt_settings: Prompt settings
            client_ip: Client IP address
            user_agent: User agent string
            accept_language_header: Accept-Language header
            vectorstore: Vectorstore instance (for short-circuit)
            index_version: Index version string (for cache key)
            endpoint_name: Endpoint name for metrics (e.g., "api/answer", "stream")
            
        Returns:
            AnswerResponse
        """
        start_time = time_func()
        
        # Get or create conversation ID
        conversation_id = await self.conversation_service.get_or_create_conversation(
            request.conversation_id
        )
        
        # Load or parse history
        chat_history_text = ""
        if request.history:
            chat_history_text = parse_history_to_text(request.history)
        elif conversation_id:
            history_list = await self.conversation_service.load_history(conversation_id, limit=20)
            if history_list:
                chat_history_text = parse_history_to_text(history_list)
        
        # Build cache key (include history signature)
        history_signature = hashlib.md5(chat_history_text.encode()).hexdigest()[:8] if chat_history_text else "no_history"
        
        # Build passthrough namespace early (needed for language selection and cache key)
        context_hint_dict = None
        if request.context_hint:
            context_hint_dict = request.context_hint.dict()
        
        passthrough_dict = {}
        if request.passthrough:
            passthrough_dict.update(request.passthrough)
        if context_hint_dict:
            passthrough_dict.update(context_hint_dict)
        
        # Select output language early (needed for cache key)
        system_namespace_preview = self.prompt_service.build_system_namespace(
            request_id,
            conversation_id,
            prompt_settings.mode,
            passthrough=passthrough_dict,
            context_hint=context_hint_dict,
            accept_language_header=accept_language_header
        )
        output_language = system_namespace_preview["output_language"]
        
        # Get template info for cache key
        template_info = get_selected_template_info()
        template_identifier = template_info.get("selected_template", "legacy")
        
        # Build cache key with template and language
        top_k_value = prompt_settings.default_top_k
        if request.retrieval and request.retrieval.top_k:
            top_k_value = request.retrieval.top_k
        
        detector_version = "v1"
        reranking_enabled = os.getenv("RERANKING_ENABLED", "0").lower() in ("1", "true", "yes")
        
        # Use provided index_version or default to empty
        index_version_str = index_version or ""
        
        settings_signature = (
            f"mode={prompt_settings.mode}_"
            f"template={template_identifier}_"
            f"lang={output_language}_"
            f"top_k={top_k_value}_"
            f"temp={prompt_settings.default_temperature}_"
            f"max_tokens={getattr(prompt_settings, 'default_max_tokens', None)}_"
            f"supported={','.join(sorted(prompt_settings.supported_languages))}_"
            f"fallback={prompt_settings.fallback_language}_"
            f"rerank={reranking_enabled}_"
            f"detector={detector_version}_"
            f"history={history_signature}_"
            f"index_version={index_version_str}"
        )
        
        cache_key = response_cache._generate_key(request.question, settings_signature)
        
        # Check cache (only if no history to avoid stale responses)
        cached_result: Optional[AnswerResponse] = None
        if not chat_history_text:
            cached_result = response_cache.get(cache_key)
        
        if cached_result:
            logger.debug(f"[{request_id}] Cache hit for query")
            latency_ms = int((time_func() - start_time) * 1000)
            
            # Update conversation_id in cached response
            cached_result.conversation_id = conversation_id
            
            # Save to conversation history if DB available
            await self.conversation_service.append_message(conversation_id, "user", request.question)
            await self.conversation_service.append_message(conversation_id, "assistant", cached_result.answer)
            
            return cached_result
        
        # Get template string
        template_str = get_prompt_template_content(prompt_settings)
        
        # Generate answer with stage timings
        top_k_override = None
        if request.retrieval and request.retrieval.top_k:
            top_k_override = request.retrieval.top_k
        
        # Stage timings
        retrieval_start = time_func()
        prompt_render_start = None
        prompt_render_end = None
        llm_start = None
        llm_end = None
        
        # For Jinja2 mode, we need context_docs to build source namespace
        if is_jinja_mode():
            # First call to get context_docs (includes retrieval)
            answer, sources, not_found, context_info = await self.generate_answer(
                rag_chain,
                request.question,
                request_id,
                prompt_settings,
                chat_history=chat_history_text,
                top_k_override=top_k_override,
                context_hint=context_hint_dict,
                system_prompt="",
                vectorstore=vectorstore
            )
            
            retrieval_end = time_func()
            
            # Build namespaces from context_docs
            context_docs = context_info.get("context_docs", [])
            system_namespace = system_namespace_preview
            source_namespace = self.prompt_service.build_source_namespace(context_docs, prompt_settings)
            
            tools_namespace = {}
            
            # Render system prompt
            prompt_render_start = time_func()
            rendered_system_prompt = self.prompt_service.render_system_prompt(
                template_str,
                system_namespace,
                source_namespace,
                passthrough_dict,
                tools_namespace,
                request_id
            )
            prompt_render_end = time_func()
            
            # Regenerate answer with rendered prompt (LLM call)
            llm_start = time_func()
            answer, sources, not_found, _ = await self.generate_answer(
                rag_chain,
                request.question,
                request_id,
                prompt_settings,
                chat_history=chat_history_text,
                top_k_override=top_k_override,
                context_hint=context_hint_dict,
                system_prompt=rendered_system_prompt,
                vectorstore=vectorstore
            )
            llm_end = time_func()
        else:
            # Legacy mode: use default system prompt
            response_language = system_namespace_preview["output_language"]
            default_system_prompt = build_system_prompt(prompt_settings, response_language=response_language)
            
            # In legacy mode, generate_answer includes both retrieval and LLM
            llm_start = time_func()
            answer, sources, not_found, _ = await self.generate_answer(
                rag_chain,
                request.question,
                request_id,
                prompt_settings,
                chat_history=chat_history_text,
                top_k_override=top_k_override,
                context_hint=context_hint_dict,
                system_prompt=default_system_prompt,
                vectorstore=vectorstore
            )
            retrieval_end = llm_end = time_func()
        
        # Calculate stage timings
        retrieval_ms = int((retrieval_end - retrieval_start) * 1000) if retrieval_end else 0
        prompt_render_ms = int((prompt_render_end - prompt_render_start) * 1000) if prompt_render_start and prompt_render_end else 0
        llm_ms = int((llm_end - llm_start) * 1000) if llm_start and llm_end else 0
        
        # Update Prometheus metrics
        from app.infra.metrics import (
            rag_retrieval_latency_seconds,
            rag_prompt_render_latency_seconds,
            rag_llm_latency_seconds,
            PROMETHEUS_AVAILABLE,
        )
        
        if PROMETHEUS_AVAILABLE:
            if rag_retrieval_latency_seconds and retrieval_ms > 0:
                rag_retrieval_latency_seconds.labels(endpoint=endpoint_name).observe(retrieval_ms / 1000.0)
            if rag_prompt_render_latency_seconds and prompt_render_ms > 0:
                rag_prompt_render_latency_seconds.labels(endpoint=endpoint_name).observe(prompt_render_ms / 1000.0)
            if rag_llm_latency_seconds and llm_ms > 0:
                rag_llm_latency_seconds.labels(endpoint=endpoint_name).observe(llm_ms / 1000.0)
        
        latency_sec = time_func() - start_time
        latency_ms = int(latency_sec * 1000)
        
        # Build response
        debug_info = None
        if request.debug and (request.debug.return_prompt or request.debug.return_chunks):
            # Add stage timings to debug if debug is enabled
            debug_info = {
                "performance": {
                    "retrieval_ms": retrieval_ms,
                    "prompt_render_ms": prompt_render_ms,
                    "llm_ms": llm_ms,
                    "total_ms": latency_ms
                }
            }
            if request.debug.return_chunks:
                debug_info["chunks"] = [{"content": s.snippet[:200], "score": s.score} for s in sources[:5]]
        
        response = AnswerResponse(
            answer=answer,
            sources=sources,
            conversation_id=conversation_id,
            request_id=request_id,
            not_found=not_found,
            metrics=MetricsPayload(
                latency_ms=latency_ms,
                cache_hit=False,
                retrieved_chunks=len(sources),
                model=None
            ),
            debug=debug_info
        )
        
        # Save to cache (only if no history)
        if not chat_history_text:
            response_cache.set(cache_key, response)
        
        # Save to conversation history
        await self.conversation_service.append_message(conversation_id, "user", request.question)
        await self.conversation_service.append_message(conversation_id, "assistant", answer)
        
        logger.info(
            f"[{request_id}] Answer generated: conversation_id={conversation_id}, "
            f"sources={len(sources)}, not_found={not_found}, latency_ms={latency_ms}"
        )
        
        return response

