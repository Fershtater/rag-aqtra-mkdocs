"""
Answer service for generating RAG answers.
"""

import asyncio
import hashlib
import logging
import os
from time import time as time_func
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

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
from app.infra.openai_utils import stream_chat_completion
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
        
        # Retrieve and prepare sources using unified method (NO chain.invoke)
        # This includes relevance filtering, so retrieved_docs are already filtered
        retrieved_docs, sources, source_content, source_namespace_dict, _ = await self._retrieve_and_prepare_sources(
            rag_chain,
            question,
            request_id,
            prompt_settings,
            top_k_override,
            vectorstore
        )
        
        # Short-circuit for strict mode + no relevant sources (after filtering)
        if prompt_settings.mode == "strict" and len(retrieved_docs) == 0:
            logger.info(f"[{request_id}] Strict mode + no relevant sources after filtering: short-circuiting LLM call (chunks=0, sources=0)")
            not_found_message = "I don't have enough information in the documentation to answer this question."
            # Return empty sources and retrieved_chunks=0
            return not_found_message, [], True, {"context_docs": [], "retrieved_chunks": 0}
        
        # Build chain input (use retrieved_docs from _retrieve_and_prepare_sources)
        # Note: sources are already normalized and filtered by relevance
        chain_input = {
            "input": question,
            "response_language": response_language,
            "chat_history": chat_history or "",
            "system_prompt": system_prompt
        }
        
        # Call RAG chain to generate answer
        try:
            result = await asyncio.to_thread(rag_chain.invoke, chain_input)
        except Exception as e:
            logger.error(f"[{request_id}] Error calling RAG chain: {e}", exc_info=True)
            raise
        
        # Extract answer
        answer = result.get("answer", "Failed to generate answer")
        
        # Use sources from _retrieve_and_prepare_sources (already filtered by relevance)
        # Don't re-normalize from chain result, as it may include unfiltered docs
        context_docs = retrieved_docs
        
        # Determine not_found based on filtered sources
        not_found_score_threshold = float(os.getenv("NOT_FOUND_SCORE_THRESHOLD", "0.20"))
        not_found = len(sources) == 0
        
        if sources:
            scores = [s.score for s in sources if s.score is not None]
            if scores:
                top_score = max(scores)
                if top_score < not_found_score_threshold:
                    not_found = True
        
        return answer, sources, not_found, {"context_docs": context_docs}
    
    def _extract_retriever_from_chain(self, rag_chain):
        """
        Extract retriever from RAG chain.
        
        Returns:
            Retriever instance or None if not found
        """
        retriever = None
        
        if hasattr(rag_chain, 'retriever'):
            retriever = rag_chain.retriever
        elif hasattr(rag_chain, 'steps') and len(rag_chain.steps) > 0:
            for step in rag_chain.steps:
                if hasattr(step, 'retriever'):
                    retriever = step.retriever
                    break
        elif hasattr(rag_chain, 'first') and hasattr(rag_chain.first, 'retriever'):
            retriever = rag_chain.first.retriever
        elif hasattr(rag_chain, 'bound') and hasattr(rag_chain.bound, 'retriever'):
            retriever = rag_chain.bound.retriever
        
        return retriever
    
    async def _retrieve_and_prepare_sources(
        self,
        rag_chain,
        question: str,
        request_id: str,
        prompt_settings: PromptSettings,
        top_k_override: Optional[int] = None,
        vectorstore=None
    ) -> Tuple[List, List[Source], str, Dict[str, Any], Dict[str, int]]:
        """
        Unified method to retrieve documents and prepare sources for both API response and prompt rendering.
        
        Uses retriever directly (NO chain.invoke) to avoid LLM calls during retrieval.
        
        Returns:
            Tuple of (retrieved_docs, normalized_sources, source_content, source_namespace_dict, timing_metrics)
            - retrieved_docs: Raw Document objects from retrieval (filtered by relevance)
            - normalized_sources: Source objects for API response
            - source_content: Concatenated content string
            - source_namespace_dict: Dict with count, content, documents for prompt rendering
            - timing_metrics: Dict with embed_query_ms, vector_search_ms, format_sources_ms, retrieval_ms
        """
        from time import time as time_func
        
        retrieved_docs = []
        normalized_sources = []
        source_content = ""
        source_namespace_dict = {"count": 0, "content": "", "documents": []}
        timing_metrics = {
            "embed_query_ms": None,
            "vector_search_ms": None,
            "format_sources_ms": None,
            "retrieval_ms": None
        }
        
        retrieval_start = time_func()
        
        try:
            effective_k = top_k_override if top_k_override else prompt_settings.default_top_k
            effective_k = max(1, min(10, effective_k))
            
            # Extract retriever from chain (NO chain.invoke)
            retriever = self._extract_retriever_from_chain(rag_chain)
            
            if retriever:
                # Use retriever directly (this will compute embedding and do vector search)
                # Note: embedding happens inside retriever via CachedEmbeddings (cache works automatically)
                # We measure total retrieval time, but can't easily separate embed_query_ms
                vector_search_start = time_func()
                retrieved_docs_raw = await asyncio.to_thread(retriever.invoke, question)
                vector_search_end = time_func()
                timing_metrics["vector_search_ms"] = int((vector_search_end - vector_search_start) * 1000)
                
                # Note: embed_query_ms is included in vector_search_ms for retriever path
                # CachedEmbeddings wrapper will use cache automatically, but we can't measure it separately
                timing_metrics["embed_query_ms"] = None  # Not measurable separately for retriever path
                
                # Filter by relevance if scores are available in metadata
                not_found_score_threshold = float(os.getenv("NOT_FOUND_SCORE_THRESHOLD", "0.20"))
                filtered_docs = []
                
                for doc in retrieved_docs_raw:
                    # Check if doc has relevance score in metadata
                    if hasattr(doc, 'metadata') and doc.metadata:
                        score = doc.metadata.get('score')
                        if score is not None:
                            try:
                                internal_relevance = float(score)
                                if internal_relevance >= not_found_score_threshold:
                                    filtered_docs.append(doc)
                            except (ValueError, TypeError):
                                # If score can't be converted, assume relevant
                                filtered_docs.append(doc)
                        else:
                            # No score: assume relevant (retriever may not provide scores)
                            filtered_docs.append(doc)
                    else:
                        # No metadata: assume relevant
                        filtered_docs.append(doc)
                
                # Apply lexical overlap gate in strict mode
                if prompt_settings.mode == "strict":
                    lexical_gate_enabled = os.getenv("STRICT_LEXICAL_GATE_ENABLED", "true").lower() in ("1", "true", "yes")
                    if lexical_gate_enabled:
                        from app.core.lexical_gate import apply_lexical_gate
                        
                        min_hits = int(os.getenv("STRICT_LEXICAL_MIN_HITS", "1"))
                        min_token_len = int(os.getenv("STRICT_LEXICAL_MIN_TOKEN_LEN", "4"))
                        
                        # Convert to (doc, relevance) format for lexical gate
                        docs_with_relevance = []
                        for doc in filtered_docs:
                            relevance = 1.0  # Default if no score
                            if hasattr(doc, 'metadata') and doc.metadata:
                                score = doc.metadata.get('score')
                                if score is not None:
                                    try:
                                        relevance = float(score)
                                    except (ValueError, TypeError):
                                        pass
                            docs_with_relevance.append((doc, relevance))
                        
                        docs_with_relevance = apply_lexical_gate(
                            docs_with_relevance,
                            question,
                            min_hits=min_hits,
                            min_token_len=min_token_len
                        )
                        
                        # Extract docs from (doc, relevance) tuples
                        filtered_docs = [doc for doc, _ in docs_with_relevance]
                        logger.debug(
                            f"[{request_id}] Retriever + lexical gate: {len(retrieved_docs_raw)} docs -> "
                            f"{len(filtered_docs)} after relevance + lexical filter"
                        )
                
                retrieved_docs = filtered_docs
                if len(filtered_docs) < len(retrieved_docs_raw):
                    logger.debug(
                        f"[{request_id}] Retriever: {len(retrieved_docs_raw)} docs, "
                        f"{len(filtered_docs)} after relevance filter (threshold={not_found_score_threshold})"
                    )
            
            elif vectorstore:
                # Fallback: use vectorstore directly
                # This allows us to measure embed_query separately and use cached embeddings
                from app.infra.embedding_cache import embedding_cache
                from app.infra.openai_utils import get_embeddings_client
                
                # Step 1: Get or compute query embedding (with cache)
                embed_query_start = time_func()
                embedding_model = "text-embedding-3-small"  # Default from get_embeddings_client
                cached_embedding = embedding_cache.get(question, embedding_model)
                
                if cached_embedding:
                    # Use cached embedding (no API call)
                    query_embedding = cached_embedding
                    embed_query_end = time_func()
                    timing_metrics["embed_query_ms"] = int((embed_query_end - embed_query_start) * 1000)
                    logger.debug(f"[{request_id}] Using cached embedding for query ({timing_metrics['embed_query_ms']}ms)")
                else:
                    # Compute embedding (this is the slow part)
                    embeddings_client = get_embeddings_client()
                    query_embedding = await asyncio.to_thread(embeddings_client.embed_query, question)
                    embed_query_end = time_func()
                    timing_metrics["embed_query_ms"] = int((embed_query_end - embed_query_start) * 1000)
                    
                    # Cache the embedding
                    embedding_cache.set(question, query_embedding, embedding_model)
                    logger.debug(f"[{request_id}] Computed and cached embedding ({timing_metrics['embed_query_ms']}ms)")
                
                # Step 2: Vector search with scores for relevance filtering
                vector_search_start = time_func()
                
                # Try to use similarity_search_with_score_by_vector if available (more efficient with cached embedding)
                # Otherwise fall back to similarity_search_with_score (which will recompute embedding)
                try:
                    if hasattr(vectorstore, 'similarity_search_with_score_by_vector'):
                        docs_with_scores = await asyncio.to_thread(
                            vectorstore.similarity_search_with_score_by_vector,
                            query_embedding,
                            k=effective_k
                        )
                    else:
                        # Fallback: use query string (will recompute embedding internally, but CachedEmbeddings will cache it)
                        docs_with_scores = await asyncio.to_thread(
                            vectorstore.similarity_search_with_score,
                            question,
                            k=effective_k
                        )
                except Exception as e:
                    logger.warning(f"[{request_id}] Error in vector search: {e}, falling back to query string")
                    docs_with_scores = await asyncio.to_thread(
                        vectorstore.similarity_search_with_score,
                        question,
                        k=effective_k
                    )
                    
                vector_search_end = time_func()
                timing_metrics["vector_search_ms"] = int((vector_search_end - vector_search_start) * 1000)
                
                # Step 3: Filter by relevance score
                # FAISS uses L2 distance: lower distance = more similar
                # Convert distance to relevance [0..1]: relevance = 1 / (1 + distance)
                # For cosine similarity: distance = 1 - cosine_sim, so relevance = cosine_sim = 1 - distance
                not_found_score_threshold = float(os.getenv("NOT_FOUND_SCORE_THRESHOLD", "0.20"))
                docs_with_relevance = []
                
                for doc, distance in docs_with_scores:
                    # Convert distance to internal_relevance [0..1]
                    # Strategy: assume L2 distance (FAISS default), convert to relevance
                    # relevance = 1 / (1 + distance) for L2 distance
                    # For very small distances (< 0.1), use: relevance = 1 - distance (cosine-like)
                    if isinstance(distance, (int, float)):
                        if distance < 0.1:
                            # Small distance: assume cosine-like (1 - distance)
                            internal_relevance = max(0.0, 1.0 - distance)
                        else:
                            # Larger distance: assume L2, use inverse relationship
                            internal_relevance = 1.0 / (1.0 + distance)
                    else:
                        internal_relevance = 1.0  # Assume relevant if no score
                    
                    if internal_relevance >= not_found_score_threshold:
                        # Store relevance in metadata for later use
                        if not hasattr(doc, 'metadata') or doc.metadata is None:
                            doc.metadata = {}
                        doc.metadata['score'] = internal_relevance
                        docs_with_relevance.append((doc, internal_relevance))
                
                # Step 4: Apply lexical overlap gate in strict mode
                if prompt_settings.mode == "strict":
                    lexical_gate_enabled = os.getenv("STRICT_LEXICAL_GATE_ENABLED", "true").lower() in ("1", "true", "yes")
                    if lexical_gate_enabled:
                        from app.core.lexical_gate import apply_lexical_gate
                        
                        min_hits = int(os.getenv("STRICT_LEXICAL_MIN_HITS", "1"))
                        min_token_len = int(os.getenv("STRICT_LEXICAL_MIN_TOKEN_LEN", "4"))
                        
                        docs_with_relevance = apply_lexical_gate(
                            docs_with_relevance,
                            question,
                            min_hits=min_hits,
                            min_token_len=min_token_len
                        )
                        logger.debug(
                            f"[{request_id}] Lexical gate applied: {len(docs_with_relevance)} docs after lexical filtering"
                        )
                
                # Extract docs from (doc, relevance) tuples
                filtered_docs = [doc for doc, _ in docs_with_relevance]
                retrieved_docs = filtered_docs
                logger.debug(
                    f"[{request_id}] Vector search: {len(docs_with_scores)} candidates, "
                    f"{len(filtered_docs)} after relevance filter (threshold={not_found_score_threshold}, "
                    f"embed_query={timing_metrics.get('embed_query_ms', 0)}ms, "
                    f"vector_search={timing_metrics.get('vector_search_ms', 0)}ms)"
                )
            else:
                logger.warning(f"[{request_id}] No retriever or vectorstore available for retrieval")
            
            # Format sources (measure timing)
            format_start = time_func()
            
            if retrieved_docs:
                # Normalize sources for API response
                result_preview = {"context": retrieved_docs, "source_documents": retrieved_docs}
                normalized_sources = self.normalize_sources(result_preview, prompt_settings)
                
                # Build source namespace for prompt rendering
                source_namespace_dict = self.prompt_service.build_source_namespace(retrieved_docs, prompt_settings)
                source_content = source_namespace_dict.get("content", "")
            
            format_end = time_func()
            timing_metrics["format_sources_ms"] = int((format_end - format_start) * 1000)
            
            retrieval_end = time_func()
            timing_metrics["retrieval_ms"] = int((retrieval_end - retrieval_start) * 1000)
            
            # Debug logging
            logger.debug(
                f"[{request_id}] Retrieved: docs={len(retrieved_docs)}, "
                f"sources={len(normalized_sources)}, "
                f"source.count={source_namespace_dict.get('count', 0)}, "
                f"source.documents={len(source_namespace_dict.get('documents', []))}, "
                f"content_len={len(source_content)}, "
                f"timing={timing_metrics}"
            )
        except Exception as e:
            logger.warning(f"[{request_id}] Error retrieving documents: {e}", exc_info=True)
            retrieval_end = time_func()
            timing_metrics["retrieval_ms"] = int((retrieval_end - retrieval_start) * 1000)
        
        return retrieved_docs, normalized_sources, source_content, source_namespace_dict, timing_metrics
    
    async def generate_answer_stream(
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
        vectorstore=None,
        stream_flush_chars: int = 50,
        preset_override: Optional[str] = None
    ) -> AsyncIterator[Tuple[str, List[Source], bool, Dict]]:
        """
        Generate answer using RAG chain with real token streaming.
        
        Yields:
            Tuples of (token_delta, sources, not_found, context_info)
            - token_delta: Incremental token text
            - sources: List of sources (available after retrieval)
            - not_found: Whether answer was not found
            - context_info: Context documents dict
            
        Note:
            Sources are yielded after retrieval, before LLM streaming starts.
            Final not_found status is yielded after streaming completes.
        """
        # Start timing
        stream_start_time = time_func()
        
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
        
        # Retrieve and prepare sources using unified method (NO chain.invoke)
        retrieved_docs, sources, source_content, source_namespace_dict, retrieval_timing = await self._retrieve_and_prepare_sources(
            rag_chain,
            question,
            request_id,
            prompt_settings,
            top_k_override,
            vectorstore
        )
        retrieval_ms = retrieval_timing.get("retrieval_ms", 0)
        embed_query_ms = retrieval_timing.get("embed_query_ms")
        vector_search_ms = retrieval_timing.get("vector_search_ms")
        format_sources_ms = retrieval_timing.get("format_sources_ms")
        
        context_docs = retrieved_docs
        
        # Short-circuit check: if strict mode and no relevant sources after filtering
        # Note: retrieved_docs are already filtered by relevance in _retrieve_and_prepare_sources
        if prompt_settings.mode == "strict" and len(retrieved_docs) == 0:
            logger.info(f"[{request_id}] Strict mode + no relevant sources after filtering: short-circuiting LLM call (chunks=0, sources=0)")
            not_found_message = "I don't have enough information in the documentation to answer this question."
            # Yield short-circuit message with empty sources and retrieved_chunks=0
            # This ensures sources=[] and retrieved_chunks=0 in end metrics
            yield (not_found_message, [], True, {
                "context_docs": [],
                "retrieved_chunks": 0,
                "retrieval_ms": retrieval_ms,
                "embed_query_ms": embed_query_ms,
                "vector_search_ms": vector_search_ms,
                "format_sources_ms": format_sources_ms,
                "prompt_render_ms": 0,
                "llm_connect_ms": 0
            })
            return
        
        # Yield sources immediately after retrieval (before LLM streaming)
        # Only yield if sources exist (filtered docs)
        if sources and len(sources) > 0:
            yield ("", sources, False, {
                "context_docs": context_docs,
                "retrieval_ms": retrieval_ms,
                "embed_query_ms": embed_query_ms,
                "vector_search_ms": vector_search_ms,
                "format_sources_ms": format_sources_ms
            })
        
        # Build system prompt using Jinja2 template rendering (same as /api/answer)
        from app.core.prompt_config import get_prompt_template_content, is_jinja_mode
        from datetime import datetime
        
        # If system_prompt is None, use Jinja2 template rendering
        # If system_prompt is provided, use it (for legacy mode or when explicitly set)
        if system_prompt is None and is_jinja_mode():
            # Use Jinja2 template rendering with source namespace (use preset_override if provided)
            template_str = get_prompt_template_content(prompt_settings, preset_override=preset_override)
            
            # Build system namespace (minimal, conversation_id will be set by caller if needed)
            system_namespace = {
                "request_id": request_id,
                "conversation_id": "",
                "now_iso": datetime.utcnow().isoformat(),
                "timezone": "UTC",
                "app_version": os.getenv("APP_VERSION", ""),
                "mode": prompt_settings.mode,
                "output_language": response_language
            }
            
            # Build passthrough namespace
            passthrough_dict = {}
            if context_hint:
                passthrough_dict.update(context_hint)
            
            tools_namespace = {}
            
            # Render system prompt with source namespace
            # Debug: verify source_namespace_dict is populated
            logger.debug(
                f"[{request_id}] Rendering prompt: source.count={source_namespace_dict.get('count', 0)}, "
                f"source.documents={len(source_namespace_dict.get('documents', []))}, "
                f"content_len={len(source_namespace_dict.get('content', ''))}"
            )
            
            # Measure prompt rendering time
            prompt_render_start = time_func()
            rendered_system_prompt = self.prompt_service.render_system_prompt(
                template_str,
                system_namespace,
                source_namespace_dict,  # Use source namespace with count, content, documents
                passthrough_dict,
                tools_namespace,
                request_id
            )
            prompt_render_end = time_func()
            prompt_render_ms = int((prompt_render_end - prompt_render_start) * 1000)
            
            # Apply PROMPT_MAX_CHARS limit
            prompt_max_chars = int(os.getenv("PROMPT_MAX_CHARS", "40000"))
            if len(rendered_system_prompt) > prompt_max_chars:
                logger.warning(
                    f"[{request_id}] Prompt too long ({len(rendered_system_prompt)} chars), "
                    f"truncating to {prompt_max_chars}"
                )
                rendered_system_prompt = rendered_system_prompt[:prompt_max_chars] + "..."
            
            # Build user message
            user_message = f"User question: {question}\n\n"
            if chat_history:
                user_message = f"Conversation history:\n{chat_history}\n\n{user_message}"
            
            messages = [
                {"role": "system", "content": rendered_system_prompt},
                {"role": "user", "content": user_message}
            ]
        else:
            # Legacy mode: no prompt rendering time measurement
            prompt_render_ms = 0
            # Legacy mode or explicit system_prompt: use simple template
            context_text = source_content if source_content else ""
            human_template = (
                "Documentation context (relevant fragments):\n\n"
                f"{context_text}\n\n"
                "---\n\n"
                f"Conversation history:\n{chat_history or ''}\n\n"
                f"User question: {question}\n\n"
                "Instructions:\n"
                "- Use ONLY the information from the context\n"
                "- Answer as clearly and structurally as possible\n"
                "- Provide examples and step-by-step instructions when helpful\n"
                "- If the context is insufficient, explain what exactly is missing"
            )
            
            # Use provided system_prompt or build default
            effective_system_prompt = system_prompt if system_prompt else build_system_prompt(prompt_settings, response_language=response_language)
            
            messages = [
                {"role": "system", "content": effective_system_prompt},
                {"role": "user", "content": human_template}
            ]
        
        # Create LLM for streaming (use same parameters as in chain)
        # Instead of extracting from chain, create a new one with same settings
        from app.infra.openai_utils import get_chat_llm
        llm = get_chat_llm(
            temperature=prompt_settings.default_temperature,
            max_tokens=prompt_settings.default_max_tokens
        )
        
        # Measure LLM connect time (time until first token)
        llm_connect_start = time_func()
        llm_connect_ms = None
        
        # Stream tokens with flush policy: first token immediately, then buffer
        buffer = ""
        full_answer = ""
        first_token_yielded = False
        
        try:
            async for token_delta in stream_chat_completion(llm, messages):
                if token_delta:
                    # Measure LLM connect time on first token
                    if llm_connect_ms is None:
                        llm_connect_end = time_func()
                        llm_connect_ms = int((llm_connect_end - llm_connect_start) * 1000)
                    
                    full_answer += token_delta
                    
                    # First token: flush immediately (no buffering)
                    if not first_token_yielded:
                        yield (token_delta, sources, False, {
                            "context_docs": context_docs,
                            "retrieval_ms": retrieval_ms,
                            "embed_query_ms": embed_query_ms,
                            "vector_search_ms": vector_search_ms,
                            "format_sources_ms": format_sources_ms,
                            "prompt_render_ms": prompt_render_ms,
                            "llm_connect_ms": llm_connect_ms
                        })
                        first_token_yielded = True
                    else:
                        # Subsequent tokens: buffer and flush when threshold reached
                        buffer += token_delta
                        if len(buffer) >= stream_flush_chars or stream_flush_chars == 0:
                            yield (buffer, sources, False, {
                                "context_docs": context_docs,
                                "retrieval_ms": retrieval_ms,
                                "embed_query_ms": embed_query_ms,
                                "vector_search_ms": vector_search_ms,
                                "format_sources_ms": format_sources_ms,
                                "prompt_render_ms": prompt_render_ms,
                                "llm_connect_ms": llm_connect_ms
                            })
                            buffer = ""
            
            # Flush remaining buffer
            if buffer:
                yield (buffer, sources, False, {
                    "context_docs": context_docs,
                    "retrieval_ms": retrieval_ms,
                    "embed_query_ms": embed_query_ms,
                    "vector_search_ms": vector_search_ms,
                    "format_sources_ms": format_sources_ms,
                    "prompt_render_ms": prompt_render_ms,
                    "llm_connect_ms": llm_connect_ms
                })
            
            # Determine not_found
            not_found_score_threshold = float(os.getenv("NOT_FOUND_SCORE_THRESHOLD", "0.20"))
            not_found = len(sources) == 0
            if sources:
                scores = [s.score for s in sources if s.score is not None]
                if scores:
                    top_score = max(scores)
                    if top_score < not_found_score_threshold:
                        not_found = True
            
            # Final yield with complete answer and final status
            yield (full_answer, sources, not_found, {
                "context_docs": context_docs,
                "retrieval_ms": retrieval_ms,
                "embed_query_ms": embed_query_ms,
                "vector_search_ms": vector_search_ms,
                "format_sources_ms": format_sources_ms,
                "prompt_render_ms": prompt_render_ms,
                "llm_connect_ms": llm_connect_ms
            })
            
        except Exception as e:
            logger.error(f"[{request_id}] Error in streaming: {e}", exc_info=True)
            yield (f"[Error: {str(e)}]", sources, True, {
                "context_docs": context_docs,
                "retrieval_ms": retrieval_ms if 'retrieval_ms' in locals() else None,
                "embed_query_ms": embed_query_ms if 'embed_query_ms' in locals() else None,
                "vector_search_ms": vector_search_ms if 'vector_search_ms' in locals() else None,
                "format_sources_ms": format_sources_ms if 'format_sources_ms' in locals() else None,
                "prompt_render_ms": prompt_render_ms if 'prompt_render_ms' in locals() else None,
                "llm_connect_ms": llm_connect_ms if 'llm_connect_ms' in locals() else None
            })
    
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
        
        # Compute effective preset (request override > passthrough > server default)
        settings = getattr(self, '_settings', None)
        if settings is None:
            # Fallback: read from env
            server_preset = os.getenv("PROMPT_PRESET", "strict").lower()
        else:
            server_preset = settings.PROMPT_PRESET.lower()
        
        effective_preset = None
        if request.preset:
            effective_preset = request.preset.lower().strip()
        elif request.passthrough and isinstance(request.passthrough, dict):
            preset_from_passthrough = request.passthrough.get("preset")
            if preset_from_passthrough:
                effective_preset = str(preset_from_passthrough).lower().strip()
        
        # Validate and normalize preset
        valid_presets = {"strict", "support", "developer"}
        if effective_preset not in valid_presets:
            effective_preset = server_preset
        
        # Update prompt_settings mode based on effective_preset
        # strict preset -> mode="strict", others -> mode="helpful"
        effective_mode = "strict" if effective_preset == "strict" else "helpful"
        if prompt_settings.mode != effective_mode:
            # Create new PromptSettings with updated mode
            from app.core.prompt_config import PromptSettings
            prompt_settings = PromptSettings(
                supported_languages=prompt_settings.supported_languages,
                fallback_language=prompt_settings.fallback_language,
                base_docs_url=prompt_settings.base_docs_url,
                not_found_message=prompt_settings.not_found_message,
                include_sources_in_text=prompt_settings.include_sources_in_text,
                mode=effective_mode,
                default_temperature=prompt_settings.default_temperature,
                default_top_k=prompt_settings.default_top_k,
                default_max_tokens=prompt_settings.default_max_tokens
            )
        
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
        
        # Build cache key (include history signature and effective_preset)
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
        
        # Get template info for cache key (use effective_preset for template selection)
        template_info = get_selected_template_info()
        template_identifier = template_info.get("selected_template", "legacy")
        # Override template identifier if effective_preset differs from server default
        if effective_preset != server_preset:
            template_identifier = f"preset:{effective_preset}"
        
        # Build cache key with template and language
        top_k_value = prompt_settings.default_top_k
        if request.retrieval and request.retrieval.top_k:
            top_k_value = request.retrieval.top_k
        
        detector_version = "v1"
        reranking_enabled = os.getenv("RERANKING_ENABLED", "0").lower() in ("1", "true", "yes")
        
        # Use provided index_version or default to empty
        index_version_str = index_version or ""
        
        # Build settings signature for cache key (include effective_preset to avoid cache collisions)
        settings_signature = (
            f"preset={effective_preset}_"
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
        
        # DEBUG: Log cache key hash and components (without sensitive data)
        cache_key_hash = hashlib.sha256(cache_key.encode()).hexdigest()[:12]
        logger.debug(
            f"[{request_id}] Cache key hash={cache_key_hash}, "
            f"components=[template={template_identifier}, lang={output_language}, "
            f"mode={prompt_settings.mode}, top_k={top_k_value}, "
            f"index_version={index_version_str[:8] if index_version_str else 'none'}, "
            f"history_len={len(chat_history_text)}, history_sig={history_signature[:8]}]"
        )
        
        # Check cache (only if no history to avoid stale responses)
        cached_result: Optional[AnswerResponse] = None
        if not chat_history_text:
            # Diagnostic: log cache and service object IDs to detect per-request recreation
            cache_id = id(response_cache)
            service_id = id(self)
            logger.debug(f"[{request_id}] Cache check: cache_id={cache_id}, service_id={service_id}")
            
            cached_result = response_cache.get(cache_key)
            if cached_result:
                logger.info(f"[{request_id}] Cache HIT: key_hash={cache_key_hash}")
            else:
                logger.debug(f"[{request_id}] Cache MISS: key_hash={cache_key_hash}")
        
        if cached_result:
            logger.debug(f"[{request_id}] Cache hit for query")
            latency_ms = int((time_func() - start_time) * 1000)
            
            # Update conversation_id in cached response (don't modify original, create copy)
            from copy import deepcopy
            cached_result_copy = deepcopy(cached_result)
            cached_result_copy.conversation_id = conversation_id
            cached_result_copy.metrics.cache_hit = True  # Ensure cache_hit flag is set
            
            # Save to conversation history if DB available
            await self.conversation_service.append_message(conversation_id, "user", request.question)
            await self.conversation_service.append_message(conversation_id, "assistant", cached_result_copy.answer)
            
            return cached_result_copy
        
        # Get template string (use effective_preset override)
        template_str = get_prompt_template_content(prompt_settings, preset_override=effective_preset)
        
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

