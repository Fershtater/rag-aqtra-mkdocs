"""
Prometheus metrics for RAG service monitoring.
"""

import logging
import time
from typing import Optional

try:
    from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("prometheus_client not installed, metrics unavailable")

# Initialize metrics (only if Prometheus is available)
if PROMETHEUS_AVAILABLE:
    # Counters
    query_requests_total = Counter(
        'rag_query_requests_total',
        'Total number of /query requests',
        ['status']  # success, error
    )
    
    update_index_requests_total = Counter(
        'rag_update_index_requests_total',
        'Total number of /update_index requests',
        ['status']  # success, error
    )
    
    rate_limit_hits_total = Counter(
        'rag_rate_limit_hits_total',
        'Total number of rate limit hits',
        ['endpoint']  # query, update_index
    )
    
    # Histograms (latency)
    query_latency_seconds = Histogram(
        'rag_query_latency_seconds',
        'Query endpoint latency in seconds',
        buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
    )
    
    update_index_duration_seconds = Histogram(
        'rag_update_index_duration_seconds',
        'Update index operation duration in seconds',
        buckets=[10.0, 30.0, 60.0, 120.0, 300.0, 600.0]
    )
    
    # Stage histograms
    rag_retrieval_latency_seconds = Histogram(
        'rag_retrieval_latency_seconds',
        'Retrieval stage latency in seconds',
        ['endpoint'],
        buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
    )
    
    rag_prompt_render_latency_seconds = Histogram(
        'rag_prompt_render_latency_seconds',
        'Prompt rendering stage latency in seconds',
        ['endpoint'],
        buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
    )
    
    rag_llm_latency_seconds = Histogram(
        'rag_llm_latency_seconds',
        'LLM generation stage latency in seconds',
        ['endpoint'],
        buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
    )
    
    rag_ttft_seconds = Histogram(
        'rag_ttft_seconds',
        'Time to first token (TTFT) in seconds',
        ['endpoint'],
        buckets=[0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
    )
    
    # Embedding cache metrics
    rag_embedding_cache_hits_total = Counter(
        'rag_embedding_cache_hits_total',
        'Total number of embedding cache hits'
    )
    
    rag_embedding_cache_misses_total = Counter(
        'rag_embedding_cache_misses_total',
        'Total number of embedding cache misses'
    )
    
    # Gauges
    documents_in_index = Gauge(
        'rag_documents_in_index',
        'Number of documents in the FAISS index'
    )
    
    chunks_in_index = Gauge(
        'rag_chunks_in_index',
        'Number of chunks in the FAISS index'
    )
else:
    # Stubs for when Prometheus is unavailable
    query_requests_total = None
    update_index_requests_total = None
    rate_limit_hits_total = None
    query_latency_seconds = None
    update_index_duration_seconds = None
    rag_retrieval_latency_seconds = None
    rag_prompt_render_latency_seconds = None
    rag_llm_latency_seconds = None
    rag_ttft_seconds = None
    rag_embedding_cache_hits_total = None
    rag_embedding_cache_misses_total = None
    documents_in_index = None
    chunks_in_index = None


def get_metrics_response():
    """Returns metrics in Prometheus format."""
    if not PROMETHEUS_AVAILABLE:
        return "Prometheus client not available", "text/plain"
    return generate_latest(), CONTENT_TYPE_LATEST


def update_index_metrics(documents_count: int, chunks_count: int):
    """Updates index metrics."""
    if PROMETHEUS_AVAILABLE and documents_in_index is not None:
        documents_in_index.set(documents_count)
        chunks_in_index.set(chunks_count)

