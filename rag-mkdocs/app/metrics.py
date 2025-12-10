"""
Метрики Prometheus для мониторинга RAG-сервиса.
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
    logger.warning("prometheus_client не установлен, метрики недоступны")

# Инициализация метрик (только если Prometheus доступен)
if PROMETHEUS_AVAILABLE:
    # Счетчики
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
    
    # Гистограммы (latency)
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
    # Заглушки для случая, когда Prometheus недоступен
    query_requests_total = None
    update_index_requests_total = None
    rate_limit_hits_total = None
    query_latency_seconds = None
    update_index_duration_seconds = None
    documents_in_index = None
    chunks_in_index = None


def get_metrics_response():
    """Возвращает метрики в формате Prometheus."""
    if not PROMETHEUS_AVAILABLE:
        return "Prometheus client not available", "text/plain"
    return generate_latest(), CONTENT_TYPE_LATEST


def update_index_metrics(documents_count: int, chunks_count: int):
    """Обновляет метрики индекса."""
    if PROMETHEUS_AVAILABLE and documents_in_index is not None:
        documents_in_index.set(documents_count)
        chunks_in_index.set(chunks_count)

