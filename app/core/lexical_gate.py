"""
Lexical overlap gate for strict mode filtering.

Provides cheap keyword-based filtering to reject documents that don't contain
relevant keywords from the question, even if vector similarity is high.
"""

import logging
import re
from typing import Set

logger = logging.getLogger(__name__)

# Default English stopwords (minimal set)
DEFAULT_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
    "to", "was", "will", "with", "the", "this", "but", "they", "have",
    "had", "what", "said", "each", "which", "their", "time", "if",
    "up", "out", "many", "then", "them", "these", "so", "some", "her",
    "would", "make", "like", "into", "him", "has", "two", "more", "very",
    "after", "words", "long", "about", "than", "first", "been", "call",
    "who", "oil", "sit", "now", "find", "down", "day", "did", "get",
    "come", "made", "may", "part"
}

# Tokens to ignore (common domain-specific words that don't add meaning)
DEFAULT_IGNORE_TOKENS = {
    "aqtra", "app", "application", "platform", "configure", "how", "create",
    "what", "where", "when", "why", "which", "can", "should", "does",
    "work", "works", "working", "use", "using", "used", "do", "does",
    "done", "make", "makes", "made", "get", "gets", "got", "set", "sets",
    "step", "steps", "component", "components", "field", "fields", "model",
    "models", "data", "flow", "flows", "workflow", "workflows"
}


def extract_keywords(
    question: str,
    min_token_len: int = 4,
    stopwords: Set[str] = None,
    ignore_tokens: Set[str] = None
) -> Set[str]:
    """
    Extract meaningful keywords from question.
    
    Args:
        question: User question string
        min_token_len: Minimum token length to include
        stopwords: Set of stopwords to exclude (defaults to DEFAULT_STOPWORDS)
        ignore_tokens: Set of tokens to ignore (defaults to DEFAULT_IGNORE_TOKENS)
        
    Returns:
        Set of lowercase keywords
    """
    if stopwords is None:
        stopwords = DEFAULT_STOPWORDS
    if ignore_tokens is None:
        ignore_tokens = DEFAULT_IGNORE_TOKENS
    
    # Split by non-alphanumeric characters
    tokens = re.findall(r'\b\w+\b', question.lower())
    
    # Filter: min length, not in stopwords, not in ignore_tokens
    keywords = {
        token for token in tokens
        if len(token) >= min_token_len
        and token not in stopwords
        and token not in ignore_tokens
    }
    
    return keywords


def lexical_hits(doc_text: str, keywords: Set[str]) -> int:
    """
    Count unique keyword occurrences in document text.
    
    Args:
        doc_text: Document text to search
        keywords: Set of keywords to search for
        
    Returns:
        Number of unique keywords found in document
    """
    if not keywords or not doc_text:
        return 0
    
    doc_lower = doc_text.lower()
    hits = sum(1 for keyword in keywords if keyword in doc_lower)
    
    return hits


def apply_lexical_gate(
    docs_with_relevance: list,
    question: str,
    min_hits: int = 1,
    min_token_len: int = 4,
    stopwords: Set[str] = None,
    ignore_tokens: Set[str] = None
) -> list:
    """
    Apply lexical overlap gate to filter documents.
    
    Args:
        docs_with_relevance: List of (doc, relevance) tuples
        question: User question
        min_hits: Minimum number of keyword hits required
        min_token_len: Minimum token length for keywords
        stopwords: Set of stopwords to exclude
        ignore_tokens: Set of tokens to ignore
        
    Returns:
        Filtered list of (doc, relevance) tuples
    """
    if not docs_with_relevance:
        return []
    
    # Extract keywords from question
    keywords = extract_keywords(question, min_token_len, stopwords, ignore_tokens)
    
    if not keywords:
        # No keywords extracted: pass all docs (fallback to relevance only)
        logger.debug(f"Lexical gate: no keywords extracted, passing all {len(docs_with_relevance)} docs")
        return docs_with_relevance
    
    # Filter by lexical hits
    filtered = []
    for doc, relevance in docs_with_relevance:
        doc_text = doc.page_content if hasattr(doc, 'page_content') else str(doc)
        hits = lexical_hits(doc_text, keywords)
        
        if hits >= min_hits:
            filtered.append((doc, relevance))
        else:
            logger.debug(
                f"Lexical gate: doc rejected (hits={hits} < min={min_hits}, "
                f"keywords={keywords}, relevance={relevance:.3f})"
            )
    
    logger.debug(
        f"Lexical gate: {len(docs_with_relevance)} docs -> {len(filtered)} docs "
        f"(keywords={keywords}, min_hits={min_hits})"
    )
    
    return filtered

