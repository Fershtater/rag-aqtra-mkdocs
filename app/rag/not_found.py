"""
Not found detection module: logic for strict mode + no sources.
"""

import logging
import os
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

NOT_FOUND_SCORE_THRESHOLD = float(os.getenv("NOT_FOUND_SCORE_THRESHOLD", "0.20"))


def check_not_found(
    sources: List,
    threshold: Optional[float] = None
) -> bool:
    """
    Check if sources indicate "not found" based on scores.
    
    Args:
        sources: List of sources (with optional score in metadata)
        threshold: Score threshold (default from NOT_FOUND_SCORE_THRESHOLD env)
        
    Returns:
        True if no relevant sources found, False otherwise
    """
    if threshold is None:
        threshold = NOT_FOUND_SCORE_THRESHOLD
    
    if not sources:
        return True
    
    # Extract scores from sources
    scores = []
    for source in sources:
        if hasattr(source, "metadata") and source.metadata:
            score = source.metadata.get("score")
            if score is not None:
                try:
                    scores.append(float(score))
                except (ValueError, TypeError):
                    pass
    
    if scores:
        top_score = max(scores)
        return top_score < threshold
    
    # If no scores available, assume sources might be relevant
    return False


def check_not_found_from_scores(
    scores: List[float],
    threshold: Optional[float] = None
) -> bool:
    """
    Check if scores indicate "not found".
    
    Args:
        scores: List of similarity scores
        threshold: Score threshold (default from NOT_FOUND_SCORE_THRESHOLD env)
        
    Returns:
        True if all scores below threshold, False otherwise
    """
    if threshold is None:
        threshold = NOT_FOUND_SCORE_THRESHOLD
    
    if not scores:
        return True
    
    max_score = max(scores)
    return max_score < threshold

