"""
Ranking and Retrieval Metrics for Offline Evaluation.
"""

from typing import Sequence, Union

def hit_at_k(recommended_indices: Sequence[Union[int, str]], target_index: Union[int, str], k: int) -> float:
    """
    Check if target item exists in top-k recommendations (Binary 0.0 or 1.0).
    """
    if k <= 0:
        raise ValueError("k must be a positive integer.")
    
    top_k_recs = recommended_indices[:k]
    return 1.0 if target_index in top_k_recs else 0.0

def recall_at_k(recommended_indices: Sequence[Union[int, str]], target_index: Union[int, str], k: int) -> float:
    """
    Compute Recall@K for single-item Leave-One-Out evaluation.
    In LOO with 1 ground-truth target, Recall@K is equivalent to Hit@K.
    """
    if k <= 0:
        raise ValueError("k must be a positive integer.")
    
    return hit_at_k(recommended_indices, target_index, k)

def precision_at_k(recommended_indices: Sequence[Union[int, str]], target_index: Union[int, str], k: int) -> float:
    """
    Compute Precision@K for single-item Leave-One-Out evaluation.
    Formula: Hit@K / K
    """
    if k <= 0:
        raise ValueError("k must be a positive integer.")
    
    return hit_at_k(recommended_indices, target_index, k) / float(k)


def reciprocal_rank_at_k(
    recommended_indices: Sequence[Union[int, str]],
    target_index: Union[int, str],
    k: int
) -> float:
    """
    Compute Reciprocal Rank (RR) within top-k recommendations.
    Returns 1 / (rank + 1) if found, else 0.0. Rank is 1-indexed.
    """
    if k <= 0:
        raise ValueError("k must be a positive integer.")
    
    top_k_recs = recommended_indices[:k]
    if target_index in top_k_recs:
        rank = top_k_recs.index(target_index) + 1
        return 1.0 / rank
    return 0.0