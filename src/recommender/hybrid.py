"""
Hybrid Recommender combining Content-Based Cosine Similarity with Global Training Popularity.

Score formulation:
    Score(u, i) = (1 - alpha) * C(u, i) + alpha * P(i)

where:
    - C(u, i) = clip(dot(feature_i, user_vector), 0.0, 1.0)
    - P(i) is the min-max normalized log1p popularity count in [0.0, 1.0]
    - alpha in [0.0, 1.0]
"""
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

class HybridRecommender:
    """
    Hybrid recommender blending user-profile content similarity and global popularity.
    """
    
    def __init__(
        self,
        feature_matrix: np.ndarray,
        train_counts: Dict[int, int],
        alpha: float = 0.3,
    ):
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be in range [0.0, 1.0], got {alpha}")

        if feature_matrix.ndim != 2:
            raise ValueError("feature_matrix must be a 2D numpy array.")
        
        self.feature_matrix = feature_matrix
        self.num_items = feature_matrix.shape[0]
        self.train_counts = train_counts
        self.alpha = float(alpha)
        
        self.normalized_pop_scores = self._compute_normalized_popularity()
        
    
    def _compute_normalized_popularity(self) -> np.ndarray:
        """
        Computes log1p of item counts, then applies min-max scaling to [0, 1].
        """
        
        raw_counts = np.array([self.train_counts.get(i, 0) for i in range(self.num_items)], dtype=float)

        log_counts = np.log1p(raw_counts)
        
        min_val = float(log_counts.min())
        max_val = float(log_counts.max())
        
        if np.isclose(max_val, min_val):
            return np.zeros(self.num_items, dtype=float)
        
        return (log_counts - min_val) / (max_val - min_val)
    
    def recommend(
        self,
        user_vector: np.ndarray,
        top_k: int = 10,
        exclude_indices: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """
        Generates top_k hybrid recommendations.
        """
        if top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got {top_k}")

        if user_vector.ndim != 1 or user_vector.shape[0] != self.feature_matrix.shape[1]:
            raise ValueError(
                f"user_vector shape {user_vector.shape} must be "
                f"({self.feature_matrix.shape[1]},)."
            )

            
        content_scores = np.clip(self.feature_matrix @ user_vector, 0.0, 1.0)

        
        hybrid_scores = (
            (1.0 - self.alpha) * content_scores
            + self.alpha * self.normalized_pop_scores
        )
        
        valid_mask = np.ones(self.num_items, dtype=bool)
        if exclude_indices is not None and len(exclude_indices) > 0:
            excluded = np.asarray(exclude_indices, dtype=np.int64)
            excluded = excluded[(excluded >= 0) & (excluded < self.num_items)]
            valid_mask[excluded] = False
            
            
        num_valid = int(valid_mask.sum())
        if num_valid == 0:
            return pd.DataFrame(columns=["item_index", "similarity_score"])
                    
        
        scores = np.where(valid_mask, hybrid_scores, -np.inf)
        n = scores.shape[0]
        k = min(top_k, num_valid)
        
        cutoff = np.partition(scores, n - k)[n - k]
        candidates = np.flatnonzero(scores >= cutoff)
        
        order = np.lexsort((candidates, -scores[candidates]))[:k]
        top = candidates[order]
        
        return pd.DataFrame(
            {
                "item_index": top.astype(int),
                "similarity_score": scores[top].astype(float),
                
            }
        )
            