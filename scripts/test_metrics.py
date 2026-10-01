"""
Unit tests for ranking and retrieval metrics in isolation.
"""
import numpy as np
from src.evaluation.metrics import hit_at_k, recall_at_k, precision_at_k, reciprocal_rank_at_k

def test_hit_at_k():
    recs = [10, 20, 30, 40, 50]
    
    # Target present within k
    assert hit_at_k(recs, target_index=10, k=1) == 1.0
    assert hit_at_k(recs, target_index=30, k=3) == 1.0
    
    # Target present after k
    assert hit_at_k(recs, target_index=30, k=2) == 0.0
    
    # Target not present in recommendations at all
    assert hit_at_k(recs, target_index=99, k=5) == 0.0
    

def test_recall_at_k():
    recs = ["louvre", "orsay", "pompidou"]

    # In single-item holdout, Recall@K is equivalent to Hit@K
    assert recall_at_k(recs, target_index="orsay", k=2) == 1.0
    assert recall_at_k(recs, target_index="pompidou", k=2) == 0.0
    
def test_precision_at_k():
    recs = [101, 102, 103, 104]
    
    # Target at rank 1: Precision@1 = 1/1 = 1.0
    assert np.isclose(precision_at_k(recs, target_index=101, k=1), 1.0)
    
    # Target at rank 2: Precision@2 = 1/2 = 0.5
    assert np.isclose(precision_at_k(recs, target_index=102, k=2), 0.5)

    # Target at rank 3: Precision@4 = 1/4 = 0.25
    assert np.isclose(precision_at_k(recs, target_index=103, k=4), 0.25)

    # Target absent: Precision@4 = 0.0
    assert precision_at_k(recs, target_index=999, k=4) == 0.0

    
def test_reciprocal_rank_at_k():
    recs = [500, 501, 502, 503, 504]
    
    # Rank 1 -> 1 / 1 = 1.0
    assert np.isclose(reciprocal_rank_at_k(recs, target_index=500, k=5), 1.0)

    # Rank 3 -> 1 / 3
    assert np.isclose(reciprocal_rank_at_k(
        recs, target_index=502, k=5), 1.0 / 3.0)

    # Rank 4 with k=3 cutoff -> not found -> 0.0
    assert reciprocal_rank_at_k(recs, target_index=503, k=3) == 0.0

    # Target completely absent -> 0.0
    assert reciprocal_rank_at_k(recs, target_index=999, k=5) == 0.0


if __name__ == "__main__":
    test_hit_at_k()
    test_recall_at_k()
    test_precision_at_k()
    test_reciprocal_rank_at_k()
    print("✅ All isolated metrics unit tests passed successfully!")
        