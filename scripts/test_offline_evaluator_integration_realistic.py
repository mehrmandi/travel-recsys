"""
Realistic End-to-End Integration Test for OfflineEvaluator.

Verifies:
1. Profile is built strictly from training interactions (no data leakage).
2. Target item is absent from training history.
3. Training items are excluded from recommendation candidates.
4. Recommendation ordering is mathematically consistent with Cosine Similarity.
5. Calculated evaluation metrics strictly match the true rank of the target item.
"""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.evaluator import OfflineEvaluator
from src.evaluation.metrics import precision_at_k, recall_at_k, reciprocal_rank_at_k
from src.recommender.user_profile import UserProfileBuilder, UserProfileRecommender

def test_last_item_holdout_end_to_end_realistic_integration(monkeypatch):
    # ------------------------------------------------------------------
    # 1. Deterministic catalog
    # ------------------------------------------------------------------
    places_df = pd.DataFrame(
        {
            "item_index": [0, 1, 2, 3, 4],
            "name_en": [
                "Training item 0",
                "Training item 1",
                "Target item 2",
                "Competitor item 3",
                "Opposite item 4",
            ],
            "city_en": ["Tehran"] * 5,
        }   
    )
    
    feature_matrix = np.asarray(
        [
            [1.0, 0.0],        # item 0: training
            [0.0, 1.0],        # item 1: training
            [0.9, 0.4358899],  # item 2: target
            [0.6, 0.8],        # item 3: competitor
            [-1.0, 0.0],       # item 4: opposite
        ],
        dtype=np.float32,
    )
    
    recommender = UserProfileRecommender(
        places_df=places_df,
        feature_matrix=feature_matrix,
    )
    
    evaluator = OfflineEvaluator(
        recommender=recommender,
        k_list=[2, 5],
    )
    
    # ------------------------------------------------------------------
    # 2. Last-item holdout interaction history
    # ------------------------------------------------------------------
    
    user_interactions ={
        "user-1": [0, 1, 2],
    }
    
    # ------------------------------------------------------------------
    # 3. Capture calls to verify integration contracts
    # ------------------------------------------------------------------
    
    captured ={
        "builder_called": 0,
        "builder_indices": None,
        "recommend_called": 0,
        "recommend_exclude_indices": None,
        "recommendations": None,
    }
    
    original_build = UserProfileBuilder.build_from_history
    original_recommend = recommender.recommend
    
    def build_spy(
        feature_matrix,
        interaction_indices,
        interaction_types=None,
        timestamps=None,
        interaction_weights_map=None,
        decay_rate=0.0,
    ):
        captured["builder_called"] += 1
        captured["builder_indices"] = list(interaction_indices)
    
    
        assert list(interaction_indices) == [0, 1]
        assert 2 not in interaction_indices
        
        return original_build(
            feature_matrix=feature_matrix,
            interaction_indices=interaction_indices,
            interaction_types=interaction_types,
            timestamps=timestamps,
            interaction_weights_map=interaction_weights_map,
            decay_rate=decay_rate,
        )   
        
    def recommend_spy(
        user_vector,
        top_k,
        exclude_indices=None,
        city_filter=None,
    ):
        
        captured["recommend_called"] += 1
        captured["recommend_exclude_indices"] = list(exclude_indices)
        
        assert set(exclude_indices) == {0, 1}
        
        result = original_recommend(
            user_vector=user_vector,
            top_k=top_k,
            exclude_indices=exclude_indices,
            city_filter=city_filter,
        )
    
        captured["recommendations"] = result["item_index"].tolist()
        return result
    
    
    monkeypatch.setattr(
        UserProfileBuilder,
        "build_from_history",
        staticmethod(build_spy),
    )
    
    monkeypatch.setattr(recommender, "recommend", recommend_spy)
    
    # ------------------------------------------------------------------
    # 4. Run the actual evaluator
    # ------------------------------------------------------------------

    result_df = evaluator.evaluate_last_item_holdout(
        user_interactions=user_interactions,
        min_interactions=3
    )
    
    # The evaluator creates one user row and one MEAN row.
    assert len(result_df) == 2
    assert result_df.iloc[0]["user_id"] == "user-1"
    assert result_df.iloc[1]["user_id"] == "MEAN"

    # ------------------------------------------------------------------
    # 5. Verify training-only profile construction and exclusion
    # ------------------------------------------------------------------
    assert captured["builder_called"] == 1
    assert captured["builder_indices"] == [0, 1]

    assert captured["recommend_called"] == 1
    assert captured["recommend_exclude_indices"] == [0, 1]

    recommendations = captured["recommendations"]

    assert 0 not in recommendations
    assert 1 not in recommendations

    # ------------------------------------------------------------------
    # 6. Independently verify cosine ranking
    # ------------------------------------------------------------------

    expected_profile = np.asarray([1.0, 1.0], dtype=np.float32)
    expected_profile /= np.linalg.norm(expected_profile)
    
    candidate_indices = [2, 3, 4]
    expected_scores = {
        index: float(np.dot(expected_profile, feature_matrix[index]))
        for index in candidate_indices
    }
    
    assert expected_scores[3] > expected_scores[2]
    assert expected_scores[2] > expected_scores[4]
    
    # The target must therefore be ranked second.
    assert recommendations[:3] == [3, 2, 4]

    target_rank = recommendations.index(2) + 1
    assert target_rank == 2

    # ------------------------------------------------------------------
    # 7. Verify metrics independently from evaluator output
    # ------------------------------------------------------------------
    
    expected_metrics = {
        "recall@2": recall_at_k(recommendations, 2, 2),
        "precision@2": precision_at_k(recommendations, 2, 2),
        "mrr@2": reciprocal_rank_at_k(recommendations, 2, 2),
        "recall@5": recall_at_k(recommendations, 2, 5),
        "precision@5": precision_at_k(recommendations, 2, 5),
        "mrr@5": reciprocal_rank_at_k(recommendations, 2, 5),
    }
    
    user_row = result_df.iloc[0]
    
    for metric_name, expected_value in expected_metrics.items():
        assert user_row[metric_name] == pytest.approx(expected_value)

    # Target rank = 2:
    # Recall@2 = 1
    # Precision@2 = 1/2
    # MRR@2 = 1/2
    # Recall@5 = 1
    # Precision@5 = 1/5
    # MRR@5 = 1/2
    assert user_row["recall@2"] == pytest.approx(1.0)
    assert user_row["precision@2"] == pytest.approx(0.5)
    assert user_row["mrr@2"] == pytest.approx(0.5)

    assert user_row["recall@5"] == pytest.approx(1.0)
    assert user_row["precision@5"] == pytest.approx(0.2)
    assert user_row["mrr@5"] == pytest.approx(0.5)

    # The MEAN row must match the only evaluated user.
    mean_row = result_df.iloc[1]

    assert mean_row["recall@2"] == pytest.approx(1.0)
    assert mean_row["precision@2"] == pytest.approx(0.5)
    assert mean_row["mrr@2"] == pytest.approx(0.5)

    assert mean_row["recall@5"] == pytest.approx(1.0)
    assert mean_row["precision@5"] == pytest.approx(0.2)
    assert mean_row["mrr@5"] == pytest.approx(0.5)
