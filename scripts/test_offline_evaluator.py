"""
Integration unit tests for OfflineEvaluator using Last-Item Holdout.
"""
import numpy as np
import pandas as pd
import pytest
from src.recommender.user_profile import UserProfileRecommender
from src.evaluation.evaluator import OfflineEvaluator

@pytest.fixture
def dummy_recommender():
    place_df = pd.DataFrame({
        "name_en": ["Louvre", "Orsay", "Central Park", "Versailles", "Pompidou"],
        "category": ["museum", "museum", "park", "palace", "museum"],
        "city_en": ["Paris", "Paris", "New York", "Paris", "Paris"]
    })
    
    # One-hot representation: [museum, park, palace]
    feature_matrix = np.array([
        [1.0, 0.0, 0.0],  # 0: Louvre (museum)
        [1.0, 0.0, 0.0],  # 1: Orsay (museum)
        [0.0, 1.0, 0.0],  # 2: Central Park (park)
        [0.0, 0.0, 1.0],  # 3: Versailles (palace)
        [1.0, 0.0, 0.0],  # 4: Pompidou (museum)
        
    ], dtype=np.float32)
    
    return UserProfileRecommender(places_df=place_df, feature_matrix=feature_matrix)

def test_offline_evaluator_last_item_holdout(dummy_recommender):
    evaluator = OfflineEvaluator(recommender=dummy_recommender, k_list=[1, 2])
    
    interactions = {
        # User 1: train=[0, 1] (museums), target=4 (Pompidou, museum)
        "user_1": [0, 1, 4],
        # User 2: less than min_interactions=3 -> should be skipped
        "user_2": [0, 2]  
    }
    
    results = evaluator.evaluate_last_item_holdout(interactions, min_interactions=3)
    
    # User 2 skipped -> 1 user row + 1 MEAN summary row = 2 rows
    assert len(results) == 2

    user_row = results[results["user_id"] == "user_1"].iloc[0]
    mean_row = results[results["user_id"] == "MEAN"].iloc[0]

    # Target 4 (Pompidou) must rank #1 among unvisited items (similarity=1.0)
    assert user_row["recall@1"] == 1.0
    assert user_row["recall@2"] == 1.0
    assert user_row["mrr@1"] == 1.0
    assert np.isclose(user_row["precision@2"], 0.5)

    # MEAN row reflects single user values
    assert mean_row["recall@1"] == 1.0
    assert mean_row["mrr@1"] == 1.0
    
def test_offline_evaluator_insufficient_interactions_raises_error(dummy_recommender):
    evaluator = OfflineEvaluator(recommender=dummy_recommender, k_list=[2])
    
    interactions = {
        "user_1": [0, 1]
    }
    
    with pytest.raises(ValueError, match="No eligible users found"):
        evaluator.evaluate_last_item_holdout(interactions, min_interactions=3)


if __name__ == "__main__":
    pytest.main([__file__])
