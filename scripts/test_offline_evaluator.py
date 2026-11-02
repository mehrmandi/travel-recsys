"""
Integration tests for OfflineEvaluator using last-unseen-venue holdout.
"""
from src.evaluation.evaluator import (
    OfflineEvaluator,
    build_last_unseen_evaluation_cases,
)
from src.recommender.user_profile import UserProfileRecommender
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def dummy_recommender():
    places_df = pd.DataFrame({
        "name_en": [
            "Louvre", "Orsay", "Central Park", "Versailles", "Pompidou"
        ],
        "category": ["museum", "museum", "park", "palace", "museum"],
        "city_en": ["Paris", "Paris", "New York", "Paris", "Paris"],
    })

    feature_matrix = np.array([
        [1.0, 0.0, 0.0],  # 0: Louvre
        [1.0, 0.0, 0.0],  # 1: Orsay
        [0.0, 1.0, 0.0],  # 2: Central Park
        [0.0, 0.0, 1.0],  # 3: Versailles
        [1.0, 0.0, 0.0],  # 4: Pompidou
    ], dtype=np.float32)

    return UserProfileRecommender(
        places_df=places_df,
        feature_matrix=feature_matrix,
    )


def test_last_unseen_case_uses_only_history_before_target():
    interactions = {"user_1": [0, 1, 0, 1, 4, 0]}

    cases = build_last_unseen_evaluation_cases(
        user_interactions=interactions,
        min_interactions=5,
    )

    assert len(cases) == 1
    case = cases[0]
    assert case.target_index == 4
    assert list(case.train_history) == [0, 1, 0, 1]
    assert case.target_index not in case.train_history


def test_offline_evaluator_last_unseen_holdout(dummy_recommender):
    evaluator = OfflineEvaluator(
        recommender=dummy_recommender,
        k_list=[1, 2],
    )

    interactions = {
        "user_1": [0, 1, 0, 1, 4, 0],
        "user_2": [0, 1, 0, 1, 0],
        "user_3": [0, 2, 0, 2],
    }

    results = evaluator.evaluate_last_unseen_holdout(
        interactions,
        min_interactions=5,
    )

    assert len(results) == 2  # user_1 و MEAN
    assert set(results["user_id"]) == {"user_1", "MEAN"}

    user_row = results.loc[results["user_id"] == "user_1"].iloc[0]
    mean_row = results.loc[results["user_id"] == "MEAN"].iloc[0]

    assert user_row["recall@1"] == 1.0
    assert user_row["recall@2"] == 1.0
    assert user_row["mrr@1"] == 1.0
    assert np.isclose(user_row["precision@2"], 0.5)

    assert mean_row["recall@1"] == 1.0
    assert mean_row["mrr@1"] == 1.0


def test_offline_evaluator_no_eligible_users_raises(dummy_recommender):
    evaluator = OfflineEvaluator(
        recommender=dummy_recommender,
        k_list=[2],
    )

    interactions = {
        "too_short": [0, 1, 0, 1],
        "no_late_new_venue": [0, 1, 0, 1, 0],
    }

    with pytest.raises(ValueError, match="No eligible users found"):
        evaluator.evaluate_last_unseen_holdout(
            interactions,
            min_interactions=5,
        )
