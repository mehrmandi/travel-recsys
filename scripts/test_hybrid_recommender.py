"""
Unit Tests for HybridRecommender.

Covers input validation checks, boundary alpha conditions (pure content vs. 
pure popularity), exclusion filters, and deterministic tie-breaking.
"""


import pytest
import numpy as np
import pandas as pd
from src.recommender.hybrid import HybridRecommender
from src.recommender.most_popular import PopularRecommender


@pytest.fixture
def sample_feature_matrix():
    return np.array([
        [1.0, 0.0, 0.0],  # item 0
        [0.0, 1.0, 0.0],  # item 1
        [0.8, 0.6, 0.0],  # item 2
        [0.0, 0.0, 1.0],  # item 3
    ], dtype=float)


@pytest.fixture
def sample_train_counts():
    return {0: 5, 1: 50, 3: 20}


def test_input_validations(sample_feature_matrix, sample_train_counts):
    with pytest.raises(ValueError, match="alpha must be in range"):
        HybridRecommender(sample_feature_matrix,
                          sample_train_counts, alpha=-0.1)

    with pytest.raises(ValueError, match="alpha must be in range"):
        HybridRecommender(sample_feature_matrix,
                          sample_train_counts, alpha=1.1)

    rec = HybridRecommender(sample_feature_matrix,
                            sample_train_counts, alpha=0.5)

    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        rec.recommend(np.array([1.0, 0.0, 0.0]), top_k=0)

    with pytest.raises(ValueError, match="shape"):
        rec.recommend(np.array([1.0, 0.0]), top_k=2)


def test_boundary_alpha_zero_pure_content(sample_feature_matrix, sample_train_counts):
    rec = HybridRecommender(sample_feature_matrix,
                            sample_train_counts, alpha=0.0)
    user_vector = np.array([1.0, 0.0, 0.0])

    df = rec.recommend(user_vector, top_k=4)

    assert list(df["item_index"]) == [0, 2, 1, 3]
    assert np.isclose(df.iloc[0]["similarity_score"], 1.0)
    assert np.isclose(df.iloc[1]["similarity_score"], 0.8)


def test_boundary_alpha_one_pure_popularity(sample_feature_matrix, sample_train_counts):
    hybrid_rec = HybridRecommender(
        sample_feature_matrix, sample_train_counts, alpha=1.0
    )

    num_items = sample_feature_matrix.shape[0]
    sample_places_df = pd.DataFrame({
        "item_index": np.arange(num_items),
        "name_en": [f"Place {i}" for i in range(num_items)],
    })
    pop_rec = PopularRecommender(sample_places_df, sample_train_counts)

    user_vector = np.array([1.0, 0.0, 0.0])
    df_hybrid = hybrid_rec.recommend(user_vector, top_k=4)
    df_pop = pop_rec.recommend(top_k=4)

    assert list(df_hybrid["item_index"]) == list(df_pop["item_index"])



def test_exclude_indices(sample_feature_matrix, sample_train_counts):
    rec = HybridRecommender(sample_feature_matrix,
                            sample_train_counts, alpha=0.5)
    user_vector = np.array([1.0, 0.0, 0.0])

    df = rec.recommend(user_vector, top_k=2, exclude_indices=[0])

    assert 0 not in list(df["item_index"])
    assert len(df) == 2


def test_deterministic_tie_breaking():
    zero_counts = {0: 10, 1: 10}
    matrix = np.array([
        [1.0, 0.0],
        [1.0, 0.0],
    ])
    rec = HybridRecommender(matrix, zero_counts, alpha=0.5)
    df = rec.recommend(np.array([1.0, 0.0]), top_k=2)

    assert list(df["item_index"]) == [0, 1]


def test_partition_tie_boundry():
    feature_matrix = np.zeros((5, 2))
    train_counts = {i: 10 for i in range(5)}
    rec = HybridRecommender(feature_matrix, train_counts, alpha=1.0)

    df = rec.recommend(np.zeros(2), top_k=2)
    assert list(df["item_index"]) == [0, 1]
