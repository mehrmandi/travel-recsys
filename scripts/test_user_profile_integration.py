"""
Integration tests: UserProfileBuilder + UserProfileRecommender

Scenarios:
1.  End-to-end pipeline with recency dominance, city filter where the excluded
    city sits in the FIRST row (to catch the "filter-then-reindex" bug class),
    exclusions, and top_k > candidate pool.
1b. Parametrized city filter robustness: "tehran", "TEHRAN", " Tehran ".
2.  Auto-generation of `item_index` when the source DataFrame lacks the column;
    tied scores -> only the top rank is asserted (no sort-stability dependence).
3.  Non-existent city filter returns an empty DataFrame with intact schema.
4.  Unfiltered pipeline with fully explicit ranking order.
5a. Out-of-bounds `exclude_indices` raises IndexError.
5b. `top_k <= 0` raises ValueError.

Contract tests (1b whitespace case, 5a, 5b)
-------------------------------------------
These encode behavior that `recommend` may not implement yet:
- city strings are normalized via str.strip().str.lower() before matching
- `exclude_indices` bounds and `top_k > 0` are validated by `recommend` itself

While CONTRACT_IMPLEMENTED is False they are marked xfail(strict=True), so:
- they do not break the suite today, and
- the moment `recommend` honors the contract they XPASS (= fail loudly), which
  tells you to flip CONTRACT_IMPLEMENTED to True and turn them into normal tests.

Note: Test 1 already relies on case-insensitive matching ("tehran" vs "Tehran"),
so the "TEHRAN" case is NOT treated as a contract test.
"""

import numpy as np
import pandas as pd
import pytest

from src.recommender.user_profile import (
    UserProfileBuilder,
    UserProfileRecommender,
)

# Flip to True once `recommend` implements whitespace normalization and
# input validation (top_k > 0, exclude_indices bounds).
CONTRACT_IMPLEMENTED = True

xfail_contract = pytest.mark.xfail(
    condition=not CONTRACT_IMPLEMENTED,
    strict=True,
    reason="Contract not yet implemented in `recommend`",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def assert_l2_normalized(vector: np.ndarray) -> None:
    """Assert that a non-zero vector has unit L2 norm."""
    assert vector.ndim == 1, f"Expected 1D vector, got shape {vector.shape}"
    norm = np.linalg.norm(vector)
    assert np.isclose(
        norm, 1.0, atol=1e-6), f"Expected unit L2 norm, got {norm:.6f}"


def assert_sorted_descending(scores: np.ndarray) -> None:
    """Assert scores are sorted non-increasingly (descending, ties allowed)."""
    scores = np.asarray(scores)
    if scores.size < 2:
        return  # nothing to compare; safe for top_k=1 or empty outputs
    assert np.all(scores[:-1] >= scores[1:]), (
        "Expected similarity_score to be sorted descending (non-increasing)."
    )


def make_feature_matrix() -> np.ndarray:
    """
    3D feature space. The Shiraz item is placed at index 0 (FIRST row) so that
    if recommend() filters by city first and then rebuilds item_index from the
    new positions, indices shift by one and the tests fail loudly.
    """
    return np.asarray(
        [
            [1.0, 1.0, 1.0],  # Item 0  (Shiraz -> must be city-filtered)
            [1.0, 0.0, 0.0],  # Item 1  (Tehran, visited, older)
            [0.0, 1.0, 0.0],  # Item 2  (Tehran, viewed, newer)
            [1.0, 1.0, 0.0],  # Item 3  (Tehran candidate: aligns with profile)
            [0.0, 0.0, 1.0],  # Item 4  (Tehran candidate: orthogonal)
        ],
        dtype=np.float32,
    )


def make_places_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "name_en": ["Shiraz Place", "Place 1", "Place 2", "Place 3", "Place 4"],
            "category": ["museum", "museum", "museum", "park", "park"],
            "city_en": ["Shiraz", "Tehran", "Tehran", "Tehran", "Tehran"],
        }
    )


# ---------------------------------------------------------------------------
# Test 1: Full pipeline (explicit weights, recency dominance, filters)
# ---------------------------------------------------------------------------

def test_full_pipeline_weighted_profile_recommend_exclude_and_city_filter():
    """
    Weights: visit=3.0 (item 1, delta=10d), view=1.0 (item 2, delta=0d).
    decay_rate=0.2  =>  w_total(item1) = 3*exp(-2) ≈ 0.406 < 1.0 = w_total(item2).
    So dimension 1 (newer, viewed) strictly dominates dimension 0 (older, visited).

    Expected profile ≈ [0.376, 0.926, 0].
    Cosine scores: item 3 ≈ 0.921, item 4 = 0.0 (item 0 / Shiraz ≈ 0.752 if not filtered).
    """
    feature_matrix = make_feature_matrix()
    places_df = make_places_df()

    interaction_indices = [1, 2]
    user_profile = UserProfileBuilder.build_from_history(
        feature_matrix=feature_matrix,
        interaction_indices=interaction_indices,
        interaction_types=["visit", "view"],
        timestamps=["2026-01-01", "2026-01-11"],
        interaction_weights_map={"visit": 3.0,
                                 "view": 1.0},  # explicit, no defaults
        decay_rate=0.2,
    )

    assert user_profile.shape == (3,)
    assert_l2_normalized(user_profile)

    # Recency dominance: newer item (dim1) strictly outweighs older item (dim0)
    assert user_profile[1] > user_profile[0], (
        f"Expected dim1 ({user_profile[1]:.4f}) > dim0 ({user_profile[0]:.4f})."
    )
    assert user_profile[2] == pytest.approx(0.0, abs=1e-6)

    recommender = UserProfileRecommender(
        places_df=places_df, feature_matrix=feature_matrix)
    recs = recommender.recommend(
        user_vector=user_profile,
        top_k=10,  # exceeds the remaining pool of 2
        exclude_indices=interaction_indices,
        city_filter="tehran",  # case-insensitive match against "Tehran"
    )

    # Schema & count
    expected_cols = {"item_index", "name_en",
                     "category", "city_en", "similarity_score"}
    assert expected_cols.issubset(set(recs.columns))
    # top_k > pool must not raise; exactly 2 candidates remain
    assert len(recs) == 2

    # City filter + exclusion policy. Shiraz is row 0, so any
    # "filter-then-reindex" bug would shift indices and break this assertion.
    assert (recs["city_en"].str.lower() == "tehran").all()
    assert set(recs["item_index"].tolist()) == {3, 4}
    assert not set(recs["item_index"]).intersection({1, 2})

    # Ranking: item 3 (aligned with profile) beats item 4 (orthogonal -> 0)
    assert recs["item_index"].tolist() == [3, 4]
    assert_sorted_descending(recs["similarity_score"].to_numpy())

    # Safe cosine bounds with float32 tolerance
    assert (recs["similarity_score"] >= -1e-6).all()
    assert (recs["similarity_score"] <= 1.0 + 1e-6).all()


# ---------------------------------------------------------------------------
# Test 1b: City filter robustness (case + whitespace), parametrized
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "city_input",
    [
        "tehran",
        "TEHRAN",
        " Tehran "

    ],
)
def test_city_filter_is_case_and_whitespace_insensitive(city_input):
    """The filter must normalize case and strip surrounding whitespace."""
    recommender = UserProfileRecommender(
        places_df=make_places_df(), feature_matrix=make_feature_matrix()
    )
    recs = recommender.recommend(
        user_vector=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        top_k=10,
        city_filter=city_input,
    )
    assert (recs["city_en"].str.lower() == "tehran").all()
    assert set(recs["item_index"].tolist()) == {1, 2, 3, 4}


# ---------------------------------------------------------------------------
# Test 2: Auto-generation of item_index from raw DataFrame
# ---------------------------------------------------------------------------

def test_recommender_synthesizes_item_index_if_missing():
    feature_matrix = np.eye(3, dtype=np.float32)
    places_df = pd.DataFrame(
        {"name_en": ["A", "B", "C"], "city_en": ["Isfahan"] * 3}
    )
    assert "item_index" not in places_df.columns

    recommender = UserProfileRecommender(places_df, feature_matrix)
    recs = recommender.recommend(
        user_vector=np.array([1.0, 0.0, 0.0], dtype=np.float32), top_k=3
    )

    assert "item_index" in recs.columns
    assert set(recs["item_index"].tolist()) == {0, 1, 2}
    assert recs.iloc[0]["item_index"] == 0
    # Items 1 & 2 tie at 0.0: only the top rank is asserted, avoiding
    # dependence on sort stability for tied scores.


# ---------------------------------------------------------------------------
# Test 3: Non-existent city filter -> empty DataFrame with intact schema
# ---------------------------------------------------------------------------

def test_non_existent_city_filter_returns_empty_dataframe_with_schema():
    recommender = UserProfileRecommender(
        places_df=make_places_df(), feature_matrix=make_feature_matrix()
    )
    recs = recommender.recommend(
        user_vector=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        top_k=5,
        city_filter="Tabriz",
    )
    assert isinstance(recs, pd.DataFrame)
    assert len(recs) == 0
    assert "item_index" in recs.columns
    assert "similarity_score" in recs.columns


# ---------------------------------------------------------------------------
# Test 4: Unfiltered pipeline with explicit ranking order
# ---------------------------------------------------------------------------

def test_full_pipeline_without_filters_keeps_all_candidates_and_ranks():
    feature_matrix = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    places_df = pd.DataFrame(
        {"name_en": ["A", "B", "C"], "city_en": ["Tehran", "Tehran", "Shiraz"]}
    )

    # Profile built from item 0 only -> [1, 0]
    profile = UserProfileBuilder.build_from_history(
        feature_matrix=feature_matrix, interaction_indices=[0]
    )
    assert_l2_normalized(profile)
    assert np.allclose(profile, [1.0, 0.0], atol=1e-6)

    recommender = UserProfileRecommender(places_df, feature_matrix)
    recs = recommender.recommend(user_vector=profile, top_k=3)

    assert len(recs) == 3
    assert set(recs["city_en"].unique()) == {"Tehran", "Shiraz"}
    assert 0 in recs["item_index"].values  # nothing excluded

    # Item 0 is identical to the profile (cos = 1.0)
    # > item 2 (cos ≈ 0.707) > item 1 (cos = 0.0). All scores are distinct,
    # so the full order is deterministic.
    assert recs["item_index"].tolist() == [0, 2, 1]
    assert_sorted_descending(recs["similarity_score"].to_numpy())


# ---------------------------------------------------------------------------
# Test 5a / 5b: Input validation in `recommend` (contract-based, split so one
# failure cannot hide the other)
# ---------------------------------------------------------------------------

@xfail_contract
def test_recommend_rejects_out_of_bounds_exclude_indices():
    recommender = UserProfileRecommender(
        places_df=make_places_df(), feature_matrix=make_feature_matrix()
    )
    user_vector = np.array([0.5, 0.5, 0.0], dtype=np.float32)

    with pytest.raises(IndexError):
        recommender.recommend(user_vector=user_vector,
                              top_k=3, exclude_indices=[99])


@xfail_contract
def test_recommend_rejects_non_positive_top_k():
    recommender = UserProfileRecommender(
        places_df=make_places_df(), feature_matrix=make_feature_matrix()
    )
    user_vector = np.array([0.5, 0.5, 0.0], dtype=np.float32)

    with pytest.raises(ValueError):
        recommender.recommend(user_vector=user_vector, top_k=0)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
