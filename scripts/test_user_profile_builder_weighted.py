"""
Unit tests for UserProfileBuilder weighted aggregation and time decay capabilities.

Coverage:
- Test 1: Baseline behavior without weights
- Test 2: Interaction-type weighting
- Test 3: Combined interaction weighting and exponential time decay
- Test 4: Mismatched interaction_types length
- Test 5: Mismatched timestamps length
- Test 6: Negative decay_rate validation
- Test 7: Negative interaction weight validation
- Test 8: Uniform fallback when sum_weights <= 0
- Test 9: Zero-vector edge case
- Test 10: Out-of-bounds interaction_indices
- Test 11: Empty or None interaction_indices
"""

import numpy as np
import pytest

from src.recommender.user_profile import UserProfileBuilder


# ---------------------------------------------------------------------------
# Shared assertion helpers
# ---------------------------------------------------------------------------

def assert_l2_normalized(vector: np.ndarray) -> None:
    """Assert that a non-zero vector has unit L2 norm."""
    assert vector.ndim == 1, (
        f"Expected a one-dimensional vector, got shape {vector.shape}"
    )

    norm = np.linalg.norm(vector)

    assert np.isclose(norm, 1.0, atol=1e-6), (
        f"Expected unit L2 norm, got {norm:.6f}"
    )


def assert_is_zero_vector(vector: np.ndarray) -> None:
    """Assert that a vector is numerically equal to the zero vector."""
    assert vector.ndim == 1, (
        f"Expected a one-dimensional vector, got shape {vector.shape}"
    )

    assert np.allclose(vector, 0.0, atol=1e-6), (
        f"Expected zero vector, got {vector}"
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def dummy_feature_matrix() -> np.ndarray:
    """
    Create a deterministic feature matrix for exact vector calculations.

    Items:
    - Item 0: [1.0, 0.0]
    - Item 1: [0.0, 1.0]
    - Item 2: [1.0, 1.0]
    """
    return np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )


@pytest.fixture
def zero_vector_feature_matrix() -> np.ndarray:
    """
    Create a feature matrix whose item vectors are all zero.
    """
    return np.zeros(
        shape=(3, 2),
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------

def test_1_baseline_behavior_no_weights(
    dummy_feature_matrix: np.ndarray,
) -> None:
    """
    Verify uniform aggregation without optional weights.
    """
    profile = UserProfileBuilder.build_from_history(
        feature_matrix=dummy_feature_matrix,
        interaction_indices=[0, 1],
    )

    # Mean:
    # ([1, 0] + [0, 1]) / 2 = [0.5, 0.5]
    #
    # L2-normalized:
    # [0.5, 0.5] / sqrt(0.5^2 + 0.5^2)
    # = [1 / sqrt(2), 1 / sqrt(2)]
    expected = np.asarray(
        [1.0, 1.0],
        dtype=np.float32,
    ) / np.sqrt(2.0)

    assert profile.shape == (2,)
    assert_l2_normalized(profile)
    assert np.allclose(profile, expected, atol=1e-6)


def test_2_interaction_weighting(
    dummy_feature_matrix: np.ndarray,
) -> None:
    """
    Verify that interaction-type weights affect the user profile.
    """
    profile = UserProfileBuilder.build_from_history(
        feature_matrix=dummy_feature_matrix,
        interaction_indices=[0, 1],
        interaction_types=["visit", "view"],
        interaction_weights_map={
            "visit": 3.0,
            "view": 1.0,
        },
        decay_rate=0.0,
    )

    # Weighted raw profile:
    # (3 * [1, 0] + 1 * [0, 1]) / (3 + 1)
    # = [0.75, 0.25]
    expected_raw = np.asarray(
        [0.75, 0.25],
        dtype=np.float32,
    )

    expected = expected_raw / np.linalg.norm(expected_raw)

    assert_l2_normalized(profile)
    assert np.allclose(profile, expected, atol=1e-6)

    # The visit interaction has the larger weight, so its dimension
    # should have the larger contribution.
    assert profile[0] > profile[1]


def test_3_interaction_weighting_plus_time_decay(
    dummy_feature_matrix: np.ndarray,
) -> None:
    """
    Verify interaction weighting and time decay are combined correctly.
    """
    decay_rate = 0.1

    profile = UserProfileBuilder.build_from_history(
        feature_matrix=dummy_feature_matrix,
        interaction_indices=[0, 1],
        interaction_types=["visit", "view"],
        interaction_weights_map={
            "visit": 3.0,
            "view": 1.0,
        },
        timestamps=[
            "2026-01-01",
            "2026-01-11",
        ],
        decay_rate=decay_rate,
    )

    # Item 0 is 10 days older than item 1.
    type_weights = np.asarray(
        [3.0, 1.0],
        dtype=np.float32,
    )

    time_weights = np.asarray(
        [
            np.exp(-decay_rate * 10.0),
            np.exp(-decay_rate * 0.0),
        ],
        dtype=np.float32,
    )

    total_weights = type_weights * time_weights

    expected_raw = (
        total_weights[0] * dummy_feature_matrix[0]
        + total_weights[1] * dummy_feature_matrix[1]
    ) / np.sum(total_weights)

    expected = expected_raw / np.linalg.norm(expected_raw)

    assert_l2_normalized(profile)
    assert np.allclose(profile, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------

def test_4_interaction_types_wrong_length(
    dummy_feature_matrix: np.ndarray,
) -> None:
    """
    Verify mismatched interaction_types length raises ValueError.
    """
    with pytest.raises(
        ValueError,
        match="interaction_types",
    ):
        UserProfileBuilder.build_from_history(
            feature_matrix=dummy_feature_matrix,
            interaction_indices=[0, 1],
            interaction_types=["visit"],
        )


def test_5_timestamps_wrong_length(
    dummy_feature_matrix: np.ndarray,
) -> None:
    """
    Verify mismatched timestamps length raises ValueError.
    """
    with pytest.raises(
        ValueError,
        match="timestamps",
    ):
        UserProfileBuilder.build_from_history(
            feature_matrix=dummy_feature_matrix,
            interaction_indices=[0, 1],
            timestamps=["2026-01-01"],
            decay_rate=0.1,
        )


def test_6_negative_decay_rate_raises(
    dummy_feature_matrix: np.ndarray,
) -> None:
    """
    Verify negative decay_rate raises ValueError.
    """
    with pytest.raises(
        ValueError,
        match="decay_rate",
    ):
        UserProfileBuilder.build_from_history(
            feature_matrix=dummy_feature_matrix,
            interaction_indices=[0, 1],
            timestamps=[
                "2026-01-01",
                "2026-01-02",
            ],
            decay_rate=-0.1,
        )


def test_7_negative_weight_raises(
    dummy_feature_matrix: np.ndarray,
) -> None:
    """
    Verify negative interaction weights raise ValueError.
    """
    with pytest.raises(
        ValueError,
        match="non-negative",
    ):
        UserProfileBuilder.build_from_history(
            feature_matrix=dummy_feature_matrix,
            interaction_indices=[0, 1],
            interaction_types=["visit", "view"],
            interaction_weights_map={
                "visit": -3.0,
                "view": 1.0,
            },
        )


# ---------------------------------------------------------------------------
# Edge-case tests
# ---------------------------------------------------------------------------

def test_8_zero_total_weights_fall_back_to_uniform(
    dummy_feature_matrix: np.ndarray,
) -> None:
    """
    Verify the sum_weights <= 0 fallback uses uniform weights.

    A very large decay_rate alone cannot guarantee sum_weights == 0 because
    the newest interaction has delta_days == 0 and therefore time weight 1.
    Zero interaction-type weights deterministically exercise the fallback path.
    """
    profile = UserProfileBuilder.build_from_history(
        feature_matrix=dummy_feature_matrix,
        interaction_indices=[0, 1],
        interaction_types=["visit", "view"],
        interaction_weights_map={
            "visit": 0.0,
            "view": 0.0,
        },
    )

    # Fallback result:
    # mean([1, 0], [0, 1]) = [0.5, 0.5]
    expected = np.asarray(
        [1.0, 1.0],
        dtype=np.float32,
    ) / np.sqrt(2.0)

    assert not np.any(np.isnan(profile))
    assert not np.any(np.isinf(profile))
    assert_l2_normalized(profile)
    assert np.allclose(profile, expected, atol=1e-6)


def test_9_zero_feature_vectors_return_zero_profile(
    zero_vector_feature_matrix: np.ndarray,
) -> None:
    """
    Verify all-zero feature vectors return a clean zero profile.

    The raw profile has zero norm, so the implementation must not divide by
    zero or return NaN values.
    """
    profile = UserProfileBuilder.build_from_history(
        feature_matrix=zero_vector_feature_matrix,
        interaction_indices=[0, 1],
    )

    assert profile.shape == (2,)
    assert not np.any(np.isnan(profile))
    assert not np.any(np.isinf(profile))
    assert_is_zero_vector(profile)


@pytest.mark.parametrize(
    "invalid_indices",
    [
        [-1],
        [3],
        [99],
        [0, 5],
        [-1, 1],
    ],
)
def test_10_out_of_bounds_indices_raise_index_error(
    dummy_feature_matrix: np.ndarray,
    invalid_indices: list[int],
) -> None:
    """
    Verify negative and too-large indices raise IndexError.

    The feature matrix contains three rows, so valid indices are 0, 1, and 2.
    """
    with pytest.raises(
        IndexError,
        match="out-of-bounds",
    ):
        UserProfileBuilder.build_from_history(
            feature_matrix=dummy_feature_matrix,
            interaction_indices=invalid_indices,
        )


@pytest.mark.parametrize(
    "empty_indices",
    [
        [],
        None,
    ],
)
def test_11_empty_interaction_indices_raise_value_error(
    dummy_feature_matrix: np.ndarray,
    empty_indices: list[int] | None,
) -> None:
    """
    Verify empty and None interaction_indices raise ValueError.
    """
    with pytest.raises(
        ValueError,
        match="interaction_indices",
    ):
        UserProfileBuilder.build_from_history(
            feature_matrix=dummy_feature_matrix,
            interaction_indices=empty_indices,
        )


if __name__ == "__main__":
    pytest.main(["-v", __file__])
