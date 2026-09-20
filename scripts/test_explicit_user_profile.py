"""
Unit tests for pure UserProfileBuilder and UserProfileRecommender (unweighted baseline).
"""

import numpy as np
import pandas as pd
from src.recommender.user_profile import UserProfileBuilder, UserProfileRecommender


def test_user_profile_unweighted_baseline():
    places_df = pd.DataFrame({
        "name_en": ["Louvre", "Orsay", "Central Park", "Versailles"],
        "category": ["museum", "museum", "park", "palace"],
        "city_en": ["Paris", "Paris", "New York", "Paris"]
    })

    # 4 items x 3 features (museum, park, palace)
    feature_matrix = np.array([
        [1.0, 0.0, 0.0],  # 0: Louvre (museum)
        [1.0, 0.0, 0.0],  # 1: Orsay (museum)
        [0.0, 1.0, 0.0],  # 2: Central Park (park)
        [0.0, 0.0, 1.0],  # 3: Versailles (palace)
    ], dtype=np.float32)

    # -------------------------------------------------------------
    # Test 1: Uniform history aggregation
    # User interacted with Louvre (0) and Central Park (2)
    # -------------------------------------------------------------
    user_vec = UserProfileBuilder.build_from_history(
        feature_matrix=feature_matrix,
        interaction_indices=[0, 2]
    )

    # Must be L2-normalized: ||u|| = 1.0
    assert np.isclose(np.linalg.norm(user_vec),
                      1.0), "Profile vector must have unit L2 norm."
    # Both selected features must have equal weight
    assert np.isclose(user_vec[0], user_vec[1]
                      ), "Pure baseline must treat interactions equally."
    assert user_vec[2] == 0.0, "Non-interacted feature must be exactly 0."

    # -------------------------------------------------------------
    # Test 2: Recommendation & Exclusion
    # -------------------------------------------------------------
    recommender = UserProfileRecommender(
        places_df=places_df, feature_matrix=feature_matrix)
    recs = recommender.recommend(
        user_vector=user_vec,
        top_k=2,
        exclude_indices=[0, 2]
    )

    recommended_names = recs["name_en"].tolist()
    assert "Louvre" not in recommended_names, "Excluded item must not appear in recommendations."
    assert "Central Park" not in recommended_names, "Excluded item must not appear in recommendations."
    assert "Orsay" in recommended_names, "Orsay (museum) must be recommended."

    print("✅ Pure UserProfile baseline assertions passed successfully!")


if __name__ == "__main__":
    test_user_profile_unweighted_baseline()
