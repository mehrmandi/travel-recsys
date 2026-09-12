"""
Unit tests for CategoryEncoder edge cases: normalization, NaNs, empty strings, and case sensitivity.
"""

import pandas as pd
import numpy as np
from src.features.category_encoder import CategoryEncoder


def test_category_encoder_normalization_and_missing_values():
    # 1. Fit encoder with standard clean categories
    train_df = pd.DataFrame({
        "category": ["museum", "park", "palace"]
    })
    encoder = CategoryEncoder(target_column="category")
    encoder.fit(train_df)

    # 2. Test dataset covering all edge cases
    test_df = pd.DataFrame({
        "category": [
            None,            # index 0: should map to unknown
            "",              # index 1: should map to unknown
            "  MUSEUM  ",    # index 2: should normalize to museum
            "UNKNOWN",       # index 3: should normalize to unknown
        ]
    })

    test_encoded = encoder.transform(test_df)

    # ------------------------------------------------------------------
    # Case 0: None / NaN -> cat_unknown = 1.0
    # ------------------------------------------------------------------
    assert test_encoded.loc[0, "cat_unknown"] == 1.0, (
        f"Assertion Failed: Row 0 (None) expected 'cat_unknown'=1.0, got {test_encoded.loc[0, 'cat_unknown']}"
    )
    assert np.isclose(test_encoded.loc[0].sum(
    ), 1.0), "Assertion Failed: Row 0 sum must be 1.0"

    # ------------------------------------------------------------------
    # Case 1: Empty String ("") -> cat_unknown = 1.0
    # ------------------------------------------------------------------
    assert test_encoded.loc[1, "cat_unknown"] == 1.0, (
        f"Assertion Failed: Row 1 ('') expected 'cat_unknown'=1.0, got {test_encoded.loc[1, 'cat_unknown']}"
    )
    assert np.isclose(test_encoded.loc[1].sum(
    ), 1.0), "Assertion Failed: Row 1 sum must be 1.0"

    # ------------------------------------------------------------------
    # Case 2: Whitespace + Uppercase ("  MUSEUM  ") -> cat_museum = 1.0
    # ------------------------------------------------------------------
    assert test_encoded.loc[2, "cat_museum"] == 1.0, (
        f"Assertion Failed: Row 2 ('  MUSEUM  ') expected 'cat_museum'=1.0, got {test_encoded.loc[2, 'cat_museum']}"
    )
    assert test_encoded.loc[2, "cat_unknown"] == 0.0, (
        "Assertion Failed: Row 2 ('  MUSEUM  ') must not be marked as unknown"
    )
    assert np.isclose(test_encoded.loc[2].sum(
    ), 1.0), "Assertion Failed: Row 2 sum must be 1.0"

    # ------------------------------------------------------------------
    # Case 3: Explicit Uppercase "UNKNOWN" -> cat_unknown = 1.0
    # ------------------------------------------------------------------
    assert test_encoded.loc[3, "cat_unknown"] == 1.0, (
        f"Assertion Failed: Row 3 ('UNKNOWN') expected 'cat_unknown'=1.0, got {test_encoded.loc[3, 'cat_unknown']}"
    )
    assert np.isclose(test_encoded.loc[3].sum(
    ), 1.0), "Assertion Failed: Row 3 sum must be 1.0"

    print("✅ All normalization and missing value assertions passed successfully!")


if __name__ == "__main__":
    test_category_encoder_normalization_and_missing_values()


