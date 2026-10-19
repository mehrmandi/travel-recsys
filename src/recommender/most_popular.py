"""
Popularity Baseline Recommender.

Ranks every place by how often it appears in the TRAINING interactions,
ignoring the user profile entirely. Its job is to be the sanity-check baseline
the content-based model must beat in offline evaluation.

Important: `train_counts` must be built from training data only (without each
user's held-out last interaction), otherwise the baseline sees the future.
"""
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

class PopularRecommender:
    """
    Non-personalized recommender ranking places by training-set frequency.

    The public `recommend` signature matches `UserProfileRecommender.recommend`,
    so it can be plugged into `OfflineEvaluator` unchanged.
    """
    
    DISPLAY_COLUMNS = [
        "item_index",
        "name_en",
        "category",
        "city_en",
        "country_en",
        "popularity_count",
    ]
    
    
    def __init__(self, places_df: pd.DataFrame, train_counts: Dict[int, int]):
        """
        Args:
            places_df: Catalog of places. If it has no 'item_index' column, the
                       positional row index is used (same convention as
                       UserProfileRecommender). If present, it must be unique.
            train_counts: Mapping item_index -> frequency in the training set.
        """
        places = places_df.copy().reset_index(drop=True)
        
        if "item_index" not in places.columns:
            places["item_index"] = places.index
            
        if places["item_index"].isna().any():
            raise ValueError("places_df['item_index'] must not contain missing values.")
        
        if places["item_index"].duplicated().any():
            raise ValueError("places_df['item_index'] must be unique.")
        places["item_index"] = places["item_index"].astype(np.int64)
        
        counts = dict(train_counts)
        
        if any(count < 0 for count in counts.values()):
            raise ValueError("train_counts must contain non-negative counts only.")

            
        if counts and not places["item_index"].isin(list(counts.keys())).any():
            raise ValueError(
                "No keys in train_counts match places_df['item_index'] (check key dtypes)."
            )
            
        places["popularity_count"] = (
            places["item_index"].map(counts).fillna(0).astype(int)
        )
        
        self.places_df = places
        self._valid_indices = places["item_index"].to_numpy()
    
        self.sorted_df = places.sort_values(
            by=["popularity_count", "item_index"],
            ascending=[False, True],
        ).reset_index(drop=True)
        
        
    def recommend(
        self,
        user_vector: Optional[np.ndarray] = None,
        top_k: int = 10,
        exclude_indices: Optional[list[int]] = None,
        city_filter: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Return the most popular places, after exclusion and city filtering.

        :param user_vector: Ignored. Accepted for interface compatibility.
        :param top_k: Number of results. Must be a positive integer.
        :param exclude_indices: item_index values to remove BEFORE truncating to
                                top_k (so top_k items are returned when available).
        :param city_filter: Optional city name (case-/whitespace-insensitive).
        :raises ValueError: top_k <= 0, or city_filter without a 'city_en' column.
        :raises IndexError: exclude_indices contains item_index values not in the catalog.
        """
        if isinstance(top_k, bool) or not isinstance(top_k, (int, np.integer)) or top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got {top_k!r}.")

        results = self.sorted_df

        if exclude_indices is not None and len(exclude_indices) > 0:
            excluded = np.asarray(exclude_indices, dtype=np.int64)
            unknown = excluded[~np.isin(excluded, self._valid_indices)]

            if unknown.size > 0:
                raise IndexError(
                    f"exclude_indices contains item_index values not in the catalog: "
                    f"{unknown.tolist()}."
                )

            results = results[~results["item_index"].isin(excluded)]

        if city_filter:
            if "city_en" not in results.columns:
                raise ValueError(
                    "city_filter was provided but places_df has no 'city_en' column."
                )

            results = results[
                results["city_en"].astype(str).str.strip().str.lower()
                == city_filter.strip().lower()
            ]

        display_cols = [c for c in self.DISPLAY_COLUMNS if c in results.columns]
        if display_cols:
            return results[display_cols].head(top_k)
        return results.head(top_k)[display_cols].reset_index(drop=True)
                
