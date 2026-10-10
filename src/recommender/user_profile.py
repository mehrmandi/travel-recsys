"""
User Profile Builder and Baseline Content-Based Recommender.

Responsibilities:
- Build L2-normalized user vectors from explicit interests (MVP cold-start approach).
- Build L2-normalized user vectors from interaction history (uniform or weighted).
- Support interaction-type weighting and exponential time decay.
- Rank candidate places using Cosine Similarity with item exclusion and city filtering.
"""

from typing import List, Optional, Union , Dict, Sequence
import numpy as np
import pandas as pd

from src.features.category_encoder import CategoryEncoder

DEFAULT_INTERACTION_WEIGHTS: Dict[str, float] = {
    "view": 1.0,
    "like": 2.0,
    "bookmark": 2.5,
    "visit": 3.0,
    "review": 3.0,
}

class UserProfileBuilder:
    """
    Constructs user preference vectors either from explicit interest categories
    or from past interaction history (uniform or weighted).
    """
    
    @staticmethod
    def build_from_interests(interests: List[str], encoder: CategoryEncoder) -> np.ndarray:
        """
        Build an L2-normalized profile vector directly from explicit category names.
        Uses the fitted CategoryEncoder to ensure 1:1 feature space alignment.

        :param interests: List of category names chosen by the user (e.g. ['museum', 'park']).
        :param encoder: A fitted instance of CategoryEncoder.
        :return: 1D normalized numpy array of length equal to total feature dimensions.
        """
        if not interests:
            raise ValueError("Interests list cannot be empty.")
        
        if encoder.categories is None:
            raise RuntimeError("CategoryEncoder must be fitted before building profiles.")

        df_interests = pd.DataFrame({encoder.target_column: interests})
        encoded_df = encoder.transform(df_interests)
        encoded_vectors = encoded_df.to_numpy(dtype=np.float32)
            
        raw_profile = np.mean(encoded_vectors , axis=0)
        
        norm = np.linalg.norm(raw_profile)
        return raw_profile / norm if norm > 0 else raw_profile
    
    @staticmethod
    def build_from_history(
        feature_matrix: np.ndarray, 
        interaction_indices: List[int], 
        interaction_types: Optional[Sequence[str]] = None,
        timestamps: Optional[Sequence[Union[pd.Timestamp, str, int, float]]] = None,
        interaction_weights_map: Optional[Dict[str, float]] = None,
        decay_rate: float = 0.0,
        ) -> np.ndarray:
        """
        Build an aggregated user profile vector with optional interaction weighting and time decay.

        :param feature_matrix: 2D numpy array of item representations (N x D).
        :param interaction_indices: Sequence of row indices for visited/interacted places.
        :param interaction_types: Optional sequence of interaction types corresponding to indices.
        :param timestamps: Optional sequence of datetime/epoch timestamps of interactions.
        :param interaction_weights_map: Dictionary mapping interaction type strings to scalar weights.
        :param decay_rate: Exponential decay parameter lambda (>= 0). If 0.0, time decay is disabled.
        :return: 1D normalized numpy array (D,).
        """
        if not interaction_indices:
            raise ValueError("interaction_indices cannot be empty.")
        
        if decay_rate < 0.0:
            raise ValueError("decay_rate must be >= 0.0")

        indices = np.asarray(interaction_indices, dtype=np.int64)
        n_items = feature_matrix.shape[0]
        
        if np.any(indices < 0) or np.any(indices >= n_items):
            out_of_bounds = indices[(indices < 0) | (indices >= n_items)]
            raise IndexError(
                f"Interaction indices contains out-of-bounds values: {out_of_bounds.tolist()} "
                f"(valid range: 0 to {n_items - 1})."
            )
            
        n_interactions = len(indices)
        weights = np.ones(n_interactions, dtype=np.float32)

        if interaction_types is not None:
            if len(interaction_types) != n_interactions:
                raise ValueError("Length of interaction_types must match interaction_indices.")
                            
            weight_mapping = interaction_weights_map or DEFAULT_INTERACTION_WEIGHTS
            type_weights = np.array(
                [weight_mapping.get(str(t).lower(), 1.0) for t in interaction_types],
                dtype=np.float32,
            )
            weights *= type_weights
            
        
        if decay_rate > 0.0 and timestamps is not None:
            if len(timestamps) != n_interactions:
                raise ValueError("Length of timestamps must match interaction_indices.")
            

            ts_series = pd.to_datetime(timestamps)
            max_ts = ts_series.max()
            delta_days = (max_ts - ts_series).total_seconds() / 86400.0
            time_weights = np.exp(-decay_rate * delta_days.to_numpy(dtype=np.float32))
            weights *= time_weights
            
        if np.any(weights < 0):
            raise ValueError("Interaction weights must be non-negative.")


        sum_weights = np.sum(weights)
        if sum_weights <= 0:
            weights = np.ones(n_interactions, dtype=np.float32)
            sum_weights = float(n_interactions)
            
        liked_vectors = feature_matrix[indices]
        raw_profile = np.dot(weights, liked_vectors) / sum_weights

        norm = np.linalg.norm(raw_profile)
        return raw_profile / norm if norm > 0 else raw_profile


class UserProfileRecommender:
    """
    Ranks candidate places against a user profile vector using Cosine Similarity.
    """

    def __init__(self, places_df: pd.DataFrame, feature_matrix: Union[pd.DataFrame, np.ndarray]):
        if isinstance(feature_matrix, pd.DataFrame):
            self.feature_matrix = feature_matrix.to_numpy(dtype=np.float32)
        else:
            self.feature_matrix = np.asarray(feature_matrix, dtype=np.float32)

        if len(places_df) != self.feature_matrix.shape[0]:
            raise ValueError(
                f"Length mismatch: places_df ({len(places_df)}) vs "
                f"feature_matrix ({self.feature_matrix.shape[0]})."
            )

        self.places_df = places_df.reset_index(drop=True)

        item_norms = np.linalg.norm(self.feature_matrix, axis=1, keepdims=True)
        item_norms[item_norms == 0] = 1e-10
        self.normalized_items = self.feature_matrix / item_norms

    def recommend(
        self,
        user_vector: np.ndarray,
        top_k: int = 10,
        exclude_indices: Optional[list[int]] = None,
        city_filter: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Rank candidate places using vectorized Cosine Similarity.

        :param user_vector: 1D profile vector (D,). It is L2-normalized internally,
                            so scores are true cosine similarities.
        :param top_k: Number of results to return. Must be a positive integer.
        :param exclude_indices: Item indices to remove from the candidates
                                (must be within [0, n_items - 1]).
        :param city_filter: Optional city name; matched case-/whitespace-insensitively.
        :raises ValueError: top_k <= 0, wrong user_vector shape, or city_filter
                            requested but the catalog has no `city_en` column.
        :raises IndexError: exclude_indices contains out-of-bounds values.
        """
        n_items, n_dims = self.feature_matrix.shape

        # ---- Input validation -------------------------------------------------
        # NOTE: pandas head(-k) silently drops the LAST k rows, so a negative
        # top_k must be rejected explicitly instead of "working" with wrong output.
        if isinstance(top_k, bool) or not isinstance(top_k, (int, np.integer)) or top_k <= 0:
            raise ValueError(
                f"top_k must be a positive integer, got {top_k!r}.")

        user_vector = np.asarray(user_vector, dtype=np.float32)
        if user_vector.shape != (n_dims,):
            raise ValueError(
                f"user_vector must have shape ({n_dims},), got {user_vector.shape}."
            )

        exclude_array = None
        if exclude_indices is not None and len(exclude_indices) > 0:
            exclude_array = np.asarray(exclude_indices, dtype=np.int64)
            out_of_bounds = exclude_array[(
                exclude_array < 0) | (exclude_array >= n_items)]
            if out_of_bounds.size > 0:
                raise IndexError(
                    f"exclude_indices contains out-of-bounds values: {out_of_bounds.tolist()} "
                    f"(valid range: 0 to {n_items - 1})."
                )

        if city_filter and "city_en" not in self.places_df.columns:
            raise ValueError(
                "city_filter was provided but places_df has no 'city_en' column.")

        # ---- Scoring ----------------------------------------------------------
        # Normalize the user vector so scores are true cosine similarities.
        # (A zero vector stays zero -> all scores 0.0, no NaNs.)
        user_norm = np.linalg.norm(user_vector)
        if user_norm > 0:
            user_vector = user_vector / user_norm

        similarities = np.dot(self.normalized_items, user_vector)

        results = self.places_df.copy()
        results["item_index"] = results.index
        results["similarity_score"] = similarities

        # ---- Filtering --------------------------------------------------------
        if exclude_array is not None:
            results = results.drop(index=exclude_array)

        if city_filter:
            results = results[
                results["city_en"].astype(str).str.strip().str.lower()
                == city_filter.strip().lower()
            ]

        # ---- Ranking ----------------------------------------------------------
        # kind="stable": tied scores keep catalog order, so rankings (and therefore
        # offline-evaluation metrics) are reproducible across runs.
        ranked_df = results.sort_values(
            by="similarity_score", ascending=False, kind="stable"
        ).head(top_k)

        display_cols = [
            col
            for col in [
                "item_index",
                "name_en",
                "category",
                "city_en",
                "country_en",
                "similarity_score",
            ]
            if col in ranked_df.columns
        ]

        return ranked_df[display_cols].reset_index(drop=True)

    
                