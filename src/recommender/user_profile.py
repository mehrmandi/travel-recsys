"""
User Profile Builder and Baseline Content-Based Recommender.

Responsibilities:
- Build L2-normalized user vectors from explicit interests (MVP cold-start approach).
- Build L2-normalized user vectors from interaction history (liked/visited places).
- Rank candidate places using Cosine Similarity with item exclusion and city filtering.
"""

from typing import List, Optional, Union
import numpy as np
import pandas as pd

from src.features.category_encoder import CategoryEncoder

class UserProfileBuilder:
    """
    Constructs user preference vectors either from explicit interest categories
    or from past interaction history.
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
    def build_from_history(feature_matrix: np.ndarray, interaction_indices: List[int]) -> np.ndarray:
        """
        Build an aggregated user profile vector by averaging vectors of interacted places.

        :param feature_matrix: 2D numpy array of item representations (N x D).
        :param interaction_indices: Row indices of visited/liked places.
        :return: 1D normalized numpy array (D,).
        """
        if not interaction_indices:
            raise ValueError("interaction_indices cannot be empty.")
        
        max_idx = feature_matrix.shape[0] - 1
        
        for idx in interaction_indices:
            if idx < 0 or idx > max_idx:
                raise IndexError(f"Interaction index {idx} out of bounds (max: {max_idx}).")

        liked_vectors = feature_matrix[interaction_indices]

            
        raw_profile = np.mean(liked_vectors, axis=0)
            
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
        
    def recommend(self, user_vector: np.ndarray, top_k: int = 10, exclude_indices: Optional[list[int]] = None, city_filter: Optional[str] = None) -> pd.DataFrame:
        """
        Rank candidate places using vectorized Cosine Similarity.
        """
        
        similarities = np.dot(self.normalized_items, user_vector)
        
        results = self.places_df.copy()
        results["item_index"] = results.index
        results["similarity_score"] = similarities
        
        if exclude_indices:
            results = results.drop(index=exclude_indices, errors="ignore")
            
        if city_filter and "city_en" in results.columns:
            results = results[
                results["city_en"].astype(str).str.strip().str.lower() == city_filter.strip().lower()
                ]
            
        ranked_df = results.sort_values(by="similarity_score", ascending=False).head(top_k)
        
        display_cols = [
            col for col in ["name_en", "category", "city_en", "country_en", "similarity_score"]
            if col in ranked_df.columns
        ]
        
        return ranked_df[display_cols].reset_index(drop=True)
    
            

    
                