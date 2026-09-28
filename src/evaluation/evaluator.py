"""
Offline Evaluator for Content-Based Recommenders using Leave-One-Out (LOO).
"""
from typing import Dict, List, Optional
import pandas as pd

from src.recommender.user_profile import UserProfileBuilder, UserProfileRecommender
from src.evaluation.metrics import recall_at_k, precision_at_k, reciprocal_rank_at_k

class OfflineEvaluator:
    """
    Executes Leave-One-Out offline evaluation across a population of users.
    """
    def __init__(self, recommender: UserProfileRecommender, k_list: Optional[List[int]] = None):
        self.recommender = recommender
        self.k_list = sorted(k_list) if k_list else [5, 10]
        
    def evaluate_leave_one_out(self, user_interactions: Dict[str, List[int]], min_interactions: int = 3, city_filter: Optional[str] = None) -> pd.DataFrame:
        """
        Runs LOO evaluation.
        For each user:
          - History[:-1] forms the training profile.
          - History[-1] is the holdout ground-truth target item.
          - Candidate ranking excludes History[:-1].
        """
        records = []
        max_k = max(self.k_list)
        
        for user_id, history in user_interactions.items():
            if len(history) < min_interactions:
                continue
            
            train_history = history[:-1]
            test_target = history[-1]
            
            # 1. Build profile from training set
            user_vec = UserProfileBuilder.build_from_history(feature_matrix=self.recommender.feature_matrix, interaction_indices=train_history)
            
            # 2. Get recommendations excluding train history
            recs_df = self.recommender.recommend(user_vector=user_vec, top_k=max_k, exclude_indices=train_history, city_filter=city_filter)
            
            # 3. Retrieve recommended indices via explicit item_index
            recommended_indices = recs_df["item_index"].tolist()
            
            user_metrics: Dict[str, object] = {"user_id": user_id}
            
            for k in self.k_list:
                user_metrics[f"recall@{k}"] = recall_at_k(recommended_indices, test_target, k)
                user_metrics[f"precision@{k}"] = precision_at_k(recommended_indices, test_target, k)
                user_metrics[f"mrr@{k}"] = reciprocal_rank_at_k(recommended_indices, test_target, k)
                
            records.append(user_metrics)
        
        if not records:
            raise ValueError("No eligible users found matching the min_interactions threshold.")
        
        metrics_df = pd.DataFrame(records)
        
        # Compute summary row (mean across users)
        summary = metrics_df.drop(columns=["user_id"]).mean().to_dict()
        summary["user_id"] = "MEAN"
        summary_df = pd.DataFrame([summary])

        return pd.concat([metrics_df, summary_df], ignore_index=True)
        
        
            