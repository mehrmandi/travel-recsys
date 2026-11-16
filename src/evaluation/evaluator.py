"""
Offline Evaluator for Content-Based Recommenders using Leave-One-Out (LOO).
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence
import pandas as pd

from src.recommender.user_profile import UserProfileBuilder, UserProfileRecommender
from src.evaluation.metrics import recall_at_k, precision_at_k, reciprocal_rank_at_k
from src.recommender.most_popular import PopularRecommender
from src.recommender.hybrid import HybridRecommender


@dataclass(frozen=True)
class EvaluationCase:
    user_id: str
    train_history: List[int]
    target_index: int

    @property
    def exclude_indices(self) -> frozenset[int]:
        return frozenset(self.train_history)


def build_last_unseen_evaluation_cases(
    user_interactions: Dict[str, Sequence[int]],
    min_interactions: int = 5,
) -> List[EvaluationCase]:
    """
    Builds one evaluation case per eligible user.

    Protocol:
      - Target is the last venue whose first visit occurs after enough history.
      - Training history contains only check-ins strictly before the target.
      - Target must not appear in training history.
      - All venues in training history are excluded from recommendations.

    `min_interactions` is the minimum total number of check-ins in the resulting
    case, including the held-out target. Therefore, at least
    `min_interactions - 1` training check-ins are required.
    """
    if min_interactions < 5:
        raise ValueError("min_interactions must be >= 5")

    cases: List[EvaluationCase] = []

    for user_id, raw_history in user_interactions.items():
        history = list(raw_history)

        if len(history) < min_interactions:
            continue

        seen: set[int] = set()
        target_position: Optional[int] = None

        for position, item_index in enumerate(history):
            is_unseen = item_index not in seen
            enough_prior_history = position >= min_interactions - 1

            if is_unseen and enough_prior_history:
                target_position = position

            seen.add(item_index)

        if target_position is None:
            continue

        train_history = history[:target_position]
        target_index = history[target_position]

        if target_index in train_history:
            raise AssertionError(
                f"Target {target_index!r} already exists in training history "
                f"for user {user_id!r}."
            )

        cases.append(
            EvaluationCase(
                user_id=user_id,
                train_history=train_history,
                target_index=target_index,
            )
        )

    return cases
class OfflineEvaluator:
    """
    Executes Leave-One-Out offline evaluation across a population of users.
    """
    def __init__(self, recommender: UserProfileRecommender, k_list: Optional[List[int]] = None):
        self.recommender = recommender
        self.k_list = sorted(k_list) if k_list else [5, 10]
        

    def evaluate_last_unseen_holdout(
        self,
        user_interactions: Dict[str, List[int]],
        min_interactions: int = 5,
        city_filter: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Runs leave-one-out evaluation using the user's last unseen venue.

        For every eligible user:
        - Training history contains check-ins strictly before the target.
        - Target is the last venue not previously seen in that history.
        - All training venues are excluded from recommendations.
        """
        records = []
        max_k = max(self.k_list)

        cases = build_last_unseen_evaluation_cases(
            user_interactions=user_interactions,
            min_interactions=min_interactions,
        )
        
        if not cases:
            raise ValueError(
                "No eligible users found for the last-unseen holdout protocol."
            )


        train_interactions = [
            item_index
            for case in cases
            for item_index in case.train_history
        ]

        if isinstance(self.recommender, (PopularRecommender, HybridRecommender)):
            self.recommender.fit(train_interactions)


        for case in cases:
            train_history = case.train_history
            test_target = case.target_index
            exclude_indices = case.exclude_indices

            if isinstance(self.recommender, PopularRecommender):
                recs_df = self.recommender.recommend(
                    top_k=max_k,
                    exclude_indices=exclude_indices,
                    city_filter=city_filter,
                )
            else:
                user_vec = UserProfileBuilder.build_from_history(
                    feature_matrix=self.recommender.feature_matrix,
                    interaction_indices=train_history,
                )

                if isinstance(self.recommender, HybridRecommender):
                    recs_df = self.recommender.recommend(
                        user_vector=user_vec,
                        top_k=max_k,
                        exclude_indices=exclude_indices,
                    )
                else:
                    recs_df = self.recommender.recommend(
                        user_vector=user_vec,
                        top_k=max_k,
                        exclude_indices=list(case.exclude_indices) if case.exclude_indices else None,
                        city_filter=city_filter,
                    )

            recommended_indices = recs_df["item_index"].tolist()

            user_metrics = {"user_id": case.user_id}

            for k in self.k_list:
                user_metrics[f"recall@{k}"] = recall_at_k(
                    recommended_indices,
                    test_target,
                    k,
                )
                user_metrics[f"precision@{k}"] = precision_at_k(
                    recommended_indices,
                    test_target,
                    k,
                )
                user_metrics[f"mrr@{k}"] = reciprocal_rank_at_k(
                    recommended_indices,
                    test_target,
                    k,
                )

            records.append(user_metrics)

        if not records:
            raise ValueError(
                "No eligible users found for the last-unseen holdout protocol."
            )

        metrics_df = pd.DataFrame(records)

        summary = metrics_df.drop(columns=["user_id"]).mean().to_dict()
        summary["user_id"] = "MEAN"

        return pd.concat(
            [metrics_df, pd.DataFrame([summary])],
            ignore_index=True,
        )

        
        
            