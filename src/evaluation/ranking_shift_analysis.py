"""
Ranking Shift Analysis Module.

Measures how the rank of a user's target item changes between the Popularity
model and one or more Hybrid models (one per alpha), over all candidate items.

Design notes
------------
* Ties are handled by rounding scores to a fixed number of decimals and then
  comparing EXACTLY. Unlike "|a - b| <= atol", rounding is an equivalence
  relation (transitive), so ranks always form a consistent permutation.
* Ties on score are broken deterministically: the smaller item id ranks first.
* Statistical tests use paired Wilcoxon signed-rank tests on non-zero
  differences only. When a test cannot be run, the p-value is NaN (never 1.0).
* The reciprocal-rank difference is NOT truncated at K, so its mean is a
  full-MRR difference and differs from an MRR@K difference.
"""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

# Alpha keys are matched with this absolute tolerance, so that e.g.
# 0.1 + 0.8 (0.9000000000000001) still finds the key 0.9.

_ALPHA_KEY_ATOL = 1e-12

@dataclass(frozen=True)
class RankingCase:
    """
    Evaluation data for a single user.

    Score vectors must be the final model scores BEFORE any top-K cut, computed
    with the same training history and normalization, and aligned element-wise
    with `item_indices`.

    `exclude_indices` is normalized to a frozenset on construction, and the keys
    of `hybrid_scores` are normalized to float.
    """
    
    user_id : Any
    target_index : int
    item_indices: Sequence[int]
    popularity_scores: Sequence[float]
    hybrid_scores: Mapping[float, Sequence[float]]
    exclude_indices: Iterable[int] = ()
    
    def __post_init__(self) -> None:
        # Consume the iterable exactly once (it may be a generator).
        object.__setattr__(self, "exclude_indices", frozenset(self.exclude_indices))
        object.__setattr__(self, "hybrid_scores", {float(k): v for k, v in self.hybrid_scores.items()})

        
def _validate_inputs(
  cases: Sequence[RankingCase],
  alphas: Sequence[float],
  top_k: int,
  decimals: Optional[int],
  ) -> None:
  if isinstance(top_k, bool) or not isinstance(top_k, (int, np.integer)) or top_k <= 0:
    raise ValueError(f"top_k must be a positive integer, got: {top_k}")
  
  if decimals is not None and (
    isinstance(decimals, bool)
    or not isinstance(decimals, (int, np.integer))
    or decimals < 0
    ):
    raise ValueError(f"decimals must be a non-negative integer or None, got: {decimals}")
    
  if len(alphas) == 0:
    raise ValueError("alphas sequence cannot be empty.")
  
  seen_alphas: Set[float] = set()
  
  for a in alphas:
    if isinstance(a, bool) or not isinstance(a, (int, float, np.floating)):
      raise ValueError(f"Invalid alpha value: {a}")
    
    if not np.isfinite(a) or not (0.0 <= a <= 1.0):
      raise ValueError(f"Alpha must be finite and in [0.0, 1.0], got: {a}")
  
    if float(a) in seen_alphas:
      raise ValueError(f"Duplicate alpha found: {a}")
    
    seen_alphas.add(float(a))
    
  if not cases:
    raise ValueError("No ranking cases provided.")
  
  seen_users: Set[Any] = set()
  
  for case in cases:
    if case.user_id in seen_users:
      raise ValueError(f"Duplicate user_id detected: {case.user_id}")
    seen_users.add(case.user_id)
    
def _get_alpha_scores(case: RankingCase, alpha: float) -> Sequence[float]:
  """Fetch the hybrid score vector for `alpha`, tolerating float key noise."""
      
  if alpha in case.hybrid_scores:
    return case.hybrid_scores[alpha]
  
  for key, value in case.hybrid_scores.items():
    if abs(key - alpha) <= _ALPHA_KEY_ATOL:
      return value
    
  raise ValueError(f"User {case.user_id}: Missing hybrid scores for alpha={alpha}.")
      
    
def _compute_ranks(
  score_matrix: np.ndarray,
  cand_ids: np.ndarray,
  target_pos: int,
  target_id: int,
) -> np.ndarray:
    """
    1-based rank of the target for every row of `score_matrix` (one row per model).

    rank = 1 + #(items with strictly higher score)
             + #(items with equal score and a smaller item id)

    Scores must already be rounded; comparisons here are exact.
    """
    
    target_scores = score_matrix[:, [target_pos]]
    higher = np.sum(score_matrix > target_scores, axis=1)
    tried_with_smaller_id = np.sum(
      (score_matrix == target_scores) & (cand_ids < target_id), axis=1
    )
    
    return (1 + higher + tried_with_smaller_id).astype(np.int64)
  
  
def _wilcoxon_test(diffs: np.ndarray) -> Tuple[float, int]:
    """
    Paired Wilcoxon signed-rank test on non-zero differences.

    Returns (p_value, n_effective). The p-value is NaN when there are no
    non-zero differences, because no test can be performed. Only that expected
    case is handled; any other error propagates.
    """
    non_zero = diffs[diffs != 0]
    n = int(non_zero.size)

    if n == 0:
      return float("nan"), 0
    
    return float(wilcoxon(non_zero).pvalue), n
  

def _bonferroni(p_value: float, n_tests: int) -> float:
  if np.isnan(p_value):
    return float("nan")
  
  return float(min(1.0, p_value * n_tests))


def analyze_ranking_shifts(
  cases: Iterable[RankingCase],
  *,
  alphas: Sequence[float] = (0.90, 0.95),
  top_k: int = 20,
  decimals: Optional[int] = 12,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Analyze target-rank shifts between Popularity and Hybrid models.

    Parameters
    ----------
    cases : iterable of RankingCase
    alphas : hybrid mixing weights to evaluate
    top_k : cutoff used for the recovery / loss flags
    decimals : scores are rounded to this many decimals before comparison so
        that float noise does not break genuine ties. None disables rounding.

    Returns
    -------
    summary_df : one row per alpha (counts, rank/MRR shifts, Wilcoxon tests)
    details_df : one row per (user, alpha)

    Multiple-comparison correction
    ------------------------------
    Two tests (rank change, reciprocal-rank difference) are run per alpha, so
    the Bonferroni family size is 2 * len(alphas). The reciprocal-rank test is
    the primary one; the rank-change test is secondary (the two are highly
    correlated because they use the same data).
    """
    
    cases_list = list(cases)
    _validate_inputs(cases_list, alphas, top_k, decimals)
    
    alpha_list = [float(a) for a in alphas]
    detail_rows: List[Dict[str, Any]] = []

    
    for case in cases_list:
      item_ids = np.asarray(case.item_indices, dtype=np.int64)
      pop_scores = np.asarray(case.popularity_scores, dtype=np.float64)
      
      # ---- per-user validation ----
      if len(item_ids) != len(pop_scores):
          raise ValueError(
              f"User {case.user_id}: item_indices and popularity_scores length mismatch."
          )
          
      if len(np.unique(item_ids)) != len(item_ids):
        raise ValueError(f"User {case.user_id}: item_indices contains duplicates.")
      
      if not np.all(np.isfinite(pop_scores)):
        raise ValueError(f"User {case.user_id}: Non-finite values in popularity_scores.")
      if case.target_index in case.exclude_indices:
          raise ValueError(
              f"User {case.user_id}: target_index {case.target_index} is in exclude_indices."
          )
      
      if not np.any(item_ids == case.target_index):
          raise ValueError(
              f"User {case.user_id}: target_index {case.target_index} not in item_indices."
          )
          
      # ---- candidate mask (exclusion by item id, not by position) ----
      if case.exclude_indices:
        valid_mask = ~np.isin(item_ids, np.fromiter(case.exclude_indices, dtype=np.int64))
      
      else:
        valid_mask = np.ones(len(item_ids), dtype=bool)
        
      cand_ids = item_ids[valid_mask]
      n_candidates = len(cand_ids)
      target_pos = int(np.flatnonzero(cand_ids == case.target_index)[0])
      
      # ---- build score matrix: row 0 = Popularity, rows 1.. = Hybrid per alpha ----
      rows = [pop_scores[valid_mask]]
      
      for a in alpha_list:
        h = np.asarray(_get_alpha_scores(case, a), dtype=np.float64)
        if len(h) != len(item_ids):
            raise ValueError(f"User {case.user_id}: hybrid_scores[{a}] shape mismatch.")

        if not np.all(np.isfinite(h)):
            raise ValueError(f"User {case.user_id}: Non-finite values in hybrid_scores[{a}].")

        rows.append(h[valid_mask])
        
      score_matrix = np.vstack(rows)
      
      if decimals is not None:
        score_matrix = np.round(score_matrix, decimals)
        
      ranks = _compute_ranks(score_matrix, cand_ids, target_pos, case.target_index)
      
      pop_rank = int(ranks[0])

      for i, a in enumerate(alpha_list):
          h_rank = int(ranks[i + 1])
          rank_change = pop_rank - h_rank  # positive = target moved up
          detail_rows.append({
              "user_id": case.user_id,
              "alpha": a,
              "target_index": int(case.target_index),
              "n_candidates": int(n_candidates),
              "popularity_rank": pop_rank,
              "hybrid_rank": h_rank,
              "rank_change": int(rank_change),
              "relative_rank_change_pct": float(rank_change / n_candidates * 100.0),
              "reciprocal_rank_difference": float(1.0 / h_rank - 1.0 / pop_rank),
              "improved": bool(rank_change > 0),
              "worsened": bool(rank_change < 0),
              "unchanged": bool(rank_change == 0),
              "top_k_recovery": bool(pop_rank > top_k and h_rank <= top_k),
              "top_k_loss": bool(pop_rank <= top_k and h_rank > top_k),
              "both_outside_top_k": bool(pop_rank > top_k and h_rank > top_k),
          })

    details_df = pd.DataFrame(detail_rows)

    # Two tests per alpha (rank change + reciprocal-rank difference).
    n_tests = 2 * len(alpha_list)
    summary_rows: List[Dict[str, Any]] = []

    for a in alpha_list:
        sub = details_df[details_df["alpha"] == a]

        p_rank, n_rank = _wilcoxon_test(sub["rank_change"].to_numpy())
        p_mrr, n_mrr = _wilcoxon_test(sub["reciprocal_rank_difference"].to_numpy())

        summary_rows.append({
            "alpha": a,
            "users": int(len(sub)),
            "improved": int(sub["improved"].sum()),
            "worsened": int(sub["worsened"].sum()),
            "unchanged": int(sub["unchanged"].sum()),
            "mean_rank_change": float(sub["rank_change"].mean()),
            "median_rank_change": float(sub["rank_change"].median()),
            "mean_relative_rank_change_pct": float(sub["relative_rank_change_pct"].mean()),
            "mean_reciprocal_rank_difference": float(sub["reciprocal_rank_difference"].mean()),
            "top_k_recovery": int(sub["top_k_recovery"].sum()),
            "top_k_loss": int(sub["top_k_loss"].sum()),
            "both_outside_top_k": int(sub["both_outside_top_k"].sum()),
            "wilcoxon_rank_p": p_rank,
            "wilcoxon_rank_adj_p": _bonferroni(p_rank, n_tests),
            "wilcoxon_rank_n": n_rank,
            "wilcoxon_mrr_p": p_mrr,
            "wilcoxon_mrr_adj_p": _bonferroni(p_mrr, n_tests),
            "wilcoxon_mrr_n": n_mrr,
        })

    return pd.DataFrame(summary_rows), details_df
        
            
        
        
        
        
  
  
       

