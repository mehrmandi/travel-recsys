"""
Pipeline to evaluate UserProfileRecommender & OfflineEvaluator on real Foursquare check-in dataset.
"""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
    
import numpy as np
import pandas as pd
from collections import Counter
from scipy.stats import wilcoxon
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


from src.recommender.most_popular import PopularRecommender
from src.recommender.hybrid import HybridRecommender
from src.recommender.user_profile import UserProfileBuilder, UserProfileRecommender
from src.evaluation.evaluator import OfflineEvaluator
    


def load_and_preprocess_foursquare(file_path: Path, min_user_interactions: int = 5):
    print(f"[*] Loading raw dataset from: {file_path}")
    df = pd.read_csv(file_path, sep=",", encoding="latin-1")
    
    # Standardize column names to lowercase/underscore to match our code
    df.columns = [c.strip().lower() for c in df.columns]
   
    df.columns = [
        "user_id",
        "venue_id",
        "category_id",
        "category",
        "latitude",
        "longitude",
        "timezone_offset",
        "utc_timestamp",
    ]

    
    print(f"    Detected columns: {list(df.columns)}")
    print(f"    Raw records: {len(df):,}")
    
    df["timestamp"] = pd.to_datetime(df["utc_timestamp"],format="%a %b %d %H:%M:%S +0000 %Y")
    df = df.sort_values(by=["user_id", "timestamp"], kind="mergesort").reset_index(drop=True)
    
    user_counts = df["user_id"].value_counts()
    eligible_users = user_counts[user_counts >= min_user_interactions].index
    df = df[df["user_id"].isin(eligible_users)].copy()
    print(f"Records after filtering users (< {min_user_interactions} interactions): {len(df):,}")
    print(f"Unique eligible users: {df['user_id'].nunique():,}")
    
    items_df = (df[["venue_id", "category", "latitude", "longitude"]].drop_duplicates(subset=["venue_id"]).reset_index(drop=True))
    items_df["item_index"] = items_df.index
    venue_to_index = dict(zip(items_df["venue_id"], items_df["item_index"]))

    print(f"Unique venues (items): {len(items_df):,}")
    
    
    tfidf = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
    raw_features = tfidf.fit_transform(items_df["category"].fillna("unknown"))
    
    feature_matrix = normalize(raw_features.toarray(), norm="l2", axis=1)

    df["item_index"] = df["venue_id"].map(venue_to_index)
    
    user_interactions = (df.groupby("user_id")["item_index"].apply(list).to_dict())
    
    return feature_matrix, items_df, user_interactions



def _compare_metric(hyb: pd.Series, pop: pd.Series, atol: float = 1e-9) -> dict:
    """Per-user comparison of a single metric: win/loss/tie counts, effect size, Wilcoxon p."""

    diff = hyb - pop
    tie = np.isclose(diff, 0.0, atol=atol, rtol=0.0)
    wins = int(((diff > 0) & ~tie).sum())
    losses = int(((diff < 0) & ~tie).sum())
    ties = int(tie.sum())
    
    nonzero = diff[~tie]
    
    if len(nonzero) > 0:
        p_value = wilcoxon(nonzero).pvalue
        
    else:
        p_value = float("nan")
        
    return {
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "mean_diff": float(diff.mean()),
        "median_diff": float(diff.median()),
        "n_effective": len(nonzero),
        "p_value": p_value,
    }
        



def diagnose_hybrid_vs_popularity(pop_eval_df, hybrid_eval_df, alpha: float | None = None, at_k: int = 20) -> dict:
    """
    Per-user comparison of Hybrid vs Popularity on MRR@k and Recall@k.
    Returns the results dict so they can be logged or tested, and prints a report.
    """
    
    metrics = {"MRR": f"mrr@{at_k}", "Recall": f"recall@{at_k}"}
    
    for name, df in (("pop_eval_df", pop_eval_df), ("hybrid_eval_df", hybrid_eval_df)):
        missing = {"user_id", *metrics.values()} - set(df.columns)
        if missing:
            raise ValueError(f"{name} is missing columns: {sorted(missing)}")
        
        
    pop = pop_eval_df[pop_eval_df["user_id"] != "MEAN"]
    hyb = hybrid_eval_df[hybrid_eval_df["user_id"] != "MEAN"]
    
    merged = pd.merge(pop, hyb, on="user_id", suffixes=("_pop", "_hyb"), validate="one_to_one")
    dropped = (len(pop) - len(merged), len(hyb) - len(merged))
    
    if any(dropped):
        print(f"[!] Warning: {dropped[0]} Popularity-only and {dropped[1]} Hybrid-only users were dropped.")
        
    results = {
        label: _compare_metric(merged[f"{col}_hyb"], merged[f"{col}_pop"])
        for label, col in metrics.items()
    }
    
    n_tests = len(metrics)
    for r in results.values():
        r["p_adj"] = min(r["p_value"] * n_tests, 1.0)
        
    title = f"Hybrid{f' (alpha={alpha:.2f})' if alpha is not None else ''} vs Popularity @ Top-{at_k}"
    print("\n" + "=" * 62)
    print(f"[*] {title}")
    print(f"  Total Evaluated Users : {len(merged)}")
    for label, r in results.items():
        print("-" * 62)
        print(f"  [{label}]")
        print(
            f"    Wins / Losses / Ties : {r['wins']} / {r['losses']} / {r['ties']}")
        print(f"    Mean diff            : {r['mean_diff']:+.6f}")
        print(f"    Median diff          : {r['median_diff']:+.6f}")
        print(
            f"    Wilcoxon (n={r['n_effective']}) : p={r['p_value']:.4e}, Bonferroni-adj p={r['p_adj']:.4e}")
    print("=" * 62 + "\n")

    return results
    

def main():
    data_path = PROJECT_ROOT / "data" / "Foursquare NYC_Tokyo" / \
        "dataset_TSMC2014_NYC.csv_2" / "dataset_TSMC2014_NYC.csv"

    if not data_path.exists():
        fallback_candidates = list(PROJECT_ROOT.glob(
            "*TSMC2014*NYC*")) + list(PROJECT_ROOT.glob("data/*NYC*"))
        if fallback_candidates:
            data_path = fallback_candidates[0]
        else:
            print(f"[!] File not found at {data_path}.")
            print(
                "    Please place the Foursquare NYC file in the 'data/' folder or update 'data_path'.")
            return

    feature_matrix, items_df, user_interactions = load_and_preprocess_foursquare(
        file_path=data_path,
        min_user_interactions=5
    )

    train_counts = Counter(
        item_index
        for history in user_interactions.values()
        for item_index in history[:-1]
    )

    # 1. Content-Based Recommender
    print("\n[*] Initializing UserProfileRecommender...")
    recommender = UserProfileRecommender(
        places_df=items_df,
        feature_matrix=feature_matrix
    )
    print("[*] Running Offline Leave-One-Out Evaluation...")
    evaluator = OfflineEvaluator(
        recommender=recommender,
        k_list=[5, 10, 20],
    )
    results_df = evaluator.evaluate_last_item_holdout(
        user_interactions=user_interactions,
        min_interactions=5
    )
    print("\nContent-Based Results:")
    print(results_df[results_df["user_id"] == "MEAN"].to_string(index=False))

    # 2. Most Popular Recommender
    print("\n[*] Initializing PopularRecommender...")
    pop_recommender = PopularRecommender(
        places_df=items_df,
        train_counts=train_counts
    )
    pop_evaluator = OfflineEvaluator(
        recommender=pop_recommender,
        k_list=[5, 10, 20],
    )
    print("[*] Running Popularity Evaluation...")
    pop_results_df = pop_evaluator.evaluate_last_item_holdout(
        user_interactions=user_interactions,
        min_interactions=5
    )
    print("\nPopularity Results:")
    print(pop_results_df[pop_results_df["user_id"]
          == "MEAN"].to_string(index=False))

    # 3. Hybrid Recommender (alpha = 0.90)
    print("\n[*] Running Hybrid Evaluation (alpha = 0.90)...")
    hybrid_rec_90 = HybridRecommender(
        feature_matrix=feature_matrix,
        train_counts=train_counts,
        alpha=0.90,
    )
    hybrid_evaluator_90 = OfflineEvaluator(
        recommender=hybrid_rec_90,
        k_list=[5, 10, 20],
    )
    hybrid_results_90 = hybrid_evaluator_90.evaluate_last_item_holdout(
        user_interactions=user_interactions,
        min_interactions=5,
    )
    print("\nHybrid (alpha=0.90) Results:")
    print(hybrid_results_90[hybrid_results_90["user_id"]
          == "MEAN"].to_string(index=False))

    # 4. Hybrid Recommender (alpha = 0.95)
    print("\n[*] Running Hybrid Evaluation (alpha = 0.95)...")
    hybrid_rec_95 = HybridRecommender(
        feature_matrix=feature_matrix,
        train_counts=train_counts,
        alpha=0.95,
    )
    hybrid_evaluator_95 = OfflineEvaluator(
        recommender=hybrid_rec_95,
        k_list=[5, 10, 20],
    )
    hybrid_results_95 = hybrid_evaluator_95.evaluate_last_item_holdout(
        user_interactions=user_interactions,
        min_interactions=5,
    )
    print("\nHybrid (alpha=0.95) Results:")
    print(hybrid_results_95[hybrid_results_95["user_id"]
          == "MEAN"].to_string(index=False))

    # 5. User-Level Statistical Diagnostics
    print("\n" + "#" * 62)
    print(" Running User-Level Statistical Diagnostics vs Popularity")
    print("#" * 62)

    diagnose_hybrid_vs_popularity(
        pop_eval_df=pop_results_df,
        hybrid_eval_df=hybrid_results_90,
        at_k=20,
        alpha=0.90
    )

    diagnose_hybrid_vs_popularity(
        pop_eval_df=pop_results_df,
        hybrid_eval_df=hybrid_results_95,
        at_k=20,
        alpha=0.95
    )


if __name__ == "__main__":
    main()
