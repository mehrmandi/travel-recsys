"""
Pipeline to evaluate UserProfileRecommender & OfflineEvaluator on real Foursquare check-in dataset.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    

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


def main():
    data_path = PROJECT_ROOT / "data" / "Foursquare NYC_Tokyo" / "dataset_TSMC2014_NYC.csv_2" / "dataset_TSMC2014_NYC.csv"

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

    # 5. Summary Results
    print("\n" + "=" * 60)
    print("           FOURSQUARE NYC OFFLINE EVALUATION RESULTS           ")
    print("=" * 60)

    # استخراج سطر میانگین به عنوان یک دیکشنری یا سریز تک‌ردیفه
    mean_row = results_df[results_df["user_id"] == "MEAN"].iloc[0]

    print("\nMacro-Averaged Metrics across Eligible Users:")
    for col, val in mean_row.items():
        if col == "user_id":
            continue
        print(f"  {col:<15}: {float(val):.6f}")

    print("=" * 60)

    # نمایش ۵ کاربر اول برای نمونه
    print("\nSample User Metrics (First 5):")
    print(results_df[results_df["user_id"] != "MEAN"].head())


if __name__ == "__main__":
    main()
