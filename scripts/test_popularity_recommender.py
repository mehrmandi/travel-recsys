"""
Evaluates PopularityRecommender Baseline on Foursquare NYC dataset
using the shared OfflineEvaluator.
"""
from collections import Counter
from pathlib import Path
import pandas as pd

from src.recommender.most_popular import PopularRecommender
from src.evaluation.evaluator import OfflineEvaluator

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "Foursquare NYC_Tokyo" / "dataset_TSMC2014_NYC.csv_2" / "dataset_TSMC2014_NYC.csv"

def load_foursquare_data(path: Path, min_interactions: int = 5):
    column_names = [
        "user_id", "venue_id", "category_id", "category",
        "latitude", "longitude", "timezone_offset", "utc_timestamp"        
    ]
    
    df = pd.read_csv(
        path,
        header=None,
        names=column_names,
        encoding="latin1",
        on_bad_lines="skip"
    )
    
    if df.iloc[0]["user_id"] == "user_id" or not str(df.iloc[0]["user_id"]).isdigit():
        df = df.iloc[1:].reset_index(drop=True)
        
    df["user_id"] = df["user_id"].astype(str)
    df["venue_id"] = df["venue_id"].astype(str)
    
    df["utc_timestamp"] = pd.to_datetime(df["utc_timestamp"], errors="coerce")
    initial_len = len(df)
    df = df.dropna(subset=["utc_timestamp"]).reset_index(drop=True)
    dropped_count = initial_len - len(df)
    if dropped_count > 0:
        print(f"[*] Dropped {dropped_count:,} rows with invalid timestamps.")

    df = df.sort_values(by=["user_id", "utc_timestamp"], kind="mergesort").reset_index(drop=True)
           
    user_counts = df["user_id"].value_counts()
    eligible_users = set(user_counts[user_counts >= min_interactions].index)
    df = df[df["user_id"].isin(eligible_users)].reset_index(drop=True)
    
    venues_df = df.drop_duplicates(subset=["venue_id"]).reset_index(drop=True)
    venues_df["item_index"] = venues_df.index
    venue_to_index = dict(zip(venues_df["venue_id"], venues_df["item_index"]))
    
    df["item_index"] = df["venue_id"].map(venue_to_index)
    user_interactions = df.groupby("user_id", sort=False)["item_index"].apply(list).to_dict()
    
    return venues_df, user_interactions


def main():
    print(f"[*] Loading raw dataset from: {DATA_PATH}")
    venues_df, user_interactions = load_foursquare_data(DATA_PATH, min_interactions=5)
    
    print(f"Unique eligible users: {len(user_interactions):,}")
    print(f"Unique venues (items): {len(venues_df):,}")
    
    print("\n[*] Computing venue frequencies strictly from training interactions...")
    train_counts = Counter()
    for history in user_interactions.values():
        train_counts.update(history[:-1])
        
    print("[*] Initializing PopularityRecommender...")
    recommender = PopularRecommender(
        places_df=venues_df,
        train_counts=dict(train_counts)
    )
    
    print("[*] Running Offline Leave-One-Out Evaluation via OfflineEvaluator...")
    evaluator = OfflineEvaluator(
        recommender=recommender,
        k_list=[5, 10, 20]
    )
    
    results_df = evaluator.evaluate_last_item_holdout(
        user_interactions=user_interactions,
        min_interactions=5
    )
    
    print("\n" + "=" * 65)
    print("        FOURSQUARE NYC MOST POPULAR BASELINE RESULTS          ")
    print("=" * 65)
    
    mean_row = results_df[results_df["user_id"] == "MEAN"].iloc[0]
    
    print("\nMacro-Averaged Metrics across Eligible Users:")
    for col, val in mean_row.items():
        if col == "user_id":
            continue
        
        print(f"  {col:<15}: {float(val):.6f}")
    print("=" * 65)


if __name__ == "__main__":
    main()
    
    