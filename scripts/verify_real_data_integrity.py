"""
Integrity & Statistics Check on Real Dataset (Foursquare NYC)
Verifies:
1. Target item is strictly NEVER seen in the prior training history.
2. All training history items are in exclude_indices.
3. Target item is NOT in exclude_indices.
4. Exact count of eligible users under the new 'last_unseen' protocol.
"""
from src.evaluation.evaluator import build_last_unseen_evaluation_cases
from pathlib import Path
import sys
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main():
    # 1. Locate the NYC dataset path
    data_dir = PROJECT_ROOT / "data"
    candidates = [p for p in data_dir.glob("*NYC*.csv") if p.is_file()]
    if not candidates:
        candidates = [p for p in PROJECT_ROOT.glob(
            "**/*NYC*.csv") if p.is_file()]

    if not candidates:
        raise FileNotFoundError("Foursquare NYC dataset file not found!")

    dataset_path = candidates[0]
    print(f"Reading dataset: {dataset_path.name} ...")

    df = pd.read_csv(dataset_path)

    # Standardize column names
    user_col = "userId" if "userId" in df.columns else df.columns[0]
    venue_col = "venueId" if "venueId" in df.columns else df.columns[1]
    time_col = "utcTimestamp" if "utcTimestamp" in df.columns else (
        "timestamp" if "timestamp" in df.columns else None
    )

    # Sort interactions chronologically
    if time_col and time_col in df.columns:
        df[time_col] = pd.to_datetime(
            df[time_col], format="mixed", errors="coerce")
        df = df.sort_values(by=[user_col, time_col], ascending=[True, True])

    # Map venue IDs to continuous integer indices
    venue_mapping = {v: idx for idx, v in enumerate(df[venue_col].unique())}
    df["item_index"] = df[venue_col].map(venue_mapping)

    # Group interactions per user in chronological order
    user_interactions = df.groupby(
        user_col)["item_index"].apply(list).to_dict()
    total_raw_users = len(user_interactions)
    total_raw_interactions = len(df)

    print(f"\n--- Raw Dataset Summary ---")
    print(f"Total Users: {total_raw_users:,}")
    print(f"Total Interactions: {total_raw_interactions:,}")
    print(f"Total Unique Venues: {len(venue_mapping):,}")

    # 2. Build evaluation cases using the Last-Unseen protocol (min_interactions=5)
    cases = build_last_unseen_evaluation_cases(
        user_interactions=user_interactions,
        min_interactions=5
    )

    eligible_users_count = len(cases)
    print(f"\n--- Protocol Check (min_interactions >= 5) ---")
    print(
        f"Eligible Users for Evaluation: {eligible_users_count:,} (out of {total_raw_users:,} total users)")

    # 3. Perform strict integrity audit checks
    target_in_history_leaks = 0
    target_in_exclude_leaks = 0
    history_not_in_exclude_leaks = 0
    history_too_short_leaks = 0

    for case in cases:
        train_set = set(case.train_history)
        exclude_set = set(case.exclude_indices)
        target = case.target_item

        # Check 1: Target must not be in the past training history
        if target in train_set:
            target_in_history_leaks += 1

        # Check 2: All training items must be included in exclude_indices
        if not train_set.issubset(exclude_set):
            history_not_in_exclude_leaks += 1

        # Check 3: Target must NOT be in exclude_indices (to allow recommendation)
        if target in exclude_set:
            target_in_exclude_leaks += 1

        # Check 4: Training history length must be at least 4 (5 = 4 train + 1 target)
        if len(case.train_history) < 4:
            history_too_short_leaks += 1

    print("\n--- Integrity Audit Results ---")
    print(
        f"Target leaked into train_history:      {target_in_history_leaks} (Expected: 0)")
    print(
        f"Target leaked into exclude_indices:    {target_in_exclude_leaks} (Expected: 0)")
    print(
        f"Train items missing in exclude_indices:{history_not_in_exclude_leaks} (Expected: 0)")
    print(
        f"Train histories shorter than 4:        {history_too_short_leaks} (Expected: 0)")

    if any([target_in_history_leaks, target_in_exclude_leaks, history_not_in_exclude_leaks, history_too_short_leaks]):
        print("\n❌ Error: Evaluation contract or data integrity violated!")
    else:
        print("\n✅ Full Verification Passed: All conditions and data strictly comply with the evaluation protocol with zero data leakage.")


if __name__ == "__main__":
    main()
