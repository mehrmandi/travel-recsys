# Model Evaluation Benchmarks

This document tracks offline evaluation results across different recommender baselines and algorithms on the **Foursquare NYC Check-in Dataset (TSMC2014)**.

## Dataset Summary
- **Source:** `dataset_TSMC2014_NYC.csv`
- **Raw Check-ins:** 227,428
- **Unique Eligible Users ($\ge 5$ interactions):** 1,083
- **Unique Venues (Items):** 38,333
- **Multi-category Handling:** Preserves the first encountered category per venue.
- **Evaluation Protocol:** Leave-One-Out (Last-item holdout per user, candidate exclusion for train items).

---

## Benchmark Results

| Model / Strategy | Recall@5 | Recall@10 | Recall@20 | Precision@5 | Precision@10 | Precision@20 | MRR@5 | MRR@10 | MRR@20 | Date |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Baseline** (Theoretical) | 0.000130 | 0.000261 | 0.000522 | 0.000026 | 0.000026 | 0.000026 | 0.000059 | 0.000076 | 0.000094 | 2026-10-17 |
| **Content-Based (TF-IDF Category)** | **0.000923** | **0.001847** | **0.003693** | **0.000185** | **0.000185** | **0.000185** | **0.000185** | **0.000339** | **0.000456** | 2026-10-17 |
| **Global Most Popular** | 0.012004 | 0.014774 | 0.016620 | 0.002401 | 0.001477 | 0.000831 | 0.005910 | 0.006248 | 0.006368 | 2026-10-19 |
| **Context-Aware (Content + Distance)** | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | - |

---

## Notes & Observations
- **Content-Based Profile:**
  - Operates purely on textual similarity of venue categories.
  - Demonstrates ~7x improvement in Recall@20 over purely random item selection.
  - Expected limitations: lack of spatial proximity awareness (geographic dispersion in NYC).

* **Global Most Popular:**

  * Ranks venues by popularity calculated from training interactions.
  * Outperforms the current Content-Based (TF-IDF Category) baseline across all reported metrics.
  * Achieves approximately 4.50x the Recall@20 of the Content-Based baseline.
  * Serves as a non-personalized baseline for evaluating future recommendation strategies.

* **Evaluation Considerations:**

  * Both models use a last-item holdout evaluation strategy.
  * Exact equivalence of preprocessing, venue indexing, and holdout targets has not been fully verified.
  * Further comparisons should preserve the evaluation protocol and avoid using held-out interactions to calculate popularity.

