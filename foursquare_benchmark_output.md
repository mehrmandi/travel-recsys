# Foursquare NYC Recommendation Benchmark Report

## 1. Dataset & Evaluation Protocol
- **Dataset**: Foursquare NYC (`dataset_TSMC2014_NYC.csv`)
- **Total Raw Interactions**: 227,428
- **Filtered Users**: 1,083 (users with `>= 5` interactions)
- **Unique Venues (Items)**: 38,333
- **Evaluation Protocol**: Last-Unseen (Leave-One-Out)
  - Target: The last first-seen venue visited by the user.
  - Leakage Prevention: Training check-ins occur strictly prior to the target; all training venues are excluded from the candidate recommendation pool.
- **Preprocessing Constraints**: `min_interactions=5`; the first category is retained for multi-category venues.

---

## 2. Overall Performance Metrics

| Model | Metric | @5 | @10 | @20 |
| :--- | :--- | :--- | :--- | :--- |
| **Content-Based** | Recall | 0.002770 | 0.003693 | 0.005540 |
| | Precision | 0.000554 | 0.000369 | 0.000277 |
| | MRR | 0.000600 | 0.000754 | 0.000871 |
| **Popularity** | Recall | 0.021237 | 0.025854 | 0.031394 |
| | Precision | 0.004247 | 0.002585 | 0.001570 |
| | MRR | 0.008433 | 0.009017 | 0.009441 |
| **Hybrid ($\alpha=0.90$)** | Recall | 0.021237 | 0.026777 | 0.032318 |
| | Precision | 0.004247 | 0.002678 | 0.001616 |
| | MRR | 0.009249 | 0.009892 | 0.010309 |
| **Hybrid ($\alpha=0.95$)** | Recall | 0.021237 | 0.025854 | 0.032318 |
| | Precision | 0.004247 | 0.002585 | 0.001616 |
| | MRR | 0.008972 | 0.009555 | 0.010038 |

---

## 3. Statistical Diagnostics (vs. Popularity Baseline @ Top-20)

### **Hybrid ($\alpha = 0.90$) vs. Popularity**
- **Total Evaluated Users**: 1,083
- **MRR Wins / Losses / Ties**: 12 / 6 / 1,065
  - Mean Difference: `+0.000868` | Median Difference: `+0.000000`
  - Wilcoxon Test ($n=18$): $p = 0.034028$, **Bonferroni-adjusted $p = 0.068055$**
- **Recall Wins / Losses / Ties**: 1 / 0 / 1,082
  - Mean Difference: `+0.000923` | Median Difference: `+0.000000`
  - Wilcoxon Test ($n=1$): $p = 1.000000$, **Bonferroni-adjusted $p = 1.000000$**

### **Hybrid ($\alpha = 0.95$) vs. Popularity**
- **Total Evaluated Users**: 1,083
- **MRR Wins / Losses / Ties**: 9 / 3 / 1,071
  - Mean Difference: `+0.000597` | Median Difference: `+0.000000`
  - Wilcoxon Test ($n=12$): $p = 0.053711$, **Bonferroni-adjusted $p = 0.107422$**
- **Recall Wins / Losses / Ties**: 1 / 0 / 1,082
  - Mean Difference: `+0.000923` | Median Difference: `+0.000000`
  - Wilcoxon Test ($n=1$): $p = 1.000000$, **Bonferroni-adjusted $p = 1.000000$**

---

## 4. Key Findings & Next Steps
1. **Popularity Dominance**: The global popularity baseline substantially outperforms the Content-Based `UserProfileRecommender` across all cutoffs.
2. **Hybrid Performance**: Hybrid models ($\alpha=0.90$ and $\alpha=0.95$) achieve metrics closely aligned with the popularity baseline, with $\alpha=0.90$ showing the strongest hybrid indicators. However, after Bonferroni correction, the improvements are not statistically significant at the $\alpha=0.05$ level.
3. **Consistency Note**: The exact same evaluation protocol and filtering constraints must be enforced for upcoming evaluations such as the Ranking Shift Analysis to maintain methodological integrity.
