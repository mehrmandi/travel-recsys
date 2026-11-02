"""
Unit tests for the Ranking Shift Analysis module.
"""
import unittest
import numpy as np

from src.evaluation.ranking_shift_analysis import RankingCase, analyze_ranking_shifts

def make_case(user_id, target, items, pop, hyb, exclude=()):
    """Small helper that builds a RankingCase using keyword arguments."""
    return RankingCase(
        user_id=user_id,
        target_index=target,
        item_indices=items,
        popularity_scores=pop,
        hybrid_scores=hyb,
        exclude_indices=exclude,
    )
    
class TestRankComputation(unittest.TestCase):
    def test_tie_breaking_determinism(self):
        # All scores equal: rank is decided by item id (smaller id first).
        scores = [1.0, 1.0, 1.0, 1.0]
        case = make_case("u1", 30, [40, 10, 30, 20], scores, {0.90: scores})
        _, d = analyze_ranking_shifts([case], alphas=[0.90], top_k=20)
        self.assertEqual(d.iloc[0]["popularity_rank"],
                         3)  # ids 10 and 20 come first
        self.assertEqual(d.iloc[0]["hybrid_rank"], 3)
        self.assertTrue(d.iloc[0]["unchanged"])
    
    def test_top_k_boundries(self):
        # Rank 21 -> 20 is a recovery; rank 21 -> 22 stays outside and worsens.
        items = list(range(1, 26))
        pop = [100 - i for i in range(25)]
        hyb_90 = list(pop)
        hyb_90[20] = hyb_90[19] + 0.5
        hyb_95 = list(pop)
        hyb_95[20] = hyb_95[21] - 0.5
        
        case = make_case("u", 21, items, pop, {0.90: hyb_90, 0.95: hyb_95})
        _, d = analyze_ranking_shifts([case], alphas=[0.90, 0.95], top_k=20)

        r90 = d[d["alpha"] == 0.90].iloc[0]
        self.assertEqual((r90["popularity_rank"], r90["hybrid_rank"]), (21, 20))
        self.assertTrue(r90["top_k_recovery"])
        self.assertFalse(r90["both_outside_top_k"])

        r95 = d[d["alpha"] == 0.95].iloc[0]
        self.assertEqual(
            (r95["popularity_rank"], r95["hybrid_rank"]), (21, 22))
        self.assertFalse(r95["top_k_recovery"])
        self.assertTrue(r95["both_outside_top_k"])
        self.assertTrue(r95["worsened"])
        
    def test_top_k_loss(self):
        items = [1, 2, 3]
        case = make_case("u", 1, items, [3.0, 2.0, 1.0], {
                         0.90: [1.0, 2.0, 3.0]})
        _, d = analyze_ranking_shifts([case], alphas=[0.90], top_k=1)
        row = d.iloc[0]
        self.assertEqual((row["popularity_rank"], row["hybrid_rank"]), (1, 3))
        self.assertTrue(row["top_k_loss"])

    def test_exclude_by_item_id_not_position(self):
        case = make_case(
            "u", 300, [100, 200, 300, 400],
            [10.0, 9.0, 8.0, 7.0], {0.90: [10.0, 9.0, 8.0, 7.0]},
            exclude=(100, 200),
        )
        _, d = analyze_ranking_shifts([case], alphas=[0.90])
        self.assertEqual(d.iloc[0]["popularity_rank"], 1)
        self.assertEqual(d.iloc[0]["n_candidates"], 2)

    def test_exclude_indices_generator_is_consumed_once(self):
        case = make_case(
            "u", 300, [100, 200, 300],
            [3.0, 2.0, 1.0], {0.90: [3.0, 2.0, 1.0]},
            exclude=(i for i in (100, 200)),
        )
        self.assertEqual(case.exclude_indices, frozenset({100, 200}))
        _, d = analyze_ranking_shifts([case], alphas=[0.90])
        self.assertEqual(d.iloc[0]["popularity_rank"], 1)
        

class TestFloatHandling(unittest.TestCase):

    def test_float_noise_is_a_tie(self):
        # 1e-15 difference is float noise: scores tie, smaller id (10) ranks first.
        pop = [1.0, 1.0 + 1e-15]
        case = make_case("u", 20, [10, 20], pop, {0.90: pop})
        _, d = analyze_ranking_shifts([case], alphas=[0.90], decimals=12)
        self.assertEqual(d.iloc[0]["popularity_rank"], 2)
        self.assertEqual(d.iloc[0]["hybrid_rank"], 2)

    def test_real_difference_is_not_a_tie(self):
        pop = [1.0, 1.0 + 1e-6]
        case = make_case("u", 20, [10, 20], pop, {0.90: pop})
        _, d = analyze_ranking_shifts([case], alphas=[0.90], decimals=12)
        self.assertEqual(d.iloc[0]["popularity_rank"], 1)

    def test_rounding_disabled_with_none(self):
        pop = [1.0, 1.0 + 1e-15]
        case = make_case("u", 20, [10, 20], pop, {0.90: pop})
        _, d = analyze_ranking_shifts([case], alphas=[0.90], decimals=None)
        self.assertEqual(d.iloc[0]["popularity_rank"], 1)  # exact comparison

    def test_ranks_form_consistent_permutation(self):
        # Scores spaced at sub-tolerance steps must still yield ranks {1, 2, 3}
        # (a tolerance-based tie rule could produce inconsistent ranks here).
        scores = [0.0, 0.8e-12, 1.6e-12]
        ranks = []
        for target in (1, 2, 3):
            case = make_case(f"u{target}", target, [
                             1, 2, 3], scores, {0.90: scores})
            _, d = analyze_ranking_shifts([case], alphas=[0.90])
            ranks.append(int(d.iloc[0]["popularity_rank"]))
        self.assertEqual(sorted(ranks), [1, 2, 3])

    def test_alpha_key_with_float_noise(self):
        key = 0.1 + 0.8  # 0.9000000000000001
        case = make_case("u", 1, [1, 2], [1.0, 2.0], {key: [2.0, 1.0]})
        _, d = analyze_ranking_shifts([case], alphas=[0.9])
        self.assertEqual(d.iloc[0]["hybrid_rank"], 1)


class TestValidation(unittest.TestCase):

    def setUp(self):
        self.ok = make_case("u1", 1, [1], [1.0], {0.90: [1.0]})

    def test_duplicate_alpha(self):
        with self.assertRaisesRegex(ValueError, "Duplicate alpha found"):
            analyze_ranking_shifts([self.ok], alphas=[0.90, 0.90])

    def test_duplicate_user_id(self):
        with self.assertRaisesRegex(ValueError, "Duplicate user_id detected"):
            analyze_ranking_shifts([self.ok, self.ok], alphas=[0.90])

    def test_alpha_out_of_range(self):
        with self.assertRaisesRegex(ValueError, "Alpha must be finite"):
            analyze_ranking_shifts([self.ok], alphas=[1.5])

    def test_invalid_top_k(self):
        with self.assertRaisesRegex(ValueError, "top_k must be a positive integer"):
            analyze_ranking_shifts([self.ok], alphas=[0.90], top_k=0)

    def test_invalid_decimals(self):
        with self.assertRaisesRegex(ValueError, "decimals must be"):
            analyze_ranking_shifts([self.ok], alphas=[0.90], decimals=-1)

    def test_empty_cases(self):
        with self.assertRaisesRegex(ValueError, "No ranking cases"):
            analyze_ranking_shifts([], alphas=[0.90])

    def test_nan_in_popularity(self):
        case = make_case("u", 1, [1, 2], [1.0, np.nan], {0.90: [1.0, 2.0]})
        with self.assertRaisesRegex(ValueError, "Non-finite values in popularity_scores"):
            analyze_ranking_shifts([case], alphas=[0.90])

    def test_nan_in_hybrid(self):
        case = make_case("u", 1, [1, 2], [1.0, 2.0], {0.90: [1.0, np.inf]})
        with self.assertRaisesRegex(ValueError, r"Non-finite values in hybrid_scores"):
            analyze_ranking_shifts([case], alphas=[0.90])

    def test_target_in_exclude(self):
        case = make_case("u", 1, [1, 2], [1.0, 2.0], {
                         0.90: [1.0, 2.0]}, exclude=[1])
        with self.assertRaisesRegex(ValueError, "target_index 1 is in exclude_indices"):
            analyze_ranking_shifts([case], alphas=[0.90])

    def test_target_not_in_items(self):
        case = make_case("u", 999, [1, 2, 3], [1.0, 2.0, 3.0], {
                         0.90: [1.0, 2.0, 3.0]})
        with self.assertRaisesRegex(ValueError, "not in item_indices"):
            analyze_ranking_shifts([case], alphas=[0.90])

    def test_missing_alpha_scores(self):
        case = make_case("u", 1, [1, 2], [1.0, 2.0], {0.90: [1.0, 2.0]})
        with self.assertRaisesRegex(ValueError, "Missing hybrid scores for alpha=0.95"):
            analyze_ranking_shifts([case], alphas=[0.90, 0.95])

    def test_hybrid_length_mismatch(self):
        case = make_case("u", 1, [1, 2], [1.0, 2.0], {0.90: [1.0]})
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            analyze_ranking_shifts([case], alphas=[0.90])

    def test_popularity_length_mismatch(self):
        case = make_case("u", 1, [1, 2], [1.0], {0.90: [1.0, 2.0]})
        with self.assertRaisesRegex(ValueError, "length mismatch"):
            analyze_ranking_shifts([case], alphas=[0.90])


class TestSummary(unittest.TestCase):

    def test_symmetric_improvement_and_loss(self):
        c1 = make_case("u1", 2, [1, 2], [10.0, 5.0], {
                       0.90: [5.0, 10.0]})   # rank 2 -> 1
        c2 = make_case("u2", 2, [1, 2], [5.0, 10.0], {
                       0.90: [10.0, 5.0]})   # rank 1 -> 2
        summary, _ = analyze_ranking_shifts([c1, c2], alphas=[0.90])

        row = summary.iloc[0]
        self.assertEqual(row["users"], 2)
        self.assertEqual(
            (row["improved"], row["worsened"], row["unchanged"]), (1, 1, 0))
        self.assertAlmostEqual(row["mean_rank_change"], 0.0)
        self.assertAlmostEqual(row["wilcoxon_rank_p"], 1.0)
        self.assertEqual(row["wilcoxon_rank_n"], 2)
        self.assertEqual(row["wilcoxon_mrr_n"], 2)

    def test_all_unchanged_gives_nan_p_value(self):
        scores = [3.0, 2.0, 1.0]
        case = make_case("u", 2, [1, 2, 3], scores, {0.90: scores})
        summary, _ = analyze_ranking_shifts([case], alphas=[0.90])
        row = summary.iloc[0]
        self.assertTrue(np.isnan(row["wilcoxon_rank_p"]))
        self.assertTrue(np.isnan(row["wilcoxon_rank_adj_p"]))
        self.assertTrue(np.isnan(row["wilcoxon_mrr_p"]))
        self.assertEqual(row["wilcoxon_rank_n"], 0)
        self.assertEqual(row["wilcoxon_mrr_n"], 0)

    def test_bonferroni_family_size(self):
        # Six users, target always moves from last place to first place, with a
        # different list length per user -> all differences positive and distinct.
        # Exact two-sided Wilcoxon p for n=6, all positive = 2 / 2**6 = 0.03125.
        cases = []
        for n_items in range(2, 8):
            items = list(range(1, n_items + 1))
            # Item 1 is the target: lowest popularity, highest hybrid score.
            pop = [0.0] + [1.0] * (n_items - 1)
            hyb = [1.0] + [0.0] * (n_items - 1)
            cases.append(make_case(f"u{n_items}", 1,
                         items, pop, {0.90: hyb, 0.95: hyb}))

        summary, _ = analyze_ranking_shifts(cases, alphas=[0.90, 0.95])
        row = summary.iloc[0]
        self.assertAlmostEqual(row["wilcoxon_mrr_p"], 0.03125)
        # 2 tests per alpha x 2 alphas = family of 4
        self.assertAlmostEqual(row["wilcoxon_mrr_adj_p"], 0.03125 * 4)
        self.assertAlmostEqual(
            row["wilcoxon_rank_adj_p"], row["wilcoxon_rank_p"] * 4)

    def test_adjusted_p_is_capped_at_one(self):
        c1 = make_case("u1", 2, [1, 2], [10.0, 5.0], {0.90: [5.0, 10.0]})
        c2 = make_case("u2", 2, [1, 2], [5.0, 10.0], {0.90: [10.0, 5.0]})
        summary, _ = analyze_ranking_shifts([c1, c2], alphas=[0.90])
        self.assertLessEqual(summary.iloc[0]["wilcoxon_rank_adj_p"], 1.0)


if __name__ == "__main__":
    unittest.main()
