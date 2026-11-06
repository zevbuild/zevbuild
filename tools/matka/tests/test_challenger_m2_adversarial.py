"""
Challenger Tier 5 Adversarial Verification Suite: Milestone 2 Audit Pipeline & Recalibration.
Empirical validation of:
1. evaluate_outcome against edge draws (00, 05, cut pairs, red doubles, boundary ranks 1/10/11/100, type coercion)
2. recalibrate_hyperparameters under extreme scenarios (0%, 100%, 100-loop saturation, boundary rates)
3. Invariants: lambda in [0.25, 0.45], weights sum strictly to 1.0
4. 3-way bit-exact SHA256 parity across mirrors for performance_log.json and prediction_data.json
5. Empirical reproduction of adversarial failure modes in evaluate_outcome and recalibrate_hyperparameters
"""

import os
import sys
import json
import hashlib
import unittest
from pathlib import Path
import numpy as np

# Ensure workspace is importable
WORKSPACE_DIR = Path(__file__).resolve().parent.parent / "kalyan_4_35_to_6_35"
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

from models import evaluate_outcome, recalibrate_hyperparameters, get_family_jodis


class ChallengerM2AdversarialTests(unittest.TestCase):
    """Adversarial stress harness for Milestone 2 audit and tuning logic."""

    def setUp(self):
        self.root_dir = Path(__file__).resolve().parent.parent
        self.kalyan_dir = self.root_dir / "kalyan_4_35_to_6_35"
        self.web_dir = self.kalyan_dir / "web"

    # =========================================================================
    # 1. Adversarial Audit Testing: evaluate_outcome
    # =========================================================================

    def test_evaluate_outcome_rank_boundaries(self):
        """Verify actual_rank and top10_hit at critical boundaries: rank 1, 10, 11, 100."""
        ranked_jodis = [f"{i:02d}" for i in range(100)]
        scores = np.linspace(1.0, 0.01, 100)

        # Rank 1: "00"
        res_r1 = evaluate_outcome(scores, top_pick="00", top_10=ranked_jodis[:10],
                                  otc_digits=[0, 5, 1, 6], actual_jodi="00", actual_open=0, actual_close=0)
        self.assertEqual(res_r1["actual_rank"], 1)
        self.assertTrue(res_r1["top10_hit"])
        self.assertAlmostEqual(res_r1["reciprocal_rank"], 1.0, places=4)

        # Rank 10: "09"
        res_r10 = evaluate_outcome(scores, top_pick="00", top_10=ranked_jodis[:10],
                                   otc_digits=[0, 5, 9, 4], actual_jodi="09", actual_open=0, actual_close=9)
        self.assertEqual(res_r10["actual_rank"], 10)
        self.assertTrue(res_r10["top10_hit"])
        self.assertAlmostEqual(res_r10["reciprocal_rank"], 0.1, places=4)

        # Rank 11: "10" (first out of top 10)
        res_r11 = evaluate_outcome(scores, top_pick="00", top_10=ranked_jodis[:10],
                                   otc_digits=[1, 6, 0, 5], actual_jodi="10", actual_open=1, actual_close=0)
        self.assertEqual(res_r11["actual_rank"], 11)
        self.assertFalse(res_r11["top10_hit"])
        self.assertAlmostEqual(res_r11["reciprocal_rank"], round(1.0 / 11, 4), places=4)

        # Rank 100: "99" (last rank)
        res_r100 = evaluate_outcome(scores, top_pick="00", top_10=ranked_jodis[:10],
                                    otc_digits=[9, 4, 2, 7], actual_jodi="99", actual_open=9, actual_close=9)
        self.assertEqual(res_r100["actual_rank"], 100)
        self.assertFalse(res_r100["top10_hit"])
        self.assertAlmostEqual(res_r100["reciprocal_rank"], 0.01, places=4)

    def test_evaluate_outcome_all_red_doubles(self):
        """Test all 10 red double jodis ('00', '11', ..., '99') to ensure symmetric evaluation."""
        red_doubles = [f"{d}{d}" for d in range(10)]
        for rd in red_doubles:
            d = int(rd[0])
            otc = [d, (d + 5) % 10, (d + 1) % 10, (d + 6) % 10]
            top_10 = [rd] + [f"{i:02d}" for i in range(9)]
            res = evaluate_outcome(top_10, top_pick=rd, top_10=top_10,
                                   otc_digits=otc, actual_jodi=rd, actual_open=d, actual_close=d)
            self.assertEqual(res["actual_rank"], 1, f"Red double {rd} should be rank 1")
            self.assertTrue(res["top10_hit"])
            self.assertTrue(res["family_cut_hit"], f"{rd} must be in its own family")
            self.assertTrue(res["otc_hit"])
            self.assertTrue(res["open_hit"])
            self.assertTrue(res["close_hit"])

    def test_evaluate_outcome_cut_pairs(self):
        """Test complementary cut pairs (e.g. '05', '50', '16', '61', '27', etc.)."""
        cut_pairs = ["05", "50", "16", "61", "27", "72", "38", "83", "49", "94"]
        for cp in cut_pairs:
            op = int(cp[0])
            cl = int(cp[1])
            top_pick = cp
            otc = [op, cl, (op + 1) % 10, ((op + 1) + 5) % 10]
            top_10 = [cp] + [f"{i:02d}" for i in range(9)]

            res = evaluate_outcome(top_10, top_pick=top_pick, top_10=top_10,
                                   otc_digits=otc, actual_jodi=cp, actual_open=op, actual_close=cl)
            self.assertTrue(res["family_cut_hit"])
            self.assertTrue(res["otc_hit"])
            self.assertTrue(res["open_hit"])
            self.assertTrue(res["close_hit"])

    def test_evaluate_outcome_orthogonal_otc_hits(self):
        """Test orthogonal combinations: open only hit, close only hit, neither hit, both hit."""
        otc = [1, 6, 2, 7]

        # Both hit: Open 1, Close 2 (Jodi 12)
        res_both = evaluate_outcome(["12"], top_pick="12", top_10=["12"], otc_digits=otc,
                                    actual_jodi="12", actual_open=1, actual_close=2)
        self.assertTrue(res_both["open_hit"])
        self.assertTrue(res_both["close_hit"])
        self.assertTrue(res_both["otc_hit"])

        # Open only hit: Open 1, Close 4 (Jodi 14)
        res_open = evaluate_outcome(["14"], top_pick="14", top_10=["14"], otc_digits=otc,
                                    actual_jodi="14", actual_open=1, actual_close=4)
        self.assertTrue(res_open["open_hit"])
        self.assertFalse(res_open["close_hit"])
        self.assertTrue(res_open["otc_hit"])

        # Close only hit: Open 8, Close 7 (Jodi 87)
        res_close = evaluate_outcome(["87"], top_pick="87", top_10=["87"], otc_digits=otc,
                                     actual_jodi="87", actual_open=8, actual_close=7)
        self.assertFalse(res_close["open_hit"])
        self.assertTrue(res_close["close_hit"])
        self.assertTrue(res_close["otc_hit"])

        # Neither hit: Open 8, Close 9 (Jodi 89)
        res_none = evaluate_outcome(["89"], top_pick="89", top_10=["89"], otc_digits=otc,
                                    actual_jodi="89", actual_open=8, actual_close=9)
        self.assertFalse(res_none["open_hit"])
        self.assertFalse(res_none["close_hit"])
        self.assertFalse(res_none["otc_hit"])

    def test_evaluate_outcome_type_coercion(self):
        """Test resilience to integer, string with whitespace, float string inputs."""
        otc = [0, 5, 2, 7]
        res = evaluate_outcome(["05"], top_pick="05", top_10=["05"], otc_digits=otc,
                               actual_jodi=5, actual_open="0", actual_close=5.0)
        self.assertEqual(res["actual_rank"], 1)
        self.assertTrue(res["top10_hit"])
        self.assertTrue(res["family_cut_hit"])
        self.assertTrue(res["otc_hit"])

    # =========================================================================
    # 2. Hyperparameter Tuning Guardrail Stress: recalibrate_hyperparameters
    # =========================================================================

    def test_recalibrate_hyperparameters_bounds_invariants(self):
        """Verify lambda in [0.25, 0.45] and sum of weights == 1.0 across extreme conditions."""
        initial_params = {
            "lambda_cut_cluster": 0.35,
            "w_recency": 0.52,
            "w_markov": 0.18,
            "w_seasonal": 0.15,
            "w_gap": 0.15,
        }

        # 100% Hit Rate -> Lambda should increase to 0.37
        log_100 = {"summary": {"family_cut_hit_rate": 1.0}}
        res_100 = recalibrate_hyperparameters(initial_params, log_100)
        self.assertAlmostEqual(res_100["lambda_cut_cluster"], 0.37, places=4)
        w_sum = sum([res_100["w_recency"], res_100["w_markov"], res_100["w_seasonal"], res_100["w_gap"]])
        self.assertAlmostEqual(w_sum, 1.0, places=4)

        # 0% Hit Rate -> Lambda should decrease to 0.33
        log_0 = {"summary": {"family_cut_hit_rate": 0.0}}
        res_0 = recalibrate_hyperparameters(initial_params, log_0)
        self.assertAlmostEqual(res_0["lambda_cut_cluster"], 0.33, places=4)
        w_sum = sum([res_0["w_recency"], res_0["w_markov"], res_0["w_seasonal"], res_0["w_gap"]])
        self.assertAlmostEqual(w_sum, 1.0, places=4)

    def test_recalibrate_hyperparameters_repeated_saturation(self):
        """Run 100 consecutive iterations to verify clamping at ceilings and floors."""
        params = {
            "lambda_cut_cluster": 0.35,
            "w_recency": 0.52,
            "w_markov": 0.18,
            "w_seasonal": 0.15,
            "w_gap": 0.15,
        }

        # 100 iterations of 100% hit rate -> must not exceed 0.45
        for _ in range(100):
            params = recalibrate_hyperparameters(params, {"summary": {"family_cut_hit_rate": 1.0}})
            self.assertLessEqual(params["lambda_cut_cluster"], 0.4501)
            self.assertGreaterEqual(params["lambda_cut_cluster"], 0.2499)
            w_sum = sum([params["w_recency"], params["w_markov"], params["w_seasonal"], params["w_gap"]])
            self.assertAlmostEqual(w_sum, 1.0, places=4)
        self.assertAlmostEqual(params["lambda_cut_cluster"], 0.45, places=4)

        # 100 iterations of 0% hit rate -> must not drop below 0.25
        for _ in range(100):
            params = recalibrate_hyperparameters(params, {"summary": {"family_cut_hit_rate": 0.0}})
            self.assertLessEqual(params["lambda_cut_cluster"], 0.4501)
            self.assertGreaterEqual(params["lambda_cut_cluster"], 0.2499)
            w_sum = sum([params["w_recency"], params["w_markov"], params["w_seasonal"], params["w_gap"]])
            self.assertAlmostEqual(w_sum, 1.0, places=4)
        self.assertAlmostEqual(params["lambda_cut_cluster"], 0.25, places=4)

    def test_recalibrate_hyperparameters_partially_safe_cases(self):
        """Test cases where recalibrate_hyperparameters correctly handles edge values."""
        base_params = {
            "lambda_cut_cluster": 0.35,
            "w_recency": 0.52,
            "w_markov": 0.18,
            "w_seasonal": 0.15,
            "w_gap": 0.15,
        }

        safe_edge_cases = [
            None,
            {},
            {"summary": {}},
            {"summary": {"family_cut_hit_rate": None}},
            {"summary": {"family_cut_hit_rate": float("nan")}},
        ]

        for clog in safe_edge_cases:
            res = recalibrate_hyperparameters(base_params, clog)
            self.assertTrue(isinstance(res, dict))
            self.assertGreaterEqual(res["lambda_cut_cluster"], 0.25)
            self.assertLessEqual(res["lambda_cut_cluster"], 0.45)
            w_sum = sum([res["w_recency"], res["w_markov"], res["w_seasonal"], res["w_gap"]])
            self.assertAlmostEqual(w_sum, 1.0, places=4)

    # =========================================================================
    # 3. Mirror Parity Verification
    # =========================================================================

    def test_performance_log_three_way_sha256_exact_match(self):
        """Verify performance_log.json exists and has identical SHA256 across all 3 mirror paths."""
        paths = [
            self.root_dir / "performance_log.json",
            self.kalyan_dir / "performance_log.json",
            self.web_dir / "performance_log.json",
        ]
        hashes = []
        for p in paths:
            self.assertTrue(p.exists(), f"Mirror missing: {p}")
            with open(p, "rb") as f:
                content = f.read()
            self.assertGreater(len(content), 0, f"File {p} is empty")
            hashes.append(hashlib.sha256(content).hexdigest())

        self.assertEqual(hashes[0], hashes[1], "Root and subfolder performance_log.json SHA256 mismatch")
        self.assertEqual(hashes[1], hashes[2], "Subfolder and web performance_log.json SHA256 mismatch")

    def test_prediction_data_three_way_sha256_exact_match(self):
        """Verify prediction_data.json exists and has identical SHA256 across all 3 mirror paths."""
        paths = [
            self.root_dir / "prediction_data.json",
            self.kalyan_dir / "prediction_data.json",
            self.web_dir / "prediction_data.json",
        ]
        hashes = []
        for p in paths:
            self.assertTrue(p.exists(), f"Mirror missing: {p}")
            with open(p, "rb") as f:
                content = f.read()
            self.assertGreater(len(content), 0, f"File {p} is empty")
            hashes.append(hashlib.sha256(content).hexdigest())

        self.assertEqual(hashes[0], hashes[1], "Root and subfolder prediction_data.json SHA256 mismatch")
        self.assertEqual(hashes[1], hashes[2], "Subfolder and web prediction_data.json SHA256 mismatch")

    # =========================================================================
    # 4. Adversarial Resilience & Defensive Guardrail Verification
    # =========================================================================

    def test_rank_11_boundary_miss(self):
        """Rank 11 boundary miss: if '00' is at rank 11, actual_rank must be 11 and top10_hit must be False."""
        ranked_jodis = [f"{i:02d}" for i in range(1, 11)] + ["00"] + [f"{i:02d}" for i in range(11, 100)]
        scores = np.zeros(100)
        for rank, jodi_str in enumerate(ranked_jodis, start=1):
            scores[int(jodi_str)] = 101 - rank
        res = evaluate_outcome(
            scores,
            top_pick="01",
            top_10=ranked_jodis[:10],
            otc_digits=[1, 6, 2, 7],
            actual_jodi="00",
            actual_open=0,
            actual_close=0,
        )
        self.assertEqual(res["actual_rank"], 11)
        self.assertFalse(res["top10_hit"])
        self.assertAlmostEqual(res["reciprocal_rank"], round(1.0 / 11, 4), places=4)

    def test_red_doubles_and_cut_pairs(self):
        """Verify red doubles and cut pairs with top 10 vs family cut expectations."""
        top_10 = ["64", "41", "14", "69", "19", "46", "96", "91", "00", "05"]
        res = evaluate_outcome(
            top_10,
            top_pick="64",
            top_10=top_10,
            otc_digits=[6, 1, 4, 9],
            actual_jodi="41",
            actual_open=4,
            actual_close=1,
        )
        self.assertTrue(res["top10_hit"])
        self.assertTrue(res["family_cut_hit"])
        self.assertTrue(res["otc_hit"])
        self.assertEqual(res["actual_rank"], 2)

        # Family-enriched Top 10: Jodi in top_10 published list but rank 15 in raw score array
        scores = np.zeros(100)
        ranked = [f"{i:02d}" for i in range(100)]
        for i, j in enumerate(ranked):
            scores[int(j)] = 100 - i
        published_top_10 = ["00", "64", "02", "03", "04", "05", "06", "07", "08", "09"]
        res_enriched = evaluate_outcome(
            scores,
            top_pick="00",
            top_10=published_top_10,
            otc_digits=[0, 5, 6, 1],
            actual_jodi="64",
            actual_open=6,
            actual_close=4,
        )
        self.assertTrue(res_enriched["top10_hit"])
        self.assertEqual(res_enriched["actual_rank"], 2)

    def test_submodel_weight_bounds(self):
        """Verify submodel weight guardrails hold strictly within specified bounds."""
        extreme_cases = [
            {"w_recency": 0.99, "w_markov": 0.001, "w_seasonal": 0.001, "w_gap": 0.001},
            {"w_recency": 0.001, "w_markov": 0.99, "w_seasonal": 0.001, "w_gap": 0.001},
            {"w_recency": 0.001, "w_markov": 0.001, "w_seasonal": 0.99, "w_gap": 0.001},
            {"w_recency": 0.001, "w_markov": 0.001, "w_seasonal": 0.001, "w_gap": 0.99},
        ]
        dummy_log = {"summary": {"family_cut_hit_rate": 0.15}}
        for ec in extreme_cases:
            tuned = recalibrate_hyperparameters(ec, dummy_log)
            self.assertTrue(0.395 <= tuned["w_recency"] <= 0.655, f"w_recency: {tuned['w_recency']}")
            self.assertTrue(0.095 <= tuned["w_markov"] <= 0.305, f"w_markov: {tuned['w_markov']}")
            self.assertTrue(0.075 <= tuned["w_seasonal"] <= 0.225, f"w_seasonal: {tuned['w_seasonal']}")
            self.assertTrue(0.075 <= tuned["w_gap"] <= 0.225, f"w_gap: {tuned['w_gap']}")
            w_sum = sum([tuned["w_recency"], tuned["w_markov"], tuned["w_seasonal"], tuned["w_gap"]])
            self.assertAlmostEqual(w_sum, 1.0, places=4)

    def test_adversarial_resilience_recalibrate_summary_none(self):
        """Verify recalibrate_hyperparameters handles summary=None without raising AttributeError."""
        base_params = {"lambda_cut_cluster": 0.35, "w_recency": 0.52, "w_markov": 0.18, "w_seasonal": 0.15, "w_gap": 0.15}
        res = recalibrate_hyperparameters(base_params, {"summary": None})
        self.assertIsInstance(res, dict)
        self.assertTrue(0.25 <= res["lambda_cut_cluster"] <= 0.45)
        w_sum = sum([res["w_recency"], res["w_markov"], res["w_seasonal"], res["w_gap"]])
        self.assertAlmostEqual(w_sum, 1.0, places=4)

    def test_adversarial_resilience_recalibrate_summary_str(self):
        """Verify recalibrate_hyperparameters handles non-dict summary string without raising AttributeError."""
        base_params = {"lambda_cut_cluster": 0.35, "w_recency": 0.52, "w_markov": 0.18, "w_seasonal": 0.15, "w_gap": 0.15}
        res = recalibrate_hyperparameters(base_params, {"summary": "corrupt_string"})
        self.assertIsInstance(res, dict)
        self.assertTrue(0.25 <= res["lambda_cut_cluster"] <= 0.45)
        w_sum = sum([res["w_recency"], res["w_markov"], res["w_seasonal"], res["w_gap"]])
        self.assertAlmostEqual(w_sum, 1.0, places=4)

    def test_adversarial_resilience_recalibrate_non_numeric_rate(self):
        """Verify recalibrate_hyperparameters handles non-numeric family_cut_hit_rate without raising TypeError."""
        base_params = {"lambda_cut_cluster": 0.35, "w_recency": 0.52, "w_markov": 0.18, "w_seasonal": 0.15, "w_gap": 0.15}
        res = recalibrate_hyperparameters(base_params, {"summary": {"family_cut_hit_rate": "invalid"}})
        self.assertIsInstance(res, dict)
        self.assertTrue(0.25 <= res["lambda_cut_cluster"] <= 0.45)
        w_sum = sum([res["w_recency"], res["w_markov"], res["w_seasonal"], res["w_gap"]])
        self.assertAlmostEqual(w_sum, 1.0, places=4)

    def test_adversarial_resilience_evaluate_outcome_none_open_or_close(self):
        """Verify evaluate_outcome handles None actual_open or actual_close during partial live draw audits."""
        res_open_none = evaluate_outcome(["00"], top_pick="00", top_10=["00"], otc_digits=[0, 5, 1, 6],
                                         actual_jodi="00", actual_open=None, actual_close=0)
        self.assertEqual(res_open_none["actual_rank"], 1)
        self.assertTrue(res_open_none["top10_hit"])
        self.assertFalse(res_open_none["open_hit"])
        self.assertTrue(res_open_none["close_hit"])
        self.assertTrue(res_open_none["otc_hit"])

        res_close_none = evaluate_outcome(["00"], top_pick="00", top_10=["00"], otc_digits=[0, 5, 1, 6],
                                          actual_jodi="00", actual_open=0, actual_close=None)
        self.assertEqual(res_close_none["actual_rank"], 1)
        self.assertTrue(res_close_none["top10_hit"])
        self.assertTrue(res_close_none["open_hit"])
        self.assertFalse(res_close_none["close_hit"])
        self.assertTrue(res_close_none["otc_hit"])


if __name__ == "__main__":
    unittest.main()
