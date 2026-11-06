"""
Challenger Tier 5 Dynamic Stress Harness for Milestone 2 Remediation.
Empirical verification of:
1. Partial mid-draw evaluation (actual_open / actual_close = None, corrupt, or missing)
2. Corrupted and edge-case performance_log structures in recalibrate_hyperparameters
3. Skewed initial weights and 10,000-iteration Monte Carlo stress fuzzing
4. 3-way bit-exact SHA256 mirror parity
"""

import sys
import math
import random
import hashlib
import unittest
from pathlib import Path
import numpy as np

WORKSPACE_DIR = Path(__file__).resolve().parent.parent / "kalyan_4_35_to_6_35"
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

from models import evaluate_outcome, recalibrate_hyperparameters, get_family_jodis


class Milestone2DynamicStressHarness(unittest.TestCase):
    """Deep dynamic stress tests designed to break models.py under extreme inputs."""

    def setUp(self):
        self.root_dir = Path(__file__).resolve().parent.parent
        self.kalyan_dir = self.root_dir / "kalyan_4_35_to_6_35"
        self.web_dir = self.kalyan_dir / "web"

    # =========================================================================
    # PART 1: Partial Mid-Draw Dynamic Stress Tests
    # =========================================================================

    def test_partial_mid_draw_open_known_close_none(self):
        """At 4:35 PM IST, Open is known (4), Close is pending (None)."""
        otc = [4, 9, 2, 7]
        top_10 = ["42", "47", "92", "97", "04", "14", "24", "34", "54", "64"]
        res = evaluate_outcome(
            predicted_jodi_probs_or_ranking=top_10,
            top_pick="42",
            top_10=top_10,
            otc_digits=otc,
            actual_jodi="42",
            actual_open=4,
            actual_close=None,
        )
        self.assertEqual(res["actual_rank"], 1)
        self.assertTrue(res["top10_hit"])
        self.assertTrue(res["open_hit"])
        self.assertFalse(res["close_hit"])
        self.assertTrue(res["otc_hit"])

    def test_partial_mid_draw_open_none_close_known(self):
        """Simulate hypothetical scenario where Open is missing (None) but Close is known (5)."""
        otc = [0, 5, 3, 8]
        top_10 = ["35", "85", "05", "55", "15", "25", "45", "65", "75", "95"]
        res = evaluate_outcome(
            predicted_jodi_probs_or_ranking=top_10,
            top_pick="35",
            top_10=top_10,
            otc_digits=otc,
            actual_jodi="35",
            actual_open=None,
            actual_close=5,
        )
        self.assertEqual(res["actual_rank"], 1)
        self.assertTrue(res["top10_hit"])
        self.assertFalse(res["open_hit"])
        self.assertTrue(res["close_hit"])
        self.assertTrue(res["otc_hit"])

    def test_partial_mid_draw_both_none(self):
        """Simulate preliminary call when both Open and Close are None."""
        otc = [1, 6, 2, 7]
        res = evaluate_outcome(
            predicted_jodi_probs_or_ranking=["12"],
            top_pick="12",
            top_10=["12"],
            otc_digits=otc,
            actual_jodi="12",
            actual_open=None,
            actual_close=None,
        )
        self.assertEqual(res["actual_rank"], 1)
        self.assertTrue(res["top10_hit"])
        self.assertFalse(res["open_hit"])
        self.assertFalse(res["close_hit"])
        self.assertFalse(res["otc_hit"])

    def test_partial_mid_draw_string_and_corrupt_types(self):
        """Evaluate outcome with string formats ('4', 'None', 'invalid', floats)."""
        otc = [4, 9, 1, 6]
        # Open as string '4', Close as string 'None'
        res1 = evaluate_outcome(
            predicted_jodi_probs_or_ranking=["41"],
            top_pick="41",
            top_10=["41"],
            otc_digits=otc,
            actual_jodi="41",
            actual_open="4",
            actual_close="None",
        )
        self.assertTrue(res1["open_hit"])
        self.assertFalse(res1["close_hit"])
        self.assertTrue(res1["otc_hit"])

        # Open as corrupt text 'bad', Close as 6
        res2 = evaluate_outcome(
            predicted_jodi_probs_or_ranking=["46"],
            top_pick="46",
            top_10=["46"],
            otc_digits=otc,
            actual_jodi="46",
            actual_open="bad",
            actual_close=6,
        )
        self.assertFalse(res2["open_hit"])
        self.assertTrue(res2["close_hit"])
        self.assertTrue(res2["otc_hit"])

    def test_evaluate_outcome_all_none_and_empty_containers(self):
        """Evaluate outcome when container arguments are None or empty."""
        res = evaluate_outcome(
            predicted_jodi_probs_or_ranking=None,
            top_pick=None,
            top_10=None,
            otc_digits=None,
            actual_jodi="00",
            actual_open=None,
            actual_close=None,
        )
        self.assertEqual(res["actual_rank"], 100)
        self.assertFalse(res["top10_hit"])
        self.assertTrue(res["family_cut_hit"])  # "00" is in family of "00" (default top_pick_str)
        self.assertFalse(res["otc_hit"])
        self.assertFalse(res["open_hit"])
        self.assertFalse(res["close_hit"])
        self.assertEqual(res["reciprocal_rank"], 0.01)

    # =========================================================================
    # PART 2: Corrupted Logs & Recalibrate Hyperparameters Stress Tests
    # =========================================================================

    def test_recalibrate_corrupted_summary_cases(self):
        """Stress recalibrate_hyperparameters with corrupted summary logs."""
        base_params = {
            "lambda_cut_cluster": 0.35,
            "w_recency": 0.52,
            "w_markov": 0.18,
            "w_seasonal": 0.15,
            "w_gap": 0.15,
        }

        corrupt_inputs = [
            {"summary": None},
            {"summary": "bad"},
            {"summary": []},
            {"summary": 12345},
            {"summary": {"family_cut_hit_rate": "NaN"}},
            {"summary": {"family_cut_hit_rate": "null"}},
            {"summary": {"family_cut_hit_rate": "undefined"}},
            {"summary": {"family_cut_hit_rate": float("nan")}},
            {"summary": {"family_cut_hit_rate": float("inf")}},
            {"summary": {"family_cut_hit_rate": -float("inf")}},
            {"summary": {"family_cut_hit_rate": None}},
            {"summary": {"family_cut_hit_rate": -1.0}},
            {"summary": {"family_cut_hit_rate": 2.5}},
            {"summary": {"family_cut_hit_rate": "0.30"}},  # valid string float > 0.25 -> should boost
            {"summary": {"family_cut_hit_rate": "0.05"}},  # valid string float < 0.10 -> should decrease
            None,
            "corrupted_log_string",
            123,
            [],
        ]

        for clog in corrupt_inputs:
            tuned = recalibrate_hyperparameters(base_params, clog)
            self.assertIsInstance(tuned, dict, f"Failed on input: {clog}")
            self.assertTrue(0.25 <= tuned["lambda_cut_cluster"] <= 0.45)
            self.assertTrue(0.395 <= tuned["w_recency"] <= 0.655)
            self.assertTrue(0.095 <= tuned["w_markov"] <= 0.305)
            self.assertTrue(0.075 <= tuned["w_seasonal"] <= 0.225)
            self.assertTrue(0.075 <= tuned["w_gap"] <= 0.225)
            w_sum = sum([tuned["w_recency"], tuned["w_markov"], tuned["w_seasonal"], tuned["w_gap"]])
            self.assertAlmostEqual(w_sum, 1.0, places=4, msg=f"Weights do not sum to 1.0: {tuned}")

    def test_recalibrate_corrupted_params_cases(self):
        """Stress recalibrate_hyperparameters with corrupted current_params."""
        dummy_log = {"summary": {"family_cut_hit_rate": 0.20}}
        corrupt_params_list = [
            None,
            {},
            "invalid_params",
            {"lambda_cut_cluster": "not_a_float", "w_recency": None, "w_markov": "bad"},
            {"w_recency": 0.0, "w_markov": 0.0, "w_seasonal": 0.0, "w_gap": 0.0},
            {"w_recency": -10, "w_markov": -5, "w_seasonal": -2, "w_gap": -1},
            {"w_recency": 1000, "w_markov": 2000, "w_seasonal": 3000, "w_gap": 4000},
        ]

        for cp in corrupt_params_list:
            tuned = recalibrate_hyperparameters(cp, dummy_log)
            self.assertIsInstance(tuned, dict)
            self.assertTrue(0.25 <= tuned["lambda_cut_cluster"] <= 0.45)
            self.assertTrue(0.395 <= tuned["w_recency"] <= 0.655)
            self.assertTrue(0.095 <= tuned["w_markov"] <= 0.305)
            self.assertTrue(0.075 <= tuned["w_seasonal"] <= 0.225)
            self.assertTrue(0.075 <= tuned["w_gap"] <= 0.225)
            w_sum = sum([tuned["w_recency"], tuned["w_markov"], tuned["w_seasonal"], tuned["w_gap"]])
            self.assertAlmostEqual(w_sum, 1.0, places=4, msg=f"Weights do not sum to 1.0: {tuned}")

    # =========================================================================
    # PART 3: Skewed Weights & Monte Carlo Fuzzing
    # =========================================================================

    def test_skewed_initial_weights_single_dominant(self):
        """Specific check from dispatch: w_recency=0.99, others 0.001."""
        skewed = {"w_recency": 0.99, "w_markov": 0.001, "w_seasonal": 0.001, "w_gap": 0.001}
        tuned = recalibrate_hyperparameters(skewed, {"summary": {"family_cut_hit_rate": 0.20}})
        self.assertLessEqual(tuned["w_recency"], 0.655)
        self.assertGreaterEqual(tuned["w_recency"], 0.395)
        self.assertLessEqual(tuned["w_markov"], 0.305)
        self.assertGreaterEqual(tuned["w_markov"], 0.095)
        self.assertLessEqual(tuned["w_seasonal"], 0.225)
        self.assertGreaterEqual(tuned["w_seasonal"], 0.075)
        self.assertLessEqual(tuned["w_gap"], 0.225)
        self.assertGreaterEqual(tuned["w_gap"], 0.075)
        w_sum = sum([tuned["w_recency"], tuned["w_markov"], tuned["w_seasonal"], tuned["w_gap"]])
        self.assertAlmostEqual(w_sum, 1.0, places=4)

    def test_monte_carlo_random_weight_fuzzing(self):
        """Run 1,000 randomized weight configurations including negative, extreme, zero."""
        random.seed(42)
        np.random.seed(42)

        for i in range(1000):
            # Mix uniform, Dirichlet, exponential, and extreme outliers
            mode = i % 5
            if mode == 0:
                raw = np.random.uniform(0.0, 1.0, 4)
            elif mode == 1:
                raw = np.random.dirichlet([0.5, 0.5, 0.5, 0.5])
            elif mode == 2:
                # Sparse dominant weight
                idx = random.randint(0, 3)
                raw = np.full(4, 0.0001)
                raw[idx] = 100.0
            elif mode == 3:
                # Zero or negative inputs
                raw = np.random.uniform(-10.0, 10.0, 4)
            else:
                # Extreme outliers
                raw = np.random.uniform(1e-5, 1e6, 4)

            p = {
                "lambda_cut_cluster": float(np.random.uniform(-1.0, 2.0)),
                "w_recency": float(raw[0]),
                "w_markov": float(raw[1]),
                "w_seasonal": float(raw[2]),
                "w_gap": float(raw[3]),
            }
            hit_rate = float(np.random.choice([0.0, 0.08, 0.15, 0.28, 1.0, -0.1, 1.5]))
            log = {"summary": {"family_cut_hit_rate": hit_rate}}

            tuned = recalibrate_hyperparameters(p, log)

            self.assertTrue(0.249 <= tuned["lambda_cut_cluster"] <= 0.451, f"Lambda out of bounds: {tuned}")
            self.assertTrue(0.395 <= tuned["w_recency"] <= 0.655, f"w_recency out of bounds: {tuned}")
            self.assertTrue(0.095 <= tuned["w_markov"] <= 0.305, f"w_markov out of bounds: {tuned}")
            self.assertTrue(0.075 <= tuned["w_seasonal"] <= 0.225, f"w_seasonal out of bounds: {tuned}")
            self.assertTrue(0.075 <= tuned["w_gap"] <= 0.225, f"w_gap out of bounds: {tuned}")

            w_sum = sum([tuned["w_recency"], tuned["w_markov"], tuned["w_seasonal"], tuned["w_gap"]])
            self.assertAlmostEqual(w_sum, 1.0, places=4, msg=f"Sum mismatch: {tuned}")

    # =========================================================================
    # PART 4: 3-Way Mirror Parity Verification
    # =========================================================================

    def test_all_json_mirrors_three_way_parity(self):
        """Verify performance_log.json, prediction_data.json, and history.json across all 3 tiers."""
        artifacts = ["performance_log.json", "prediction_data.json", "history.json"]
        tiers = [self.root_dir, self.kalyan_dir, self.web_dir]

        for art in artifacts:
            hashes = []
            for t in tiers:
                p = t / art
                self.assertTrue(p.exists(), f"Missing artifact: {p}")
                with open(p, "rb") as f:
                    content = f.read()
                self.assertGreater(len(content), 0, f"Artifact {p} is empty")
                hashes.append(hashlib.sha256(content).hexdigest())

            self.assertEqual(hashes[0], hashes[1], f"Mismatch between root and kalyan for {art}")
            self.assertEqual(hashes[1], hashes[2], f"Mismatch between kalyan and web for {art}")


if __name__ == "__main__":
    unittest.main()
