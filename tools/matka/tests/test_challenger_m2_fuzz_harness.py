"""
Empirical Challenger Fuzz and Stress Harness for Milestone 2.
Adversarial fuzzing of evaluate_outcome, recalibrate_hyperparameters, and submodel bounds.
"""

import sys
import random
import unittest
from pathlib import Path
import numpy as np

WORKSPACE_DIR = Path(__file__).resolve().parent.parent / "kalyan_4_35_to_6_35"
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

from models import recalibrate_hyperparameters, evaluate_outcome, get_family_jodis


class ChallengerM2FuzzStressTest(unittest.TestCase):
    """Deep fuzz harness challenging invariant compliance under hostile inputs."""

    def test_recalibrate_hyperparameters_10000_fuzz(self):
        """Run 10,000 randomized and pathological inputs through recalibrate_hyperparameters."""
        bounds = {
            "w_recency": (0.40, 0.65),
            "w_markov": (0.10, 0.30),
            "w_seasonal": (0.08, 0.22),
            "w_gap": (0.08, 0.22),
        }

        for i in range(10000):
            mode = i % 8
            if mode == 0:
                params = {}
            elif mode == 1:
                params = None
            elif mode == 2:
                params = "invalid_string"
            elif mode == 3:
                params = {
                    "w_recency": random.choice([0.0, 1.0, -10.0, 100.0, -1.0]),
                    "w_markov": random.choice([0.0, 1.0, -10.0, 100.0, -1.0]),
                    "w_seasonal": random.choice([0.0, 1.0, -10.0, 100.0, -1.0]),
                    "w_gap": random.choice([0.0, 1.0, -10.0, 100.0, -1.0]),
                    "lambda_cut_cluster": random.choice([0.0, 1.0, -1.0, 50.0]),
                }
            elif mode == 4:
                params = {"w_recency": "str", "w_markov": None, "w_seasonal": [], "w_gap": {}}
            else:
                params = {
                    "w_recency": random.uniform(-10, 10),
                    "w_markov": random.uniform(-10, 10),
                    "w_seasonal": random.uniform(-10, 10),
                    "w_gap": random.uniform(-10, 10),
                    "lambda_cut_cluster": random.uniform(-10, 10),
                }

            log_mode = (i // 8) % 8
            if log_mode == 0:
                log = None
            elif log_mode == 1:
                log = {}
            elif log_mode == 2:
                log = "corrupt_string"
            elif log_mode == 3:
                log = {"summary": None}
            elif log_mode == 4:
                log = {"summary": "corrupt_summary"}
            elif log_mode == 5:
                log = {
                    "summary": {
                        "family_cut_hit_rate": random.choice(
                            ["invalid", None, float("nan"), -1.0, 0.0, 0.05, 0.20, 0.35, 1.0, 100.0]
                        )
                    }
                }
            elif log_mode == 6:
                log = {"summary": {"family_cut_hit_rate": random.uniform(-10, 10)}}
            else:
                log = [1, 2, 3]

            res = recalibrate_hyperparameters(params, log)
            self.assertIsInstance(res, dict)
            self.assertTrue(0.2499 <= res["lambda_cut_cluster"] <= 0.4501)
            w_sum = sum([res["w_recency"], res["w_markov"], res["w_seasonal"], res["w_gap"]])
            self.assertAlmostEqual(w_sum, 1.0, places=3)
            # Verify each submodel weight stays within its guardrail box (+- 0.005 for rounding/slack)
            self.assertTrue(
                bounds["w_recency"][0] - 0.005 <= res["w_recency"] <= bounds["w_recency"][1] + 0.005,
                f"w_recency {res['w_recency']} violated ({bounds['w_recency']})",
            )
            self.assertTrue(
                bounds["w_markov"][0] - 0.005 <= res["w_markov"] <= bounds["w_markov"][1] + 0.005,
                f"w_markov {res['w_markov']} violated ({bounds['w_markov']})",
            )
            self.assertTrue(
                bounds["w_seasonal"][0] - 0.005 <= res["w_seasonal"] <= bounds["w_seasonal"][1] + 0.005,
                f"w_seasonal {res['w_seasonal']} violated ({bounds['w_seasonal']})",
            )
            self.assertTrue(
                bounds["w_gap"][0] - 0.005 <= res["w_gap"] <= bounds["w_gap"][1] + 0.005,
                f"w_gap {res['w_gap']} violated ({bounds['w_gap']})",
            )

    def test_evaluate_outcome_10000_fuzz(self):
        """Run 10,000 hostile edge cases through evaluate_outcome."""
        for i in range(10000):
            # Adversarial actual_jodi
            jodi_choices = [
                "00", "05", "64", "40", "99", 0, 5, 64, 40.0, None, "", "  ", "??", "***", 105, "abc"
            ]
            act_jodi = random.choice(jodi_choices)

            # Adversarial actual_open, actual_close
            open_choices = [0, 4, 6, 9, "0", "6", 4.0, None, "", "***", -1, 10, "bad"]
            close_choices = [0, 1, 4, 8, "0", "8", 1.0, None, "", "***", -1, 10, "bad"]
            act_o = random.choice(open_choices)
            act_c = random.choice(close_choices)

            # Adversarial top_pick
            top_pick_choices = ["00", "64", 64, 0, 5, "40", "99", None]
            tp = random.choice(top_pick_choices)

            # Adversarial top_10
            top_10_choices = [
                None,
                [],
                [f"{x:02d}" for x in range(10)],
                ["64", "40", "00", "05", "11", "22", "33", "44", "55", "66"],
                [None, "invalid", 64, 0, 5.0],
                ["00"] * 10,
            ]
            t10 = random.choice(top_10_choices)

            # Adversarial otc_digits
            otc_choices = [
                None,
                [],
                [1, 6, 4, 9],
                ["1", "6", "4", "9"],
                [0, 5, 2, 7],
                [0, 1, 2, 3, 4, 5],
            ]
            otc = random.choice(otc_choices)

            # Adversarial predicted_jodi_probs_or_ranking
            pred_choices = [
                np.ones(100) / 100.0,
                np.zeros(100),
                np.linspace(1.0, 0.0, 100),
                [f"{x:02d}" for x in range(100)],
                [f"{x:02d}" for x in reversed(range(100))],
                None,
                [],
                ["64", "40", "00"],
            ]
            pred = random.choice(pred_choices)

            res = evaluate_outcome(pred, tp, t10, otc, act_jodi, act_o, act_c)

            self.assertIsInstance(res, dict)
            self.assertIn("actual_rank", res)
            self.assertIn("top10_hit", res)
            self.assertIn("family_cut_hit", res)
            self.assertIn("otc_hit", res)
            self.assertIn("open_hit", res)
            self.assertIn("close_hit", res)
            self.assertIn("reciprocal_rank", res)

            self.assertIsInstance(res["actual_rank"], int)
            self.assertTrue(1 <= res["actual_rank"] <= 100)
            self.assertIsInstance(res["top10_hit"], bool)
            self.assertIsInstance(res["family_cut_hit"], bool)
            self.assertIsInstance(res["otc_hit"], bool)
            self.assertIsInstance(res["open_hit"], bool)
            self.assertIsInstance(res["close_hit"], bool)
            self.assertIsInstance(res["reciprocal_rank"], float)
            self.assertTrue(0.0 <= res["reciprocal_rank"] <= 1.0)


if __name__ == "__main__":
    unittest.main()
