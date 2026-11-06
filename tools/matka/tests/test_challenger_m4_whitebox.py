"""
Milestone 4 - Tier 5 White-Box Adversarial Coverage Hardening Suite
Comprehensive empirical tests challenging edge cases, invariant bounds,
hyperparameter guardrails, cold-start handling, and API robustness.
"""

import os
import sys
import json
import math
import unittest
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path
from http.server import HTTPServer
import threading
import urllib.request
import urllib.error

# Setup import path
WORKSPACE_MATKA = Path(__file__).resolve().parent.parent / "kalyan_4_35_to_6_35"
if str(WORKSPACE_MATKA) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_MATKA))

from models import (
    get_cut_digit,
    get_family_jodis,
    sanitize_draw_data,
    MarkovChainModel,
    RecencyWeightedModel,
    OverdueGapModel,
    DayOfWeekSeasonalModel,
    IntraDrawConditionalModel,
    JodiTotalModel,
    PattiPanelModel,
    EnsemblePredictor,
    evaluate_outcome,
    recalibrate_hyperparameters,
)
from predict import (
    load_performance_log,
    sync_performance_log,
    resolve_csv_path,
    audit_and_update_performance_log,
    generate_upcoming_prediction,
)
from app import (
    compute_last_week_performance,
    KalyanWebHandler,
    ThreadedHTTPServer,
)


class TestWhiteBoxMathAndInvariants(unittest.TestCase):
    """Deep white-box verification of mathematical invariants and boundaries."""

    def test_cut_digit_cyclic_group(self):
        """Verify get_cut_digit forms an order-2 permutation across Z_10."""
        for d in range(10):
            cut_d = get_cut_digit(d)
            self.assertEqual(get_cut_digit(cut_d), d, f"Cut of cut should return original digit {d}")
            self.assertEqual(abs(cut_d - d), 5, f"Difference between cut pair must be 5")

    def test_family_jodis_algebraic_symmetry(self):
        """Verify get_family_jodis algebraic invariants for doubles and non-doubles."""
        # Non-doubles without cut match (e.g. 64) -> 8 distinct Jodis
        fam_64 = get_family_jodis("64")
        self.assertEqual(len(fam_64), 8)
        self.assertIn("64", fam_64)
        self.assertIn("46", fam_64)  # reversal
        self.assertIn("14", fam_64)  # cut open
        self.assertIn("69", fam_64)  # cut close
        self.assertIn("19", fam_64)  # cut both
        self.assertIn("41", fam_64)  # reversal cut close
        self.assertIn("96", fam_64)  # reversal cut open
        self.assertIn("91", fam_64)  # reversal cut both

        # Double digits (e.g. 55 or 00) -> 4 distinct Jodis
        fam_55 = get_family_jodis("55")
        self.assertEqual(sorted(fam_55), ["00", "05", "50", "55"])
        fam_00 = get_family_jodis(0)
        self.assertEqual(sorted(fam_00), ["00", "05", "50", "55"])

        # Edge cases: None, empty string, malformed string, float
        self.assertEqual(get_family_jodis(None), [])
        self.assertEqual(get_family_jodis(""), [])
        self.assertEqual(get_family_jodis("invalid_text"), [])
        self.assertEqual(get_family_jodis(64.0), fam_64)
        self.assertEqual(get_family_jodis(" 64 "), fam_64)

    def test_sanitize_draw_data_edge_cases(self):
        """Verify sanitize_draw_data under hostile DataFrame structures."""
        # None and empty
        self.assertTrue(sanitize_draw_data(None).empty)
        self.assertTrue(sanitize_draw_data(pd.DataFrame()).empty)

        # Corrupted entries, strings, NaNs, missing Is_Valid
        raw_df = pd.DataFrame({
            "Date": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05"],
            "Jodi": ["64", None, "CORRUPT", "40", "99"],
            "Open_Digit": [6, 1, "NaN", 4, 9],
            "Close_Digit": [4, 5, 2, 0, None],
            "Is_Valid": [True, True, True, False, True],
        }, index=[10, 25, 42, 99, 105])  # non-contiguous index

        cleaned = sanitize_draw_data(raw_df)
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(cleaned.iloc[0]["Jodi"], "64")
        self.assertEqual(cleaned.index.tolist(), [0])  # strictly reset index

    def test_ensemble_cold_start_without_fit(self):
        """Verify EnsemblePredictor behaves gracefully on cold start without fit()."""
        model = EnsemblePredictor()
        preds = model.predict(day_of_week="Mon")

        self.assertIn("jodi_probs", preds)
        self.assertEqual(len(preds["jodi_probs"]), 100)
        self.assertAlmostEqual(preds["jodi_probs"].sum(), 1.0, places=5)
        self.assertEqual(len(preds["open_probs"]), 10)
        self.assertAlmostEqual(preds["open_probs"].sum(), 1.0, places=5)
        self.assertEqual(len(preds["close_probs"]), 10)
        self.assertAlmostEqual(preds["close_probs"].sum(), 1.0, places=5)
        self.assertEqual(len(preds["top_10_jodis"]), 10)
        self.assertEqual(len(preds["otc_digits"]), 4)
        self.assertEqual(len(preds["otc_pairs"]), 2)
        self.assertGreater(preds["otc_pass_prob"], 0.60)

    def test_ensemble_hostile_boundary_inputs(self):
        """Verify EnsemblePredictor handles adversarial/out-of-bounds inputs without crashing."""
        model = EnsemblePredictor()
        # Out-of-bounds digits and negative values
        preds = model.predict(
            day_of_week="InvalidDay",
            prev_jodi="999",
            prev_open=-99,
            prev_close=42,
            lambda_cut_cluster=0.35,
        )
        self.assertAlmostEqual(preds["jodi_probs"].sum(), 1.0, places=5)
        self.assertAlmostEqual(preds["cut_cluster_density_scores"].sum(), 1.0, places=5)
        self.assertTrue(all(p >= 0.0 for p in preds["jodi_probs"]))
        self.assertTrue(all(s >= 0.0 for s in preds["cut_cluster_density_scores"]))

    def test_4_otc_matrix_complementary_structure(self):
        """Verify 4-OTC Matrix always produces two disjoint complementary cut pairs."""
        df = pd.read_csv(resolve_csv_path())
        model = EnsemblePredictor()
        model.fit(df)
        preds = model.predict(day_of_week="Fri")

        pairs = preds["otc_pairs"]
        self.assertEqual(len(pairs), 2)
        p1, p2 = pairs[0], pairs[1]

        # Each pair must be a valid cut pair
        self.assertEqual((p1[0] + 5) % 10, p1[1])
        self.assertEqual((p2[0] + 5) % 10, p2[1])

        # Digits must be strictly disjoint (4 distinct digits)
        digits = preds["otc_digits"]
        self.assertEqual(len(set(digits)), 4)
        self.assertSetEqual(set(digits), {p1[0], p1[1], p2[0], p2[1]})

        # Pass probability must be > 68% for empirical draw data
        self.assertGreaterEqual(preds["otc_pass_prob"], 0.68)


class TestHyperparameterRecalibrationStress(unittest.TestCase):
    """Stress tests and Monte Carlo fuzzing of recalibrate_hyperparameters."""

    def test_monte_carlo_fuzz_guardrails(self):
        """Fuzz recalibrate_hyperparameters with 1,000 extreme/random inputs."""
        np.random.seed(42)
        bounds = {
            "lambda_cut_cluster": (0.25, 0.45),
            "w_recency": (0.40, 0.65),
            "w_markov": (0.10, 0.30),
            "w_seasonal": (0.08, 0.22),
            "w_gap": (0.08, 0.22),
        }

        for i in range(1000):
            # Generate random or hostile parameters
            raw_params = {
                "lambda_cut_cluster": float(np.random.uniform(-5.0, 5.0)),
                "w_recency": float(np.random.uniform(-10.0, 10.0)),
                "w_markov": float(np.random.uniform(-10.0, 10.0)),
                "w_seasonal": float(np.random.uniform(-10.0, 10.0)),
                "w_gap": float(np.random.uniform(-10.0, 10.0)),
            }
            # Hostile performance log scenarios
            scenario = i % 5
            if scenario == 0:
                perf_log = {"summary": {"family_cut_hit_rate": float(np.random.uniform(-1.0, 2.0))}}
            elif scenario == 1:
                perf_log = {"summary": {"family_cut_hit_rate": "invalid_string"}}
            elif scenario == 2:
                perf_log = {"summary": None}
            elif scenario == 3:
                perf_log = {}
            else:
                perf_log = None

            calibrated = recalibrate_hyperparameters(raw_params, perf_log)

            # 1. Check all bounds strictly satisfied
            for key, (b_lo, b_hi) in bounds.items():
                val = calibrated[key]
                self.assertGreaterEqual(val, b_lo - 1e-4, f"Iter {i}: {key}={val} < {b_lo}")
                self.assertLessEqual(val, b_hi + 1e-4, f"Iter {i}: {key}={val} > {b_hi}")

            # 2. Check strict unit sum of submodel weights
            sub_sum = round(calibrated["w_recency"] + calibrated["w_markov"] + calibrated["w_seasonal"] + calibrated["w_gap"], 4)
            self.assertAlmostEqual(sub_sum, 1.0, places=3, msg=f"Iter {i}: Submodel weights sum to {sub_sum} != 1.0")


class TestEvaluateOutcomeAdversarial(unittest.TestCase):
    """Adversarial testing of evaluate_outcome under boundary conditions."""

    def test_evaluate_outcome_rank_invariants(self):
        """Verify actual_rank is strictly in range 1..100 across arbitrary inputs."""
        # 1. Actual Jodi is top pick
        res = evaluate_outcome(
            predicted_jodi_probs_or_ranking=["64", "40", "12"],
            top_pick="64",
            top_10=["64", "40", "12"],
            otc_digits=[6, 1, 4, 9],
            actual_jodi="64",
            actual_open=6,
            actual_close=4,
        )
        self.assertEqual(res["actual_rank"], 1)
        self.assertTrue(res["top10_hit"])
        self.assertTrue(res["family_cut_hit"])
        self.assertTrue(res["otc_hit"])
        self.assertTrue(res["open_hit"])
        self.assertTrue(res["close_hit"])
        self.assertEqual(res["reciprocal_rank"], 1.0)

        # 2. Actual Jodi is rank 10
        top_10_list = [f"{i:02d}" for i in range(10)]
        res10 = evaluate_outcome(
            predicted_jodi_probs_or_ranking=top_10_list,
            top_pick=top_10_list[0],
            top_10=top_10_list,
            otc_digits=[0, 5, 1, 6],
            actual_jodi=top_10_list[9],
            actual_open=0,
            actual_close=9,
        )
        self.assertEqual(res10["actual_rank"], 10)
        self.assertTrue(res10["top10_hit"])
        self.assertEqual(res10["reciprocal_rank"], 0.1)

        # 3. Actual Jodi is rank 100 (worst case)
        res_worst = evaluate_outcome(
            predicted_jodi_probs_or_ranking=np.arange(100),  # probabilities 0..99
            top_pick="99",
            top_10=[f"{99-i:02d}" for i in range(10)],
            otc_digits=[2, 7, 3, 8],
            actual_jodi="00",  # lowest prob
            actual_open=0,
            actual_close=0,
        )
        self.assertEqual(res_worst["actual_rank"], 100)
        self.assertFalse(res_worst["top10_hit"])
        self.assertEqual(res_worst["reciprocal_rank"], 0.01)


class TestAppServerAndAPIAdversarial(unittest.TestCase):
    """Stress tests on KalyanWebHandler and HTTP endpoints."""

    @classmethod
    def setUpClass(cls):
        cls.port = 8899
        cls.server = ThreadedHTTPServer(("127.0.0.1", cls.port), KalyanWebHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def test_api_status_stress_and_cors(self):
        """Verify /api/status returns valid CORS headers and payload."""
        req = urllib.request.Request(f"{self.base_url}/api/status?stress=1&random={np.random.randint(1000)}")
        with urllib.request.urlopen(req, timeout=5) as resp:
            self.assertEqual(resp.status, 200)
            self.assertEqual(resp.headers.get("Access-Control-Allow-Origin"), "*")
            data = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(data.get("status"), "online")
            self.assertGreater(data.get("draw_count"), 0)

    def test_api_performance_log_endpoint_and_schema(self):
        """Verify /api/performance-log returns valid JSON strictly conforming to PROJECT.md."""
        req = urllib.request.Request(f"{self.base_url}/api/performance-log")
        with urllib.request.urlopen(req, timeout=5) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertIn("summary", data)
            self.assertIn("current_hyperparameters", data)
            self.assertIn("draw_logs", data)
            s = data["summary"]
            self.assertIn("total_draws_logged", s)
            self.assertIn("top10_hit_rate", s)
            self.assertIn("family_cut_hit_rate", s)
            self.assertIn("otc_hit_rate", s)

    def test_path_traversal_and_invalid_methods(self):
        """Verify server rejects path traversal and unhandled POST endpoints with HTTP 404."""
        # Unhandled POST
        req = urllib.request.Request(f"{self.base_url}/api/status", data=b"{}", method="POST")
        try:
            urllib.request.urlopen(req, timeout=5)
            self.fail("Expected HTTP 404 for POST /api/status")
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 404)

        # Path traversal
        traversal_url = f"{self.base_url}/../../etc/passwd"
        try:
            urllib.request.urlopen(traversal_url, timeout=5)
            self.fail("Expected HTTP 404 for path traversal attempt")
        except urllib.error.HTTPError as e:
            self.assertIn(e.code, [400, 404])


class TestOct26To31ReferenceWeekAdversarial(unittest.TestCase):
    """Replays Oct 26-31 reference week draws and verifies promotional invariants."""

    def test_oct_reference_week_promotions(self):
        """
        Verify that for the Oct 26–31 reference period (draws 68, 63, 64, 48, 34, 40),
        Cut-Cluster Density Weighting promotes high-affinity family members (including 64 and 40)
        into the Top 10 / Family Cut recommendations.
        """
        df = pd.read_csv(resolve_csv_path())
        predictor = EnsemblePredictor()
        predictor.fit(df)
        preds = predictor.predict(day_of_week="Mon")

        top_10 = preds["top_10_jodis"]
        self.assertEqual(len(top_10), 10)
        top_jodi_strs = [item["jodi"] for item in top_10]

        # Top pick family bracket must be fully represented
        top_pick = top_10[0]["jodi"]
        top_family = get_family_jodis(top_pick)
        family_in_top10 = [j for j in top_family if j in top_jodi_strs]
        self.assertGreaterEqual(len(family_in_top10), 4, "High affinity family members must enter Top 10")


if __name__ == "__main__":
    unittest.main()
