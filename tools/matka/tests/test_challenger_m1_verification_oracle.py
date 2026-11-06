r"""
Milestone 1 Challenger Empirical Verification Oracle & Mathematical Stress Suite
================================================================================
Empirical Challenger: challenger_m1_2_3
Roles: critic, specialist

Rigorous automated verification of:
1. Reference Week Replay (Oct 26–31, 2026: 68, 63, 64, 48, 34, 40)
   - High-affinity family members and numbers 64 and 40 representation in Top 10.
2. Mathematical Invariants:
   - Coherent probability normalization (Open, Close, Jodi, Density Scores sum to 1.0 within 1e-6).
   - Cut-Cluster Density Weighting formula adherence: S(j) = (1 - \lambda) * P(j) + \lambda * Mean(Family(j)).
   - Orthogonal two-partition cross-transition scaling ensuring post-norm mass == 0.2070.
   - Dual cut-pair harmonic resonance structure [(d1, cut(d1)), (d2, cut(d2))].
3. Adversarial Edge Cases:
   - Holiday / NaN / string-float corrupted DataFrames.
   - Zero-length / single-row history DataFrames.
   - Dirac delta and uniform distribution stress testing.
   - Prev_open == Prev_close collision scenarios (e.g., Jodi 55).
   - Boundary lambda values (\lambda = 0.0 and \lambda = 1.0).
"""

import math
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
MATKA_ROOT = REPO_ROOT / "tools" / "matka"
MATKA_ENGINE = MATKA_ROOT / "kalyan_4_35_to_6_35"

for p in [str(REPO_ROOT), str(MATKA_ROOT), str(MATKA_ENGINE)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from models import (
    EnsemblePredictor,
    get_family_jodis,
    get_cut_digit,
    sanitize_draw_data,
)


def independent_oracle_family(jodi_str: str) -> list:
    val = int(float(jodi_str))
    o = val // 10
    c = val % 10
    co = (o + 5) % 10
    cc = (c + 5) % 10
    pairs = [
        (o, c), (o, cc), (co, c), (co, cc),
        (c, o), (cut_c := cc, o), (c, co), (cc, co),
    ]
    seen = set()
    res = []
    for d1, d2 in pairs:
        s = f"{d1}{d2}"
        if s not in seen:
            seen.add(s)
            res.append(s)
    return sorted(res)


def independent_cut_cluster_formula(jodi_probs: np.ndarray, lam: float = 0.35) -> np.ndarray:
    scores = np.zeros(100, dtype=float)
    for j in range(100):
        fam = independent_oracle_family(f"{j:02d}")
        fam_indices = [int(x) for x in fam]
        scores[j] = (1.0 - lam) * jodi_probs[j] + lam * float(np.mean(jodi_probs[fam_indices]))
    return scores / scores.sum()


class TestReferenceWeekReplayVerification(unittest.TestCase):
    """Empirically replay and verify the Oct 26–31 reference week."""

    @classmethod
    def setUpClass(cls):
        csv_file = MATKA_ROOT / "kalyan_historical_data.csv"
        cls.df = pd.read_csv(csv_file)
        cls.predictor = EnsemblePredictor()
        cls.predictor.fit(cls.df)

        cls.ref_draws = [
            {"date": "2026-10-26", "day": "Mon", "actual_j": "68", "actual_o": 6, "actual_c": 8, "prev_j": "91", "prev_o": 9, "prev_c": 1},
            {"date": "2026-10-27", "day": "Tue", "actual_j": "63", "actual_o": 6, "actual_c": 3, "prev_j": "68", "prev_o": 6, "prev_c": 8},
            {"date": "2026-10-28", "day": "Wed", "actual_j": "64", "actual_o": 6, "actual_c": 4, "prev_j": "63", "prev_o": 6, "prev_c": 3},
            {"date": "2026-10-29", "day": "Thu", "actual_j": "48", "actual_o": 4, "actual_c": 8, "prev_j": "64", "prev_o": 6, "prev_c": 4},
            {"date": "2026-10-30", "day": "Fri", "actual_j": "34", "actual_o": 3, "actual_c": 4, "prev_j": "48", "prev_o": 4, "prev_c": 8},
            {"date": "2026-10-31", "day": "Sat", "actual_j": "40", "actual_o": 4, "actual_c": 0, "prev_j": "34", "prev_o": 3, "prev_c": 4},
        ]

    def test_reference_week_jodi_representation(self):
        """Verify Jodi 64 and 40 are successfully represented in Top 10 recommendations."""
        oct_28_res = self.predictor.predict(
            day_of_week="Wed",
            prev_jodi="63",
            prev_open=6,
            prev_close=3
        )
        oct_28_top10 = [x["jodi"] for x in oct_28_res["top_10_jodis"]]
        self.assertIn("64", oct_28_top10, "Reference week draw 64 MUST be in Top 10 recommendations on 2026-10-28")
        self.assertEqual(oct_28_top10[0], "64", "64 should be Top Pick (Rank 1) on 2026-10-28")

        oct_31_res = self.predictor.predict(
            day_of_week="Sat",
            prev_jodi="34",
            prev_open=3,
            prev_close=4
        )
        oct_31_top10 = [x["jodi"] for x in oct_31_res["top_10_jodis"]]
        self.assertIn("40", oct_31_top10, "Reference week draw 40 MUST be in Top 10 recommendations on 2026-10-31")

    def test_reference_week_full_hit_rates(self):
        """Verify empirical performance across all 6 reference week days."""
        otc_hits = 0
        top10_hits = 0

        for d in self.ref_draws:
            res = self.predictor.predict(
                day_of_week=d["day"],
                prev_jodi=d["prev_j"],
                prev_open=d["prev_o"],
                prev_close=d["prev_c"]
            )
            top10_list = [x["jodi"] for x in res["top_10_jodis"]]
            otc = res["otc_digits"]

            if d["actual_j"] in top10_list:
                top10_hits += 1
            if d["actual_o"] in otc or d["actual_c"] in otc:
                otc_hits += 1

            self.assertGreaterEqual(res["otc_pass_prob"], 0.60, f"Modeled OTC pass prob {res['otc_pass_prob']} below 0.60 on {d['date']}")
            self.assertEqual(len(res["otc_pairs"]), 2, "Must output exactly 2 cut pairs")
            self.assertEqual(len(res["otc_digits"]), 4, "Must output exactly 4 OTC digits")

        self.assertEqual(top10_hits, 6, "All 6 reference week draws (68, 63, 64, 48, 34, 40) should be in Top 10")
        self.assertEqual(otc_hits, 6, "All 6 reference week draws should hit the 4-OTC matrix")


class TestMathematicalInvariants(unittest.TestCase):
    """Verify strict mathematical invariants across distributions and algorithms."""

    def setUp(self):
        csv_file = MATKA_ROOT / "kalyan_historical_data.csv"
        self.df = pd.read_csv(csv_file)
        self.predictor = EnsemblePredictor()
        self.predictor.fit(self.df)

    def test_probability_sums_to_unity(self):
        """Verify open_probs, close_probs, jodi_probs, and density scores sum to 1.0 within 1e-6 across 50 draw configs."""
        days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
        for o in range(10):
            for c in [0, 3, 5, 8, 9]:
                d = days[(o + c) % len(days)]
                res = self.predictor.predict(
                    day_of_week=d,
                    prev_open=o,
                    prev_close=c,
                    prev_jodi=f"{o}{c}"
                )
                self.assertAlmostEqual(res["open_probs"].sum(), 1.0, places=6, msg=f"open_probs failed unity for o={o}, c={c}")
                self.assertAlmostEqual(res["close_probs"].sum(), 1.0, places=6, msg=f"close_probs failed unity for o={o}, c={c}")
                self.assertAlmostEqual(res["jodi_probs"].sum(), 1.0, places=6, msg=f"jodi_probs failed unity for o={o}, c={c}")
                self.assertAlmostEqual(res["cut_cluster_density_scores"].sum(), 1.0, places=6, msg=f"density_scores failed unity for o={o}, c={c}")

                # Non-negativity
                self.assertTrue(np.all(res["open_probs"] >= 0.0))
                self.assertTrue(np.all(res["close_probs"] >= 0.0))
                self.assertTrue(np.all(res["jodi_probs"] >= 0.0))
                self.assertTrue(np.all(res["cut_cluster_density_scores"] >= 0.0))

    def test_cut_cluster_density_weighting_formula_strict_adherence(self):
        r"""Verify that S(j) strictly equals (1 - \lambda)*P(j) + \lambda*Mean(Family(j))."""
        res = self.predictor.predict(day_of_week="Wed", prev_open=6, prev_close=3, prev_jodi="63")
        jodi_probs = res["jodi_probs"]
        model_scores = res["cut_cluster_density_scores"]

        expected_scores = independent_cut_cluster_formula(jodi_probs, lam=0.35)
        np.testing.assert_allclose(
            model_scores,
            expected_scores,
            atol=1e-6,
            err_msg="Cut-Cluster density score deviates from mathematical formula"
        )

    def test_cross_transition_standalone_without_open_boost(self):
        """
        Verify that orthogonal two-partition scaling achieves 0.2070 when
        the prior vector sums to 1.0 (no open repetition boost distortion).
        """
        p_isolated = EnsemblePredictor()
        p_isolated.fit(self.df)
        p_isolated.last_open_digit = None  # Ensure no fallback open boost

        for prev_c in range(10):
            res = p_isolated.predict(day_of_week="Tue", prev_open=None, prev_close=prev_c, prev_jodi=None)
            open_probs = res["open_probs"]
            c_dir = prev_c
            c_cut = (prev_c + 5) % 10
            post_cross_mass = open_probs[c_dir] + open_probs[c_cut]
            self.assertGreaterEqual(
                post_cross_mass,
                0.2069,
                f"Isolated post-normalization cross mass {post_cross_mass} fell below 0.2070 for prev_close={prev_c}"
            )

    def test_cross_transition_invariant_when_integrated_with_open_boost(self):
        """
        POST-PATCH EMPIRICAL INVARIANT TEST:
        When prev_open is provided, open_probs[prev_open] *= 1.15 is normalized before
        cross-transition scaling. rem_curr accurately reflects remaining digits mass,
        ensuring post-normalization cross mass achieves exactly 0.2070.
        """
        res = self.predictor.predict(day_of_week="Tue", prev_open=1, prev_close=0, prev_jodi="10")
        open_probs = res["open_probs"]
        post_cross_mass = float(open_probs[0] + open_probs[5])
        self.assertAlmostEqual(
            post_cross_mass,
            0.2070,
            places=6,
            msg=f"Expected exact post-normalization mass 0.2070, got {post_cross_mass}"
        )

    def test_complementary_4_otc_matrix_cut_pair_geometry(self):
        """Verify 4-OTC output always has 2 complementary harmonic cut pairs."""
        for prev_o in range(10):
            for prev_c in range(10):
                res = self.predictor.predict(prev_open=prev_o, prev_close=prev_c, prev_jodi=f"{prev_o}{prev_c}")
                pairs = res["otc_pairs"]
                digits = res["otc_digits"]

                self.assertEqual(len(pairs), 2, "Must output exactly 2 cut pairs")
                self.assertEqual(len(digits), 4, "Must output exactly 4 unique digits")
                for p in pairs:
                    self.assertEqual((p[0] + 5) % 10, p[1], f"Pair {p} is not a harmonic cut pair")
                for p in pairs:
                    self.assertIn(p[0], digits)
                    self.assertIn(p[1], digits)


class TestAdversarialInputsAndRobustness(unittest.TestCase):
    """Stress-test models against corrupted, extreme, and malformed inputs."""

    def test_sanitize_draw_data_with_holiday_and_corrupted_rows(self):
        """Verify sanitize_draw_data cleanly strips invalid rows and NaNs."""
        dirty_df = pd.DataFrame([
            {"Date": "2026-01-01", "Day_Of_Week": "Thu", "Jodi": np.nan, "Open_Digit": np.nan, "Close_Digit": np.nan, "Is_Valid": False},
            {"Date": "2026-01-02", "Day_Of_Week": "Fri", "Jodi": "68", "Open_Digit": 6, "Close_Digit": 8, "Is_Valid": True},
            {"Date": "2026-01-03", "Day_Of_Week": "Sat", "Jodi": "corrupt", "Open_Digit": "X", "Close_Digit": "Y", "Is_Valid": True},
            {"Date": "2026-01-04", "Day_Of_Week": "Sun", "Jodi": "34.0", "Open_Digit": "3.0", "Close_Digit": "4.0", "Is_Valid": True},
        ])
        clean = sanitize_draw_data(dirty_df)
        self.assertEqual(len(clean), 2, "Should retain exactly the 2 valid numeric rows")
        self.assertEqual(clean.iloc[0]["Jodi"], "68")
        self.assertEqual(clean.iloc[1]["Jodi"], "34.0")

    def test_fit_and_predict_with_corrupted_data(self):
        """Verify predictor fits and predicts without exception on corrupt datasets."""
        dirty_df = pd.DataFrame([
            {"Date": "2026-01-01", "Day_Of_Week": "Thu", "Jodi": np.nan, "Open_Digit": np.nan, "Close_Digit": np.nan, "Is_Valid": False},
            {"Date": "2026-01-02", "Day_Of_Week": "Fri", "Jodi": "68", "Open_Digit": 6, "Close_Digit": 8, "Is_Valid": True},
            {"Date": "2026-01-03", "Day_Of_Week": "Sat", "Jodi": "12", "Open_Digit": 1, "Close_Digit": 2, "Is_Valid": True},
        ])
        p = EnsemblePredictor()
        p.fit(dirty_df)
        res = p.predict()
        self.assertAlmostEqual(res["jodi_probs"].sum(), 1.0, places=6)
        self.assertEqual(len(res["top_10_jodis"]), 10)

    def test_open_close_collision_scenario(self):
        """Stress-test scenario where prev_open == prev_close (e.g. Jodi 55, 00)."""
        p = EnsemblePredictor()
        csv_file = MATKA_ROOT / "kalyan_historical_data.csv"
        p.fit(pd.read_csv(csv_file))

        for collision_digit in [0, 5, 9]:
            res = p.predict(
                day_of_week="Mon",
                prev_open=collision_digit,
                prev_close=collision_digit,
                prev_jodi=f"{collision_digit}{collision_digit}"
            )
            # Must remain fully normalized without NaN or Inf
            self.assertFalse(np.any(np.isnan(res["open_probs"])))
            self.assertFalse(np.any(np.isinf(res["open_probs"])))
            self.assertAlmostEqual(res["open_probs"].sum(), 1.0, places=6)
            self.assertAlmostEqual(res["jodi_probs"].sum(), 1.0, places=6)

    def test_boundary_lambda_values(self):
        """Test lambda = 0.0 (pure raw) and lambda = 1.0 (pure family mean)."""
        p = EnsemblePredictor()
        csv_file = MATKA_ROOT / "kalyan_historical_data.csv"
        p.fit(pd.read_csv(csv_file))

        # Lambda = 0.0
        res_zero = p.predict(lambda_cut_cluster=0.0)
        np.testing.assert_allclose(
            res_zero["cut_cluster_density_scores"],
            res_zero["jodi_probs"],
            atol=1e-6,
            err_msg="When lambda=0, cut_cluster_density_scores must equal jodi_probs"
        )

        # Lambda = 1.0: all family members must have identical density scores
        res_one = p.predict(lambda_cut_cluster=1.0)
        scores_one = res_one["cut_cluster_density_scores"]
        for j in range(100):
            fam = get_family_jodis(f"{j:02d}")
            fam_vals = [scores_one[int(x)] for x in fam]
            self.assertAlmostEqual(
                min(fam_vals),
                max(fam_vals),
                places=6,
                msg=f"When lambda=1, family members of {j} must share identical density score"
            )


if __name__ == "__main__":
    unittest.main()
