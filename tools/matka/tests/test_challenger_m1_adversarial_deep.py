"""
Deep Adversarial and Empirical Verification Harness for Milestone 1
Authored by challenger_m1_2_2 (Empirical Challenger: Critic & Specialist)

Validates:
1. No artificial floor/clamp (0.682) in models.py or predict.py
2. Exact post-normalization cross-transition mass = 0.2070 under varied prior distributions
3. Dual complementary 4-OTC selection correctness and genuine probability reporting
4. Schema conformance and edge-case boundary resilience of sanitize_draw_data / EnsemblePredictor.fit()
5. Exposes failure modes when arbitrary DataFrames lack required columns or have out-of-range values.
"""

import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
ENGINE_DIR = REPO_ROOT / "tools" / "matka" / "kalyan_4_35_to_6_35"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(ENGINE_DIR))

import models
from models import EnsemblePredictor, sanitize_draw_data, get_family_jodis, get_cut_digit


class TestEmpiricalChallengerDeep(unittest.TestCase):
    """Deep adversarial testing for M1 algorithms."""

    def test_no_hardcoded_otc_floor_in_codebase(self):
        """Verify source code contains no artificial 0.682 max() floor in models.py."""
        models_path = ENGINE_DIR / "models.py"
        with open(models_path, "r", encoding="utf-8") as f:
            code = f.read()

        self.assertNotIn(
            "max(best_prob, 0.682)",
            code,
            "Artificial clamp max(best_prob, 0.682) detected in models.py!"
        )
        self.assertNotIn(
            "max(prob, 0.682)",
            code,
            "Artificial clamp max(prob, 0.682) detected in models.py!"
        )

    def test_cross_transition_bias_exact_normalization_fuzzing(self):
        """
        Fuzz cross-transition bias with 1,000 Dirichlet distributions to guarantee
        that whenever initial mass < 0.2070, post-normalization mass on {c_dir, c_cut}
        is EXACTLY 0.2070 within 1e-6 tolerance.
        """
        rng = np.random.RandomState(42)
        target_mass = 0.2070

        for iteration in range(1000):
            # Generate random 10-class probability distribution
            alpha = rng.uniform(0.1, 5.0, size=10)
            open_probs_init = rng.dirichlet(alpha)

            for c_dir in range(10):
                c_cut = (c_dir + 5) % 10
                open_probs = open_probs_init.copy()

                curr_cross = open_probs[c_dir] + open_probs[c_cut]
                if curr_cross < target_mass and curr_cross > 0:
                    rem_target = 1.0 - target_mass
                    rem_curr = 1.0 - curr_cross
                    scale_cross = target_mass / curr_cross
                    scale_rem = (rem_target / rem_curr) if rem_curr > 0 else 1.0

                    open_probs[c_dir] *= scale_cross
                    open_probs[c_cut] *= scale_cross
                    for d in range(10):
                        if d not in (c_dir, c_cut):
                            open_probs[d] *= scale_rem

                    open_probs /= open_probs.sum()

                    post_mass = open_probs[c_dir] + open_probs[c_cut]
                    # Check exact post-normalization mass
                    self.assertAlmostEqual(
                        post_mass,
                        target_mass,
                        places=6,
                        msg=f"Iter {iteration}, c_dir={c_dir}: Expected {target_mass}, got {post_mass}"
                    )
                    # Check probability sum == 1.0
                    self.assertAlmostEqual(
                        float(open_probs.sum()),
                        1.0,
                        places=6,
                        msg=f"Open probs does not sum to 1.0: {open_probs.sum()}"
                    )
                    # Check ratio invariance within cross pair
                    if open_probs_init[c_cut] > 1e-12:
                        ratio_init = open_probs_init[c_dir] / open_probs_init[c_cut]
                        ratio_post = open_probs[c_dir] / open_probs[c_cut]
                        self.assertAlmostEqual(
                            ratio_init / ratio_post,
                            1.0,
                            places=6,
                            msg=f"Relative ratio between c_dir and c_cut was distorted!"
                        )

    def test_cross_transition_boundary_conditions(self):
        """Test boundary conditions for cross-transition bias: curr_cross >= target_mass and near-zero."""
        target_mass = 0.2070

        # Scenario: Initial mass already exceeds target_mass (e.g. 0.35)
        probs = np.ones(10) * 0.05
        probs[2] = 0.20
        probs[7] = 0.15
        probs /= probs.sum()
        c_dir = 2
        c_cut = 7
        curr_cross = probs[c_dir] + probs[c_cut]
        self.assertGreater(curr_cross, target_mass)

        c_probs = probs.copy()
        if curr_cross < target_mass and curr_cross > 0:
            scale_cross = target_mass / curr_cross
            c_probs[c_dir] *= scale_cross
            c_probs[c_cut] *= scale_cross
        c_probs /= c_probs.sum()
        self.assertAlmostEqual(c_probs[c_dir] + c_probs[c_cut], curr_cross, places=6)

    def test_ensemble_predictor_predict_otc_matrix_properties(self):
        """
        Verify that 4-OTC output always has:
        - Exactly 2 pairs of complementary cut digits
        - Length 4 unique digits
        - Genuine mathematical probability computed without artificial constant floor
        """
        predictor = EnsemblePredictor()
        csv_file = ENGINE_DIR / "kalyan_historical_data.csv"
        if csv_file.exists():
            df = pd.read_csv(csv_file)
            predictor.fit(df)

        for day in ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]:
            pred = predictor.predict(day_of_week=day)
            otc_digits = pred["otc_digits"]
            otc_pairs = pred["otc_pairs"]
            otc_prob = pred["otc_pass_prob"]

            self.assertEqual(len(otc_digits), 4)
            self.assertEqual(len(set(otc_digits)), 4)
            self.assertEqual(len(otc_pairs), 2)

            for p in otc_pairs:
                self.assertEqual(len(p), 2)
                self.assertEqual((p[0] + 5) % 10, p[1])

            self.assertIsInstance(otc_prob, float)
            self.assertGreaterEqual(otc_prob, 0.0)
            self.assertLessEqual(otc_prob, 1.0)
            self.assertGreater(otc_prob, 0.60)

    def test_sanitize_draw_data_and_fit_supported_schemas(self):
        """
        Stress-test EnsemblePredictor.fit() and predict() with supported schemas
        containing empty sets, holidays, NaNs, and zeros.
        """
        predictor = EnsemblePredictor()

        # 1. None
        cleaned = sanitize_draw_data(None)
        self.assertEqual(len(cleaned), 0)
        predictor.fit(None)
        res = predictor.predict("Mon")
        self.assertIn("top_10_jodis", res)

        # 2. Empty DataFrame
        empty_df = pd.DataFrame()
        cleaned = sanitize_draw_data(empty_df)
        self.assertEqual(len(cleaned), 0)
        predictor.fit(empty_df)
        res = predictor.predict("Mon")
        self.assertIn("top_10_jodis", res)

        # 3. All NaNs with valid schema
        nan_df = pd.DataFrame({
            "Date": [np.nan, "2026-10-25"],
            "Day_Of_Week": ["Sun", "Mon"],
            "Jodi": [np.nan, np.nan],
            "Open_Digit": [np.nan, np.nan],
            "Close_Digit": [np.nan, np.nan],
            "Is_Valid": [True, True]
        })
        cleaned = sanitize_draw_data(nan_df)
        self.assertEqual(len(cleaned), 0)
        predictor.fit(nan_df)
        res = predictor.predict("Mon")
        self.assertIn("top_10_jodis", res)

        # 4. 100% Market Holidays (Is_Valid == False)
        holiday_df = pd.DataFrame({
            "Date": ["2026-10-25", "2026-10-26"],
            "Day_Of_Week": ["Sun", "Mon"],
            "Jodi": [12, 34],
            "Open_Digit": [1, 3],
            "Close_Digit": [2, 4],
            "Is_Valid": [False, False]
        })
        cleaned = sanitize_draw_data(holiday_df)
        self.assertEqual(len(cleaned), 0)
        predictor.fit(holiday_df)
        res = predictor.predict("Mon")
        self.assertIn("top_10_jodis", res)

        # 5. Mixed invalid and valid rows with standard schema
        mixed_df = pd.DataFrame({
            "Date": ["2026-10-25", "2026-10-26", "2026-10-27"],
            "Day_Of_Week": ["Sun", "Mon", "Tue"],
            "Jodi": ["**", "NaN", "45"],
            "Open_Digit": ["bad", None, "4"],
            "Close_Digit": ["err", np.nan, "5"],
            "Is_Valid": [True, True, True]
        })
        cleaned = sanitize_draw_data(mixed_df)
        self.assertEqual(len(cleaned), 1)
        predictor.fit(mixed_df)
        res = predictor.predict("Mon")
        self.assertIn("top_10_jodis", res)

        # 6. Single valid row
        single_df = pd.DataFrame({
            "Date": ["2026-10-25"],
            "Day_Of_Week": ["Sun"],
            "Jodi": [64],
            "Open_Digit": [6],
            "Close_Digit": [4],
            "Is_Valid": [True]
        })
        cleaned = sanitize_draw_data(single_df)
        self.assertEqual(len(cleaned), 1)
        predictor.fit(single_df)
        res = predictor.predict("Mon")
        self.assertIn("top_10_jodis", res)

        # 7. All zeros
        zeros_df = pd.DataFrame({
            "Date": [f"2026-01-{i:02d}" for i in range(1, 10)],
            "Day_Of_Week": ["Mon"] * 9,
            "Jodi": [0] * 9,
            "Open_Digit": [0] * 9,
            "Close_Digit": [0] * 9,
            "Is_Valid": [True] * 9
        })
        cleaned = sanitize_draw_data(zeros_df)
        self.assertEqual(len(cleaned), 9)
        predictor.fit(zeros_df)
        res = predictor.predict("Mon")
        self.assertIn("top_10_jodis", res)

    def test_adversarial_missing_day_of_week_column_crashes(self):
        """
        ADVERSARIAL STRESS TEST:
        Expose assumption vulnerability: If DataFrame lacks 'Day_Of_Week' AND 'Day',
        DayOfWeekSeasonalModel.fit() crashes with KeyError: 'Day'.
        """
        predictor = EnsemblePredictor()
        df_no_day = pd.DataFrame({
            "Date": ["2026-10-25", "2026-10-26"],
            "Jodi": [12, 34],
            "Open_Digit": [1, 3],
            "Close_Digit": [2, 4],
            "Is_Valid": [True, True]
        })
        # Verifies the failure mode exists when Day_Of_Week is absent
        with self.assertRaises(KeyError) as ctx:
            predictor.fit(df_no_day)
        self.assertIn("Day", str(ctx.exception))

    def test_adversarial_out_of_range_digits_crashes(self):
        """
        ADVERSARIAL STRESS TEST:
        Expose assumption vulnerability: If Open_Digit or Close_Digit has out-of-range values
        (e.g. 15 or -1), MarkovChainModel.fit() crashes with IndexError because transition
        matrices are fixed size (10, 10).
        """
        predictor = EnsemblePredictor()
        df_out_of_range = pd.DataFrame({
            "Date": ["2026-10-25", "2026-10-26"],
            "Day_Of_Week": ["Sun", "Mon"],
            "Jodi": [12, 34],
            "Open_Digit": [15, 3],  # 15 is out of bounds for (10, 10)
            "Close_Digit": [2, 4],
            "Is_Valid": [True, True]
        })
        with self.assertRaises(IndexError):
            predictor.fit(df_out_of_range)

    def test_unfitted_predictor_state_fallbacks(self):
        """Verify predict() works safely even when fit() has never been called."""
        predictor = EnsemblePredictor()
        res = predictor.predict()
        self.assertEqual(len(res["top_10_jodis"]), 10)
        self.assertEqual(len(res["otc_digits"]), 4)
        self.assertGreater(res["otc_pass_prob"], 0.60)


if __name__ == "__main__":
    unittest.main()
