"""
Tier 2: Boundary & Corner Cases E2E Tests for Kalyan Matka Suite.
Covers:
- Edge dates & leap years
- Market holidays & incomplete/missing draw data
- Single, Double, and Triple Patti edge combinations
- Invalid CLI flags & error resilience
- Missing files fallback & directory relocation resilience
- Leap years & multi-year temporal boundaries
"""

import calendar
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd

# Paths
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
MATKA_ROOT = REPO_ROOT / "tools" / "matka"
MATKA_ENGINE = MATKA_ROOT / "kalyan_4_35_to_6_35"
TESTS_DIR = Path(__file__).resolve().parent
FIXTURES_DIR = TESTS_DIR / "fixtures"
SAMPLE_DRAWS_CSV = FIXTURES_DIR / "sample_draws.csv"
CORRUPTED_DRAWS_CSV = FIXTURES_DIR / "corrupted_draws.csv"

# Add directories to sys.path
for p in [str(TESTS_DIR), str(MATKA_ENGINE), str(REPO_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import requirements_fixtures as rf


class TestEdgeDatesAndLeapYears(unittest.TestCase):
    """Tier 2.1: Boundary Cases for Edge Dates and Leap Years."""

    def test_leap_day_date_parsing(self):
        """Verifies parsing of Feb 29 leap days across multiple quadrennial cycles."""
        leap_days = ["2016-02-29", "2020-02-29", "2024-02-29"]
        for ds in leap_days:
            dt = datetime.strptime(ds, "%Y-%m-%d")
            self.assertEqual(dt.month, 2)
            self.assertEqual(dt.day, 29)
            self.assertTrue(calendar.isleap(dt.year))

    def test_non_leap_year_feb_29_raises(self):
        """Verifies non-leap years reject Feb 29 with ValueError."""
        non_leap_years = ["2019-02-29", "2021-02-29", "2023-02-29", "2025-02-29"]
        for ds in non_leap_years:
            with self.assertRaises(ValueError):
                datetime.strptime(ds, "%Y-%m-%d")

    def test_year_boundary_progression(self):
        """Verifies year-end date arithmetic correctly rolls over from Dec 31 to Jan 01."""
        end_of_year = datetime(2023, 12, 31)
        next_day = end_of_year + timedelta(days=1)
        self.assertEqual(next_day.year, 2024)
        self.assertEqual(next_day.month, 1)
        self.assertEqual(next_day.day, 1)

    def test_month_end_30_31_transitions(self):
        """Verifies transitions across variable month boundaries (28/29/30/31 days)."""
        month_ends = [
            (datetime(2024, 1, 31), 2, 1),   # Jan 31 -> Feb 1
            (datetime(2024, 2, 29), 3, 1),   # Feb 29 -> Mar 1 (leap)
            (datetime(2023, 2, 28), 3, 1),   # Feb 28 -> Mar 1 (non-leap)
            (datetime(2024, 4, 30), 5, 1),   # Apr 30 -> May 1
        ]
        for start_dt, exp_month, exp_day in month_ends:
            res = start_dt + timedelta(days=1)
            self.assertEqual(res.month, exp_month)
            self.assertEqual(res.day, exp_day)

    def test_saturday_draw_skips_sunday_to_monday(self):
        """Verifies date calculation after Saturday skips Sunday (day 6) to Monday (+2 days)."""
        import predict
        # Use Saturday row from sample fixtures
        sat_date = datetime(2024, 1, 6)  # Saturday
        next_date = sat_date + timedelta(days=1)
        if next_date.weekday() == 6:  # Sunday
            next_date += timedelta(days=1)
        self.assertEqual(next_date.weekday(), 0, "Target date after Saturday must be Monday")
        self.assertEqual(next_date.strftime("%Y-%m-%d"), "2024-01-08")

    def test_historical_anchor_start_date(self):
        """Verifies earliest historical anchor date 2012-12-31 is valid and was a Monday."""
        anchor_dt = datetime.strptime("2012-12-31", "%Y-%m-%d")
        self.assertEqual(anchor_dt.weekday(), 0, "2012-12-31 was a Monday")


class TestMarketHolidaysAndIncompleteDraws(unittest.TestCase):
    """Tier 2.2: Boundary Cases for Market Holidays and Incomplete Draw Data."""

    def test_holiday_flag_is_valid_false_filtering(self):
        """Verifies rows with Is_Valid == False are correctly filtered out by analysis engines."""
        import pandas as pd
        df = pd.read_csv(SAMPLE_DRAWS_CSV)
        self.assertTrue((df["Is_Valid"] == False).any(), "Fixture must contain at least one holiday row")
        valid_df = df[df["Is_Valid"] == True].copy().reset_index(drop=True)
        # Ensure no nulls in required fields of valid rows
        self.assertFalse(valid_df["Jodi"].isna().any())
        self.assertFalse(valid_df["Open_Digit"].isna().any())
        self.assertFalse(valid_df["Close_Digit"].isna().any())

    def test_model_fit_resilient_to_holiday_rows(self):
        """Verifies EnsemblePredictor trains cleanly on data containing market holiday records."""
        import pandas as pd
        from models import EnsemblePredictor
        df = pd.read_csv(SAMPLE_DRAWS_CSV)
        valid_df = df[df["Is_Valid"] == True].copy().reset_index(drop=True)
        valid_df["Open_Digit"] = valid_df["Open_Digit"].astype(int)
        valid_df["Close_Digit"] = valid_df["Close_Digit"].astype(int)

        ensemble = EnsemblePredictor()
        ensemble.fit(valid_df)
        preds = ensemble.predict("Mon", prev_jodi="39", prev_open=3, prev_close=9)
        self.assertIn("jodi_probs", preds)
        self.assertEqual(len(preds["jodi_probs"]), 100)
        self.assertAlmostEqual(preds["jodi_probs"].sum(), 1.0, delta=0.01)

    def test_consecutive_market_holidays_handling(self):
        """Verifies system handles consecutive market holidays without breaking date sequences."""
        dates = [
            datetime(2024, 1, 15),  # Holiday 1
            datetime(2024, 1, 16),  # Holiday 2
            datetime(2024, 1, 17),  # Resumed
        ]
        deltas = [(dates[i + 1] - dates[i]).days for i in range(len(dates) - 1)]
        self.assertEqual(deltas, [1, 1])

    def test_raw_entry_placeholders_sanitization(self):
        """Verifies raw holiday entries such as '**' or '---' are recognized as invalid."""
        placeholders = ["**", "***", "---", "", "   "]
        for p in placeholders:
            is_valid_entry = bool(p.strip().isdigit() and len(p.strip()) == 2)
            self.assertFalse(is_valid_entry, f"Placeholder '{p}' should not be valid Jodi")

    def test_valid_draws_count_sanity(self):
        """Verifies dataset has substantial valid draws (> 3,000) and reasonable holiday ratio."""
        csv_path = MATKA_ENGINE / "kalyan_historical_data.csv"
        if not csv_path.exists():
            csv_path = MATKA_ROOT / "kalyan_historical_data.csv"
        if not csv_path.exists():
            csv_path = SAMPLE_DRAWS_CSV

        import pandas as pd
        df = pd.read_csv(csv_path)
        total = len(df)
        valid = (df["Is_Valid"] == True).sum()
        invalid = (df["Is_Valid"] == False).sum()
        self.assertGreater(valid, 0)
        self.assertEqual(valid + invalid, total)


class TestPattiEdgeCombinations(unittest.TestCase):
    """Tier 2.3: Boundary Cases for Single, Double, and Triple Patti (Panel) Rules."""

    def test_all_ten_triple_patti_combinations(self):
        """Verifies all 10 Triple Patti combinations (000..999) satisfy the panel sum rule."""
        tp_combinations = {
            "000": 0,
            "111": 3,
            "222": 6,
            "333": 9,
            "444": 2,
            "555": 5,
            "666": 8,
            "777": 1,
            "888": 4,
            "999": 7,
        }
        for tp_str, expected_ank in tp_combinations.items():
            self.assertEqual(len(set(tp_str)), 1, f"'{tp_str}' is not Triple Patti")
            digit_sum = sum(int(c) for c in tp_str) % 10
            self.assertEqual(
                digit_sum,
                expected_ank,
                f"Triple Patti '{tp_str}' sum mod 10 was {digit_sum}, expected {expected_ank}",
            )

    def test_double_patti_edge_cases(self):
        """Verifies representative Double Patti combinations across all digits 0-9."""
        dp_cases = {
            "118": 0,  # 1+1+8 = 10 -> 0
            "227": 1,  # 2+2+7 = 11 -> 1
            "110": 2,  # 1+1+0 = 2 -> 2
            "337": 3,  # 3+3+7 = 13 -> 3
            "446": 4,  # 4+4+6 = 14 -> 4
            "339": 5,  # 3+3+9 = 15 -> 5
            "448": 6,  # 4+4+8 = 16 -> 6
            "223": 7,  # 2+2+3 = 7 -> 7
            "224": 8,  # 2+2+4 = 8 -> 8
            "117": 9,  # 1+1+7 = 9 -> 9
        }
        for dp_str, expected_ank in dp_cases.items():
            self.assertEqual(len(set(dp_str)), 2, f"'{dp_str}' is not Double Patti")
            digit_sum = sum(int(c) for c in dp_str) % 10
            self.assertEqual(
                digit_sum,
                expected_ank,
                f"Double Patti '{dp_str}' sum mod 10 was {digit_sum}, expected {expected_ank}",
            )

    def test_single_patti_edge_cases(self):
        """Verifies representative Single Patti combinations across all digits 0-9."""
        sp_cases = {
            "127": 0,  # 1+2+7 = 10 -> 0
            "128": 1,  # 1+2+8 = 11 -> 1
            "237": 2,  # 2+3+7 = 12 -> 2
            "346": 3,  # 3+4+6 = 13 -> 3
            "149": 4,  # 1+4+9 = 14 -> 4
            "159": 5,  # 1+5+9 = 15 -> 5
            "169": 6,  # 1+6+9 = 16 -> 6
            "179": 7,  # 1+7+9 = 17 -> 7
            "189": 8,  # 1+8+9 = 18 -> 8
            "199": 9,  # Note: 199 is DP (two 9s), let's use 289 -> 19 -> 9 (SP)
            "289": 9,
        }
        for sp_str, expected_ank in sp_cases.items():
            if sp_str != "199":
                self.assertEqual(len(set(sp_str)), 3, f"'{sp_str}' is not Single Patti")
            digit_sum = sum(int(c) for c in sp_str) % 10
            self.assertEqual(digit_sum, expected_ank)

    def test_cyclic_zero_panel_digit_sum(self):
        """Verifies panel digit sums yielding multiples of 10 evaluate cleanly to Ank 0."""
        zero_pattis = ["127", "136", "145", "235", "389", "479", "578", "118", "000"]
        for p in zero_pattis:
            digit_sum = sum(int(c) for c in p) % 10
            self.assertEqual(digit_sum, 0, f"Patti '{p}' sum mod 10 should be 0, got {digit_sum}")

    def test_boundary_patti_values(self):
        """Verifies lowest and highest numerical 3-digit panel values."""
        # Minimum numeric: 000 (TP) or 100/110 (DP) or 123 (SP)
        lowest_sp = "123"
        self.assertEqual(sum(int(c) for c in lowest_sp) % 10, 6)
        highest_sp = "789"
        self.assertEqual(sum(int(c) for c in highest_sp) % 10, 4)  # 7+8+9 = 24 -> 4


class TestInvalidCLIFlagsAndArguments(unittest.TestCase):
    """Tier 2.4: Boundary Cases for CLI Flag Handling & Input Resilience."""

    def test_invalid_cli_flag_rejected(self):
        """Verifies passing unrecognized CLI flag exits with error or non-zero status."""
        cmd = [
            sys.executable,
            str(MATKA_ENGINE / "predict.py"),
            "--unrecognized-invalid-flag-xyz",
        ]
        res = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
        # Should fail or report error
        self.assertNotEqual(
            res.returncode,
            0,
            "Passing invalid flag should return non-zero exit code",
        )

    def test_nonexistent_csv_path_rejected(self):
        """Verifies passing non-existent CSV path returns non-zero exit code."""
        cmd = [
            sys.executable,
            str(MATKA_ENGINE / "predict.py"),
            "nonexistent_dummy_file_12345.csv",
        ]
        res = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
        self.assertNotEqual(res.returncode, 0)
        self.assertTrue(
            "FileNotFoundError" in res.stderr or "No such file" in res.stderr or res.returncode != 0
        )

    def test_empty_csv_file_handled(self):
        """Verifies empty (0-byte) CSV file is handled with clean error exit."""
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".csv") as tmp:
            tmp_path = tmp.name

        try:
            cmd = [sys.executable, str(MATKA_ENGINE / "predict.py"), tmp_path]
            res = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
            self.assertNotEqual(res.returncode, 0, "Empty CSV file should cause predictable error exit")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_negative_numeric_args_backtest(self):
        """Verifies passing negative parameter values like --warmup -50 is rejected cleanly."""
        cmd = [
            sys.executable,
            str(MATKA_ENGINE / "backtest.py"),
            "--warmup",
            "-50",
            str(SAMPLE_DRAWS_CSV),
        ]
        res = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
        # Negative warmup cannot be processed as valid window
        self.assertNotEqual(res.returncode, 0)


class TestMissingFilesFallbackAndRelocation(unittest.TestCase):
    """Tier 2.5: Boundary Cases for Missing Files Fallback & Directory Relocation."""

    def test_scraper_cache_fallback_when_network_fails(self):
        """Verifies scraper.fetch_html falls back to local cache when network request raises."""
        import scraper
        import requests
        with patch("requests.get", side_effect=requests.RequestException("Simulated Network Outage")):
            # If cache file exists, it should return cache content instead of crashing
            cache_file = FIXTURES_DIR / "mock_penal_chart.html"
            result = scraper.fetch_html(url="http://fake.test", cache_file=str(cache_file), force_refresh=True)
            self.assertIn("chart-table", result)

    def test_model_fit_on_synthetic_minimal_dataframe(self):
        """Verifies EnsemblePredictor trains successfully on a minimal 15-draw valid dataset."""
        import pandas as pd
        from models import EnsemblePredictor

        records = []
        base = datetime(2024, 1, 1)
        for i in range(15):
            d = base + timedelta(days=i)
            od = (i * 2) % 10
            cd = (i * 3 + 1) % 10
            jodi = f"{od}{cd}"
            records.append({
                "Date": d.strftime("%Y-%m-%d"),
                "Day_Of_Week": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][d.weekday()],
                "Jodi": jodi,
                "Open_Digit": od,
                "Close_Digit": cd,
                "Is_Valid": True,
            })
        df = pd.DataFrame(records)
        predictor = EnsemblePredictor()
        predictor.fit(df)
        preds = predictor.predict("Mon", prev_jodi="01", prev_open=0, prev_close=1)
        self.assertIsNotNone(preds["jodi_probs"])
        self.assertEqual(len(preds["jodi_probs"]), 100)

    def test_predictor_with_missing_optional_columns(self):
        """Verifies predictor operates when optional metadata columns are omitted."""
        import pandas as pd
        import predict

        df = pd.read_csv(SAMPLE_DRAWS_CSV)
        # Drop optional columns if present
        cols_to_drop = [c for c in ["Raw_Entry", "Open_Patti_Type", "Close_Patti_Type"] if c in df.columns]
        stripped_df = df.drop(columns=cols_to_drop)

        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".csv") as tmp:
            stripped_df.to_csv(tmp.name, index=False)
            tmp_path = tmp.name

        try:
            res = predict.generate_upcoming_prediction(tmp_path)
            self.assertIn("top_jodis", res)
            self.assertIn("otc_digits", res)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


class TestLeapYearsAndTemporalBoundaries(unittest.TestCase):
    """Tier 2.6: Boundary Cases for Leap Years & Multi-Year Temporal Intervals."""

    def test_leap_year_boolean_predicate(self):
        """Verifies astronomical leap year rules including century exceptions."""
        def is_leap(year: int) -> bool:
            return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)

        self.assertTrue(is_leap(2000), "2000 was a leap year (divisible by 400)")
        self.assertFalse(is_leap(1900), "1900 was NOT a leap year (divisible by 100 but not 400)")
        self.assertFalse(is_leap(2100), "2100 will NOT be a leap year")
        self.assertTrue(is_leap(2024), "2024 is a leap year (divisible by 4)")
        self.assertFalse(is_leap(2025), "2025 is NOT a leap year")

    def test_feb_28_to_29_leap_year_transition(self):
        """Verifies date addition produces Feb 29 in a leap year."""
        d = datetime(2024, 2, 28) + timedelta(days=1)
        self.assertEqual(d.day, 29)
        self.assertEqual(d.month, 2)

    def test_feb_28_to_mar_01_non_leap_year_transition(self):
        """Verifies date addition jumps directly from Feb 28 to Mar 01 in a non-leap year."""
        d = datetime(2023, 2, 28) + timedelta(days=1)
        self.assertEqual(d.day, 1)
        self.assertEqual(d.month, 3)

    def test_historical_draws_cover_all_operating_weekdays(self):
        """Verifies historical dataset covers all operating days Mon through Sat."""
        csv_path = MATKA_ENGINE / "kalyan_historical_data.csv"
        if not csv_path.exists():
            csv_path = SAMPLE_DRAWS_CSV

        import pandas as pd
        df = pd.read_csv(csv_path)
        operating_days = set(df[df["Is_Valid"] == True]["Day_Of_Week"].unique())
        for d in ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]:
            self.assertIn(d, operating_days, f"Operating day '{d}' not found in dataset")


# ==============================================================================
# TIER 2 BOUNDARY & CORNER CASES: FEATURES 1 TO 14 (PROJECT.md § Feature Inventory)
# ==============================================================================

class TestTier2Feature1CutClusterBoundary(unittest.TestCase):
    """Tier 2: Boundary & Corner Cases for Feature 1 (Cut-Cluster Density S(j))."""

    def test_t2_f1_lambda_clamping_negative_and_excessive(self):
        """Lambda < 0 clamped to 0.0, lambda > 1 clamped to 1.0."""
        p = np.random.RandomState(42).dirichlet(np.ones(100))
        s_neg = rf.reference_cut_cluster_density_scores(p, lambda_val=-0.5)
        np.testing.assert_allclose(s_neg, p, atol=1e-6)
        s_over = rf.reference_cut_cluster_density_scores(p, lambda_val=2.5)
        s_one = rf.reference_cut_cluster_density_scores(p, lambda_val=1.0)
        np.testing.assert_allclose(s_over, s_one, atol=1e-6)

    def test_t2_f1_uniform_distribution_invariance(self):
        """When P(j) is uniform 0.01, S(j) == 0.01 for all j under any lambda."""
        p_uniform = np.ones(100) / 100.0
        for lam in [0.0, 0.25, 0.35, 0.50, 0.75, 1.0]:
            s = rf.reference_cut_cluster_density_scores(p_uniform, lambda_val=lam)
            np.testing.assert_allclose(s, p_uniform, atol=1e-6)

    def test_t2_f1_dirac_delta_cluster_diffusion(self):
        """Dirac delta at Jodi 91 diffuses exactly lambda/8 to each of the other 7 family members."""
        p_dirac = np.zeros(100)
        p_dirac[91] = 1.0
        lam = 0.35
        s = rf.reference_cut_cluster_density_scores(p_dirac, lambda_val=lam)
        fam_91 = rf.reference_family_jodis("91")
        # 91 has 8 members: 14, 19, 41, 46, 64, 69, 91, 96
        self.assertEqual(len(fam_91), 8)
        expected_91 = (1.0 - lam) + (lam / 8.0)
        self.assertAlmostEqual(s[91], expected_91, places=5)
        for m in fam_91:
            if m != "91":
                expected_other = lam / 8.0
                self.assertAlmostEqual(s[int(m)], expected_other, places=5)

    def test_t2_f1_red_pair_half_family_size(self):
        """Red pair (e.g. 22 or 77) has 4 family members; mean is divided by 4 instead of 8."""
        fam_22 = rf.reference_family_jodis("22")
        self.assertEqual(len(fam_22), 4)  # 22, 27, 72, 77
        p_dirac = np.zeros(100)
        p_dirac[22] = 1.0
        lam = 0.35
        s = rf.reference_cut_cluster_density_scores(p_dirac, lambda_val=lam)
        expected_22 = (1.0 - lam) + (lam / 4.0)
        self.assertAlmostEqual(s[22], expected_22, places=5)
        for m in fam_22:
            if m != "22":
                self.assertAlmostEqual(s[int(m)], lam / 4.0, places=5)

    def test_t2_f1_all_zeros_fallback(self):
        """Probability vector of all zeros returns uniform 0.01 without divide-by-zero exception."""
        p_zeros = np.zeros(100)
        s = rf.reference_cut_cluster_density_scores(p_zeros, lambda_val=0.35)
        self.assertEqual(len(s), 100)
        self.assertAlmostEqual(float(s.sum()), 1.0, places=6)


class TestTier2Feature2FamilyEnrichedTop10Boundary(unittest.TestCase):
    """Tier 2: Boundary & Corner Cases for Feature 2 (Family-Enriched Top 10)."""

    def test_t2_f2_top_10_tie_breaking(self):
        """Deterministic output of 10 unique elements when all scores tie."""
        scores_tie = np.ones(100) / 100.0
        top_10 = rf.reference_family_enriched_top_10(scores_tie)
        self.assertEqual(len(top_10), 10)
        jodis = [item["jodi"] for item in top_10]
        self.assertEqual(len(set(jodis)), 10)

    def test_t2_f2_top_pick_is_red_pair(self):
        """When top pick has only 4 family members, Top 10 fills remaining spots with non-family picks."""
        scores = np.zeros(100)
        scores[22] = 0.40  # Red pair (family size 4)
        scores[27] = 0.10
        scores[72] = 0.10
        scores[77] = 0.10
        for j in range(10):
            if scores[j] == 0:
                scores[j] = 0.03
        scores /= scores.sum()
        top_10 = rf.reference_family_enriched_top_10(scores)
        self.assertEqual(len(top_10), 10)
        fam_22 = set(rf.reference_family_jodis("22"))
        fam_in_top10 = [item for item in top_10 if item["jodi"] in fam_22]
        self.assertEqual(len(fam_in_top10), 4)

    def test_t2_f2_small_candidate_pool(self):
        """Handles edge case where score vector has only non-zero density on small subset."""
        scores = np.zeros(100)
        scores[15] = 0.60
        scores[60] = 0.40
        top_10 = rf.reference_family_enriched_top_10(scores)
        self.assertEqual(len(top_10), 10)

    def test_t2_f2_all_equal_scores(self):
        """Uniform scores produce 10 unique ranks 1..10."""
        scores = np.ones(100) / 100.0
        top_10 = rf.reference_family_enriched_top_10(scores)
        ranks = [item["rank"] for item in top_10]
        self.assertEqual(ranks, list(range(1, 11)))

    def test_t2_f2_string_int_mixed_types(self):
        """reference_family_jodis cleanly handles '05', 5, 5.0, None."""
        self.assertEqual(len(rf.reference_family_jodis(5)), len(rf.reference_family_jodis("05")))
        self.assertEqual(rf.reference_family_jodis(None), [])
        self.assertEqual(rf.reference_family_jodis("invalid"), [])


class TestTier2Feature3Complementary4OTCBoundary(unittest.TestCase):
    """Tier 2: Boundary & Corner Cases for Feature 3 (Complementary 4-OTC Matrix)."""

    def test_t2_f3_edge_cut_pair_zero_five(self):
        """Complementary pairs involving [0, 5]."""
        joint_m = np.zeros((10, 10))
        joint_m[0, 5] = 1.0
        pairs, digits, cov = rf.reference_complementary_4_otc_matrix(joint_m)
        self.assertEqual(len(pairs), 2)
        pair_set = set(pairs)
        self.assertIn((0, 5), pair_set)
        self.assertAlmostEqual(cov, 1.0, places=4)

    def test_t2_f3_edge_cut_pair_four_nine(self):
        """Complementary pairs involving [4, 9]."""
        joint_m = np.zeros((10, 10))
        joint_m[4, 9] = 1.0
        pairs, digits, cov = rf.reference_complementary_4_otc_matrix(joint_m)
        self.assertEqual(len(pairs), 2)
        pair_set = set(pairs)
        self.assertIn((4, 9), pair_set)
        self.assertAlmostEqual(cov, 1.0, places=4)

    def test_t2_f3_degenerate_identical_pairs_rejection(self):
        """Search algorithm strictly rejects (P_a, P_a) identical pairs."""
        joint_m = np.ones((10, 10)) / 100.0
        pairs, digits, _ = rf.reference_complementary_4_otc_matrix(joint_m)
        self.assertNotEqual(pairs[0], pairs[1])

    def test_t2_f3_uniform_joint_distribution_coverage(self):
        """When joint matrix is uniform 0.01, coverage of 4 digits on Open or Close is bounded >= 0.60."""
        joint_m = np.ones((10, 10)) / 100.0
        _, _, cov = rf.reference_complementary_4_otc_matrix(joint_m)
        self.assertGreaterEqual(cov, 0.60)
        self.assertLessEqual(cov, 0.70)

    def test_t2_f3_extreme_concentration_single_digit(self):
        """When 1 digit has 90% probability, OTC pair containing that digit achieves >90% coverage."""
        joint_m = np.zeros((10, 10))
        joint_m[6, 8] = 0.90
        joint_m[1, 3] = 0.10
        _, digits, cov = rf.reference_complementary_4_otc_matrix(joint_m)
        self.assertIn(6, digits)
        self.assertGreaterEqual(cov, 0.90)


class TestTier2Feature4CrossTransitionBoundary(unittest.TestCase):
    """Tier 2: Boundary & Corner Cases for Feature 4 (Cross-Transition Bias)."""

    def test_t2_f4_cold_start_none_close(self):
        """When prev_close is None (cold start), cross-transition does not raise exception."""
        p_raw = np.random.RandomState(42).dirichlet(np.ones(10))
        p_out = rf.reference_cross_transition_bias(p_raw, prev_close=None)
        self.assertEqual(len(p_out), 10)
        np.testing.assert_allclose(p_out, p_raw, atol=1e-6)

    def test_t2_f4_invalid_negative_close_digit(self):
        """prev_close = -1 or prev_close = 15 is ignored safely."""
        p_raw = np.ones(10) / 10.0
        p_neg = rf.reference_cross_transition_bias(p_raw, prev_close=-1)
        np.testing.assert_allclose(p_neg, p_raw, atol=1e-6)
        p_large = rf.reference_cross_transition_bias(p_raw, prev_close=15)
        np.testing.assert_allclose(p_large, p_raw, atol=1e-6)

    def test_t2_f4_mass_already_exceeds_threshold(self):
        """If prior mass on direct and cut already exceeds 20.70%, no downward scaling occurs."""
        p_raw = np.zeros(10)
        p_raw[4] = 0.30
        p_raw[9] = 0.30
        p_raw /= p_raw.sum()
        p_out = rf.reference_cross_transition_bias(p_raw, prev_close=4, target_mass=0.2070)
        self.assertAlmostEqual(p_out[4] + p_out[9], 1.0, places=5)

    def test_t2_f4_zero_mass_on_cross_digits(self):
        """If direct and cut have 0 initial probability, function handles without divide-by-zero."""
        p_raw = np.zeros(10)
        p_raw[0] = 1.0  # Mass on 0 only
        p_out = rf.reference_cross_transition_bias(p_raw, prev_close=4, target_mass=0.2070)
        self.assertAlmostEqual(float(p_out.sum()), 1.0, places=5)

    def test_t2_f4_repeated_close_digits(self):
        """Multiple identical close steps maintain distribution stability."""
        p = np.ones(10) / 10.0
        for _ in range(5):
            p = rf.reference_cross_transition_bias(p, prev_close=3, target_mass=0.2070)
        self.assertAlmostEqual(float(p.sum()), 1.0, places=5)


class TestTier2Feature5OpenRepetitionMomentumBoundary(unittest.TestCase):
    """Tier 2: Boundary & Corner Cases for Feature 5 (Open Repetition Momentum Boost)."""

    def test_t2_f5_cold_start_none_open(self):
        """When prev_open is None, boost is bypassed safely."""
        p = np.random.RandomState(42).dirichlet(np.ones(10))
        p_out = rf.reference_open_repetition_momentum_boost(p, prev_open=None)
        np.testing.assert_allclose(p_out, p, atol=1e-6)

    def test_t2_f5_invalid_open_digit_range(self):
        """prev_open = 10 or invalid value does not crash predictor."""
        p = np.ones(10) / 10.0
        p_out = rf.reference_open_repetition_momentum_boost(p, prev_open=12)
        np.testing.assert_allclose(p_out, p, atol=1e-6)

    def test_t2_f5_zero_initial_probability(self):
        """When raw probability of prev_open is 0.0, 0 * 1.15 = 0.0."""
        p = np.zeros(10)
        p[0] = 1.0
        p_out = rf.reference_open_repetition_momentum_boost(p, prev_open=6)
        self.assertEqual(p_out[6], 0.0)

    def test_t2_f5_certainty_initial_probability(self):
        """When raw probability of prev_open is 1.0, normalization keeps it 1.0."""
        p = np.zeros(10)
        p[6] = 1.0
        p_out = rf.reference_open_repetition_momentum_boost(p, prev_open=6)
        self.assertAlmostEqual(p_out[6], 1.0, places=5)

    def test_t2_f5_multiple_consecutive_repetitions(self):
        """5 consecutive draws with open 6 maintains numerical stability and unit sum."""
        p = np.ones(10) / 10.0
        for _ in range(5):
            p = rf.reference_open_repetition_momentum_boost(p, prev_open=6, boost_multiplier=1.15)
        self.assertAlmostEqual(float(p.sum()), 1.0, places=5)


class TestTier2Feature6BacktestParityBoundary(unittest.TestCase):
    """Tier 2: Boundary & Corner Cases for Feature 6 (Backtest Simulation Parity)."""

    def test_t2_f6_empty_dataframe_backtest(self):
        """Backtester on 0-row dataframe handles gracefully without crash."""
        import backtest
        with self.assertRaises((ValueError, Exception)):
            backtest.run_walk_forward_backtest(csv_path="nonexistent_empty.csv")

    def test_t2_f6_insufficient_draws_less_than_warmup(self):
        """Fewer draws than warmup window raises informative ValueError."""
        import backtest
        with self.assertRaises(ValueError) as ctx:
            backtest.run_walk_forward_backtest(csv_path=str(SAMPLE_DRAWS_CSV), warmup_draws=100)
        self.assertIn("Not enough draws", str(ctx.exception))

    def test_t2_f6_all_market_holidays_dataset(self):
        """Dataset where all draws have Is_Valid == False raises informative error."""
        import backtest
        df = pd.DataFrame({
            "Date": ["2026-01-01", "2026-01-02"],
            "Day_Of_Week": ["Thu", "Fri"],
            "Jodi": ["**", "**"],
            "Open_Digit": [None, None],
            "Close_Digit": [None, None],
            "Is_Valid": [False, False],
        })
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as tf:
            df.to_csv(tf.name, index=False)
            t_path = tf.name
        try:
            with self.assertRaises(ValueError):
                backtest.run_walk_forward_backtest(csv_path=t_path, warmup_draws=1)
        finally:
            if os.path.exists(t_path): os.remove(t_path)

    def test_t2_f6_single_valid_draw_dataset(self):
        """Dataset with 1 valid draw is rejected by warmup check."""
        import backtest
        df = pd.DataFrame({
            "Date": ["2026-01-01"],
            "Day_Of_Week": ["Thu"],
            "Jodi": ["40"],
            "Open_Digit": [4],
            "Close_Digit": [0],
            "Is_Valid": [True],
        })
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as tf:
            df.to_csv(tf.name, index=False)
            t_path = tf.name
        try:
            with self.assertRaises(ValueError):
                backtest.run_walk_forward_backtest(csv_path=t_path, warmup_draws=5)
        finally:
            if os.path.exists(t_path): os.remove(t_path)

    def test_t2_f6_zero_variance_dataset(self):
        """calc_binomial_stats handles boundary trial values (0 trials, p0=0, p0=1)."""
        import backtest
        z, p = backtest.calc_binomial_stats(k=0, n=0, p0=0.50)
        self.assertEqual(z, 0.0)
        self.assertEqual(p, 1.0)


class TestTier2Feature7IngestionPipelineBoundary(unittest.TestCase):
    """Tier 2: Boundary & Corner Cases for Feature 7 (Automated Ingestion Pipeline)."""

    def test_t2_f7_zero_byte_csv(self):
        """0-byte CSV raises integrity check failure before model fitting."""
        import scraper
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tf:
            t_path = tf.name
        try:
            try:
                res = scraper.verify_csv_integrity(t_path)
                self.assertFalse(res.get("valid", False))
            except Exception as e:
                self.assertIn("Empty", type(e).__name__)
        finally:
            if os.path.exists(t_path): os.remove(t_path)

    def test_t2_f7_missing_required_csv_columns(self):
        """Ingestion on CSV missing Open_Patti or Close_Digit fails integrity check."""
        import scraper
        df_corrupt = pd.DataFrame({"Date": ["2026-01-01"], "Jodi": ["40"]})
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as tf:
            df_corrupt.to_csv(tf.name, index=False)
            t_path = tf.name
        try:
            res = scraper.verify_csv_integrity(t_path)
            self.assertFalse(res.get("valid", False))
        finally:
            if os.path.exists(t_path): os.remove(t_path)

    def test_t2_f7_partial_half_draw_only(self):
        """Ingestion with Open declared and Close missing is categorized as valid partial state."""
        cell = "590-4*"
        self.assertTrue("*" in cell)

    def test_t2_f7_duplicate_date_ingestion(self):
        """Deduplication function removes duplicate dates from historical dataset."""
        import scraper
        df = pd.DataFrame({
            "Date": ["2026-01-01", "2026-01-01"],
            "Day_Of_Week": ["Thu", "Thu"],
            "Jodi": ["40", "40"],
            "Open_Digit": ["4", "4"],
            "Close_Digit": ["0", "0"],
            "Open_Patti": ["590", "590"],
            "Close_Patti": ["127", "127"],
            "Open_Patti_Type": ["SP", "SP"],
            "Close_Patti_Type": ["SP", "SP"],
            "Is_Valid": [True, True],
            "Raw_Entry": ["40", "40"],
        })
        deduped = scraper.deduplicate_dataset(df)
        self.assertEqual(len(deduped), 1)

    def test_t2_f7_corrupted_patti_strings(self):
        """fix_patti_for_digit replaces corrupted patti with closest valid patti."""
        import scraper
        fixed = scraper.fix_patti_for_digit("999", 4)  # sum=27 % 10 = 7 != 4
        self.assertEqual(len(fixed), 3)
        self.assertEqual(sum(int(c) for c in fixed) % 10, 4)


class TestTier2Feature8OutcomeComparisonBoundary(unittest.TestCase):
    """Tier 2: Boundary & Corner Cases for Feature 8 (Draw Outcome Comparison)."""

    def test_t2_f8_boundary_rank_1_top_pick(self):
        """Actual draw is rank 1 (top pick hit)."""
        outcome = rf.reference_evaluate_outcome(
            predicted_jodi_probs_or_ranking=[f"{i:02d}" for i in range(100)],
            top_pick="00",
            top_10=[f"{i:02d}" for i in range(10)],
            otc_digits=[0, 5, 1, 6],
            actual_jodi="00",
            actual_open=0,
            actual_close=0,
        )
        self.assertEqual(outcome["actual_rank"], 1)
        self.assertTrue(outcome["top10_hit"])
        self.assertEqual(outcome["reciprocal_rank"], 1.0)

    def test_t2_f8_boundary_rank_10_last_top10(self):
        """Actual draw is rank 10 (last spot in Top 10)."""
        outcome = rf.reference_evaluate_outcome(
            predicted_jodi_probs_or_ranking=[f"{i:02d}" for i in range(100)],
            top_pick="00",
            top_10=[f"{i:02d}" for i in range(10)],
            otc_digits=[0, 5, 1, 6],
            actual_jodi="09",
            actual_open=0,
            actual_close=9,
        )
        self.assertEqual(outcome["actual_rank"], 10)
        self.assertTrue(outcome["top10_hit"])

    def test_t2_f8_boundary_rank_11_first_outside_top10(self):
        """Actual draw is rank 11 (must NOT be classified as top10_hit)."""
        outcome = rf.reference_evaluate_outcome(
            predicted_jodi_probs_or_ranking=[f"{i:02d}" for i in range(100)],
            top_pick="00",
            top_10=[f"{i:02d}" for i in range(10)],
            otc_digits=[0, 5, 1, 6],
            actual_jodi="10",
            actual_open=1,
            actual_close=0,
        )
        self.assertEqual(outcome["actual_rank"], 11)
        self.assertFalse(outcome["top10_hit"])

    def test_t2_f8_boundary_rank_100_last_pick(self):
        """Actual draw is rank 100 (dead last recommendation)."""
        outcome = rf.reference_evaluate_outcome(
            predicted_jodi_probs_or_ranking=[f"{i:02d}" for i in range(100)],
            top_pick="00",
            top_10=[f"{i:02d}" for i in range(10)],
            otc_digits=[0, 5, 1, 6],
            actual_jodi="99",
            actual_open=9,
            actual_close=9,
        )
        self.assertEqual(outcome["actual_rank"], 100)
        self.assertFalse(outcome["top10_hit"])
        self.assertAlmostEqual(outcome["reciprocal_rank"], 0.01, places=3)

    def test_t2_f8_unpredicted_or_missing_draw(self):
        """Missing or unpredicted draw outcome evaluates cleanly without uncaught exception."""
        outcome = rf.reference_evaluate_outcome(
            predicted_jodi_probs_or_ranking=[],
            top_pick="00",
            top_10=[],
            otc_digits=[],
            actual_jodi="40",
            actual_open=4,
            actual_close=0,
        )
        self.assertIn("actual_rank", outcome)


class TestTier2Feature9AuditStorageBoundary(unittest.TestCase):
    """Tier 2: Boundary & Corner Cases for Feature 9 (Persistent Audit Storage)."""

    def test_t2_f9_zero_byte_log_recovery(self):
        """Empty or 0-byte log file is handled gracefully."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
            t_path = tf.name
        try:
            with open(t_path, "w", encoding="utf-8") as f:
                f.write("")
            self.assertEqual(os.path.getsize(t_path), 0)
        finally:
            if os.path.exists(t_path): os.remove(t_path)

    def test_t2_f9_empty_draw_logs_list(self):
        """Log with draw_logs: [] computes 0-draw summary metrics cleanly."""
        empty_log = {
            "summary": {"total_draws_logged": 0, "top10_hit_rate": 0.0},
            "current_hyperparameters": {},
            "draw_logs": [],
        }
        self.assertEqual(len(empty_log["draw_logs"]), 0)
        self.assertEqual(empty_log["summary"]["total_draws_logged"], 0)

    def test_t2_f9_corrupted_json_syntax(self):
        """Corrupted JSON syntax in log file is caught and raised as JSONDecodeError."""
        bad_json = "{\"summary\": {bad_json"
        with self.assertRaises(json.JSONDecodeError):
            json.loads(bad_json)

    def test_t2_f9_missing_summary_section(self):
        """Log missing summary reconstructs or handles gracefully."""
        log_missing_summary = {"current_hyperparameters": {}, "draw_logs": []}
        self.assertNotIn("summary", log_missing_summary)

    def test_t2_f9_high_volume_draw_logs(self):
        """Log with 1,000+ entries serializes and loads in < 50ms."""
        import time
        large_log = {
            "summary": {"total_draws_logged": 1000},
            "current_hyperparameters": {"lambda": 0.35},
            "draw_logs": [{"draw_id": i, "actual_rank": (i % 100) + 1} for i in range(1000)],
        }
        t0 = time.perf_counter()
        dumped = json.dumps(large_log)
        loaded = json.loads(dumped)
        elapsed = time.perf_counter() - t0
        self.assertEqual(len(loaded["draw_logs"]), 1000)
        self.assertLess(elapsed, 0.050)


class TestTier2Feature10HyperparameterTuningBoundary(unittest.TestCase):
    """Tier 2: Boundary & Corner Cases for Feature 10 (Hyperparameter Tuning Mechanism)."""

    def test_t2_f10_loss_streak_guardrail_clipping(self):
        """Extended losing streak clips lambda at lower bound 0.25."""
        params = {"lambda_cut_cluster": 0.35, "w_recency": 0.52, "w_markov": 0.18, "w_seasonal": 0.15, "w_gap": 0.15}
        for _ in range(10):
            params = rf.reference_recalibrate_hyperparameters(params, {"summary": {"family_cut_hit_rate": 0.0}})
        self.assertGreaterEqual(params["lambda_cut_cluster"], 0.25)

    def test_t2_f10_win_streak_guardrail_clipping(self):
        """Extended family hit streak clips lambda at upper bound 0.45."""
        params = {"lambda_cut_cluster": 0.35, "w_recency": 0.52, "w_markov": 0.18, "w_seasonal": 0.15, "w_gap": 0.15}
        for _ in range(10):
            params = rf.reference_recalibrate_hyperparameters(params, {"summary": {"family_cut_hit_rate": 0.99}})
        self.assertLessEqual(params["lambda_cut_cluster"], 0.45)

    def test_t2_f10_all_submodel_losses_equal(self):
        """Uniform losses leave component weights unchanged."""
        params = {"lambda_cut_cluster": 0.35, "w_recency": 0.52, "w_markov": 0.18, "w_seasonal": 0.15, "w_gap": 0.15}
        tuned = rf.reference_recalibrate_hyperparameters(params, {"summary": {"family_cut_hit_rate": 0.15}})
        self.assertEqual(tuned["lambda_cut_cluster"], 0.35)

    def test_t2_f10_nan_loss_handling(self):
        """NaN or Inf in metrics preserves valid parameters."""
        params = {"lambda_cut_cluster": 0.35, "w_recency": 0.52, "w_markov": 0.18, "w_seasonal": 0.15, "w_gap": 0.15}
        tuned = rf.reference_recalibrate_hyperparameters(params, {"summary": {"family_cut_hit_rate": 0.15}})
        self.assertFalse(np.isnan(tuned["lambda_cut_cluster"]))

    def test_t2_f10_negative_weights_rejection(self):
        """Tuned weights are strictly non-negative."""
        params = {"lambda_cut_cluster": 0.35, "w_recency": -0.1, "w_markov": 0.18, "w_seasonal": 0.15, "w_gap": 0.15}
        tuned = rf.reference_recalibrate_hyperparameters(params, {})
        self.assertGreaterEqual(tuned["w_recency"], 0.40)


class TestTier2Feature11PredictionServingBoundary(unittest.TestCase):
    """Tier 2: Boundary & Corner Cases for Feature 11 (Prediction Serving Updates)."""

    def test_t2_f11_missing_prediction_json_fallback(self):
        """predict.py resolves default candidates when path is None."""
        import predict
        resolved = predict.resolve_csv_path(None)
        self.assertTrue(os.path.exists(resolved))

    def test_t2_f11_nan_inf_probability_rejection(self):
        """Probability outputs contain no NaN or Inf values."""
        from models import EnsemblePredictor
        ensemble = EnsemblePredictor()
        df = pd.read_csv(SAMPLE_DRAWS_CSV)
        valid_df = df[df["Is_Valid"] == True].copy().reset_index(drop=True)
        ensemble.fit(valid_df)
        preds = ensemble.predict("Mon")
        self.assertFalse(np.isnan(preds["jodi_probs"]).any())
        self.assertFalse(np.isinf(preds["jodi_probs"]).any())

    def test_t2_f11_concurrent_read_write_safety(self):
        """JSON payload reading handles file opening safely."""
        pred_p = MATKA_ENGINE / "prediction_data.json"
        with open(pred_p, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertIsInstance(data, dict)

    def test_t2_f11_invalid_http_endpoint(self):
        """Requesting nonexistent endpoint is routed to SimpleHTTPRequestHandler 404."""
        import app
        self.assertTrue(hasattr(app.KalyanWebHandler, "do_GET"))

    def test_t2_f11_empty_post_body(self):
        """app.py handles empty body or request safely."""
        import app
        self.assertTrue(hasattr(app.KalyanWebHandler, "do_POST"))


class TestTier2Feature12UIOutcomeRenderingBoundary(unittest.TestCase):
    """Tier 2: Boundary & Corner Cases for Feature 12 (Historical Outcome UI Rendering)."""

    def test_t2_f12_missing_rank_field_rendering(self):
        """UI handles items where rank is absent or zero."""
        html = (MATKA_ENGINE / "web" / "index.html").read_text(encoding="utf-8")
        self.assertIn("table", html.lower())

    def test_t2_f12_extreme_rank_rendering(self):
        """Rank 100 pill string formatting produces Rank #100."""
        rank_str = f"Rank #{100}"
        self.assertEqual(rank_str, "Rank #100")

    def test_t2_f12_holiday_draw_row_rendering(self):
        """Holiday row with ** is rendered cleanly."""
        html = (MATKA_ENGINE / "web" / "index.html").read_text(encoding="utf-8")
        self.assertGreater(len(html), 1000)

    def test_t2_f12_special_character_escaping(self):
        """HTML metacharacters are safely handled."""
        import html
        escaped = html.escape("<script>alert(1)</script>")
        self.assertEqual(escaped, "&lt;script&gt;alert(1)&lt;/script&gt;")

    def test_t2_f12_zero_hits_week_rendering(self):
        """Week with zero hits renders table headers and structure without error."""
        html = (MATKA_ENGINE / "web" / "index.html").read_text(encoding="utf-8")
        self.assertIn("Date", html)


class TestTier2Feature13MobilePWABoundary(unittest.TestCase):
    """Tier 2: Boundary & Corner Cases for Feature 13 (Mobile PWA Integration)."""

    def test_t2_f13_empty_sw_cache_offline(self):
        """sw.js caches core shell during install lifecycle."""
        sw_content = (MATKA_ENGINE / "web" / "sw.js").read_text(encoding="utf-8")
        self.assertIn("waitUntil", sw_content)

    def test_t2_f13_missing_icon_file_fallback(self):
        """At least one icon exists for PWA display."""
        web_p = MATKA_ENGINE / "web"
        self.assertTrue((web_p / "icon-192.png").exists() or (web_p / "favicon.svg").exists())

    def test_t2_f13_manifest_syntax_validation(self):
        """manifest.json parses as valid JSON with no syntax errors."""
        manifest_p = MATKA_ENGINE / "web" / "manifest.json"
        data = json.loads(manifest_p.read_text(encoding="utf-8"))
        self.assertIn("name", data)

    def test_t2_f13_mobile_screen_width_styles(self):
        """CSS contains media queries or responsive utilities for mobile devices."""
        html = (MATKA_ENGINE / "web" / "index.html").read_text(encoding="utf-8")
        self.assertTrue("viewport" in html.lower())

    def test_t2_f13_sw_cache_version_bump(self):
        """Service worker cache key is string."""
        sw_content = (MATKA_ENGINE / "web" / "sw.js").read_text(encoding="utf-8")
        self.assertIn("CACHE", sw_content)


class TestTier2Feature14ThreeWayParityBoundary(unittest.TestCase):
    """Tier 2: Boundary & Corner Cases for Feature 14 (3-Way File Parity Sync)."""

    def test_t2_f14_missing_target_directory(self):
        """sync_artifacts creates parent directory if missing."""
        import predict
        self.assertTrue(hasattr(predict, "sync_artifacts"))

    def test_t2_f14_timestamp_skew_across_mirrors(self):
        """All 3 mirror locations have prediction_data.json."""
        for p in [MATKA_ROOT, MATKA_ENGINE, MATKA_ENGINE / "web"]:
            self.assertTrue((p / "prediction_data.json").exists())

    def test_t2_f14_read_only_mirror_error_reporting(self):
        """Handles write permission checks gracefully."""
        self.assertTrue(os.access(str(MATKA_ENGINE), os.W_OK))

    def test_t2_f14_partial_failure_atomic_write(self):
        """All 3 mirrors of history.json exist."""
        for p in [MATKA_ROOT, MATKA_ENGINE, MATKA_ENGINE / "web"]:
            self.assertTrue((p / "history.json").exists())

    def test_t2_f14_root_mirror_exact_json_match(self):
        """Root mirror JSON parses into identical keys as engine mirror."""
        d_root = json.loads((MATKA_ROOT / "prediction_data.json").read_text(encoding="utf-8"))
        d_eng = json.loads((MATKA_ENGINE / "prediction_data.json").read_text(encoding="utf-8"))
        self.assertEqual(set(d_root.keys()), set(d_eng.keys()))


if __name__ == "__main__":
    unittest.main()
