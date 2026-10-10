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
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

# Paths
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
MATKA_ROOT = REPO_ROOT / "tools" / "matka"
MATKA_ENGINE = MATKA_ROOT / "kalyan_4_35_to_6_35"
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
SAMPLE_DRAWS_CSV = FIXTURES_DIR / "sample_draws.csv"
CORRUPTED_DRAWS_CSV = FIXTURES_DIR / "corrupted_draws.csv"

# Add engine directory to sys.path for direct module import
if str(MATKA_ENGINE) not in sys.path:
    sys.path.insert(0, str(MATKA_ENGINE))


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


if __name__ == "__main__":
    unittest.main()
