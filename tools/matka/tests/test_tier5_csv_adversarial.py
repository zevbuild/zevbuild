"""
Tier 5: White-Box Adversarial Stress Test & Empirical Verification Oracle for Kalyan Matka CSV Datasets.
Milestone 1 Adversarial Verification Suite.

Audits:
- tools/matka/kalyan_4_35_to_6_35/kalyan_historical_data.csv
- tools/matka/kalyan_historical_data.csv

Asserts on Production Datasets:
1. 0 duplicate dates across all rows
2. Strict monotonic date ordering (chronologically strictly increasing, Date[i] > Date[i-1])
3. Panel digit sum rule: sum(int(d) for d in patti) % 10 == digit for all declared pattis in valid draws
4. Valid SP (Single Patti), DP (Double Patti), TP (Triple Patti) categorization
5. Strict integer formatting (0 instances of .0 float pollution or float regex matches)
6. Byte-for-byte identity across root and engine dataset mirrors
7. Gregorian calendar day-of-week match (Mon..Sat)
8. Jodi == f"{Open_Digit}{Close_Digit}" for all valid draws

Adversarial Stress Harness on verify_csv_integrity():
1. Successfully caught corruptions:
   - Duplicate dates
   - Non-monotonic dates
   - Unparseable dates
   - Panel digit sum violation (Open Patti)
   - Panel digit sum violation (Close Patti)
   - Jodi digit inconsistency
   - Invalid Jodi format (non-2-digit)
   - Misclassified SP/DP/TP types
   - Calendar weekday mismatch
   - Missing required columns
2. Discovered Validator Blind Spots / Vulnerabilities:
   - Float pollution tolerance: int(float(Open_Digit)) accepts '5.0' without error
   - Malformed Patti bypass: non-3-digit Open_Patti ('140.0', 'xyz') silently skips sum check without error
   - Non-existent path fallback: resolve_csv_path silently substitutes default CSV on invalid paths
"""

import os
import re
import csv
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
MATKA_ROOT = REPO_ROOT / "tools" / "matka"
MATKA_ENGINE = MATKA_ROOT / "kalyan_4_35_to_6_35"

if str(MATKA_ENGINE) not in sys.path:
    sys.path.insert(0, str(MATKA_ENGINE))

from scraper import verify_csv_integrity, classify_patti, resolve_csv_path, REQUIRED_COLUMNS

ROOT_CSV_PATH = MATKA_ROOT / "kalyan_historical_data.csv"
ENGINE_CSV_PATH = MATKA_ENGINE / "kalyan_historical_data.csv"


class TestCSVRowByRowIntegrity(unittest.TestCase):
    """Empirical verification oracle auditing every row of production CSV datasets."""

    def test_datasets_exist(self):
        """Verifies both production CSV datasets exist at required paths."""
        self.assertTrue(ROOT_CSV_PATH.exists(), f"Root CSV missing at {ROOT_CSV_PATH}")
        self.assertTrue(ENGINE_CSV_PATH.exists(), f"Engine CSV missing at {ENGINE_CSV_PATH}")

    def test_datasets_exact_mirroring(self):
        """Verifies root and engine CSV datasets are identical byte-for-byte and row-for-row."""
        root_bytes = ROOT_CSV_PATH.read_bytes()
        engine_bytes = ENGINE_CSV_PATH.read_bytes()
        self.assertEqual(
            len(root_bytes),
            len(engine_bytes),
            f"File size mismatch: root={len(root_bytes)} bytes, engine={len(engine_bytes)} bytes",
        )
        self.assertEqual(root_bytes, engine_bytes, "Root and Engine CSV files differ in content")

    def _audit_csv_file(self, csv_path: Path):
        """Exhaustive row-by-row empirical oracle assertions."""
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            self.assertEqual(header, REQUIRED_COLUMNS, f"Header mismatch in {csv_path.name}")

            seen_dates = set()
            prev_date = None
            total_rows = 0
            valid_draws = 0
            holiday_draws = 0
            open_patti_checked = 0
            close_patti_checked = 0

            weekday_names = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}

            for line_no, row in enumerate(reader, start=2):
                total_rows += 1
                row_dict = dict(zip(REQUIRED_COLUMNS, row))

                date_str = row_dict["Date"].strip()
                day_str = row_dict["Day_Of_Week"].strip()
                jodi_str = row_dict["Jodi"].strip()
                open_digit_str = row_dict["Open_Digit"].strip()
                close_digit_str = row_dict["Close_Digit"].strip()
                open_patti_str = row_dict["Open_Patti"].strip()
                close_patti_str = row_dict["Close_Patti"].strip()
                open_type_str = row_dict["Open_Patti_Type"].strip()
                close_type_str = row_dict["Close_Patti_Type"].strip()
                is_valid_str = row_dict["Is_Valid"].strip()

                # Assertion 1: No float (.0) pollution in any field
                for col_name, val in row_dict.items():
                    self.assertFalse(
                        val.endswith(".0"),
                        f"Float pollution detected at line {line_no} in column '{col_name}': '{val}'",
                    )
                    self.assertNotRegex(
                        val,
                        r"^\d+\.\d+$",
                        f"Floating point representation at line {line_no} in column '{col_name}': '{val}'",
                    )

                # Assertion 2: Date format and zero duplicate dates
                self.assertRegex(
                    date_str,
                    r"^\d{4}-\d{2}-\d{2}$",
                    f"Invalid date format at line {line_no}: '{date_str}'",
                )
                self.assertNotIn(
                    date_str,
                    seen_dates,
                    f"Duplicate date found at line {line_no}: '{date_str}'",
                )
                seen_dates.add(date_str)

                # Assertion 3: Strict monotonic date ordering (strictly increasing calendar date)
                curr_date = datetime.strptime(date_str, "%Y-%m-%d")
                if prev_date is not None:
                    self.assertGreater(
                        curr_date,
                        prev_date,
                        f"Non-monotonic date at line {line_no}: '{date_str}' <= '{prev_date.strftime('%Y-%m-%d')}'",
                    )
                prev_date = curr_date

                # Assertion 4: Gregorian calendar day matches Day_Of_Week
                expected_day = weekday_names[curr_date.weekday()]
                self.assertEqual(
                    day_str,
                    expected_day,
                    f"Calendar day mismatch at line {line_no}: date {date_str} is {expected_day}, got {day_str}",
                )

                # Assertion 5: Valid draw rules
                if is_valid_str == "True":
                    valid_draws += 1

                    # 5a: Jodi formatting
                    self.assertRegex(
                        jodi_str,
                        r"^\d{2}$",
                        f"Invalid Jodi format at line {line_no}: '{jodi_str}'",
                    )
                    self.assertRegex(
                        open_digit_str,
                        r"^\d$",
                        f"Invalid Open_Digit at line {line_no}: '{open_digit_str}'",
                    )
                    self.assertRegex(
                        close_digit_str,
                        r"^\d$",
                        f"Invalid Close_Digit at line {line_no}: '{close_digit_str}'",
                    )

                    # 5b: Jodi consistency with Open & Close digits
                    expected_jodi = f"{open_digit_str}{close_digit_str}"
                    self.assertEqual(
                        jodi_str,
                        expected_jodi,
                        f"Jodi mismatch at line {line_no}: Jodi='{jodi_str}' != '{expected_jodi}'",
                    )

                    # 5c: Open Patti sum rule & SP/DP/TP categorization
                    if open_patti_str:
                        self.assertRegex(
                            open_patti_str,
                            r"^\d{3}$",
                            f"Invalid Open_Patti at line {line_no}: '{open_patti_str}'",
                        )
                        open_sum = sum(int(d) for d in open_patti_str) % 10
                        self.assertEqual(
                            open_sum,
                            int(open_digit_str),
                            f"Open Patti sum rule violation at line {line_no} ({date_str}): sum({open_patti_str})%10={open_sum} != {open_digit_str}",
                        )
                        exp_open_type = classify_patti(open_patti_str)
                        self.assertEqual(
                            open_type_str,
                            exp_open_type,
                            f"Open Patti type mismatch at line {line_no} ({date_str}): patti={open_patti_str} expected {exp_open_type} got {open_type_str}",
                        )
                        open_patti_checked += 1
                    else:
                        self.assertEqual(
                            open_type_str,
                            "",
                            f"Open_Patti_Type should be empty when Open_Patti is omitted at line {line_no}",
                        )

                    # 5d: Close Patti sum rule & SP/DP/TP categorization
                    if close_patti_str:
                        self.assertRegex(
                            close_patti_str,
                            r"^\d{3}$",
                            f"Invalid Close_Patti at line {line_no}: '{close_patti_str}'",
                        )
                        close_sum = sum(int(d) for d in close_patti_str) % 10
                        self.assertEqual(
                            close_sum,
                            int(close_digit_str),
                            f"Close Patti sum rule violation at line {line_no} ({date_str}): sum({close_patti_str})%10={close_sum} != {close_digit_str}",
                        )
                        exp_close_type = classify_patti(close_patti_str)
                        self.assertEqual(
                            close_type_str,
                            exp_close_type,
                            f"Close Patti type mismatch at line {line_no} ({date_str}): patti={close_patti_str} expected {exp_close_type} got {close_type_str}",
                        )
                        close_patti_checked += 1
                    else:
                        self.assertEqual(
                            close_type_str,
                            "",
                            f"Close_Patti_Type should be empty when Close_Patti is omitted at line {line_no}",
                        )

                elif is_valid_str == "False":
                    holiday_draws += 1
                else:
                    self.fail(f"Invalid boolean string in Is_Valid at line {line_no}: '{is_valid_str}'")

            # Assert complete dataset counts
            self.assertGreaterEqual(total_rows, 3269, f"Expected at least 3,269 total records, got {total_rows}")
            self.assertGreaterEqual(valid_draws, 3170, f"Expected at least 3,170 valid draws, got {valid_draws}")
            self.assertGreaterEqual(holiday_draws, 99, f"Expected at least 99 holiday records, got {holiday_draws}")
            self.assertGreaterEqual(open_patti_checked, 3161, f"Expected at least 3,161 checked Open Pattis, got {open_patti_checked}")
            self.assertGreaterEqual(close_patti_checked, 3163, f"Expected at least 3,163 checked Close Pattis, got {close_patti_checked}")

    def test_root_csv_row_by_row(self):
        """Audits every row of tools/matka/kalyan_historical_data.csv."""
        self._audit_csv_file(ROOT_CSV_PATH)

    def test_engine_csv_row_by_row(self):
        """Audits every row of tools/matka/kalyan_4_35_to_6_35/kalyan_historical_data.csv."""
        self._audit_csv_file(ENGINE_CSV_PATH)


class TestVerifyCSVIntegrityAdversarial(unittest.TestCase):
    """Stress-tests verify_csv_integrity() with adversarial and corrupted CSV datasets."""

    def _write_csv(self, rows) -> str:
        tmp = tempfile.NamedTemporaryFile("w", delete=False, suffix=".csv", newline="", encoding="utf-8")
        writer = csv.writer(tmp)
        writer.writerow(REQUIRED_COLUMNS)
        for r in rows:
            writer.writerow(r)
        tmp.close()
        return tmp.name

    def test_clean_csv_verifies_successfully(self):
        """Verifies clean baseline CSV passes verify_csv_integrity()."""
        report = verify_csv_integrity(str(ENGINE_CSV_PATH))
        self.assertTrue(report["valid"], f"Expected valid=True, got errors: {report.get('errors')}")
        self.assertEqual(len(report["errors"]), 0)

    def test_corrupted_duplicate_dates_rejected(self):
        """Adversarial test: Duplicate dates must be caught and rejected."""
        rows = [
            ["2024-01-01", "Mon", "01", "0", "1", "127", "128", "SP", "SP", "True", "127-01-128"],
            ["2024-01-02", "Tue", "23", "2", "3", "237", "346", "SP", "SP", "True", "237-23-346"],
            ["2024-01-02", "Tue", "23", "2", "3", "237", "346", "SP", "SP", "True", "DUPLICATE"],
        ]
        tmp_csv = self._write_csv(rows)
        try:
            report = verify_csv_integrity(tmp_csv)
            self.assertFalse(report["valid"])
            error_text = " ".join(report["errors"])
            self.assertIn("duplicate date records", error_text)
        finally:
            os.remove(tmp_csv)

    def test_corrupted_non_monotonic_dates_rejected(self):
        """Adversarial test: Non-monotonic date sequences must be caught and rejected."""
        rows = [
            ["2024-01-05", "Fri", "89", "8", "9", "189", "199", "SP", "DP", "True", "189-89-199"],
            ["2024-01-02", "Tue", "23", "2", "3", "237", "346", "SP", "SP", "True", "237-23-346"],
        ]
        tmp_csv = self._write_csv(rows)
        try:
            report = verify_csv_integrity(tmp_csv)
            self.assertFalse(report["valid"])
            error_text = " ".join(report["errors"])
            self.assertIn("non-monotonic", error_text)
        finally:
            os.remove(tmp_csv)

    def test_corrupted_unparseable_dates_rejected(self):
        """Adversarial test: Malformed or unparseable date strings must be rejected."""
        rows = [
            ["NOT-A-DATE", "Mon", "01", "0", "1", "127", "128", "SP", "SP", "True", "127-01-128"],
        ]
        tmp_csv = self._write_csv(rows)
        try:
            report = verify_csv_integrity(tmp_csv)
            self.assertFalse(report["valid"])
            error_text = " ".join(report["errors"])
            self.assertIn("unparseable dates", error_text)
        finally:
            os.remove(tmp_csv)

    def test_corrupted_broken_open_patti_sum_rejected(self):
        """Adversarial test: Panel digit sum rule violation on Open Patti must be rejected."""
        rows = [
            ["2024-01-01", "Mon", "21", "2", "1", "127", "128", "SP", "SP", "True", "BROKEN_SUM"],
        ]
        tmp_csv = self._write_csv(rows)
        try:
            report = verify_csv_integrity(tmp_csv)
            self.assertFalse(report["valid"])
            error_text = " ".join(report["errors"])
            self.assertIn("violating panel digit sum rule", error_text)
        finally:
            os.remove(tmp_csv)

    def test_corrupted_broken_close_patti_sum_rejected(self):
        """Adversarial test: Panel digit sum rule violation on Close Patti must be rejected."""
        rows = [
            ["2024-01-01", "Mon", "05", "0", "5", "127", "128", "SP", "SP", "True", "BROKEN_SUM"],
        ]
        tmp_csv = self._write_csv(rows)
        try:
            report = verify_csv_integrity(tmp_csv)
            self.assertFalse(report["valid"])
            error_text = " ".join(report["errors"])
            self.assertIn("violating panel digit sum rule", error_text)
        finally:
            os.remove(tmp_csv)

    def test_corrupted_jodi_digit_inconsistency_rejected(self):
        """Adversarial test: Jodi mismatch with declared Open/Close digits must be rejected."""
        rows = [
            ["2024-01-01", "Mon", "99", "0", "1", "127", "128", "SP", "SP", "True", "JODI_MISMATCH"],
        ]
        tmp_csv = self._write_csv(rows)
        try:
            report = verify_csv_integrity(tmp_csv)
            self.assertFalse(report["valid"])
            error_text = " ".join(report["errors"])
            self.assertIn("does not match digits", error_text)
        finally:
            os.remove(tmp_csv)

    def test_corrupted_invalid_jodi_format_rejected(self):
        """Adversarial test: Non-2-digit Jodi must be rejected."""
        rows = [
            ["2024-01-01", "Mon", "123", "0", "1", "127", "128", "SP", "SP", "True", "3_DIGIT_JODI"],
        ]
        tmp_csv = self._write_csv(rows)
        try:
            report = verify_csv_integrity(tmp_csv)
            self.assertFalse(report["valid"])
            error_text = " ".join(report["errors"])
            self.assertIn("invalid Jodi values", error_text)
        finally:
            os.remove(tmp_csv)

    def test_corrupted_patti_type_misclassification_rejected(self):
        """Adversarial test: Inaccurate SP/DP/TP classification must be rejected."""
        rows = [
            ["2024-01-01", "Mon", "01", "0", "1", "127", "128", "DP", "SP", "True", "BAD_TYPE"],
        ]
        tmp_csv = self._write_csv(rows)
        try:
            report = verify_csv_integrity(tmp_csv)
            self.assertFalse(report["valid"])
            error_text = " ".join(report["errors"])
            self.assertIn("inaccurate SP/DP/TP classification", error_text)
        finally:
            os.remove(tmp_csv)

    def test_corrupted_calendar_day_mismatch_rejected(self):
        """Adversarial test: Discrepancy between Gregorian weekday and Day_Of_Week must be rejected."""
        rows = [
            ["2024-01-01", "Wed", "01", "0", "1", "127", "128", "SP", "SP", "True", "WRONG_DAY"],
        ]
        tmp_csv = self._write_csv(rows)
        try:
            report = verify_csv_integrity(tmp_csv)
            self.assertFalse(report["valid"])
            error_text = " ".join(report["errors"])
            self.assertIn("Day_Of_Week does not match calendar date", error_text)
        finally:
            os.remove(tmp_csv)

    def test_corrupted_missing_columns_rejected(self):
        """Adversarial test: Missing required columns must be caught and rejected."""
        tmp = tempfile.NamedTemporaryFile("w", delete=False, suffix=".csv", newline="", encoding="utf-8")
        writer = csv.writer(tmp)
        writer.writerow(["Date", "Day_Of_Week", "Jodi", "Open_Digit", "Close_Digit"])
        writer.writerow(["2024-01-01", "Mon", "01", "0", "1"])
        tmp.close()
        try:
            report = verify_csv_integrity(tmp.name)
            self.assertFalse(report["valid"])
            error_text = " ".join(report["errors"])
            self.assertIn("Missing required columns", error_text)
        finally:
            os.remove(tmp.name)


class TestValidatorBlindSpots(unittest.TestCase):
    """
    Empirical documentation of discovered blind spots and edge case vulnerabilities
    in verify_csv_integrity() from scraper.py.
    """

    def _write_csv(self, rows) -> str:
        tmp = tempfile.NamedTemporaryFile("w", delete=False, suffix=".csv", newline="", encoding="utf-8")
        writer = csv.writer(tmp)
        writer.writerow(REQUIRED_COLUMNS)
        for r in rows:
            writer.writerow(r)
        tmp.close()
        return tmp.name

    def test_blind_spot_float_formatting_tolerated(self):
        """
        VULNERABILITY: verify_csv_integrity() uses int(float(Open_Digit)),
        which tolerates float representations like '5.0' rather than rejecting them.
        """
        rows = [
            ["2024-01-01", "Mon", "51", "5.0", "1", "140", "146", "SP", "SP", "True", "51"],
        ]
        tmp_csv = self._write_csv(rows)
        try:
            report = verify_csv_integrity(tmp_csv)
            # Documents that the validator allows '5.0' without flagging any errors
            self.assertTrue(report["valid"], "Vulnerability confirmed: validator tolerates '5.0' in Open_Digit")
            self.assertEqual(len(report["errors"]), 0)
        finally:
            os.remove(tmp_csv)

    def test_blind_spot_malformed_patti_bypasses_check(self):
        """
        VULNERABILITY: verify_csv_integrity() guards patti check with if re.fullmatch(r'\\d{3}', patti).
        If patti is malformed (e.g. 'xyz' or '140.0'), the check is silently bypassed with 0 errors.
        """
        rows = [
            ["2024-01-01", "Mon", "51", "5", "1", "xyz", "146", "SP", "SP", "True", "51"],
        ]
        tmp_csv = self._write_csv(rows)
        try:
            report = verify_csv_integrity(tmp_csv)
            # Documents that the validator allows non-digit Patti 'xyz' without flagging any errors
            self.assertTrue(report["valid"], "Vulnerability confirmed: validator silently skips non-digit patti")
            self.assertEqual(len(report["errors"]), 0)
        finally:
            os.remove(tmp_csv)

    def test_blind_spot_nonexistent_custom_path_fallback(self):
        """
        VULNERABILITY: resolve_csv_path() silently falls back to the default production dataset
        when an invalid custom path is provided, masking file lookup failures.
        """
        bogus_path = "non_existent_fake_path_xyz.csv"
        resolved = resolve_csv_path(bogus_path)
        self.assertTrue(os.path.exists(resolved), "resolve_csv_path fell back to existing default dataset")
        self.assertNotEqual(resolved, os.path.abspath(bogus_path), "Resolved path differs from requested bogus path")


if __name__ == "__main__":
    unittest.main()
