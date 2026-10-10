"""
Challenger Tier 5 Adversarial Stress Test Suite: Milestone 1 (M1)
================================================================
Exhaustively challenges and stress-tests:
1. CLI path resolution & handling of nonexistent/invalid input/output arguments.
2. Scraper network resilience, exponential backoff, anti-bot payload rejection, and cache decoupling.
3. Sync and push pipeline safety, REPO_ROOT precision, and dirty CSV abort guards.
4. Historical CSV multi-point integrity verifier against synthetic corruptions.
5. Three-way artifact schema and synchronization parity.
"""

import os
import sys
import tempfile
import unittest
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock
import requests
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
ENGINE_DIR = REPO_ROOT / "tools" / "matka" / "kalyan_4_35_to_6_35"
MATKA_ROOT = REPO_ROOT / "tools" / "matka"

# Add engine dir to sys.path
for p in [str(REPO_ROOT), str(ENGINE_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import scraper
import predict
import backtest
import sync_and_push


class TestCLIArgumentAdversarial(unittest.TestCase):
    """Adversarial stress-testing of CLI path handling and error behavior."""

    def test_predict_rejects_nonexistent_positional_csv(self):
        """
        VULNERABILITY CHALLENGE:
        Passing a nonexistent CSV path to predict.py should exit with non-zero code.
        Silent fallback to production data when the user specified a custom path is unsafe.
        """
        cmd = [sys.executable, str(ENGINE_DIR / "predict.py"), "nonexistent_dummy_draws_9999.csv"]
        res = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
        self.assertNotEqual(
            res.returncode,
            0,
            f"predict.py silently succeeded with exit code 0 on nonexistent file! Stderr: {res.stderr}"
        )

    def test_predict_rejects_nonexistent_flag_csv(self):
        """
        VULNERABILITY CHALLENGE:
        Passing --csv nonexistent.csv should exit with non-zero code.
        """
        cmd = [sys.executable, str(ENGINE_DIR / "predict.py"), "--csv", "nonexistent_dummy_draws_9999.csv"]
        res = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
        self.assertNotEqual(
            res.returncode,
            0,
            f"predict.py --csv silently succeeded with exit code 0 on nonexistent file! Stderr: {res.stderr}"
        )

    def test_backtest_rejects_nonexistent_positional_csv(self):
        """
        VULNERABILITY CHALLENGE:
        Passing a nonexistent CSV path to backtest.py should exit with non-zero code.
        """
        cmd = [sys.executable, str(ENGINE_DIR / "backtest.py"), "nonexistent_dummy_draws_9999.csv"]
        res = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
        self.assertNotEqual(
            res.returncode,
            0,
            f"backtest.py silently succeeded with exit code 0 on nonexistent file! Stderr: {res.stderr}"
        )

    def test_backtest_rejects_nonexistent_flag_csv(self):
        """
        VULNERABILITY CHALLENGE:
        Passing --csv nonexistent.csv to backtest.py should exit with non-zero code.
        """
        cmd = [sys.executable, str(ENGINE_DIR / "backtest.py"), "--csv", "nonexistent_dummy_draws_9999.csv"]
        res = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
        self.assertNotEqual(
            res.returncode,
            0,
            f"backtest.py --csv silently succeeded with exit code 0 on nonexistent file! Stderr: {res.stderr}"
        )

    def test_scraper_resolve_csv_path_does_not_hijack_new_output_path(self):
        """
        VULNERABILITY CHALLENGE:
        When exporting to a new output path that does not exist yet (e.g. --output new_dataset.csv),
        resolve_csv_path must NOT fall back to kalyan_historical_data.csv and overwrite production data!
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            intended_target = os.path.join(tmpdir, "new_custom_dataset.csv")
            self.assertFalse(os.path.exists(intended_target))

            resolved = scraper.resolve_csv_path(intended_target)
            self.assertEqual(
                os.path.abspath(resolved),
                os.path.abspath(intended_target),
                f"resolve_csv_path hijacked non-existent output path to: {resolved}"
            )

    def test_predict_handles_float_serialized_jodi_strings(self):
        """
        VULNERABILITY CHALLENGE:
        If a CSV contains float-formatted strings (e.g. '1.0' instead of '01'),
        predict.py should gracefully cast float strings instead of crashing with:
        ValueError: invalid literal for int() with base 10: '1.0'.
        """
        dirty_records = [
            {
                "Date": f"2024-01-{i:02d}",
                "Day_Of_Week": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][(i - 1) % 7],
                "Jodi": f"{i % 10}.0",
                "Open_Digit": f"{(i % 10)}.0",
                "Close_Digit": "0.0",
                "Open_Patti": "123",
                "Close_Patti": "145",
                "Open_Patti_Type": "SP",
                "Close_Patti_Type": "SP",
                "Is_Valid": True,
                "Raw_Entry": "123-01-145",
            }
            for i in range(1, 25)
        ]
        df = pd.DataFrame(dirty_records)
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".csv") as tmp:
            df.to_csv(tmp.name, index=False)
            tmp_path = tmp.name

        try:
            res = predict.generate_upcoming_prediction(tmp_path)
            self.assertIn("top_jodis", res)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


class TestScraperNetworkResilienceAndDecoupling(unittest.TestCase):
    """Adversarial stress-testing of scraper network behavior and cache decoupling."""

    def test_scraper_retry_on_3_consecutive_timeouts(self):
        """Verifies 3 attempts and exponential backoff [1, 2] on persistent timeouts."""
        attempts = []

        def mock_timeout(*args, **kwargs):
            attempts.append(kwargs.get("timeout"))
            raise requests.exceptions.Timeout("Simulated gateway timeout")

        with patch("requests.get", side_effect=mock_timeout):
            with patch("time.sleep") as mock_sleep:
                content = scraper.fetch_html(force_refresh=True)
                self.assertEqual(len(attempts), 3, "Scraper did not retry 3 times")
                self.assertEqual(attempts, [10, 15, 20], "Timeout escalation mismatch")
                sleep_delays = [call.args[0] for call in mock_sleep.call_args_list]
                self.assertEqual(sleep_delays, [1, 2], "Exponential backoff delays mismatch")
                self.assertIn("chart-table", content, "Failed to fall back to decoupled cache")

    def test_scraper_runtime_error_when_no_cache_and_network_fails(self):
        """Verifies RuntimeError is raised if all 3 network attempts fail and no cache exists."""
        def mock_error(*args, **kwargs):
            raise requests.exceptions.ConnectionError("Simulated total blackout")

        with tempfile.TemporaryDirectory() as tmpdir:
            phantom_cache = os.path.join(tmpdir, "nonexistent_cache.html")
            with patch("requests.get", side_effect=mock_error):
                with patch("time.sleep"):
                    with self.assertRaises(RuntimeError):
                        scraper.fetch_html(cache_file=phantom_cache, force_refresh=True)

    def test_scraper_rejects_antibot_html_without_tables(self):
        """
        Adversarial test: If server returns HTTP 200 with Cloudflare/anti-bot challenge
        that lacks table markup, scraper must reject the payload, retry, and fall back to cache.
        """
        antibot_html = (
            "<!DOCTYPE html><html><head><title>Attention Required! | Cloudflare</title></head>"
            "<body><p>Please complete the security check to access dpbossx.net</p></body></html>"
        )
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.text = antibot_html

        with patch("requests.get", return_value=mock_resp):
            with patch("time.sleep"):
                content = scraper.fetch_html(force_refresh=True)
                # Content returned must be the cached content containing chart-table, NOT the anti-bot HTML
                self.assertIn("chart-table", content)
                self.assertNotIn("Attention Required", content)

    def test_user_facing_penal_chart_is_never_overwritten_by_scraper(self):
        """
        Verifies decoupled cache invariant:
        Running scraper never touches or overwrites tools/matka/kalyan_4_35_to_6_35/kalyan_penal_chart.html.
        """
        ui_html_path = ENGINE_DIR / "kalyan_penal_chart.html"
        self.assertTrue(ui_html_path.exists(), "User UI kalyan_penal_chart.html must exist")
        mtime_before = os.path.getmtime(ui_html_path)

        # Execute scrape
        scraper.fetch_html(force_refresh=False)

        mtime_after = os.path.getmtime(ui_html_path)
        self.assertEqual(
            mtime_before,
            mtime_after,
            "CRITICAL: User-facing kalyan_penal_chart.html was touched/modified by scraper!"
        )


class TestSyncAndPushPipelineSafety(unittest.TestCase):
    """Stress-testing repo root calculation and dirty-CSV safety guards."""

    def test_repo_root_points_to_workspace_root(self):
        """
        Verifies REPO_ROOT in sync_and_push.py is exactly the workspace root
        and NOT tools/ or tools/matka/.
        """
        expected_root = REPO_ROOT
        actual_root = Path(sync_and_push.REPO_ROOT)
        self.assertEqual(
            actual_root.resolve(),
            expected_root.resolve(),
            f"REPO_ROOT calculation error: expected {expected_root}, got {actual_root}"
        )

    def test_no_push_flag_prevents_git_commands(self):
        """Verifies --no-push executes local pipeline without triggering git commit or push."""
        with patch("subprocess.run") as mock_subproc:
            # Call sync_and_push with push=False
            sync_and_push.sync_and_push(push=False)
            # Ensure git commit and git push were NEVER called
            for call in mock_subproc.call_args_list:
                args = call[0][0]
                self.assertNotIn("commit", args)
                self.assertNotIn("push", args)

    def test_sync_aborts_immediately_if_csv_integrity_fails(self):
        """
        Safety guard: If CSV integrity check fails, sync_and_push must exit(1)
        before reaching any Git commands.
        """
        with patch("sync_and_push.verify_csv_integrity", return_value={"valid": False, "errors": ["Corrupted panel"]}) as mock_verify:
            with patch("sync_and_push.clean_and_deduplicate_file"):
                with patch("sys.exit") as mock_exit:
                    sync_and_push.sync_and_push(push=True)
                    mock_exit.assert_called_with(1)


class TestCSVIntegrityValidatorAdversarial(unittest.TestCase):
    """Adversarial stress-testing of verify_csv_integrity against corrupt data."""

    def setUp(self):
        self.clean_df = pd.read_csv(scraper.DEFAULT_OUTPUT_CSV)

    def test_catches_duplicate_dates(self):
        """Verifies duplicate calendar dates are caught."""
        df_dirty = self.clean_df.head(20).copy()
        df_dirty.loc[1, "Date"] = df_dirty.loc[0, "Date"]
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".csv") as tmp:
            df_dirty.to_csv(tmp.name, index=False)
            tmp_path = tmp.name

        try:
            report = scraper.verify_csv_integrity(tmp_path)
            self.assertFalse(report["valid"])
            self.assertTrue(any("duplicate" in err.lower() for err in report["errors"]))
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_catches_non_monotonic_date_order(self):
        """Verifies out-of-order dates are caught."""
        df_dirty = self.clean_df.head(20).copy()
        # Swap rows 5 and 6
        df_dirty.iloc[5], df_dirty.iloc[6] = df_dirty.iloc[6].copy(), df_dirty.iloc[5].copy()
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".csv") as tmp:
            df_dirty.to_csv(tmp.name, index=False)
            tmp_path = tmp.name

        try:
            report = scraper.verify_csv_integrity(tmp_path)
            self.assertFalse(report["valid"])
            self.assertTrue(any("non-monotonic" in err.lower() for err in report["errors"]))
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_catches_panel_digit_sum_violation(self):
        """Verifies patti sum != digit is caught."""
        df_dirty = self.clean_df.head(20).copy()
        # Set Open_Patti to 123 (sum 6) for an Open_Digit of 0
        df_dirty.loc[0, "Open_Digit"] = "0"
        df_dirty.loc[0, "Open_Patti"] = "123"
        df_dirty.loc[0, "Is_Valid"] = True
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".csv") as tmp:
            df_dirty.to_csv(tmp.name, index=False)
            tmp_path = tmp.name

        try:
            report = scraper.verify_csv_integrity(tmp_path)
            self.assertFalse(report["valid"])
            self.assertTrue(any("sum" in err.lower() for err in report["errors"]))
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_catches_day_of_week_calendar_mismatch(self):
        """Verifies Day_Of_Week mismatching astronomical calendar day is caught."""
        df_dirty = self.clean_df.head(20).copy()
        df_dirty.loc[0, "Day_Of_Week"] = "Sun" if df_dirty.loc[0, "Day_Of_Week"] != "Sun" else "Mon"
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".csv") as tmp:
            df_dirty.to_csv(tmp.name, index=False)
            tmp_path = tmp.name

        try:
            report = scraper.verify_csv_integrity(tmp_path)
            self.assertFalse(report["valid"])
            self.assertTrue(any("day_of_week" in err.lower() for err in report["errors"]))
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


class TestArtifactMirroringParity(unittest.TestCase):
    """Verifies atomic parity across root tools/matka/, subfolder, and web/ mirrors."""

    def test_prediction_data_json_mirrors_exist_and_match(self):
        """Verifies prediction_data.json exists and is synchronized across all 3 tiers."""
        p_root = MATKA_ROOT / "prediction_data.json"
        p_sub = ENGINE_DIR / "prediction_data.json"
        p_web = ENGINE_DIR / "web" / "prediction_data.json"

        self.assertTrue(p_root.exists(), "Root prediction_data.json missing")
        self.assertTrue(p_sub.exists(), "Subfolder prediction_data.json missing")
        self.assertTrue(p_web.exists(), "Web prediction_data.json missing")

        with open(p_root, "r", encoding="utf-8") as f:
            d_root = json.load(f)
        with open(p_sub, "r", encoding="utf-8") as f:
            d_sub = json.load(f)
        with open(p_web, "r", encoding="utf-8") as f:
            d_web = json.load(f)

        self.assertEqual(d_root["latest_draw"], d_sub["latest_draw"])
        self.assertEqual(d_root["latest_draw"], d_web["latest_draw"])

    def test_history_json_mirrors_exist_and_match(self):
        """Verifies history.json exists and is synchronized across all 3 tiers."""
        h_root = MATKA_ROOT / "history.json"
        h_sub = ENGINE_DIR / "history.json"
        h_web = ENGINE_DIR / "web" / "history.json"

        self.assertTrue(h_root.exists(), "Root history.json missing")
        self.assertTrue(h_sub.exists(), "Subfolder history.json missing")
        self.assertTrue(h_web.exists(), "Web history.json missing")

        with open(h_root, "r", encoding="utf-8") as f:
            d_root = json.load(f)
        with open(h_sub, "r", encoding="utf-8") as f:
            d_sub = json.load(f)
        with open(h_web, "r", encoding="utf-8") as f:
            d_web = json.load(f)

        self.assertEqual(len(d_root), len(d_sub))
        self.assertEqual(len(d_root), len(d_web))
        self.assertEqual(d_root[-1], d_sub[-1])
        self.assertEqual(d_root[-1], d_web[-1])


if __name__ == "__main__":
    unittest.main()
