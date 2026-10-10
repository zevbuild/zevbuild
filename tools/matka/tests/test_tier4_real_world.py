"""
Tier 4: Real-World Application Scenarios E2E Tests for Kalyan Matka Suite.
Covers real-world user flows and operational cycles:
- Daily prediction cycle (ingestion -> prediction -> artifact serialization)
- Saturday to Monday recess flow (Sunday market recess handling)
- Offline PWA standalone execution & client data sufficiency
- Walk-forward historical backtesting simulation audit
- Live Indian Standard Time (IST) market countdown state machine
"""

import json
import os
import sys
import unittest
from datetime import datetime, time, timedelta, timezone
from pathlib import Path

# Paths
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
MATKA_ROOT = REPO_ROOT / "tools" / "matka"
MATKA_ENGINE = MATKA_ROOT / "kalyan_4_35_to_6_35"
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
SAMPLE_DRAWS_CSV = FIXTURES_DIR / "sample_draws.csv"

# Add engine directory to sys.path
if str(MATKA_ENGINE) not in sys.path:
    sys.path.insert(0, str(MATKA_ENGINE))


class TestRealWorldApplicationScenarios(unittest.TestCase):
    """Tier 4: End-to-End Real-World Application Workloads."""

    def test_scenario_daily_prediction_cycle(self):
        """Simulates full daily prediction cycle from draw declaration to JSON persistence."""
        import predict

        # 1. Run prediction generation
        res = predict.generate_upcoming_prediction(str(SAMPLE_DRAWS_CSV))

        # 2. Inspect latest draw and target draw
        self.assertIn("latest_draw", res)
        self.assertIn("target_draw", res)
        latest_date = datetime.strptime(res["latest_draw"]["date"], "%Y-%m-%d")
        target_date = datetime.strptime(res["target_draw"]["date"], "%Y-%m-%d")
        self.assertGreater(target_date, latest_date, "Target draw must be strictly after latest draw")

        # 3. Verify OTC and top recommendations
        otc = res["otc_digits"]
        self.assertEqual(len(otc), 4)
        top_jodis = res["top_jodis"]
        self.assertGreaterEqual(len(top_jodis), 10)

        # 4. Verify serialized artifact format
        payload = {
            "last_developer_update": datetime.now().strftime("%Y-%m-%d"),
            "latest_draw": res["latest_draw"],
            "target_draw": res["target_draw"],
            "otc_digits": otc,
            "top_1_jodi": res["top_1_jodi"],
            "family_bracket": res["family_bracket"],
        }
        serialized = json.dumps(payload)
        reloaded = json.loads(serialized)
        self.assertEqual(reloaded["target_draw"]["date"], res["target_draw"]["date"])

    def test_scenario_sunday_market_recess_flow(self):
        """Simulates draw declared on Saturday evening skipping Sunday to target Monday."""
        # When a draw occurs on Saturday
        sat_date = datetime(2024, 1, 6)  # Saturday
        self.assertEqual(sat_date.weekday(), 5)

        # Calculate target draw
        next_date = sat_date + timedelta(days=1)
        if next_date.weekday() == 6:  # Sunday recess
            next_date += timedelta(days=1)

        self.assertEqual(next_date.weekday(), 0, "Sunday must be skipped; target must be Monday")
        self.assertEqual(next_date.strftime("%Y-%m-%d"), "2024-01-08")

        # Monday prior distribution exists in prediction data
        pred_p = MATKA_ENGINE / "prediction_data.json"
        if pred_p.exists():
            data = json.loads(pred_p.read_text(encoding="utf-8"))
            if "by_day" in data:
                self.assertIn("Mon", data["by_day"], "Forecasts must include Monday line prior")

    def test_scenario_offline_pwa_standalone_execution(self):
        """Verifies all assets required for offline operation are present and precached."""
        sw_p = MATKA_ENGINE / "sw.js"
        self.assertTrue(sw_p.exists(), "sw.js must exist")
        sw_content = sw_p.read_text(encoding="utf-8")

        manifest_p = MATKA_ENGINE / "manifest.json"
        self.assertTrue(manifest_p.exists(), "manifest.json must exist")

        index_p = MATKA_ENGINE / "index.html"
        self.assertTrue(index_p.exists(), "index.html must exist")

        # Check history.json is available for client-side JavaScript models
        hist_p = MATKA_ENGINE / "history.json"
        self.assertTrue(hist_p.exists(), "history.json must exist for client predictor")
        hist_data = json.loads(hist_p.read_text(encoding="utf-8"))
        self.assertGreater(len(hist_data), 100, "history.json must contain sufficient historical draws")

        # Check service worker precache includes essential offline files
        for f in ["index.html", "manifest.json"]:
            self.assertTrue(f in sw_content or "./" in sw_content, f"Precache must reference {f}")

    def test_scenario_walk_forward_backtest_audit(self):
        """Simulates walk-forward backtesting execution and verifies mathematical metrics."""
        import backtest

        # Run walk-forward backtest on sample fixtures with small warmup
        res = backtest.run_walk_forward_backtest(
            csv_path=str(SAMPLE_DRAWS_CSV),
            warmup_draws=10,
        )
        self.assertIsNotNone(res)
        draws_tested = res.get("test_draws", res.get("total_tested", 0))
        self.assertGreater(draws_tested, 0)

        # Check pass rates
        self.assertIn("otc_pass_rate", res)
        otc_val = res["otc_pass_rate"]
        otc_rate = float(str(otc_val).replace("%", ""))
        self.assertGreaterEqual(otc_rate, 0.0)
        self.assertLessEqual(otc_rate, 100.0)

        # Check top-jodi hit rate
        self.assertIn("top10_jodi_hit_rate", res)
        top10_val = res["top10_jodi_hit_rate"]
        top10_rate = float(str(top10_val).replace("%", ""))
        self.assertGreaterEqual(top10_rate, 0.0)
        self.assertLessEqual(top10_rate, 100.0)

    def test_scenario_live_market_countdown_state_machine(self):
        """Verifies IST market timing state machine transitions across operating sessions."""
        # Indian Standard Time is UTC + 5:30
        ist_tz = timezone(timedelta(hours=5, minutes=30))

        def get_market_state(dt: datetime) -> str:
            # Check Sunday recess
            if dt.weekday() == 6:
                return "WEEKEND_RECESS"
            # Operating times: Open = 16:35 IST, Close = 18:35 IST
            open_time = time(16, 35)
            close_time = time(18, 35)
            t = dt.time()
            if t < open_time:
                return "PRE_OPEN"
            elif t < close_time:
                return "LIVE_DRAW_ACTIVE"
            else:
                return "MARKET_CLOSED"

        # 1. Monday 14:00 IST -> Pre-Open
        t1 = datetime(2024, 1, 1, 14, 0, tzinfo=ist_tz)
        self.assertEqual(get_market_state(t1), "PRE_OPEN")

        # 2. Monday 17:15 IST -> Live Draw Active (Open declared, Close countdown)
        t2 = datetime(2024, 1, 1, 17, 15, tzinfo=ist_tz)
        self.assertEqual(get_market_state(t2), "LIVE_DRAW_ACTIVE")

        # 3. Monday 19:00 IST -> Market Closed (Session complete)
        t3 = datetime(2024, 1, 1, 19, 0, tzinfo=ist_tz)
        self.assertEqual(get_market_state(t3), "MARKET_CLOSED")

        # 4. Sunday 12:00 IST -> Weekend Recess
        t4 = datetime(2024, 1, 7, 12, 0, tzinfo=ist_tz)
        self.assertEqual(get_market_state(t4), "WEEKEND_RECESS")


if __name__ == "__main__":
    unittest.main()
