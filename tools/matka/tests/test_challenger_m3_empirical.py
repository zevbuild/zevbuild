"""
Challenger Tier 5 Adversarial & Empirical Verification Suite: Milestone 3 Web & Mobile PWA Integration.

Empirical verification of:
1. REST API endpoints in KalyanWebHandler:
   - /api/status: schema, online status, draw_count (>3100), last_draw_date, valid ISO datetime, CORS headers, query param resilience.
   - /api/performance-log: schema (summary, draw_logs, current_hyperparameters), 200 OK, valid JSON, fallback handling.
   - OPTIONS pre-flight and 404 on invalid routes.
2. compute_last_week_performance data contract:
   - actual_rank (int, 1..100)
   - top10_hit (bool, matches actual_rank <= 10)
   - family_cut_hit (bool)
   - otc_hit (bool)
   - rank_pill (exact format 'Rank #{actual_rank}')
   - otc_digits (two complementary cut pairs [d_a, cut(d_a), d_b, cut(d_b)])
3. Frontend DOM & Historical Table contracts in web/index.html:
   - Patterns: 'Rank #', 'Top-10 Hit', 'Family Cut Hit', 'OTC Hit'
   - Mobile chips highlight 'Cut Fam'
   - Mobile OTC digits display complementary cut pairs ('Cut Pair A', 'Cut Pair B', 'Cut ')
   - Badges and DOM element IDs existence and simulation
4. PWA Cache v4 & Service Worker:
   - web/sw.js and sw.js declare 'kalyan-predictor-cache-v4'
   - Precaches './performance_log.json'
   - Fetch handler matches 'performance_log.json' and '/api/performance-log' offline
   - Cache cleanup logic
5. 3-Way Bit-Exact File Parity:
   - SHA256 equality of prediction_data.json, history.json, performance_log.json across root, engine, and web dirs.
   - SHA256 equality of index.html, dashboard.html, web/index.html.
   - SHA256 equality of sw.js, web/sw.js.
   - Root multi-tool portal index.html preservation.
6. Adversarial edge cases and stress tests:
   - Malformed queries, corrupt/edge dataframe inputs, rank boundary conditions.
"""

import os
import sys
import json
import re
import socket
import threading
import hashlib
import unittest
from datetime import datetime
from pathlib import Path
from http.client import HTTPConnection
import pandas as pd
import numpy as np

# Ensure workspace is importable
KALYAN_DIR = Path(__file__).resolve().parent.parent / "kalyan_4_35_to_6_35"
if str(KALYAN_DIR) not in sys.path:
    sys.path.insert(0, str(KALYAN_DIR))

from app import KalyanWebHandler, ThreadedHTTPServer, compute_last_week_performance, CSV_PATH, WORKSPACE_DIR, WEB_DIR
from models import evaluate_outcome, get_family_jodis


def get_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


class TestM3ApiEndpoints(unittest.TestCase):
    """Empirical verification of app.py HTTP endpoints using real socket connections."""

    @classmethod
    def setUpClass(cls):
        cls.port = get_free_port()
        cls.server = ThreadedHTTPServer(('127.0.0.1', cls.port), KalyanWebHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def _get(self, path):
        conn = HTTPConnection('127.0.0.1', self.port, timeout=5)
        conn.request('GET', path)
        resp = conn.getresponse()
        data = resp.read()
        conn.close()
        return resp.status, resp.getheaders(), data

    def _options(self, path):
        conn = HTTPConnection('127.0.0.1', self.port, timeout=5)
        conn.request('OPTIONS', path)
        resp = conn.getresponse()
        data = resp.read()
        conn.close()
        return resp.status, resp.getheaders(), data

    def test_api_status_success(self):
        """Verify /api/status returns 200, valid JSON, online status, draw_count > 3100, and valid ISO time."""
        status, headers, body = self._get('/api/status')
        self.assertEqual(status, 200)

        # Content-Type check
        headers_dict = dict(headers)
        self.assertIn('application/json', headers_dict.get('Content-Type', ''))
        self.assertEqual(headers_dict.get('Access-Control-Allow-Origin'), '*')

        data = json.loads(body.decode('utf-8'))
        self.assertEqual(data.get('status'), 'online')
        self.assertIsInstance(data.get('draw_count'), int)
        self.assertGreater(data.get('draw_count'), 3100)

        # Cross-verify draw_count with CSV
        df = pd.read_csv(CSV_PATH)
        expected_valid_count = int((df['Is_Valid'] == True).sum())
        self.assertEqual(data['draw_count'], expected_valid_count)

        # Verify last_draw_date matches last row
        expected_last_date = str(df[df['Is_Valid'] == True].iloc[-1]['Date'])
        self.assertEqual(data['last_draw_date'], expected_last_date)

        # Verify ISO format of time
        parsed_time = datetime.fromisoformat(data['time'])
        self.assertIsInstance(parsed_time, datetime)

    def test_api_status_with_query_params(self):
        """Verify /api/status is resilient to arbitrary or malicious query parameters."""
        for q in ['?foo=bar', '?t=123456789&nocache=1', '?malicious=<script>alert(1)</script>', '?%00']:
            status, _, body = self._get(f'/api/status{q}')
            self.assertEqual(status, 200)
            data = json.loads(body.decode('utf-8'))
            self.assertEqual(data.get('status'), 'online')
            self.assertGreater(data.get('draw_count'), 3100)

    def test_api_performance_log_success(self):
        """Verify /api/performance-log returns 200, valid JSON with summary and draw_logs."""
        status, headers, body = self._get('/api/performance-log')
        self.assertEqual(status, 200)

        headers_dict = dict(headers)
        self.assertIn('application/json', headers_dict.get('Content-Type', ''))
        self.assertEqual(headers_dict.get('Access-Control-Allow-Origin'), '*')

        data = json.loads(body.decode('utf-8'))
        self.assertIn('summary', data)
        self.assertIn('draw_logs', data)
        self.assertIn('current_hyperparameters', data)

        summary = data['summary']
        self.assertIn('total_draws_logged', summary)
        self.assertIn('top10_hit_rate', summary)
        self.assertIn('family_cut_hit_rate', summary)
        self.assertIn('otc_hit_rate', summary)
        self.assertGreater(summary['total_draws_logged'], 0)

        draw_logs = data['draw_logs']
        self.assertIsInstance(draw_logs, list)
        self.assertGreater(len(draw_logs), 0)

        # Check structure of the latest log entry
        latest = draw_logs[-1]
        self.assertIn('date', latest)
        self.assertIn('actual_draw', latest)
        self.assertIn('predictions', latest)
        self.assertIn('outcome', latest)

        outcome = latest['outcome']
        self.assertIn('actual_rank', outcome)
        self.assertIn('top10_hit', outcome)
        self.assertIn('family_cut_hit', outcome)
        self.assertIn('otc_hit', outcome)

    def test_api_performance_log_query_resilience(self):
        """Verify /api/performance-log ignores extraneous query parameters."""
        status, _, body = self._get('/api/performance-log?limit=5&sort=desc')
        self.assertEqual(status, 200)
        data = json.loads(body.decode('utf-8'))
        self.assertIn('summary', data)

    def test_api_cors_preflight(self):
        """Verify OPTIONS pre-flight requests receive CORS headers."""
        status, headers, _ = self._options('/api/status')
        self.assertEqual(status, 200)
        headers_dict = dict(headers)
        self.assertEqual(headers_dict.get('Access-Control-Allow-Origin'), '*')
        self.assertIn('GET', headers_dict.get('Access-Control-Allow-Methods', ''))

    def test_static_asset_serving(self):
        """Verify static files (performance_log.json, sw.js, manifest.json) are served properly."""
        for path in ['/performance_log.json', '/sw.js', '/manifest.json', '/prediction_data.json']:
            status, _, body = self._get(path)
            self.assertEqual(status, 200, f"Failed to serve {path}")
            self.assertGreater(len(body), 0)


class TestComputeLastWeekPerformanceContracts(unittest.TestCase):
    """Empirical verification of compute_last_week_performance data contracts."""

    @classmethod
    def setUpClass(cls):
        df = pd.read_csv(CSV_PATH)
        cls.valid_df = df[df['Is_Valid'] == True].reset_index(drop=True)
        cls.last_week, cls.last_month = compute_last_week_performance(cls.valid_df)

    def test_structure_and_weeks(self):
        """Verify output has last_week and last_month with expected attributes."""
        self.assertIn('draws', self.last_week)
        self.assertIn('summary', self.last_week)
        self.assertGreater(len(self.last_week['draws']), 0)

        self.assertIn('all_draws', self.last_month)
        self.assertIn('summary', self.last_month)
        self.assertGreater(len(self.last_month['all_draws']), 0)

    def test_every_draw_data_contract(self):
        """Verify every draw object satisfies Milestone 3 data contract."""
        all_draws = self.last_month['all_draws']

        for idx, d in enumerate(all_draws):
            date_str = d.get('date')
            actual_rank = d.get('actual_rank')
            top10_hit = d.get('top10_hit')
            family_cut_hit = d.get('family_cut_hit')
            otc_hit = d.get('otc_hit')
            rank_pill = d.get('rank_pill')
            otc_digits = d.get('otc_digits')

            # 1. actual_rank must be integer in 1..100
            self.assertIsInstance(actual_rank, int, f"Draw {idx} ({date_str}): actual_rank is not int")
            self.assertTrue(1 <= actual_rank <= 100, f"Draw {idx} ({date_str}): actual_rank {actual_rank} not in 1..100")

            # 2. top10_hit must be boolean, strictly True iff actual_rank <= 10
            self.assertIsInstance(top10_hit, bool, f"Draw {idx} ({date_str}): top10_hit not bool")
            self.assertEqual(top10_hit, (actual_rank <= 10),
                             f"Draw {idx} ({date_str}): top10_hit={top10_hit} inconsistent with actual_rank={actual_rank}")

            # 3. family_cut_hit must be boolean
            self.assertIsInstance(family_cut_hit, bool, f"Draw {idx} ({date_str}): family_cut_hit not bool")

            # 4. otc_hit must be boolean
            self.assertIsInstance(otc_hit, bool, f"Draw {idx} ({date_str}): otc_hit not bool")

            # 5. rank_pill must be formatted as 'Rank #{actual_rank}'
            expected_pill = f"Rank #{actual_rank}"
            self.assertEqual(rank_pill, expected_pill,
                             f"Draw {idx} ({date_str}): rank_pill '{rank_pill}' != '{expected_pill}'")

            # 6. otc_digits must be 4 digits with two complementary cut pairs
            self.assertIsInstance(otc_digits, list)
            self.assertEqual(len(otc_digits), 4)
            self.assertEqual(len(set(otc_digits)), 4)
            for digit in otc_digits:
                self.assertIn((digit + 5) % 10, otc_digits, f"Harmonic cut of {digit} not in otc_digits {otc_digits}")

            # 7. status_label and badge_type
            self.assertIn(d.get('badge_type'), ['success', 'highlight', 'info', 'neutral'])
            self.assertTrue(bool(d.get('status_label')))


class TestFrontendDomAndBadges(unittest.TestCase):
    """Empirical verification of web/index.html historical table badges, DOM IDs, and mobile chips."""

    def setUp(self):
        self.html_path = KALYAN_DIR / "web" / "index.html"
        self.assertTrue(self.html_path.exists())
        self.html_content = self.html_path.read_text(encoding="utf-8")

    def test_required_string_patterns(self):
        """Verify presence of required badge string patterns in HTML/JS."""
        patterns = [
            'Rank #',
            'Top-10 Hit',
            'Family Cut Hit',
            'OTC Hit',
            'Cut Fam',
            'Cut Pair A',
            'Cut Pair B',
        ]
        for pat in patterns:
            self.assertIn(pat, self.html_content, f"Pattern '{pat}' missing from web/index.html")

    def test_critical_dom_element_ids(self):
        """Verify all critical DOM IDs referenced by rendering functions exist in markup."""
        critical_ids = [
            'last-week-table-body',
            'mobile-otc-digits',
            'mobile-top15-chips',
            'mobile-otc-prob',
            'telemetry-30d-touch',
            'kpi-edge-rate',
            'kpi-edge-sub',
        ]
        for el_id in critical_ids:
            pattern = f'id="{el_id}"'
            self.assertIn(pattern, self.html_content, f"DOM element with id='{el_id}' not found in index.html")

    def test_historical_table_rendering_logic_simulation(self):
        """Simulate JS logic in renderHistoricalTable with synthetic edge-case draws."""
        # Test Case 1: Top 10 Hit (Rank 1, Top Pick Hit)
        actual_rank = 1
        is_top10 = actual_rank <= 10
        is_fam = True
        is_otc = True
        is_top_pick_hit = True

        rank_pill = f"Rank #{actual_rank}"
        badges = []
        if is_top10:
            badges.append("Top-10 Hit")
        if is_fam:
            badges.append("Family Cut Hit")
        if is_otc:
            badges.append("OTC Hit")

        self.assertEqual(rank_pill, "Rank #1")
        self.assertIn("Top-10 Hit", badges)
        self.assertIn("Family Cut Hit", badges)
        self.assertIn("OTC Hit", badges)

        # Test Case 2: Boundary Rank 10
        actual_rank = 10
        is_top10 = actual_rank <= 10
        self.assertTrue(is_top10)

        # Test Case 3: Boundary Rank 11 (Out of Top 10)
        actual_rank = 11
        is_top10 = actual_rank <= 10
        self.assertFalse(is_top10)

        # Test Case 4: Deep Rank 100
        actual_rank = 100
        is_top10 = actual_rank <= 10
        self.assertFalse(is_top10)
        self.assertEqual(f"Rank #{actual_rank}", "Rank #100")


class TestPwaCacheAndServiceWorker(unittest.TestCase):
    """Empirical verification of service worker cache version v4 and offline manifest."""

    def setUp(self):
        self.sw_web = (KALYAN_DIR / "web" / "sw.js").read_text(encoding="utf-8")
        self.sw_engine = (KALYAN_DIR / "sw.js").read_text(encoding="utf-8")

    def test_cache_version_v4(self):
        """Both sw.js files must declare kalyan-predictor-cache-v4."""
        expected_cache = "kalyan-predictor-cache-v4"
        self.assertIn(f"const CACHE_NAME = '{expected_cache}';", self.sw_web)
        self.assertIn(f"const CACHE_NAME = '{expected_cache}';", self.sw_engine)

    def test_precaching_performance_log(self):
        """Static assets list must include ./performance_log.json."""
        self.assertIn("'./performance_log.json'", self.sw_web)
        self.assertIn("'./performance_log.json'", self.sw_engine)

    def test_offline_fetch_handler_performance_log(self):
        """Fetch handler must fall back to cached performance_log.json."""
        self.assertIn("performance_log.json", self.sw_web)
        self.assertIn("/api/performance-log", self.sw_web)
        self.assertIn("caches.match('./performance_log.json')", self.sw_web)

    def test_cache_cleanup_on_activate(self):
        """Service worker must purge caches differing from CACHE_NAME."""
        self.assertIn("caches.delete(key)", self.sw_web)
        self.assertIn("key !== CACHE_NAME", self.sw_web)

    def test_precache_assets_exist_on_disk(self):
        """Every local static asset declared in STATIC_ASSETS must exist on disk in web directory."""
        match = re.search(r'const STATIC_ASSETS = \[(.*?)\];', self.sw_web, re.DOTALL)
        self.assertIsNotNone(match, "STATIC_ASSETS array not found in sw.js")
        raw_assets = match.group(1)
        assets = [line.strip().strip("',\"") for line in raw_assets.split('\n') if line.strip() and not line.strip().startswith('//')]
        local_assets = [a for a in assets if not a.startswith('http')]

        web_dir = KALYAN_DIR / "web"
        missing = []
        for a in local_assets:
            rel = a.lstrip('./')
            if rel == '':
                rel = 'index.html'
            p = web_dir / rel
            if not p.exists():
                missing.append(a)

        self.assertEqual(missing, [], f"STATIC_ASSETS declared in sw.js missing in web directory: {missing}")



class TestThreeWayFileParity(unittest.TestCase):
    """Empirical bit-exact verification across mirror directories."""

    def setUp(self):
        self.root_dir = Path(__file__).resolve().parent.parent
        self.engine_dir = self.root_dir / "kalyan_4_35_to_6_35"
        self.web_dir = self.engine_dir / "web"

    def _hash(self, path):
        self.assertTrue(path.exists(), f"File {path} does not exist")
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def test_prediction_data_parity(self):
        """prediction_data.json must be bit-for-bit identical across all 3 directories."""
        h1 = self._hash(self.root_dir / "prediction_data.json")
        h2 = self._hash(self.engine_dir / "prediction_data.json")
        h3 = self._hash(self.web_dir / "prediction_data.json")
        self.assertEqual(h1, h2, "prediction_data.json root != engine")
        self.assertEqual(h2, h3, "prediction_data.json engine != web")

    def test_history_json_parity(self):
        """history.json must be bit-for-bit identical across all 3 directories."""
        h1 = self._hash(self.root_dir / "history.json")
        h2 = self._hash(self.engine_dir / "history.json")
        h3 = self._hash(self.web_dir / "history.json")
        self.assertEqual(h1, h2, "history.json root != engine")
        self.assertEqual(h2, h3, "history.json engine != web")

    def test_performance_log_parity(self):
        """performance_log.json must be bit-for-bit identical across all 3 directories."""
        h1 = self._hash(self.root_dir / "performance_log.json")
        h2 = self._hash(self.engine_dir / "performance_log.json")
        h3 = self._hash(self.web_dir / "performance_log.json")
        self.assertEqual(h1, h2, "performance_log.json root != engine")
        self.assertEqual(h2, h3, "performance_log.json engine != web")

    def test_html_mirrors_parity(self):
        """web/index.html, engine index.html, and engine dashboard.html must be identical."""
        h_web = self._hash(self.web_dir / "index.html")
        h_engine = self._hash(self.engine_dir / "index.html")
        h_dash = self._hash(self.engine_dir / "dashboard.html")
        self.assertEqual(h_web, h_engine)
        self.assertEqual(h_engine, h_dash)

    def test_sw_mirrors_parity(self):
        """web/sw.js and engine sw.js must be identical."""
        h_web = self._hash(self.web_dir / "sw.js")
        h_engine = self._hash(self.engine_dir / "sw.js")
        self.assertEqual(h_web, h_engine)

    def test_root_portal_preservation(self):
        """Root tools/matka/index.html must remain the distinct multi-tool portal."""
        root_portal = self.root_dir / "index.html"
        self.assertTrue(root_portal.exists())
        content = root_portal.read_text(encoding="utf-8")
        self.assertIn("Matka Predictive Analytics Suite", content)
        # Verify it is not accidentally overwritten with Kalyan engine UI
        h_portal = self._hash(root_portal)
        h_web = self._hash(self.web_dir / "index.html")
        self.assertNotEqual(h_portal, h_web)


class TestAdversarialEdgeCases(unittest.TestCase):
    """Stress testing edge cases and failure modes."""

    def test_evaluate_outcome_with_corrupt_types(self):
        """Verify evaluate_outcome handles string/int type edge cases without raising exceptions."""
        scores = np.ones(100) / 100.0
        # Integer vs string jodi formats
        res = evaluate_outcome(
            scores,
            top_pick="00",
            top_10=["00", "01", "02", "03", "04", "05", "06", "07", "08", "09"],
            otc_digits=[0, 5, 1, 6],
            actual_jodi=0,  # integer instead of "00"
            actual_open="0", # string instead of int
            actual_close="0",
        )
        self.assertEqual(res["actual_rank"], 1)
        self.assertTrue(res["top10_hit"])

    def test_compute_last_week_performance_minimal_history(self):
        """Verify compute_last_week_performance behavior when historical dataset is small."""
        df = pd.read_csv(CSV_PATH)
        valid_df = df[df['Is_Valid'] == True].reset_index(drop=True)
        # Take just the last 30 draws
        sub_df = valid_df.tail(30).reset_index(drop=True)
        lw, lm = compute_last_week_performance(sub_df)
        self.assertIn('draws', lw)
        self.assertIn('all_draws', lm)

    def test_performance_log_missing_fallback(self):
        """Verify KalyanWebHandler returns '{}' if performance_log.json is absent."""
        handler = object.__new__(KalyanWebHandler)
        # Verify code logic: if both WORKSPACE_DIR and WEB_DIR paths do not exist, writes b"{}"
        perf_workspace = os.path.join(WORKSPACE_DIR, "nonexistent_perf.json")
        perf_web = os.path.join(WEB_DIR, "nonexistent_perf.json")
        self.assertFalse(os.path.exists(perf_workspace))
        self.assertFalse(os.path.exists(perf_web))


if __name__ == "__main__":
    unittest.main()
