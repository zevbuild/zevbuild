"""
Milestone 3 Iteration 2 Challenger Stress & Route Isolation Harness.
Author: challenger_m3_2_2 (Empirical Challenger)
Role: critic, specialist

Rigorous stress validation verifying:
1. compute_last_week_performance robustness against arbitrary subsets, non-contiguous indices,
   duplicate indices, string indices, sliced dataframes, None inputs, and edge boundary conditions.
2. KalyanWebHandler exact route isolation (/api/status, /api/performance-log, /api/fetch-and-predict,
   /dashboard.html, /index.html, /), rejecting nested subpaths with 404, query string survival,
   POST route restrictions, CORS options, and high-concurrency request storming.
"""

import json
import os
import sys
import threading
import time
import unittest
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from http.client import HTTPConnection
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import pandas as pd
import numpy as np

# Path setup
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
MATKA_ROOT = REPO_ROOT / "tools" / "matka"
MATKA_ENGINE = MATKA_ROOT / "kalyan_4_35_to_6_35"
WEB_DIR = MATKA_ENGINE / "web"

for p in [str(MATKA_ENGINE), str(MATKA_ROOT), str(REPO_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from app import KalyanWebHandler, ThreadedHTTPServer, compute_last_week_performance, CSV_PATH


class TestComputeLastWeekPerformanceStress(unittest.TestCase):
    """Stress tests verifying compute_last_week_performance index safety and slicing robustness."""

    @classmethod
    def setUpClass(cls):
        cls.raw_df = pd.read_csv(CSV_PATH)

    def test_default_none_input(self):
        """compute_last_week_performance(None) automatically loads CSV_PATH without error."""
        last_week, perf_data = compute_last_week_performance(None)
        self.assertIsInstance(last_week, dict)
        self.assertIsInstance(perf_data, dict)
        self.assertIn("all_draws", perf_data)
        self.assertGreater(len(perf_data["all_draws"]), 0)

    def test_unreset_filtered_boolean_mask(self):
        """compute_last_week_performance on df[df['Is_Valid'] == True] with unreset index."""
        # Unreset index has missing gap indices (e.g. index 0..3291 with ~3193 rows)
        filtered = self.raw_df[self.raw_df["Is_Valid"] == True]
        self.assertNotEqual(len(filtered), filtered.index[-1] + 1)  # Gaps exist
        last_week, perf_data = compute_last_week_performance(filtered)
        self.assertIsInstance(perf_data, dict)
        self.assertGreater(len(perf_data["all_draws"]), 0)

    def test_non_contiguous_large_gap_indices(self):
        """Index with huge arbitrary non-contiguous numbers (e.g. 10000, 20000...)."""
        filtered = self.raw_df[self.raw_df["Is_Valid"] == True].copy()
        filtered.index = [i * 1000 for i in range(len(filtered))]
        last_week, perf_data = compute_last_week_performance(filtered)
        self.assertIsInstance(perf_data, dict)
        self.assertGreater(len(perf_data["all_draws"]), 0)

    def test_negative_indices(self):
        """Index with negative integers."""
        filtered = self.raw_df[self.raw_df["Is_Valid"] == True].copy()
        filtered.index = [-i for i in range(len(filtered))]
        last_week, perf_data = compute_last_week_performance(filtered)
        self.assertIsInstance(perf_data, dict)
        self.assertGreater(len(perf_data["all_draws"]), 0)

    def test_string_labels_as_index(self):
        """Index with string date labels."""
        filtered = self.raw_df[self.raw_df["Is_Valid"] == True].copy()
        filtered.index = [f"row_{d}_{i}" for i, d in enumerate(filtered["Date"])]
        last_week, perf_data = compute_last_week_performance(filtered)
        self.assertIsInstance(perf_data, dict)
        self.assertGreater(len(perf_data["all_draws"]), 0)

    def test_duplicate_indices(self):
        """Index containing duplicate integer values."""
        filtered = self.raw_df[self.raw_df["Is_Valid"] == True].copy()
        filtered.index = [i // 2 for i in range(len(filtered))]
        last_week, perf_data = compute_last_week_performance(filtered)
        self.assertIsInstance(perf_data, dict)
        self.assertGreater(len(perf_data["all_draws"]), 0)

    def test_tail_slice(self):
        """Passing only the last 50 draws."""
        filtered = self.raw_df[self.raw_df["Is_Valid"] == True].tail(50)
        last_week, perf_data = compute_last_week_performance(filtered)
        self.assertIsInstance(perf_data, dict)
        self.assertIn("all_draws", perf_data)

    def test_middle_slice(self):
        """Passing an arbitrary middle slice of 100 draws."""
        filtered = self.raw_df[self.raw_df["Is_Valid"] == True].iloc[500:600]
        last_week, perf_data = compute_last_week_performance(filtered)
        self.assertIsInstance(perf_data, dict)
        self.assertIn("all_draws", perf_data)

    def test_missing_is_valid_column(self):
        """Valid DataFrame slice missing 'Is_Valid' column."""
        valid_df = self.raw_df[self.raw_df["Is_Valid"] == True]
        filtered = valid_df.drop(columns=["Is_Valid"]).tail(100)
        last_week, perf_data = compute_last_week_performance(filtered)
        self.assertIsInstance(perf_data, dict)
        self.assertIn("all_draws", perf_data)

    def test_all_draws_invariants(self):
        """Check all_draws invariants on standard run."""
        _, perf_data = compute_last_week_performance(None)
        draws = perf_data["all_draws"]
        for d in draws:
            self.assertTrue(1 <= d["actual_rank"] <= 100)
            self.assertEqual(d["top10_hit"], d["actual_rank"] <= 10)
            self.assertEqual(len(d["otc_digits"]), 4)
            self.assertEqual(len(set(d["otc_digits"])), 4)
            # Harmonic cut check
            for x in d["otc_digits"]:
                self.assertIn((x + 5) % 10, d["otc_digits"])


class TestKalyanWebHandlerRouteIsolation(unittest.TestCase):
    """Stress tests verifying KalyanWebHandler URL routing, isolation, and concurrency."""

    @classmethod
    def setUpClass(cls):
        cls.server = ThreadedHTTPServer(("127.0.0.1", 0), KalyanWebHandler)
        cls.port = cls.server.server_address[1]
        cls.server_thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.server_thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.port}"
        time.sleep(0.1)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def test_status_exact_and_nested_404(self):
        """GET /api/status returns 200, but nested subpaths return 404."""
        # Exact route
        with urlopen(f"{self.base_url}/api/status", timeout=5) as res:
            self.assertEqual(res.status, 200)

        # Query param returns 200
        with urlopen(f"{self.base_url}/api/status?refresh=1&timestamp=12345", timeout=5) as res:
            self.assertEqual(res.status, 200)

        # Nested invalid subpaths MUST return 404
        for subpath in ["/api/status/", "/api/status/nested", "/api/status/invalid/nested", "/api/status/123"]:
            with self.subTest(subpath=subpath):
                with self.assertRaises(HTTPError) as ctx:
                    urlopen(f"{self.base_url}{subpath}", timeout=5)
                self.assertEqual(ctx.exception.code, 404)

    def test_performance_log_exact_and_nested_404(self):
        """GET /api/performance-log returns 200, nested subpaths return 404."""
        # Exact route
        with urlopen(f"{self.base_url}/api/performance-log", timeout=5) as res:
            self.assertEqual(res.status, 200)

        # Query param returns 200
        with urlopen(f"{self.base_url}/api/performance-log?page=1&limit=50", timeout=5) as res:
            self.assertEqual(res.status, 200)

        # Nested invalid subpaths MUST return 404
        for subpath in ["/api/performance-log/", "/api/performance-log/v1", "/api/performance-log/all/deep"]:
            with self.subTest(subpath=subpath):
                with self.assertRaises(HTTPError) as ctx:
                    urlopen(f"{self.base_url}{subpath}", timeout=5)
                self.assertEqual(ctx.exception.code, 404)

    def test_dashboard_and_index_serving(self):
        """GET /dashboard.html, /index.html, / all serve 200 OK."""
        for path in ["/", "/index.html", "/dashboard.html", "/dashboard.html?v=4", "/index.html?t=1"]:
            with self.subTest(path=path):
                with urlopen(f"{self.base_url}{path}", timeout=5) as res:
                    self.assertEqual(res.status, 200)
                    body = res.read()
                    self.assertGreater(len(body), 50000)

    def test_dashboard_nested_subpath_404(self):
        """GET /dashboard.html/nested must return 404."""
        with self.assertRaises(HTTPError) as ctx:
            urlopen(f"{self.base_url}/dashboard.html/nested", timeout=5)
        self.assertEqual(ctx.exception.code, 404)

    def test_post_disallowed_on_readonly_routes(self):
        """POST to /api/status, /api/performance-log, /index.html returns 404."""
        for path in ["/api/status", "/api/performance-log", "/index.html", "/dashboard.html"]:
            with self.subTest(path=path):
                conn = HTTPConnection("127.0.0.1", self.port)
                conn.request("POST", path, body=b"{}")
                res = conn.getresponse()
                self.assertEqual(res.status, 404)
                conn.close()

    def test_options_cors_headers(self):
        """OPTIONS request to KalyanWebHandler returns CORS headers."""
        conn = HTTPConnection("127.0.0.1", self.port)
        conn.request("OPTIONS", "/api/performance-log")
        res = conn.getresponse()
        self.assertEqual(res.status, 200)
        self.assertEqual(res.getheader("Access-Control-Allow-Origin"), "*")
        self.assertIn("GET", res.getheader("Access-Control-Allow-Methods"))
        self.assertIn("POST", res.getheader("Access-Control-Allow-Methods"))
        conn.close()

    def test_heavy_concurrent_request_storm(self):
        """Stress server with 60 concurrent requests across mixed valid and 404 endpoints."""
        endpoints = [
            ("/api/status", 200),
            ("/api/status?query=stress", 200),
            ("/api/performance-log", 200),
            ("/dashboard.html", 200),
            ("/api/status/invalid/nested", 404),
            ("/api/performance-log/nonexistent", 404),
        ]
        tasks = endpoints * 10  # 60 total concurrent requests

        def fire_req(item):
            url_path, expected_status = item
            req = Request(f"{self.base_url}{url_path}")
            try:
                with urlopen(req, timeout=10) as res:
                    return (res.status == expected_status)
            except HTTPError as e:
                return (e.code == expected_status)

        with ThreadPoolExecutor(max_workers=12) as pool:
            results = list(pool.map(fire_req, tasks))

        self.assertEqual(len(results), 60)
        self.assertTrue(all(results), f"Some requests failed: {results.count(False)} failures")


if __name__ == "__main__":
    unittest.main()
