"""
Milestone 3 Challenger Stress & Robustness Test Suite.
Author: challenger_m3_2 (Empirical Challenger)
Role: critic, specialist

Focus areas:
1. Handler Stress & Robustness (KalyanWebHandler, /api/status, /api/performance-log,
   malformed parameters/headers, route fuzzing, 404 verification, concurrency).
2. Walk-forward Output Integrity (draw_obj schema, NaN/null checks, actual_rank vs top10_hit invariant,
   otc_digits complementary pairs, family_bracket invariants).
3. Offline PWA Manifest & Service Worker Validation (syntax with Node.js, precache asset resolution,
   cache v4, manifest icon disk existence).
4. Parity & File Preservation (tools/matka/index.html preservation, 3-way JSON hash parity).
"""

import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from http.client import HTTPConnection
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import numpy as np

# Path setup
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
MATKA_ROOT = REPO_ROOT / "tools" / "matka"
MATKA_ENGINE = MATKA_ROOT / "kalyan_4_35_to_6_35"
WEB_DIR = MATKA_ENGINE / "web"

for p in [str(MATKA_ENGINE), str(MATKA_ROOT), str(REPO_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from app import KalyanWebHandler, ThreadedHTTPServer, compute_last_week_performance


class TestHandlerStressAndRobustness(unittest.TestCase):
    """Adversarial stress and concurrency verification of KalyanWebHandler."""

    @classmethod
    def setUpClass(cls):
        # Bind to a free ephemeral port
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

    def test_api_status_success(self):
        """GET /api/status returns 200 with online status, draw_count, and last_draw_date."""
        req = Request(f"{self.base_url}/api/status")
        with urlopen(req, timeout=5) as res:
            self.assertEqual(res.status, 200)
            self.assertEqual(res.headers.get("Content-Type"), "application/json")
            self.assertEqual(res.headers.get("Access-Control-Allow-Origin"), "*")
            data = json.loads(res.read().decode("utf-8"))
            self.assertEqual(data.get("status"), "online")
            self.assertIsInstance(data.get("draw_count"), int)
            self.assertGreater(data.get("draw_count"), 0)
            self.assertTrue(len(data.get("last_draw_date")) >= 10)

    def test_api_status_malformed_query_params(self):
        """GET /api/status with adversarial/malformed query strings does not crash (200 OK)."""
        malformed_queries = [
            "?%ZZ%invalid",
            "?foo=bar&baz=1&empty=&===&&&",
            "?" + "a" * 8192,
            "?param=" + "%20" * 500,
            "?unicode=" + urlencode({"q": " कल्याण मटका 123 !@#$"}),
        ]
        for q in malformed_queries:
            with self.subTest(query=q[:30]):
                req = Request(f"{self.base_url}/api/status{q}")
                with urlopen(req, timeout=5) as res:
                    self.assertEqual(res.status, 200)
                    data = json.loads(res.read().decode("utf-8"))
                    self.assertEqual(data.get("status"), "online")

    def test_api_performance_log_success(self):
        """GET /api/performance-log returns 200 with valid JSON performance log schema."""
        req = Request(f"{self.base_url}/api/performance-log")
        with urlopen(req, timeout=5) as res:
            self.assertEqual(res.status, 200)
            self.assertEqual(res.headers.get("Content-Type"), "application/json")
            data = json.loads(res.read().decode("utf-8"))
            self.assertIn("summary", data)
            self.assertIn("current_hyperparameters", data)
            self.assertIn("draw_logs", data)

    def test_api_performance_log_malformed_queries(self):
        """GET /api/performance-log with malformed parameters does not crash (200 OK)."""
        req = Request(f"{self.base_url}/api/performance-log?%XXbad_query=1")
        with urlopen(req, timeout=5) as res:
            self.assertEqual(res.status, 200)
            data = json.loads(res.read().decode("utf-8"))
            self.assertIn("summary", data)

    def test_cors_preflight_options(self):
        """OPTIONS request to KalyanWebHandler returns 200 with CORS headers."""
        conn = HTTPConnection("127.0.0.1", self.port)
        conn.request("OPTIONS", "/api/status")
        res = conn.getresponse()
        self.assertEqual(res.status, 200)
        self.assertEqual(res.getheader("Access-Control-Allow-Origin"), "*")
        self.assertIn("GET", res.getheader("Access-Control-Allow-Methods"))
        conn.close()

    def test_unknown_routes_return_404(self):
        """GET requests to completely unknown endpoints must return 404."""
        unknown_endpoints = [
            "/api/nonexistent",
            "/api/unknown_route_123",
            "/secret_admin_panel",
            "/static/missing_bundle.js",
        ]
        for ep in unknown_endpoints:
            with self.subTest(endpoint=ep):
                with self.assertRaises(HTTPError) as ctx:
                    urlopen(f"{self.base_url}{ep}", timeout=5)
                self.assertEqual(ctx.exception.code, 404)

    def test_subpath_routing_behavior(self):
        """Test whether KalyanWebHandler isolates exact route or permits arbitrary subpaths due to .startswith()."""
        # Document actual behavior for forensic reporting
        req = Request(f"{self.base_url}/api/status/invalid/nested")
        try:
            with urlopen(req, timeout=5) as res:
                # If startswith("/api/status") was used, it returns 200 rather than 404
                self.handler_leaks_subpaths = (res.status == 200)
        except HTTPError as e:
            self.handler_leaks_subpaths = False

    def test_invalid_http_methods_on_routes(self):
        """POST to read-only endpoints like /api/status returns 404 per KalyanWebHandler."""
        conn = HTTPConnection("127.0.0.1", self.port)
        conn.request("POST", "/api/status", body=b"{}")
        res = conn.getresponse()
        self.assertEqual(res.status, 404)
        conn.close()

    def test_concurrent_request_stress(self):
        """Spam server with 40 concurrent requests across endpoints and ensure 100% success."""
        endpoints = [
            "/api/status",
            "/api/performance-log",
            "/prediction_data.json",
            "/history.json",
            "/manifest.json",
        ]
        tasks = endpoints * 8  # 40 total requests

        def make_request(ep):
            req = Request(f"{self.base_url}{ep}")
            with urlopen(req, timeout=10) as res:
                return res.status

        with ThreadPoolExecutor(max_workers=10) as pool:
            results = list(pool.map(make_request, tasks))

        self.assertEqual(len(results), 40)
        self.assertTrue(all(status == 200 for status in results))

    def test_sw_static_assets_all_serve_200(self):
        """Verify that EVERY asset precached by sw.js actually returns 200 OK from KalyanWebHandler."""
        sw_file = WEB_DIR / "sw.js"
        self.assertTrue(sw_file.exists())
        content = sw_file.read_text(encoding="utf-8")

        # Extract STATIC_ASSETS array
        match = re.search(r"const\s+STATIC_ASSETS\s*=\s*\[(.*?)\];", content, re.DOTALL)
        self.assertIsNotNone(match, "STATIC_ASSETS array not found in sw.js")
        raw_items = match.group(1)
        assets = [
            item.strip().strip("'\"")
            for item in raw_items.split(",")
            if item.strip() and not item.strip().startswith("//")
        ]

        missing_assets = []
        for asset in assets:
            if asset.startswith("http://") or asset.startswith("https://"):
                continue  # External CDN
            clean_path = asset.lstrip(".")
            if not clean_path.startswith("/"):
                clean_path = "/" + clean_path

            req_url = f"{self.base_url}{clean_path}"
            try:
                with urlopen(req_url, timeout=5) as res:
                    if res.status != 200:
                        missing_assets.append((asset, res.status))
            except HTTPError as e:
                missing_assets.append((asset, e.code))
            except Exception as e:
                missing_assets.append((asset, str(e)))

        self.assertEqual(
            missing_assets,
            [],
            f"Precached assets in sw.js failed to serve 200 from KalyanWebHandler: {missing_assets}",
        )


class TestWalkForwardOutputIntegrity(unittest.TestCase):
    """Adversarial validation of compute_last_week_performance() output schema & invariants."""

    @classmethod
    def setUpClass(cls):
        import pandas as pd
        csv_p = MATKA_ENGINE / "kalyan_historical_data.csv"
        df = pd.read_csv(csv_p)
        valid_df = df[df["Is_Valid"] == True].copy().reset_index(drop=True)
        valid_df["Open_Digit"] = valid_df["Open_Digit"].apply(lambda x: int(float(x)))
        valid_df["Close_Digit"] = valid_df["Close_Digit"].apply(lambda x: int(float(x)))
        valid_df["Jodi"] = valid_df["Jodi"].apply(lambda x: f"{int(float(x)):02d}")
        cls.last_week, cls.perf_output = compute_last_week_performance(valid_df)

    def test_return_structure(self):
        """Verify compute_last_week_performance() returns valid weeks and all_draws."""
        self.assertIn("weeks", self.perf_output)
        self.assertIn("all_draws", self.perf_output)
        self.assertIsInstance(self.perf_output["weeks"], list)
        self.assertIsInstance(self.perf_output["all_draws"], list)
        self.assertGreater(len(self.perf_output["all_draws"]), 0)

    def test_draw_obj_invariants_and_no_nulls(self):
        """Every draw_obj must have no NaN/nulls and uphold mathematical and hit-flag invariants."""
        all_draws = self.perf_output["all_draws"]
        required_fields = [
            "date",
            "day",
            "actual_jodi",
            "actual_open",
            "actual_close",
            "predicted_top_pick",
            "predicted_top_5",
            "predicted_top_open",
            "predicted_top_close",
            "family_bracket",
            "otc_digits",
            "otc_hit",
            "open_hit",
            "close_hit",
            "jodi_hit",
            "status_label",
            "badge_type",
            "actual_rank",
            "top10_hit",
            "family_cut_hit",
            "rank_pill",
            "touched",
        ]

        for i, d in enumerate(all_draws):
            date_label = d.get("date", f"idx_{i}")
            for field in required_fields:
                self.assertIn(
                    field,
                    d,
                    f"Missing field '{field}' in draw_obj for {date_label}",
                )
                val = d[field]
                self.assertIsNotNone(
                    val,
                    f"Field '{field}' is None in draw_obj for {date_label}",
                )
                if isinstance(val, float):
                    import math
                    self.assertFalse(
                        math.isnan(val),
                        f"Field '{field}' is NaN in draw_obj for {date_label}",
                    )

            # Invariant: actual_rank in 1..100
            actual_rank = d["actual_rank"]
            self.assertIsInstance(actual_rank, int)
            self.assertTrue(
                1 <= actual_rank <= 100,
                f"actual_rank {actual_rank} out of bounds (1..100) for {date_label}",
            )

            # Invariant: whenever actual_rank <= 10, top10_hit MUST BE True
            top10_hit = d["top10_hit"]
            self.assertIsInstance(top10_hit, (bool, np.bool_))
            if actual_rank <= 10:
                self.assertTrue(
                    top10_hit,
                    f"actual_rank={actual_rank} <= 10 but top10_hit is False for {date_label}!",
                )
            else:
                self.assertFalse(
                    top10_hit,
                    f"actual_rank={actual_rank} > 10 but top10_hit is True for {date_label}!",
                )

            # Invariant: rank_pill string matches actual_rank
            self.assertEqual(
                d["rank_pill"],
                f"Rank #{actual_rank}",
                f"rank_pill mismatch for {date_label}",
            )

            # Invariant: otc_digits has 4 unique digits
            otc = d["otc_digits"]
            self.assertEqual(
                len(otc),
                4,
                f"otc_digits length != 4 for {date_label}",
            )
            self.assertEqual(
                len(set(otc)),
                4,
                f"otc_digits contains duplicates for {date_label}",
            )

            # Invariant: otc_digits consists of two complementary harmonic cut pairs (d, (d+5)%10)
            cuts = [(x + 5) % 10 for x in otc]
            # Each digit's cut must be in otc
            for digit in otc:
                self.assertIn(
                    (digit + 5) % 10,
                    otc,
                    f"otc_digits {otc} missing harmonic cut for {digit} on {date_label}",
                )

            # Invariant: otc_hit matches actual_open or actual_close in otc_digits
            expected_otc_hit = (d["actual_open"] in otc) or (d["actual_close"] in otc)
            self.assertEqual(
                bool(d["otc_hit"]),
                expected_otc_hit,
                f"otc_hit {d['otc_hit']} != expected {expected_otc_hit} for {date_label}",
            )

            # Invariant: family_bracket has 4 or 8 jodis (4 for doubles/half-red, 8 for standard) and contains predicted_top_pick
            fam = d["family_bracket"]
            self.assertIn(
                len(fam),
                (4, 8),
                f"family_bracket length {len(fam)} not in (4, 8) for {date_label}",
            )
            self.assertIn(
                d["predicted_top_pick"],
                fam,
                f"predicted_top_pick {d['predicted_top_pick']} not in its own family_bracket for {date_label}",
            )


class TestOfflinePWAManifestAndSW(unittest.TestCase):
    """Adversarial validation of PWA manifest.json and sw.js offline resilience."""

    def test_manifest_json_validity(self):
        """manifest.json must be valid JSON, display: standalone, and icons must exist on disk."""
        manifest_p = WEB_DIR / "manifest.json"
        self.assertTrue(manifest_p.exists(), "web/manifest.json does not exist")
        data = json.loads(manifest_p.read_text(encoding="utf-8"))

        self.assertEqual(
            data.get("display"),
            "standalone",
            "PWA manifest display must be 'standalone'",
        )
        self.assertIn("start_url", data)
        self.assertIn("icons", data)
        self.assertIsInstance(data["icons"], list)
        self.assertGreater(len(data["icons"]), 0)

        for icon in data["icons"]:
            src = icon.get("src")
            self.assertIsNotNone(src, "Icon missing 'src'")
            clean_src = src.lstrip("./")
            icon_disk_path = WEB_DIR / clean_src
            self.assertTrue(
                icon_disk_path.exists(),
                f"Manifest icon '{src}' does not exist on disk at {icon_disk_path}",
            )
            self.assertGreater(
                icon_disk_path.stat().st_size,
                0,
                f"Manifest icon '{src}' on disk is empty (0 bytes)",
            )

    def test_sw_js_syntax_via_node(self):
        """sw.js syntax must compile cleanly via Node.js --check."""
        for sw_path in [WEB_DIR / "sw.js", MATKA_ENGINE / "sw.js"]:
            self.assertTrue(sw_path.exists(), f"{sw_path} does not exist")
            res = subprocess.run(
                ["node", "--check", str(sw_path)],
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                res.returncode,
                0,
                f"Node.js syntax check failed for {sw_path}:\n{res.stderr}",
            )

    def test_sw_js_cache_v4_and_offline_routes(self):
        """sw.js must use kalyan-predictor-cache-v4 and handle /api/performance-log offline."""
        content = (WEB_DIR / "sw.js").read_text(encoding="utf-8")
        self.assertIn(
            "kalyan-predictor-cache-v4",
            content,
            "sw.js must define cache name 'kalyan-predictor-cache-v4'",
        )
        self.assertIn(
            "performance_log.json",
            content,
            "sw.js must precache performance_log.json",
        )
        self.assertIn(
            "/api/performance-log",
            content,
            "sw.js offline fallback must route /api/performance-log to performance_log.json",
        )

    def test_sw_precache_assets_disk_existence(self):
        """Every local asset in sw.js STATIC_ASSETS must exist on disk in web/ directory."""
        content = (WEB_DIR / "sw.js").read_text(encoding="utf-8")
        match = re.search(r"const\s+STATIC_ASSETS\s*=\s*\[(.*?)\];", content, re.DOTALL)
        self.assertIsNotNone(match, "STATIC_ASSETS array missing in sw.js")
        raw_items = match.group(1)
        assets = [
            item.strip().strip("'\"")
            for item in raw_items.split(",")
            if item.strip() and not item.strip().startswith("//")
        ]

        missing_on_disk = []
        for asset in assets:
            if asset.startswith("http://") or asset.startswith("https://"):
                continue
            if asset in (".", "./"):
                disk_file = WEB_DIR / "index.html"
            else:
                clean_name = asset.lstrip("./")
                disk_file = WEB_DIR / clean_name
            if not disk_file.exists():
                missing_on_disk.append((asset, str(disk_file)))

        self.assertEqual(
            missing_on_disk,
            [],
            f"Precached assets declared in sw.js missing on disk in {WEB_DIR}: {missing_on_disk}",
        )


class TestParityAndFilePreservation(unittest.TestCase):
    """Adversarial verification of file preservation, multi-tool integrity, and 3-way mirrors."""

    def test_root_matka_index_html_preservation(self):
        """tools/matka/index.html must NOT be overwritten by the engine index.html."""
        root_index = MATKA_ROOT / "index.html"
        engine_index = WEB_DIR / "index.html"

        self.assertTrue(root_index.exists(), "tools/matka/index.html missing")
        self.assertTrue(engine_index.exists(), "engine index.html missing")

        root_content = root_index.read_text(encoding="utf-8")
        engine_content = engine_index.read_text(encoding="utf-8")

        # Byte size comparison: root portal is ~21KB, engine is ~260KB+
        self.assertNotEqual(
            len(root_content),
            len(engine_content),
            "tools/matka/index.html has identical length to engine index.html!",
        )
        self.assertLess(
            len(root_content),
            50000,
            "tools/matka/index.html appears overwritten (size > 50KB)",
        )

        # Title check: root portal is Matka Tools Hub, NOT Kalyan Auto-Predictor
        self.assertNotIn(
            "Kalyan 4:35 PM to 6:35 PM Quantitative Predictor",
            root_content,
            "tools/matka/index.html has engine title - indicates overwrite corruption!",
        )
        self.assertIn(
            "Matka",
            root_content,
            "tools/matka/index.html missing expected portal content",
        )

    def test_three_way_json_mirror_parity(self):
        """prediction_data.json, history.json, and performance_log.json must have 100% SHA256 parity."""
        mirrors = [
            MATKA_ROOT,
            MATKA_ENGINE,
            WEB_DIR,
        ]
        json_files = ["prediction_data.json", "history.json", "performance_log.json"]

        for jf in json_files:
            hashes = {}
            for d in mirrors:
                fpath = d / jf
                self.assertTrue(
                    fpath.exists(),
                    f"File {jf} does not exist in mirror directory {d}",
                )
                content = fpath.read_bytes()
                self.assertGreater(len(content), 0, f"{fpath} is empty")
                # Ensure valid JSON
                parsed = json.loads(content.decode("utf-8"))
                h = hashlib.sha256(content).hexdigest()
                hashes[str(d)] = h

            unique_hashes = set(hashes.values())
            self.assertEqual(
                len(unique_hashes),
                1,
                f"SHA256 mismatch across mirrors for {jf}: {hashes}",
            )

    def test_engine_and_web_html_and_sw_parity(self):
        """index.html and sw.js in MATKA_ENGINE and WEB_DIR must have identical SHA256 hashes."""
        for fname in ["index.html", "sw.js"]:
            p1 = MATKA_ENGINE / fname
            p2 = WEB_DIR / fname
            self.assertTrue(p1.exists(), f"{p1} missing")
            self.assertTrue(p2.exists(), f"{p2} missing")
            h1 = hashlib.sha256(p1.read_bytes()).hexdigest()
            h2 = hashlib.sha256(p2.read_bytes()).hexdigest()
            self.assertEqual(
                h1,
                h2,
                f"Hash mismatch between {p1} and {p2}",
            )


if __name__ == "__main__":
    unittest.main()
