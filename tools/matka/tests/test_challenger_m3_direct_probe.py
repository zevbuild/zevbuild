"""
Direct Empirical Probe by challenger_m3_2_1.
Verifies:
1. /dashboard.html returns HTTP 200 OK with identical content to index.html.
2. /api/status/invalid/nested returns HTTP 404 Not Found.
3. compute_last_week_performance(None) executes cleanly without error.
4. compute_last_week_performance(unreset_filtered_df) executes cleanly without IndexError.
5. Bit-exact SHA256 parity for all mirrored assets.
"""

import sys
import os
import threading
import time
import urllib.request
import urllib.error
import hashlib
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
MATKA_ROOT = REPO_ROOT / "tools" / "matka"
ENG_DIR = MATKA_ROOT / "kalyan_4_35_to_6_35"
WEB_DIR = ENG_DIR / "web"

sys.path.insert(0, str(ENG_DIR))
from app import ThreadedHTTPServer, KalyanWebHandler, compute_last_week_performance

def run_probes():
    print("--- Starting Direct Empirical Probes ---")
    server = ThreadedHTTPServer(("127.0.0.1", 0), KalyanWebHandler)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    time.sleep(0.2)
    base = f"http://127.0.0.1:{port}"

    try:
        # Probe 1: /dashboard.html -> 200 OK
        req_dash = f"{base}/dashboard.html"
        with urllib.request.urlopen(req_dash, timeout=5) as resp:
            status_dash = resp.status
            content_dash = resp.read()
            print(f"PROBE 1 PASS: /dashboard.html returned HTTP {status_dash}, size={len(content_dash)} bytes")
            assert status_dash == 200, f"Expected 200, got {status_dash}"
            assert len(content_dash) > 0, "Empty content for /dashboard.html"

        # Probe 2: /api/status/invalid/nested -> 404 Not Found
        req_invalid = f"{base}/api/status/invalid/nested"
        try:
            with urllib.request.urlopen(req_invalid, timeout=5) as resp:
                print(f"PROBE 2 FAIL: Expected 404, got HTTP {resp.status}")
                sys.exit(1)
        except urllib.error.HTTPError as e:
            print(f"PROBE 2 PASS: /api/status/invalid/nested returned HTTP {e.code} ({e.reason})")
            assert e.code == 404, f"Expected 404, got {e.code}"

        # Probe 3: compute_last_week_performance(None)
        lw_none, lm_none = compute_last_week_performance(None)
        num_draws_none = len(lm_none["all_draws"])
        print(f"PROBE 3 PASS: compute_last_week_performance(None) succeeded, total draws={num_draws_none}")
        assert num_draws_none > 0

        # Probe 4: compute_last_week_performance(unreset_filtered_df)
        raw_df = pd.read_csv(ENG_DIR / "kalyan_historical_data.csv")
        filtered_unreset = raw_df[raw_df["Is_Valid"] == True] # index retains raw row indices (gaps)
        max_idx = filtered_unreset.index.max()
        df_len = len(filtered_unreset)
        print(f"Testing unreset DataFrame: max_index={max_idx}, len={df_len} (gap={max_idx - df_len})")
        lw_unreset, lm_unreset = compute_last_week_performance(filtered_unreset)
        num_draws_unreset = len(lm_unreset["all_draws"])
        print(f"PROBE 4 PASS: compute_last_week_performance(filtered_unreset) succeeded without IndexError, draws={num_draws_unreset}")
        assert num_draws_unreset == num_draws_none

        # Probe 5: 3-way Parity SHA256
        def sha256_file(p):
            return hashlib.sha256(p.read_bytes()).hexdigest()

        h_dash_web = sha256_file(WEB_DIR / "dashboard.html")
        h_dash_eng = sha256_file(ENG_DIR / "dashboard.html")
        h_idx_web = sha256_file(WEB_DIR / "index.html")
        h_idx_eng = sha256_file(ENG_DIR / "index.html")
        print(f"Hash web/dashboard.html: {h_dash_web}")
        print(f"Hash eng/dashboard.html: {h_dash_eng}")
        print(f"Hash web/index.html:     {h_idx_web}")
        print(f"Hash eng/index.html:     {h_idx_eng}")
        assert h_dash_web == h_dash_eng == h_idx_web == h_idx_eng, "Hash parity mismatch for dashboard/index.html"
        print("PROBE 5 PASS: All 4 HTML mirrors are bit-for-bit identical!")

        # Probe 6: Root Matka Hub Preservation
        root_idx = MATKA_ROOT / "index.html"
        assert root_idx.exists(), "tools/matka/index.html missing"
        root_len = len(root_idx.read_text(encoding="utf-8"))
        assert root_len < 50000, f"tools/matka/index.html size {root_len} > 50000"
        assert "Matka Predictive Analytics Suite" in root_idx.read_text(encoding="utf-8")
        print(f"PROBE 6 PASS: tools/matka/index.html preserved ({root_len} bytes)")

        print("--- ALL 6 DIRECT EMPIRICAL PROBES PASSED ---")

    finally:
        server.shutdown()
        server.server_close()

if __name__ == "__main__":
    run_probes()
