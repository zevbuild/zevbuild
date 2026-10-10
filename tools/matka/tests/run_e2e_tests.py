#!/usr/bin/env python3
"""
Zevbuild Kalyan Matka Comprehensive Opaque-Box E2E Test Suite Runner
===================================================================
Executes 4-Tier requirement-driven opaque-box tests covering:
- Tier 1: Feature Coverage (CLI, Predictions, CSV, Web Assets, Links, Schemas, PWA)
- Tier 2: Boundary & Corner Cases (Edge Dates, Holidays, Patti Combinations, Flags, Fallbacks, Leap Years)
- Tier 3: Cross-Feature Combinations (Scraper -> CSV -> ML -> Schema -> Mirroring -> Link Audit)
- Tier 4: Real-World Workload Scenarios (Daily Prediction Cycle, Sunday Recess, Offline PWA, Backtest Audit)

Usage:
------
    python tools/matka/tests/run_e2e_tests.py [options]

Options:
--------
    --tier <1,2,3,4>    Run specific test tier(s), comma-separated (default: all)
    --verbose, -v       Detailed per-test output
    --report            Save machine-readable audit report to test_audit_results.json
    --help, -h          Show this help message

Exit Codes:
-----------
    0: All executed tests passed cleanly (or clearly marked pending future milestones).
    1: One or more test assertions failed or encountered unhandled errors.
"""

import argparse
import io
import json
import os
import sys
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Base repository paths
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
TESTS_DIR = Path(__file__).resolve().parent
ENGINE_DIR = REPO_ROOT / "tools" / "matka" / "kalyan_4_35_to_6_35"

# Add directories to python sys.path
for p in [str(REPO_ROOT), str(TESTS_DIR), str(ENGINE_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ANSI terminal formatting
USE_COLOR = sys.stdout.isatty() and os.name != "nt" or "TERM" in os.environ


def color(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if USE_COLOR else text


def green(t: str) -> str: return color(t, "32;1")
def red(t: str) -> str: return color(t, "31;1")
def yellow(t: str) -> str: return color(t, "33;1")
def cyan(t: str) -> str: return color(t, "36;1")
def bold(t: str) -> str: return color(t, "1")
def dim(t: str) -> str: return color(t, "2")


class DetailedTestResult(unittest.TestResult):
    """Tracks test outcomes with duration, tier categorization, and skip reasons."""

    def __init__(self, verbosity: int = 1):
        super().__init__()
        self.verbosity = verbosity
        self.test_records: List[Dict[str, any]] = []
        self._start_time: float = 0.0

    def startTest(self, test: unittest.TestCase):
        super().startTest(test)
        self._start_time = time.perf_counter()
        if self.verbosity >= 2:
            test_id = test.id().split(".")[-1]
            cls_name = test.__class__.__name__
            sys.stdout.write(f"  -> {cls_name}.{test_id} ... ")
            sys.stdout.flush()

    def addSuccess(self, test: unittest.TestCase):
        super().addSuccess(test)
        elapsed = time.perf_counter() - self._start_time
        self.test_records.append({
            "name": test.id(),
            "status": "PASS",
            "duration": elapsed,
            "message": "",
        })
        if self.verbosity >= 2:
            sys.stdout.write(f"{green('PASS')} ({elapsed:.3f}s)\n")

    def addSkip(self, test: unittest.TestCase, reason: str):
        super().addSkip(test, reason)
        elapsed = time.perf_counter() - self._start_time
        self.test_records.append({
            "name": test.id(),
            "status": "SKIP",
            "duration": elapsed,
            "message": reason,
        })
        if self.verbosity >= 2:
            sys.stdout.write(f"{yellow('SKIP')} ({reason})\n")

    def addFailure(self, test: unittest.TestCase, err: Tuple):
        super().addFailure(test, err)
        elapsed = time.perf_counter() - self._start_time
        msg = self._exc_info_to_string(err, test)
        self.test_records.append({
            "name": test.id(),
            "status": "FAIL",
            "duration": elapsed,
            "message": msg,
        })
        if self.verbosity >= 2:
            sys.stdout.write(f"{red('FAIL')}\n")

    def addError(self, test: unittest.TestCase, err: Tuple):
        super().addError(test, err)
        elapsed = time.perf_counter() - self._start_time
        msg = self._exc_info_to_string(err, test)
        self.test_records.append({
            "name": test.id(),
            "status": "ERROR",
            "duration": elapsed,
            "message": msg,
        })
        if self.verbosity >= 2:
            sys.stdout.write(f"{red('ERROR')}\n")


def load_tier_suite(tier_num: int) -> unittest.TestSuite:
    """Dynamically loads tests for specified tier number."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    tier_files = {
        1: "test_tier1_feature_coverage.py",
        2: "test_tier2_boundary_corner.py",
        3: "test_tier3_cross_feature.py",
        4: "test_tier4_real_world.py",
    }

    filename = tier_files.get(tier_num)
    if not filename:
        raise ValueError(f"Invalid tier number: {tier_num}")

    module_path = TESTS_DIR / filename
    if not module_path.exists():
        raise FileNotFoundError(f"Tier test file not found: {module_path}")

    # Load from module
    mod_name = filename[:-3]
    try:
        if mod_name in sys.modules:
            del sys.modules[mod_name]
        module = __import__(mod_name)
        tier_suite = loader.loadTestsFromModule(module)
        suite.addTests(tier_suite)
    except Exception as e:
        print(f"[ERROR] Failed to load {filename}: {e}")
        raise

    return suite


def run_e2e_suite(tiers: List[int], verbosity: int = 1, save_report: bool = False) -> int:
    """Executes the test suite across selected tiers and renders summary report."""
    print("=" * 80)
    print(bold("       ZEVBUILD KALYAN MATKA E2E TEST RUNNER — 4-TIER AUDIT"))
    print("=" * 80)
    print(f"Repository Root:   {REPO_ROOT}")
    print(f"Test Suite Dir:    {TESTS_DIR}")
    print(f"Active Tiers:      {', '.join(f'Tier {t}' for t in tiers)}")
    print(f"Timestamp:         {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')}")
    print("-" * 80)

    overall_start = time.perf_counter()
    tier_results = {}
    all_records = []
    total_passed = 0
    total_skipped = 0
    total_failed = 0
    total_errors = 0

    tier_titles = {
        1: "Tier 1: Feature Coverage (CLI, Predictor, CSV, Web, Links, Schemas, PWA)",
        2: "Tier 2: Boundary & Corner Cases (Dates, Holidays, Pattis, Flags, Fallbacks)",
        3: "Tier 3: Cross-Feature Combinations (Scraper -> CSV -> ML -> Schemas -> Sync)",
        4: "Tier 4: Real-World Workload Scenarios (Daily Cycle, Sunday Recess, Backtest)",
    }

    for tier_num in sorted(tiers):
        print(f"\n[{bold(f'TIER {tier_num}')}] {tier_titles[tier_num]}")
        suite = load_tier_suite(tier_num)
        runner_result = DetailedTestResult(verbosity=verbosity)

        t_start = time.perf_counter()
        suite.run(runner_result)
        t_elapsed = time.perf_counter() - t_start

        tier_records = runner_result.test_records
        all_records.extend(tier_records)

        p = len([r for r in tier_records if r["status"] == "PASS"])
        s = len([r for r in tier_records if r["status"] == "SKIP"])
        f = len([r for r in tier_records if r["status"] == "FAIL"])
        e = len([r for r in tier_records if r["status"] == "ERROR"])

        total_passed += p
        total_skipped += s
        total_failed += f
        total_errors += e

        tier_status = "PASS" if (f == 0 and e == 0) else "FAIL"
        tier_results[tier_num] = {
            "title": tier_titles[tier_num],
            "status": tier_status,
            "passed": p,
            "skipped": s,
            "failed": f,
            "errors": e,
            "elapsed": t_elapsed,
        }

        status_str = green("[PASS]") if tier_status == "PASS" else red("[FAIL]")
        print(
            f"  Result: {status_str}  Total: {len(tier_records)} | "
            f"Pass: {green(str(p))} | Skip: {yellow(str(s))} | Fail: {red(str(f))} | Error: {red(str(e))} "
            f"({t_elapsed:.2f}s)"
        )

    overall_elapsed = time.perf_counter() - overall_start

    # Render Skipped Tests summary (Milestone pending inventory)
    skipped_records = [r for r in all_records if r["status"] == "SKIP"]
    if skipped_records:
        print("\n" + "-" * 80)
        print(bold("PENDING / SKIPPED TEST INVENTORY (Awaiting Future Milestones):"))
        print("-" * 80)
        for r in skipped_records:
            t_name = r["name"].split(".")[-1]
            print(f"  • {cyan(t_name)}: {yellow(r['message'])}")

    # Render Failures summary if any
    failed_records = [r for r in all_records if r["status"] in ("FAIL", "ERROR")]
    if failed_records:
        print("\n" + "=" * 80)
        print(bold(red("TEST FAILURES & ERRORS:")))
        print("=" * 80)
        for r in failed_records:
            print(f"\n{red('FAIL')}: {bold(r['name'])}")
            print(r["message"])

    # Final Summary Table
    print("\n" + "=" * 80)
    print(bold("                          TEST EXECUTION SUMMARY"))
    print("=" * 80)
    print(f"{'Tier':<10} | {'Status':<8} | {'Passed':<8} | {'Skipped':<8} | {'Failed':<8} | {'Time (s)':<8}")
    print("-" * 80)
    for t_num, res in tier_results.items():
        st_color = green("PASS") if res["status"] == "PASS" else red("FAIL")
        print(
            f"Tier {t_num:<5} | {st_color:<17} | {res['passed']:<8} | {res['skipped']:<8} | {res['failed']:<8} | {res['elapsed']:<8.2f}"
        )
    print("-" * 80)
    total_count = total_passed + total_skipped + total_failed + total_errors
    final_status = "PASSED" if (total_failed == 0 and total_errors == 0) else "FAILED"
    final_color = green if final_status == "PASSED" else red
    print(
        f"TOTAL: {total_count} tests across {len(tiers)} tier(s). "
        f"Passed: {green(str(total_passed))}, Skipped: {yellow(str(total_skipped))}, Failed: {red(str(total_failed + total_errors))}."
    )
    print(f"OVERALL STATUS: {final_color(final_status)} in {overall_elapsed:.2f}s")
    print("=" * 80)

    # Save report if requested
    if save_report:
        report_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "repository_root": str(REPO_ROOT),
            "tiers": tier_results,
            "total": total_count,
            "passed": total_passed,
            "skipped": total_skipped,
            "failed": total_failed + total_errors,
            "overall_status": final_status,
            "elapsed_seconds": overall_elapsed,
            "tests": all_records,
        }
        report_path = REPO_ROOT / "test_audit_results.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)
        print(f"\n[+] Audit results saved to: {report_path}")

    return 0 if (total_failed == 0 and total_errors == 0) else 1


def main():
    parser = argparse.ArgumentParser(
        description="Zevbuild Kalyan Matka Comprehensive Opaque-Box E2E Test Suite Runner"
    )
    parser.add_argument(
        "--tier",
        type=str,
        default="1,2,3,4",
        help="Comma-separated tier numbers to execute (e.g. '1,2,3,4' or '1,3')",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose test-level output",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Save test audit results JSON file",
    )

    args = parser.parse_args()

    try:
        tier_list = [int(t.strip()) for t in args.tier.split(",") if t.strip()]
    except ValueError:
        print("[ERROR] --tier must be integers between 1 and 4 (e.g. --tier 1,2,3,4)")
        sys.exit(1)

    for t in tier_list:
        if t not in (1, 2, 3, 4):
            print(f"[ERROR] Tier {t} is invalid. Supported tiers are 1, 2, 3, 4.")
            sys.exit(1)

    verbosity_level = 2 if args.verbose else 1
    exit_code = run_e2e_suite(
        tiers=tier_list,
        verbosity=verbosity_level,
        save_report=args.report,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
