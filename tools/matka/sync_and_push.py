"""
Kalyan Matka Automated Predictor, Data Updater & Web Push Script.
Fetches latest chart data, retrains ensemble models, updates prediction files,
and pushes changes to GitHub to trigger automatic Cloudflare Pages deployment.
"""

import os
import sys
import subprocess
import argparse
from datetime import datetime

# Add matka directory to python path
MATKA_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(MATKA_DIR, "..", ".."))

if MATKA_DIR not in sys.path:
    sys.path.insert(0, MATKA_DIR)

from app import compute_all_predictions


def run_git_cmd(args, cwd=REPO_ROOT):
    """Run a git command and return stdout string or raise exception."""
    res = subprocess.run(["git"] + args, cwd=cwd, capture_output=True, text=True, check=True)
    return res.stdout.strip()


def sync_and_push(push: bool = True, custom_msg: str = None):
    print("=" * 65)
    print("  KALYAN MATKA: LIVE PREDICTION SYNC & WEB DEPLOYMENT")
    print("=" * 65)

    print("\n[STEP 1/3] Scraping latest results & recalibrating models...")
    try:
        payload = compute_all_predictions(force_refresh=True)
    except Exception as e:
        print(f"[ERROR] Failed during prediction computation: {e}")
        sys.exit(1)

    latest_draw = payload.get("latest_draw", {})
    latest_date = latest_draw.get("date", datetime.now().strftime("%Y-%m-%d"))
    latest_jodi = latest_draw.get("jodi", "--")
    dev_update = payload.get("last_update_by_developer", datetime.now().strftime("%d %b %Y"))
    print(f" -> Last Developer Update : {dev_update}")
    print(f" -> Latest Recorded Draw : {latest_draw.get('day')} {latest_date} (Jodi: {latest_jodi})")
    print(f" -> Total Historical Draws: {payload.get('valid_records', 0)}")
    print(f" -> Files Updated:")
    print(f"    - tools/matka/kalyan_historical_data.csv")
    print(f"    - tools/matka/kalyan_penal_chart.html")
    print(f"    - tools/matka/prediction_data.json & web/prediction_data.json")
    print(f"    - tools/matka/history.json & web/history.json")

    if not push:
        print("\n[INFO] --no-push requested. Local files updated successfully without Git push.")
        return

    print("\n[STEP 2/3] Checking Git status and staging Matka changes...")
    try:
        # Stage matka tool updates
        subprocess.run(["git", "add", "tools/matka/"], cwd=REPO_ROOT, check=True)

        # Check if there are staged changes
        diff_res = subprocess.run(["git", "diff", "--staged", "--name-only"], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
        staged_files = [f for f in diff_res.stdout.splitlines() if f.strip()]

        if not staged_files:
            print(" -> [NOTICE] No new draw changes detected in Matka tools. Git workspace is already clean.")
            return

        print(f" -> Staged {len(staged_files)} modified file(s):")
        for f in staged_files:
            print(f"    * {f}")

        # Create commit
        commit_msg = custom_msg or f"Update Kalyan Matka predictions & historical chart ({latest_date})"
        print(f"\n[STEP 3/3] Committing and pushing to origin main...")
        subprocess.run(["git", "commit", "-m", commit_msg], cwd=REPO_ROOT, check=True)
        print(f" -> Committed: {commit_msg}")

        # Push to remote
        push_res = subprocess.run(["git", "push", "origin", "main"], cwd=REPO_ROOT, capture_output=True, text=True)
        if push_res.returncode != 0:
            print(f"[WARN] Git push failed or remote was rejected: {push_res.stderr}")
            print("You can manually run: git push origin main")
            return

        print(" -> [SUCCESS] Changes pushed to GitHub origin/main!")
        print(" -> Cloudflare Pages deployment triggered automatically.")
        print(f" -> Live Web URL: https://zevbuild.pages.dev/tools/matka/")

    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Git operation failed: {e}")
        if hasattr(e, "stderr") and e.stderr:
            print(e.stderr)
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sync Kalyan predictions and automatically push to web")
    parser.add_argument("--no-push", action="store_true", help="Update local files only, do not commit or push to Git")
    parser.add_argument("-m", "--message", type=str, default=None, help="Custom Git commit message")
    args = parser.parse_args()

    sync_and_push(push=not args.no_push, custom_msg=args.message)
