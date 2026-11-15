"""
Kalyan Historical Data Pipeline Orchestrator.
Coordinates scraping, data hygiene, dataset export, statistical EDA,
predictive modeling, walk-forward backtesting, and the Web UI server.
"""

import os
import sys
import argparse
from pathlib import Path
import pandas as pd
from scraper import fetch_html, parse_kalyan_chart, get_preview_table, export_to_csv
from eda import analyze_frequencies, print_eda_report
from predict import generate_upcoming_prediction, print_prediction_report
from backtest import run_walk_forward_backtest, print_backtest_report

SCRIPT_DIR = Path(__file__).resolve().parent


def resolve_csv_path(custom_path: str = None) -> str:
    """Dynamically resolve kalyan_historical_data.csv across CLI and parent directories."""
    if custom_path is not None and str(custom_path).strip() != "":
        if os.path.exists(custom_path):
            return os.path.abspath(custom_path)
        cand = SCRIPT_DIR / custom_path
        if cand.exists():
            return str(cand.resolve())
        raise FileNotFoundError(f"Specified CSV dataset not found: {custom_path}")

    candidates = [
        SCRIPT_DIR / "kalyan_historical_data.csv",
        SCRIPT_DIR.parent / "kalyan_historical_data.csv",
        SCRIPT_DIR.parent.parent.parent / "tools" / "matka" / "kalyan_historical_data.csv",
        SCRIPT_DIR.parent.parent.parent / "tools" / "matka" / "kalyan_4_35_to_6_35" / "kalyan_historical_data.csv",
    ]
    for cand in candidates:
        if cand.exists():
            return str(cand.resolve())
    return str((SCRIPT_DIR / "kalyan_historical_data.csv").resolve())


def run_pipeline(
    preview_only: bool = False,
    export_only: bool = False,
    eda_only: bool = False,
    predict_only: bool = False,
    backtest_only: bool = False,
    web_server: bool = False,
    port: int = 8080,
    output_csv: str = None,
):
    print("=" * 75)
    print("      KALYAN HISTORICAL DATA SCRAPING & AUTOMATION PIPELINE")
    print("=" * 75)

    output_csv = resolve_csv_path(output_csv)

    if web_server:
        from app import start_server
        start_server(port=port, open_browser=True)
        return

    if eda_only:
        if not os.path.exists(output_csv):
            print(f"[ERROR] Cannot run EDA: '{output_csv}' does not exist.")
            sys.exit(1)
        df = pd.read_csv(output_csv, dtype={"Jodi": str, "Raw_Entry": str})
        results = analyze_frequencies(df)
        print_eda_report(results)
        return

    if predict_only:
        if not os.path.exists(output_csv):
            print(f"[ERROR] Cannot run Prediction: '{output_csv}' does not exist.")
            sys.exit(1)
        res = generate_upcoming_prediction(output_csv)
        print_prediction_report(res)
        return

    if backtest_only:
        if not os.path.exists(output_csv):
            print(f"[ERROR] Cannot run Backtest: '{output_csv}' does not exist.")
            sys.exit(1)
        res = run_walk_forward_backtest(output_csv)
        print_backtest_report(res)
        return

    # Default full pipeline
    # Step 1: Scrape & Parse
    print("\n[STEP 1/4] Fetching and parsing historical chart...")
    html = fetch_html()
    df = parse_kalyan_chart(html)

    total_records = len(df)
    valid_records = int(df["Is_Valid"].sum())
    invalid_records = total_records - valid_records

    print(f" -> Successfully extracted {total_records} historical calendar records.")
    print(f" -> Valid Jodi Draws (00-99): {valid_records} ({valid_records/total_records*100:.1f}%)")
    print(f" -> Market Holidays / Placeholders: {invalid_records} ({invalid_records/total_records*100:.1f}%)")

    # Step 2: Preview
    print("\n[STEP 2/4] Previewing first 20 chronological rows:")
    print(get_preview_table(df, 20))

    if preview_only:
        print("\n[INFO] Preview mode complete. No files were written to disk.")
        return

    # Step 3: Export to CSV
    print(f"\n[STEP 3/4] Exporting dataset to '{output_csv}'...")
    export_to_csv(df, output_csv)

    if not export_only:
        # Step 4: Run EDA
        results = analyze_frequencies(df)
        print_eda_report(results)

        # Step 5: Run Upcoming Prediction
        pred_res = generate_upcoming_prediction(output_csv)
        print_prediction_report(pred_res)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kalyan Matka Data Pipeline, EDA, Prediction & Web Engine")
    parser.add_argument("--preview", action="store_true", help="Preview first 20 rows without writing to disk")
    parser.add_argument("--export", action="store_true", help="Scrape and export CSV only")
    parser.add_argument("--eda", action="store_true", help="Run frequency EDA only from existing CSV")
    parser.add_argument("--predict", action="store_true", help="Generate upcoming prediction forecast")
    parser.add_argument("--backtest", action="store_true", help="Run historical walk-forward backtest")
    parser.add_argument("--web", action="store_true", help="Launch real-time browser Web application")
    parser.add_argument("--port", type=int, default=8080, help="Port for web server (default: 8080)")
    parser.add_argument("--output", default=None, help="Output CSV file path")

    args = parser.parse_args()
    run_pipeline(
        preview_only=args.preview,
        export_only=args.export,
        eda_only=args.eda,
        predict_only=args.predict,
        backtest_only=args.backtest,
        web_server=args.web,
        port=args.port,
        output_csv=args.output,
    )
