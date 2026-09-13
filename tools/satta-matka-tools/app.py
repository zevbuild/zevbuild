"""
Real-Time Kalyan Prediction Web Application Server.
Provides automated HTTP server, real-time scraping API, live prediction engine,
and serves the web browser UI.
"""

import os
import sys
import json
import webbrowser
import argparse
from http.server import HTTPServer, SimpleHTTPRequestHandler
from socketserver import ThreadingMixIn
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Ensure workspace is in path
WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
if WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, WORKSPACE_DIR)

from scraper import fetch_html, parse_kalyan_chart, export_to_csv
from models import EnsemblePredictor
from predict import get_family_jodis

WEB_DIR = os.path.join(WORKSPACE_DIR, "web")
CSV_PATH = os.path.join(WORKSPACE_DIR, "kalyan_historical_data.csv")


def compute_all_predictions():
    """Execute live scraping/cache read, retrain models, and generate predictions."""
    # 1. Fetch & parse
    html = fetch_html()
    df = parse_kalyan_chart(html)
    export_to_csv(df, CSV_PATH)

    valid_df = df[df["Is_Valid"] == True].copy().reset_index(drop=True)
    valid_df["Open_Digit"] = valid_df["Open_Digit"].astype(int)
    valid_df["Close_Digit"] = valid_df["Close_Digit"].astype(int)

    latest_row = valid_df.iloc[-1]
    latest_date_str = latest_row["Date"]
    latest_date = datetime.strptime(latest_date_str, "%Y-%m-%d")
    latest_jodi = str(latest_row["Jodi"]).zfill(2)
    latest_open = int(latest_row["Open_Digit"])
    latest_close = int(latest_row["Close_Digit"])

    # Fit Ensemble Model
    ensemble = EnsemblePredictor(w_markov=0.30, w_recency=0.40, w_seasonal=0.30)
    ensemble.fit(valid_df)

    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
    predictions_by_day = {}

    cur_date = latest_date + timedelta(days=1)
    if cur_date.weekday() == 6:
        cur_date += timedelta(days=1)

    for offset in range(6):
        d_date = cur_date + timedelta(days=offset)
        if d_date.weekday() == 6:
            d_date += timedelta(days=1)
        day_name = days[offset % 6]

        preds = ensemble.predict(
            day_of_week=day_name,
            prev_jodi=latest_jodi,
            prev_open=latest_open,
            prev_close=latest_close,
        )

        jodi_probs = preds["jodi_probs"]
        open_probs = preds["open_probs"]
        close_probs = preds["close_probs"]

        sorted_j = np.argsort(jodi_probs)[::-1]
        sorted_o = np.argsort(open_probs)[::-1]
        sorted_c = np.argsort(close_probs)[::-1]

        top_jodis = []
        for rank, idx in enumerate(sorted_j[:10], start=1):
            j_str = f"{idx:02d}"
            p = round(float(jodi_probs[idx]) * 100.0, 2)
            top_jodis.append({
                "rank": rank,
                "jodi": j_str,
                "prob": p,
                "family": get_family_jodis(j_str),
            })

        top_open = []
        for rank, idx in enumerate(sorted_o, start=1):
            top_open.append({
                "rank": rank,
                "digit": int(idx),
                "prob": round(float(open_probs[idx]) * 100.0, 2),
            })

        top_close = []
        for rank, idx in enumerate(sorted_c, start=1):
            top_close.append({
                "rank": rank,
                "digit": int(idx),
                "prob": round(float(close_probs[idx]) * 100.0, 2),
            })

        predictions_by_day[day_name] = {
            "date": d_date.strftime("%Y-%m-%d"),
            "day": day_name,
            "top_jodis": top_jodis,
            "open_digits": top_open,
            "close_digits": top_close,
            "top_pick": top_jodis[0]["jodi"],
            "top_family": top_jodis[0]["family"],
        }

    return {
        "status": "success",
        "sync_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_records": len(df),
        "valid_records": len(valid_df),
        "latest_draw": {
            "date": latest_date_str,
            "day": latest_row["Day_Of_Week"],
            "jodi": latest_jodi,
            "open": latest_open,
            "close": latest_close,
        },
        "by_day": predictions_by_day,
    }


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in separate threads for non-blocking browser interactions."""
    daemon_threads = True


class KalyanWebHandler(SimpleHTTPRequestHandler):
    """Custom HTTP handler serving web assets and JSON REST APIs."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=WEB_DIR, **kwargs)

    def _set_cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(200)
        self._set_cors()
        self.end_headers()

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            return super().do_GET()
        elif self.path.startswith("/api/status"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._set_cors()
            self.end_headers()
            resp = {"status": "online", "time": datetime.now().isoformat()}
            self.wfile.write(json.dumps(resp).encode("utf-8"))
        elif self.path.startswith("/api/fetch-and-predict"):
            self.handle_fetch_and_predict()
        else:
            return super().do_GET()

    def do_POST(self):
        if self.path.startswith("/api/fetch-and-predict"):
            self.handle_fetch_and_predict()
        else:
            self.send_response(404)
            self.end_headers()

    def handle_fetch_and_predict(self):
        try:
            data = compute_all_predictions()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._set_cors()
            self.end_headers()
            self.wfile.write(json.dumps(data).encode("utf-8"))
        except Exception as e:
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self._set_cors()
            self.end_headers()
            self.wfile.write(json.dumps({"status": "error", "message": str(e)}).encode("utf-8"))


def start_server(port: int = 8080, open_browser: bool = True):
    server_address = ("127.0.0.1", port)
    httpd = ThreadedHTTPServer(server_address, KalyanWebHandler)
    url = f"http://127.0.0.1:{port}"
    print("=" * 75)
    print(f"  🚀 KALYAN AUTO-PREDICTOR WEB APPLICATION RUNNING")
    print("=" * 75)
    print(f"  Local Web UI URL:  {url}")
    print(f"  API Endpoint:      {url}/api/fetch-and-predict")
    print(f"  Auto-Sync Status:  ACTIVE (Realtime web browser interface)")
    print("-" * 75)
    print("  Press Ctrl+C to stop the server.")

    if open_browser:
        webbrowser.open(url)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[INFO] Shutting down web server...")
        httpd.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start Kalyan Web Predictor Server")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind (default: 8080)")
    parser.add_argument("--no-browser", action="store_true", help="Do not open browser automatically")
    args = parser.parse_args()

    start_server(port=args.port, open_browser=not args.no_browser)
