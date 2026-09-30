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

from bs4 import BeautifulSoup
from scraper import fetch_html, parse_kalyan_chart, export_to_csv
from models import EnsemblePredictor
from predict import get_family_jodis

WEB_DIR = os.path.join(WORKSPACE_DIR, "web")
CSV_PATH = os.path.join(WORKSPACE_DIR, "kalyan_historical_data.csv")


def compute_all_predictions(force_refresh: bool = False):
    """Execute live scraping/cache read, retrain models, and generate predictions."""
    # 1. Fetch & parse
    html = fetch_html(force_refresh=force_refresh)
    df = parse_kalyan_chart(html)
    export_to_csv(df, CSV_PATH)

    valid_df = df[df["Is_Valid"] == True].copy().reset_index(drop=True)
    valid_df["Open_Digit"] = valid_df["Open_Digit"].astype(int)
    valid_df["Close_Digit"] = valid_df["Close_Digit"].astype(int)

    latest_row = valid_df.iloc[-1]
    latest_date_str = str(latest_row["Date"])
    latest_date = datetime.strptime(latest_date_str, "%Y-%m-%d")
    latest_jodi = str(latest_row["Jodi"]).zfill(2)
    latest_open = int(latest_row["Open_Digit"])
    latest_close = int(latest_row["Close_Digit"])

    # Extract latest pattis if available
    open_patti = "---"
    close_patti = "---"
    try:
        soup = BeautifulSoup(html, "html.parser")
        table = soup.find("table", class_=lambda c: c and "chart-table" in c) or soup.find("table")
        if table:
            rows = table.find_all("tr")
            for row in reversed(rows):
                tds = row.find_all("td")
                if len(tds) >= 4:
                    num_days = (len(tds) - 1) // 3
                    if num_days > 0:
                        last_col = 1 + (num_days - 1) * 3
                        open_patti = tds[last_col].get_text(strip=True)
                        if last_col + 2 < len(tds):
                            close_patti = tds[last_col + 2].get_text(strip=True)
                    break
    except Exception:
        pass

    # Fit Ensemble Model
    ensemble = EnsemblePredictor(w_markov=0.30, w_recency=0.40, w_seasonal=0.30)
    ensemble.fit(valid_df)

    day_abbrs = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat"}
    predictions_by_day = {}

    d_date = latest_date
    draw_days = []
    while len(draw_days) < 6:
        d_date += timedelta(days=1)
        if d_date.weekday() != 6:  # Skip Sunday
            draw_days.append(d_date)

    for target_d in draw_days:
        day_name = day_abbrs[target_d.weekday()]
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
            "date": target_d.strftime("%Y-%m-%d"),
            "day": day_name,
            "top_jodis": top_jodis,
            "open_digits": top_open,
            "close_digits": top_close,
            "top_pick": top_jodis[0]["jodi"],
            "top_family": top_jodis[0]["family"],
        }

    output_payload = {
        "status": "success",
        "sync_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_records": len(df),
        "valid_records": len(valid_df),
        "latest_draw": {
            "date": latest_date_str,
            "day": str(latest_row["Day_Of_Week"]),
            "jodi": latest_jodi,
            "open": latest_open,
            "close": latest_close,
            "openPatti": open_patti,
            "closePatti": close_patti,
        },
        "draw": {
            "date": latest_date_str,
            "day": str(latest_row["Day_Of_Week"]),
            "jodi": latest_jodi,
            "open_digit": latest_open,
            "close_digit": latest_close,
            "open_panna": open_patti,
            "close_panna": close_patti,
            "status": "FULL_JODI_DECLARED",
        },
        "raw_result": f"{open_patti}-{latest_jodi}-{close_patti}",
        "by_day": predictions_by_day,
    }

    # Persist updated prediction_data.json
    pred_data_to_save = {
        "latest_draw": output_payload["latest_draw"],
        "by_day": predictions_by_day,
    }
    pred_path_root = os.path.join(WORKSPACE_DIR, "prediction_data.json")
    pred_path_web = os.path.join(WEB_DIR, "prediction_data.json")
    with open(pred_path_root, "w", encoding="utf-8") as f:
        json.dump(pred_data_to_save, f, indent=2)
    with open(pred_path_web, "w", encoding="utf-8") as f:
        json.dump(pred_data_to_save, f, indent=2)

    # Persist updated history.json for client
    history_list = []
    for _, row in valid_df.iterrows():
        history_list.append([
            str(row["Date"]),
            str(row["Day_Of_Week"]),
            int(row["Jodi"]),
            int(row["Open_Digit"]),
            int(row["Close_Digit"])
        ])
    hist_path_root = os.path.join(WORKSPACE_DIR, "history.json")
    hist_path_web = os.path.join(WEB_DIR, "history.json")
    with open(hist_path_root, "w", encoding="utf-8") as f:
        json.dump(history_list, f)
    with open(hist_path_web, "w", encoding="utf-8") as f:
        json.dump(history_list, f)

    return output_payload


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
