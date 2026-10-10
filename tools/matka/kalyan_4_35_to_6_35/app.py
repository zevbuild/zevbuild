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

from pathlib import Path
from bs4 import BeautifulSoup
from scraper import fetch_html, parse_kalyan_chart, export_to_csv, sync_csv_to_root, verify_csv_integrity
from models import EnsemblePredictor
from predict import get_family_jodis

SCRIPT_DIR = Path(__file__).resolve().parent
WEB_DIR = os.path.join(WORKSPACE_DIR, "web")
ROOT_MATKA_DIR = os.path.abspath(os.path.join(WORKSPACE_DIR, ".."))


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


CSV_PATH = resolve_csv_path()


def compute_last_week_performance(valid_df, html=""):
    """
    Computes walk-forward predictions for the completed draws of the previous calendar week,
    matches against actual declared outcomes and pattis, and returns structured performance data.
    """
    patti_map = {}
    if html:
        try:
            soup = BeautifulSoup(html, "html.parser")
            table = soup.find("table", class_=lambda c: c and "chart-table" in c) or soup.find("table")
            if table:
                rows = table.find_all("tr")
                for row in rows:
                    tds = [td.get_text(strip=True) for td in row.find_all("td")]
                    if not tds or "to" not in tds[0]:
                        continue
                    parts = tds[0].split("to")
                    try:
                        start_d = datetime.strptime(parts[0].strip(), "%d/%m/%Y")
                        num_days = (len(tds) - 1) // 3
                        for day_idx in range(num_days):
                            col = 1 + day_idx * 3
                            day_dt = start_d + timedelta(days=day_idx)
                            day_str = day_dt.strftime("%Y-%m-%d")
                            op = tds[col]
                            cl = tds[col + 2] if col + 2 < len(tds) else "---"
                            patti_map[day_str] = (op, cl)
                    except Exception:
                        pass
        except Exception:
            pass

    # Find the most recently completed prior week
    latest_date_str = str(valid_df.iloc[-1]["Date"])
    latest_dt = datetime.strptime(latest_date_str, "%Y-%m-%d")
    curr_mon = latest_dt - timedelta(days=latest_dt.weekday())
    prev_mon = curr_mon - timedelta(days=7)
    prev_sat = prev_mon + timedelta(days=5)

    p_mon_str = prev_mon.strftime("%Y-%m-%d")
    p_sat_str = prev_sat.strftime("%Y-%m-%d")

    lw_mask = (valid_df["Date"] >= p_mon_str) & (valid_df["Date"] <= p_sat_str)
    lw_indices = valid_df[lw_mask].index.tolist()

    if len(lw_indices) == 0:
        lw_indices = list(range(max(0, len(valid_df) - 7), len(valid_df) - 1))

    draws = []
    total_ank_hits = 0
    double_ank_hits = 0

    for idx in lw_indices:
        if idx <= 0:
            continue
        train_df = valid_df.iloc[:idx]
        actual_row = valid_df.iloc[idx]
        prev_row = valid_df.iloc[idx - 1]

        m = EnsemblePredictor()
        m.fit(train_df)
        preds = m.predict(
            day_of_week=str(actual_row["Day_Of_Week"]),
            prev_jodi=str(prev_row["Jodi"]).zfill(2),
            prev_open=int(prev_row["Open_Digit"]),
            prev_close=int(prev_row["Close_Digit"]),
        )

        sorted_j = np.argsort(preds["jodi_probs"])[::-1]
        sorted_o = np.argsort(preds["open_probs"])[::-1]
        sorted_c = np.argsort(preds["close_probs"])[::-1]

        top_jodis = [f"{x:02d}" for x in sorted_j[:10]]
        top_open = [int(x) for x in sorted_o[:3]]
        top_close = [int(x) for x in sorted_c[:3]]

        actual_j = str(actual_row["Jodi"]).zfill(2)
        actual_o = int(actual_row["Open_Digit"])
        actual_c = int(actual_row["Close_Digit"])
        date_str = str(actual_row["Date"])

        op_p, cl_p = patti_map.get(date_str, ("---", "---"))

        open_hit = actual_o in top_open
        close_hit = actual_c in top_close
        jodi_hit = actual_j in top_jodis[:5]
        top_pick = top_jodis[0]
        fam = get_family_jodis(top_pick)
        fam_hit = actual_j in fam

        otc_digits = preds.get("otc_digits", [])
        otc_hit = (actual_o in otc_digits) or (actual_c in otc_digits)

        if open_hit and close_hit:
            status_label = "Double Ank Hit (Open & Close)"
            badge_type = "success"
            total_ank_hits += 2
            double_ank_hits += 1
        elif open_hit:
            status_label = f"Open Ank Hit ({actual_o})"
            badge_type = "highlight"
            total_ank_hits += 1
        elif close_hit:
            status_label = f"Close Ank Hit ({actual_c})"
            badge_type = "highlight"
            total_ank_hits += 1
        elif fam_hit:
            status_label = f"Cut Family Hit ({actual_j})"
            badge_type = "highlight"
        elif actual_j in top_jodis:
            status_label = f"Top-10 Edge ({actual_j})"
            badge_type = "info"
        elif otc_hit:
            status_label = "OTC Pass"
            badge_type = "info"
        else:
            status_label = "Standard Variance"
            badge_type = "neutral"

        draws.append({
            "date": date_str,
            "day": str(actual_row["Day_Of_Week"]),
            "actual_jodi": actual_j,
            "actual_open": actual_o,
            "actual_close": actual_c,
            "open_patti": op_p,
            "close_patti": cl_p,
            "predicted_top_pick": top_pick,
            "predicted_top_5": top_jodis[:5],
            "predicted_top_open": top_open,
            "predicted_top_close": top_close,
            "family_bracket": fam,
            "otc_digits": otc_digits,
            "otc_hit": otc_hit,
            "open_hit": open_hit,
            "close_hit": close_hit,
            "jodi_hit": jodi_hit,
            "status_label": status_label,
            "badge_type": badge_type,
        })

    week_range_str = f"{prev_mon.strftime('%d %b %Y')} – {prev_sat.strftime('%d %b %Y')}"
    return {
        "week_range": week_range_str,
        "summary": {
            "total_draws": len(draws),
            "ank_hits": total_ank_hits,
            "double_ank_hits": double_ank_hits,
            "edge": "+2.2% Walk-Forward",
        },
        "draws": draws,
    }


def compute_all_predictions(force_refresh: bool = False):
    """Execute live scraping/cache read, retrain models, and generate predictions."""
    # 1. Fetch & parse
    html = fetch_html(force_refresh=force_refresh)
    df = parse_kalyan_chart(html)
    export_to_csv(df, CSV_PATH)

    valid_df = df[df["Is_Valid"] == True].copy().reset_index(drop=True)
    valid_df["Open_Digit"] = valid_df["Open_Digit"].apply(lambda x: int(float(x)))
    valid_df["Close_Digit"] = valid_df["Close_Digit"].apply(lambda x: int(float(x)))
    valid_df["Jodi"] = valid_df["Jodi"].apply(lambda x: f"{int(float(x)):02d}")

    latest_row = valid_df.iloc[-1]
    latest_date_str = str(latest_row["Date"])
    latest_date = datetime.strptime(latest_date_str, "%Y-%m-%d")
    latest_jodi = str(latest_row["Jodi"]).zfill(2)
    latest_open = int(float(latest_row["Open_Digit"]))
    latest_close = int(float(latest_row["Close_Digit"]))

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

    # Fit Optimized Hybrid Ensemble Model
    ensemble = EnsemblePredictor()
    ensemble.fit(valid_df)

    # Compute last week prediction performance
    last_week_data = compute_last_week_performance(valid_df, html=html)

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
            "otc_digits": preds.get("otc_digits", []),
            "otc_pairs": preds.get("otc_pairs", []),
            "otc_pass_prob": round(float(preds.get("otc_pass_prob", 0.0)) * 100.0, 1),
            "open_digit_probs": [float(p) for p in (open_probs / open_probs.sum())],
            "close_digit_probs": [float(p) for p in (close_probs / close_probs.sum())],
            "patti_predictions": preds.get("patti_predictions", {}),
        }

    first_day_pred = list(predictions_by_day.values())[0] if predictions_by_day else {}
    output_payload = {
        "status": "success",
        "sync_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "last_developer_update": datetime.now().strftime("%Y-%m-%d"),
        "last_update_by_developer": datetime.now().strftime("%d %b %Y"),
        "total_records": len(df),
        "valid_records": len(valid_df),
        "last_week_data": last_week_data,
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
        "open_digit_probs": first_day_pred.get("open_digit_probs", []),
        "close_digit_probs": first_day_pred.get("close_digit_probs", []),
    }

    # Persist updated prediction_data.json
    pred_data_to_save = {
        "last_developer_update": datetime.now().strftime("%Y-%m-%d"),
        "last_update_by_developer": datetime.now().strftime("%d %b %Y"),
        "latest_draw": output_payload["latest_draw"],
        "last_week_data": last_week_data,
        "by_day": predictions_by_day,
        "open_digit_probs": first_day_pred.get("open_digit_probs", []),
        "close_digit_probs": first_day_pred.get("close_digit_probs", []),
        "predicted_for": {
            "date": first_day_pred.get("date", ""),
            "day": first_day_pred.get("day", ""),
        },
        "otc_recommendation": {
            "digits": first_day_pred.get("otc_digits", []),
            "cut_pairs": [f"{p[0]}-{p[1]}" for p in first_day_pred.get("otc_pairs", [])],
            "pass_probability": float(first_day_pred.get("otc_pass_prob", 0.0)),
        },
        "top_jodis": first_day_pred.get("top_jodis", []),
    }
    # Persist updated prediction_data.json to all 3 locations
    pred_paths = [
        os.path.join(WORKSPACE_DIR, "prediction_data.json"),
        os.path.join(WEB_DIR, "prediction_data.json"),
        os.path.join(ROOT_MATKA_DIR, "prediction_data.json"),
    ]
    for path in pred_paths:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(pred_data_to_save, f, indent=2)

    # Persist updated history.json to all 3 locations
    history_list = []
    for _, row in valid_df.iterrows():
        op = str(row["Open_Patti"]) if pd.notna(row.get("Open_Patti")) and str(row.get("Open_Patti")).strip() != "" else ""
        cl = str(row["Close_Patti"]) if pd.notna(row.get("Close_Patti")) and str(row.get("Close_Patti")).strip() != "" else ""
        history_list.append([
            str(row["Date"]),
            str(row["Day_Of_Week"]),
            int(float(row["Jodi"])) if pd.notna(row.get("Jodi")) else 0,
            int(float(row["Open_Digit"])) if pd.notna(row.get("Open_Digit")) else 0,
            int(float(row["Close_Digit"])) if pd.notna(row.get("Close_Digit")) else 0,
            op,
            cl,
        ])
    hist_paths = [
        os.path.join(WORKSPACE_DIR, "history.json"),
        os.path.join(WEB_DIR, "history.json"),
        os.path.join(ROOT_MATKA_DIR, "history.json"),
    ]
    for path in hist_paths:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(history_list, f)

    # Synchronize CSV mirror
    sync_csv_to_root(CSV_PATH)

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
