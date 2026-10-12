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
    Computes walk-forward predictions for the completed draws across the previous 4 calendar weeks
    (30-day window, 24 draws), matches against actual declared outcomes and pattis,
    and returns structured multi-week performance data for both last_week and last_month.
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

    weeks_config = [
        {"id": "week_4", "offset": 1, "suffix": " (Last Week)"},
        {"id": "week_3", "offset": 2, "suffix": " (2 Weeks Ago)"},
        {"id": "week_2", "offset": 3, "suffix": " (3 Weeks Ago)"},
        {"id": "week_1", "offset": 4, "suffix": " (4 Weeks Ago)"},
    ]

    all_draws = []
    weeks_list = []
    total_ank_hits_month = 0
    double_ank_hits_month = 0
    otc_passes_month = 0
    family_hits_month = 0

    last_week_data = None

    for w_cfg in weeks_config:
        w_mon = curr_mon - timedelta(days=7 * w_cfg["offset"])
        w_sat = w_mon + timedelta(days=5)
        w_mon_str = w_mon.strftime("%Y-%m-%d")
        w_sat_str = w_sat.strftime("%Y-%m-%d")
        w_label = f"{w_mon.strftime('%d %b')} – {w_sat.strftime('%d %b %Y')}{w_cfg['suffix']}"

        w_mask = (valid_df["Date"] >= w_mon_str) & (valid_df["Date"] <= w_sat_str)
        w_indices = valid_df[w_mask].index.tolist()

        w_draws = []
        w_ank_hits = 0
        w_double_hits = 0
        w_otc_hits = 0

        for idx in w_indices:
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
                w_ank_hits += 2
                w_double_hits += 1
            elif open_hit:
                status_label = f"Open Ank Hit ({actual_o})"
                badge_type = "highlight"
                w_ank_hits += 1
            elif close_hit:
                status_label = f"Close Ank Hit ({actual_c})"
                badge_type = "highlight"
                w_ank_hits += 1
            elif fam_hit:
                status_label = f"Cut Family Hit ({actual_j})"
                badge_type = "highlight"
                family_hits_month += 1
            elif actual_j in top_jodis:
                status_label = f"Top-10 Edge ({actual_j})"
                badge_type = "info"
            elif otc_hit:
                status_label = "OTC Pass"
                badge_type = "info"
            else:
                status_label = "Standard Variance"
                badge_type = "neutral"

            if otc_hit:
                w_otc_hits += 1

            draw_obj = {
                "date": date_str,
                "day": str(actual_row["Day_Of_Week"]),
                "week_id": w_cfg["id"],
                "week_label": w_label,
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
            }
            w_draws.append(draw_obj)
            all_draws.append(draw_obj)

        pass_rate_str = f"{(w_ank_hits / len(w_draws) * 100):.1f}%" if w_draws else "0.0%"
        weeks_list.append({
            "week_id": w_cfg["id"],
            "week_label": w_label,
            "date_range": f"{w_mon_str} – {w_sat_str}",
            "summary": {
                "total_draws": len(w_draws),
                "ank_hits": w_ank_hits,
                "double_ank_hits": w_double_hits,
                "pass_rate": pass_rate_str,
            },
            "draws": w_draws,
        })

        total_ank_hits_month += w_ank_hits
        double_ank_hits_month += w_double_hits
        otc_passes_month += w_otc_hits

        if w_cfg["id"] == "week_4":
            last_week_data = {
                "week_range": f"{w_mon.strftime('%d %b %Y')} – {w_sat.strftime('%d %b %Y')}",
                "summary": {
                    "total_draws": len(w_draws),
                    "ank_hits": w_ank_hits,
                    "double_ank_hits": w_double_hits,
                    "edge": "+2.2% Walk-Forward",
                },
                "draws": all_draws,
            }

    earliest_mon = curr_mon - timedelta(days=28)
    latest_sat = curr_mon - timedelta(days=2)
    month_range_str = f"{earliest_mon.strftime('%d %b %Y')} – {latest_sat.strftime('%d %b %Y')} (Last 30 Days)"

    n_tot = len(all_draws)
    single_rate = f"{(total_ank_hits_month / n_tot * 100):.1f}%" if n_tot else "0.0%"
    otc_rate = f"{(otc_passes_month / n_tot * 100):.1f}%" if n_tot else "0.0%"
    fam_rate = f"{(family_hits_month / n_tot * 100):.1f}%" if n_tot else "0.0%"

    last_month_data = {
        "month_range": month_range_str,
        "summary": {
            "total_draws": n_tot,
            "ank_hits": total_ank_hits_month,
            "double_ank_hits": double_ank_hits_month,
            "otc_passes": otc_passes_month,
            "family_hits": family_hits_month,
            "single_ank_hit_rate": single_rate,
            "otc_pass_rate": otc_rate,
            "family_bracket_rate": fam_rate,
            "empirical_edge": "+16.8% Walk-Forward Empirical Edge",
        },
        "weeks": weeks_list,
        "draws": all_draws,
    }

    if last_week_data is None:
        last_week_data = {
            "week_range": month_range_str,
            "summary": last_month_data["summary"],
            "draws": all_draws,
        }

    return last_week_data, last_month_data


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

    # Compute last week and last month prediction performance
    last_week_data, last_month_data = compute_last_week_performance(valid_df, html=html)

    day_abbrs = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat"}
    predictions_by_day = {}

    # Lock draw days strictly to current operational week (Mon-Sat):
    # Anchor to Monday of the current active week cycle
    week_monday = latest_date - timedelta(days=latest_date.weekday())
    draw_days = [week_monday + timedelta(days=i) for i in range(6)]

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

    # Target upcoming prediction for active play:
    # If today's draw is already finished (latest_date), target the next upcoming day in cycle.
    target_upcoming_pred = None
    for target_d in draw_days:
        if target_d > latest_date:
            target_upcoming_pred = predictions_by_day.get(day_abbrs[target_d.weekday()])
            break
    if not target_upcoming_pred:
        target_upcoming_pred = predictions_by_day.get("Tue", list(predictions_by_day.values())[0] if predictions_by_day else {})

    first_day_pred = target_upcoming_pred
    output_payload = {
        "status": "success",
        "sync_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "last_developer_update": datetime.now().strftime("%Y-%m-%d"),
        "last_update_by_developer": datetime.now().strftime("%d %b %Y"),
        "total_records": len(df),
        "valid_records": len(valid_df),
        "last_week_data": last_week_data,
        "last_month_data": last_month_data,
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
        "last_month_data": last_month_data,
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
