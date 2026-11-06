"""
Kalyan Matka Upcoming Draw Predictor & Analytical Forecast Engine.
Generates ranked Jodi recommendations, Single Open/Close Digits (Ank),
Family/Cut brackets, and provides transparent mathematical risk metrics.
"""

import os
import sys
import json
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tabulate import tabulate
from models import EnsemblePredictor, get_family_jodis, evaluate_outcome, recalibrate_hyperparameters

SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = str(SCRIPT_DIR)
ROOT_MATKA_DIR = str(SCRIPT_DIR.parent)
WEB_DIR = str(SCRIPT_DIR / "web")


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


def sync_performance_log(log_payload: dict):
    """Synchronize performance_log.json across all 3 mirror locations with identical binary encoding."""
    perf_paths = [
        os.path.join(WORKSPACE_DIR, "performance_log.json"),
        os.path.join(WEB_DIR, "performance_log.json"),
        os.path.join(ROOT_MATKA_DIR, "performance_log.json"),
    ]
    json_bytes = (json.dumps(log_payload, indent=2) + "\n").encode("utf-8")
    for p in perf_paths:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb") as f:
            f.write(json_bytes)
    print(f"[SUCCESS] performance_log.json synchronized across 3 mirror locations.")


def load_performance_log(custom_path: str = None) -> dict:
    """Load performance_log.json from candidate mirror locations or initialize default."""
    candidates = []
    if custom_path:
        candidates.append(Path(custom_path))
    candidates.extend([
        Path(WORKSPACE_DIR) / "performance_log.json",
        Path(ROOT_MATKA_DIR) / "performance_log.json",
        Path(WEB_DIR) / "performance_log.json",
    ])
    for p in candidates:
        if p.exists() and p.stat().st_size > 0:
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict) and "summary" in data and "current_hyperparameters" in data and "draw_logs" in data:
                    return data
            except Exception:
                continue

    # Default template conforming strictly to PROJECT.md schema
    return {
        "summary": {
            "total_draws_logged": 0,
            "top10_hit_rate": 0.0,
            "family_cut_hit_rate": 0.0,
            "otc_hit_rate": 0.0,
            "last_updated": datetime.now().strftime("%Y-%m-%d"),
        },
        "current_hyperparameters": {
            "lambda_cut_cluster": 0.35,
            "w_recency": 0.52,
            "w_markov": 0.18,
            "w_seasonal": 0.15,
            "w_gap": 0.15,
        },
        "draw_logs": [],
    }


def audit_and_update_performance_log(csv_path: str = None, lookback_draws: int = 60) -> dict:
    """
    Audits prior predictions against actual draw outcomes (Features 7, 8, 9, 10).
    Backfills or incrementally updates performance_log.json without lookahead data leakage.
    Recalibrates hyperparameters and synchronizes 3-way mirrors.
    """
    resolved_csv = resolve_csv_path(csv_path)
    df = pd.read_csv(resolved_csv, dtype={"Jodi": str, "Open_Patti": str, "Close_Patti": str, "Raw_Entry": str})
    valid_df = df[df["Is_Valid"] == True].copy().reset_index(drop=True)
    valid_df["Open_Digit"] = valid_df["Open_Digit"].apply(lambda x: int(float(x)))
    valid_df["Close_Digit"] = valid_df["Close_Digit"].apply(lambda x: int(float(x)))
    valid_df["Jodi"] = valid_df["Jodi"].apply(lambda x: f"{int(float(x)):02d}")

    total_valid = len(valid_df)
    if total_valid < 15:
        log = load_performance_log()
        sync_performance_log(log)
        return log

    log = load_performance_log()
    existing_logs = log.get("draw_logs", [])
    logged_dates = {entry["date"] for entry in existing_logs if isinstance(entry, dict) and "date" in entry}

    if not existing_logs:
        start_idx = max(10, total_valid - lookback_draws)
    else:
        last_logged_date = existing_logs[-1]["date"]
        matches = valid_df.index[valid_df["Date"] == last_logged_date].tolist()
        if matches:
            last_idx = matches[-1]
            if last_idx >= total_valid - 1:
                sync_performance_log(log)
                return log
            start_idx = last_idx + 1
        else:
            start_idx = max(10, total_valid - lookback_draws)

    current_params = dict(log.get("current_hyperparameters", {
        "lambda_cut_cluster": 0.35,
        "w_recency": 0.52,
        "w_markov": 0.18,
        "w_seasonal": 0.15,
        "w_gap": 0.15,
    }))

    for idx in range(start_idx, total_valid):
        actual_row = valid_df.iloc[idx]
        d_str = str(actual_row["Date"])
        if d_str in logged_dates:
            continue

        train_slice = valid_df.iloc[:idx]
        prev_row = valid_df.iloc[idx - 1]

        prev_jodi = str(prev_row["Jodi"]).zfill(2)
        prev_open = int(prev_row["Open_Digit"])
        prev_close = int(prev_row["Close_Digit"])
        day_of_week = str(actual_row["Day_Of_Week"])

        predictor = EnsemblePredictor(hyperparameters=current_params)
        predictor.fit(train_slice)

        preds = predictor.predict(
            day_of_week=day_of_week,
            prev_jodi=prev_jodi,
            prev_open=prev_open,
            prev_close=prev_close,
            hyperparameters=current_params,
        )

        top_10_items = preds.get("top_10_jodis", [])
        top_10_jodis = [item["jodi"] for item in top_10_items]
        top_pick = top_10_jodis[0] if top_10_jodis else "00"
        otc_digits = [int(d) for d in preds.get("otc_digits", [])]

        act_j = str(actual_row["Jodi"]).zfill(2)
        act_o = int(actual_row["Open_Digit"])
        act_c = int(actual_row["Close_Digit"])

        density_scores = preds.get("cut_cluster_density_scores", preds.get("jodi_probs"))
        outcome = evaluate_outcome(
            predicted_jodi_probs_or_ranking=density_scores,
            top_pick=top_pick,
            top_10=top_10_jodis,
            otc_digits=otc_digits,
            actual_jodi=act_j,
            actual_open=act_o,
            actual_close=act_c,
        )

        entry = {
            "date": d_str,
            "day": day_of_week,
            "actual_draw": {
                "jodi": act_j,
                "open": act_o,
                "close": act_c,
            },
            "predictions": {
                "top_pick": top_pick,
                "top_10": top_10_jodis,
                "top_family_bracket": get_family_jodis(top_pick),
                "otc_digits": otc_digits,
            },
            "outcome": {
                "actual_rank": outcome["actual_rank"],
                "top10_hit": outcome["top10_hit"],
                "family_cut_hit": outcome["family_cut_hit"],
                "otc_hit": outcome["otc_hit"],
                "open_hit": outcome["open_hit"],
                "close_hit": outcome["close_hit"],
            },
        }
        existing_logs.append(entry)
        logged_dates.add(d_str)

        if len(existing_logs) % 10 == 0 or idx == total_valid - 1:
            n_logged = len(existing_logs)
            t10_rate = sum(1 for e in existing_logs if e["outcome"]["top10_hit"]) / n_logged
            fam_rate = sum(1 for e in existing_logs if e["outcome"]["family_cut_hit"]) / n_logged
            otc_rate = sum(1 for e in existing_logs if e["outcome"]["otc_hit"]) / n_logged
            interim_summary = {
                "total_draws_logged": n_logged,
                "top10_hit_rate": round(t10_rate, 4),
                "family_cut_hit_rate": round(fam_rate, 4),
                "otc_hit_rate": round(otc_rate, 4),
                "last_updated": datetime.now().strftime("%Y-%m-%d"),
            }
            current_params = recalibrate_hyperparameters(current_params, {"summary": interim_summary})

    n_logged = len(existing_logs)
    t10_rate = sum(1 for e in existing_logs if e["outcome"]["top10_hit"]) / n_logged if n_logged > 0 else 0.0
    fam_rate = sum(1 for e in existing_logs if e["outcome"]["family_cut_hit"]) / n_logged if n_logged > 0 else 0.0
    otc_rate = sum(1 for e in existing_logs if e["outcome"]["otc_hit"]) / n_logged if n_logged > 0 else 0.0

    final_summary = {
        "total_draws_logged": n_logged,
        "top10_hit_rate": round(t10_rate, 4),
        "family_cut_hit_rate": round(fam_rate, 4),
        "otc_hit_rate": round(otc_rate, 4),
        "last_updated": datetime.now().strftime("%Y-%m-%d"),
    }
    tuned_params = recalibrate_hyperparameters(current_params, {"summary": final_summary})

    updated_log = {
        "summary": final_summary,
        "current_hyperparameters": tuned_params,
        "draw_logs": existing_logs,
    }
    sync_performance_log(updated_log)
    return updated_log


def sync_artifacts(prediction_payload: dict, valid_df: pd.DataFrame, sync_perf: bool = True):
    """Synchronize prediction_data.json, history.json, and performance_log.json across all 3 locations."""
    pred_paths = [
        os.path.join(WORKSPACE_DIR, "prediction_data.json"),
        os.path.join(WEB_DIR, "prediction_data.json"),
        os.path.join(ROOT_MATKA_DIR, "prediction_data.json"),
    ]
    base_path = pred_paths[0]
    merged_payload = {}
    if os.path.exists(base_path):
        try:
            with open(base_path, "r", encoding="utf-8") as f:
                merged_payload = json.load(f)
        except Exception:
            merged_payload = {}
    merged_payload.update(prediction_payload)

    for p in pred_paths:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(merged_payload, f, indent=2)

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
    for p in hist_paths:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(history_list, f)

    if sync_perf:
        perf = load_performance_log()
        sync_performance_log(perf)

    print(f"[SUCCESS] Prediction artifacts synchronized across 3 locations (tools/matka/, subfolder, web/).")


if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def get_cut_digit(digit: int) -> int:
    """Standard Matka Cut digit (adding 5 mod 10)."""
    return (int(digit) + 5) % 10


def get_family_jodis(jodi_val) -> list:
    """
    Generate the Matka Family / Cut Jodis for a given 2-digit number (4 or 8 Jodis).
    Uses open digit, close digit, cut open, cut close.
    Handles string, integer, float, and edge case inputs cleanly.
    """
    if jodi_val is None:
        return []
    try:
        s = f"{int(float(jodi_val)):02d}"
    except (ValueError, TypeError):
        s = str(jodi_val).strip()
        if len(s) < 2 or not s[:2].isdigit():
            return []
        s = s[:2]

    o = int(s[0])
    c = int(s[1])
    cut_o = get_cut_digit(o)
    cut_c = get_cut_digit(c)

    # 4 digit pairs + their reversals
    base_pairs = [
        (o, c), (o, cut_c), (cut_o, c), (cut_o, cut_c),
        (c, o), (cut_c, o), (c, cut_o), (cut_c, cut_o),
    ]
    seen = set()
    result = []
    for d1, d2 in base_pairs:
        val = f"{d1}{d2}"
        if val not in seen:
            seen.add(val)
            result.append(val)
    return sorted(result)


def make_ascii_bar(percent: float, max_len: int = 15) -> str:
    filled = int(round((percent / 100.0) * max_len * 3.5))
    filled = min(filled, max_len)
    return "#" * filled + "-" * (max_len - filled)


def generate_by_day_predictions(
    ensemble: EnsemblePredictor,
    latest_date: datetime,
    latest_jodi: str,
    latest_open: int,
    latest_close: int,
) -> dict:
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
        j_probs = preds["jodi_probs"]
        o_probs = preds["open_probs"]
        c_probs = preds["close_probs"]

        sorted_j = np.argsort(j_probs)[::-1]
        sorted_o = np.argsort(o_probs)[::-1]
        sorted_c = np.argsort(c_probs)[::-1]

        top_10 = preds.get("top_10_jodis", [])
        top_j = []
        if top_10:
            for item in top_10:
                j_str = item["jodi"]
                p = round(float(item["score"]) * 100.0, 2)
                top_j.append({
                    "rank": item["rank"],
                    "jodi": j_str,
                    "prob": p,
                    "confidence": p,
                    "family": get_family_jodis(j_str),
                    "score": item["score"],
                    "raw_prob": item.get("raw_prob", float(j_probs[int(j_str)])),
                    "is_family_affinity": item.get("is_family_affinity", False),
                })
        else:
            for rank, idx in enumerate(sorted_j[:10], start=1):
                j_str = f"{idx:02d}"
                p = round(float(j_probs[idx]) * 100.0, 2)
                top_j.append({
                    "rank": rank,
                    "jodi": j_str,
                    "prob": p,
                    "confidence": p,
                    "family": get_family_jodis(j_str),
                })

        top_o = [
            {"rank": rank, "digit": int(idx), "prob": round(float(o_probs[idx]) * 100.0, 2)}
            for rank, idx in enumerate(sorted_o, start=1)
        ]
        top_c = [
            {"rank": rank, "digit": int(idx), "prob": round(float(c_probs[idx]) * 100.0, 2)}
            for rank, idx in enumerate(sorted_c, start=1)
        ]

        norm_o = [float(p) for p in (o_probs / o_probs.sum())]
        norm_c = [float(p) for p in (c_probs / c_probs.sum())]

        predictions_by_day[day_name] = {
            "date": target_d.strftime("%Y-%m-%d"),
            "day": day_name,
            "top_jodis": top_j,
            "open_digits": top_o,
            "close_digits": top_c,
            "top_pick": top_j[0]["jodi"],
            "top_family": top_j[0]["family"],
            "otc_digits": preds.get("otc_digits", []),
            "otc_pairs": preds.get("otc_pairs", []),
            "otc_pass_prob": round(float(preds.get("otc_pass_prob", 0.0)) * 100.0, 1),
            "open_digit_probs": norm_o,
            "close_digit_probs": norm_c,
            "patti_predictions": preds.get("patti_predictions", {}),
        }
    return predictions_by_day


def generate_upcoming_prediction(csv_path: str = None) -> dict:
    target_csv = resolve_csv_path(csv_path)
    df = pd.read_csv(target_csv, dtype={"Jodi": str, "Open_Patti": str, "Close_Patti": str, "Raw_Entry": str})
    valid_df = df[df["Is_Valid"] == True].copy().reset_index(drop=True)
    valid_df["Open_Digit"] = valid_df["Open_Digit"].apply(lambda x: int(float(x)))
    valid_df["Close_Digit"] = valid_df["Close_Digit"].apply(lambda x: int(float(x)))
    valid_df["Jodi"] = valid_df["Jodi"].apply(lambda x: f"{int(float(x)):02d}")

    # Identify the latest known draw
    latest_row = valid_df.iloc[-1]
    latest_date_str = latest_row["Date"]
    latest_date = datetime.strptime(latest_date_str, "%Y-%m-%d")
    latest_day = latest_row["Day_Of_Week"]
    latest_jodi = str(latest_row["Jodi"]).zfill(2)
    latest_open = int(float(latest_row["Open_Digit"]))
    latest_close = int(float(latest_row["Close_Digit"]))

    # Determine target next draw date (skip Sunday if next is Sunday)
    next_date = latest_date + timedelta(days=1)
    if next_date.weekday() == 6:  # Sunday
        next_date += timedelta(days=1)
    
    day_map = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Mon"}
    target_day = day_map[next_date.weekday()]
    target_date_str = next_date.strftime("%Y-%m-%d")

    # Load calibrated hyperparameters from performance_log.json if available
    perf_log = load_performance_log()
    current_params = perf_log.get("current_hyperparameters", None)

    # Fit Optimized Hybrid Ensemble Predictor on full historical data
    ensemble = EnsemblePredictor(hyperparameters=current_params)
    ensemble.fit(valid_df)

    # Generate probabilities for target draw
    preds = ensemble.predict(
        day_of_week=target_day,
        prev_jodi=latest_jodi,
        prev_open=latest_open,
        prev_close=latest_close,
        hyperparameters=current_params,
    )

    jodi_probs = preds["jodi_probs"]
    open_probs = preds["open_probs"]
    close_probs = preds["close_probs"]

    norm_open = [float(p) for p in (open_probs / open_probs.sum())]
    norm_close = [float(p) for p in (close_probs / close_probs.sum())]

    # Top Jodis (Cut-Cluster Density Weighting & Family-Enriched Top 10)
    top_10 = preds.get("top_10_jodis")
    cut_cluster_density_scores = preds.get("cut_cluster_density_scores", jodi_probs)
    top_jodis_list = []
    if top_10:
        for item in top_10:
            rank = item["rank"]
            j_str = item["jodi"]
            score = item["score"]
            raw_p = item.get("raw_prob", jodi_probs[int(j_str)])
            p = score * 100.0
            edge = (p / 1.0) - 1.0  # relative edge over random 1%
            top_jodis_list.append({
                "Rank": rank,
                "Jodi": j_str,
                "Probability": f"{p:.2f}%",
                "Rel_Edge": f"{'+' if edge >= 0 else ''}{edge * 100:.1f}%",
                "Bar": make_ascii_bar(p, 12),
                "Score": score,
                "Raw_Prob": raw_p,
                "Is_Family_Affinity": item.get("is_family_affinity", False),
                "Family": item.get("family", get_family_jodis(j_str)),
            })
    else:
        sorted_jodi_indices = np.argsort(cut_cluster_density_scores)[::-1]
        for rank, idx in enumerate(sorted_jodi_indices[:10], start=1):
            j_str = f"{idx:02d}"
            p = cut_cluster_density_scores[idx] * 100.0
            edge = (p / 1.0) - 1.0  # relative edge over random 1%
            top_jodis_list.append({
                "Rank": rank,
                "Jodi": j_str,
                "Probability": f"{p:.2f}%",
                "Rel_Edge": f"{'+' if edge >= 0 else ''}{edge * 100:.1f}%",
                "Bar": make_ascii_bar(p, 12),
                "Score": float(cut_cluster_density_scores[idx]),
                "Raw_Prob": float(jodi_probs[idx]),
                "Is_Family_Affinity": False,
                "Family": get_family_jodis(j_str),
            })
    top_jodi_df = pd.DataFrame(top_jodis_list)

    # Open Digits Ranking
    sorted_open_indices = np.argsort(open_probs)[::-1]
    open_digits_list = []
    for rank, idx in enumerate(sorted_open_indices, start=1):
        p = open_probs[idx] * 100.0
        open_digits_list.append({
            "Rank": rank,
            "Digit": idx,
            "Probability": f"{p:.2f}%",
            "Bar": make_ascii_bar(p, 12),
        })
    open_df = pd.DataFrame(open_digits_list)

    # Close Digits Ranking
    sorted_close_indices = np.argsort(close_probs)[::-1]
    close_digits_list = []
    for rank, idx in enumerate(sorted_close_indices, start=1):
        p = close_probs[idx] * 100.0
        close_digits_list.append({
            "Rank": rank,
            "Digit": idx,
            "Probability": f"{p:.2f}%",
            "Bar": make_ascii_bar(p, 12),
        })
    close_df = pd.DataFrame(close_digits_list)

    # Recommended Top-1 Jodi and its Family / Cut bracket
    top_1_jodi = str(top_jodis_list[0]["Jodi"]).zfill(2)
    family_bracket = get_family_jodis(top_1_jodi)

    # Generate 6-day forecasts for full schedule
    by_day_forecasts = generate_by_day_predictions(
        ensemble, latest_date, latest_jodi, latest_open, latest_close
    )

    return {
        "latest_draw": {
            "date": latest_date_str,
            "day": latest_day,
            "jodi": latest_jodi,
            "open": latest_open,
            "close": latest_close,
        },
        "target_draw": {
            "date": target_date_str,
            "day": target_day,
        },
        "top_jodis": top_jodi_df,
        "open_digits": open_df,
        "close_digits": close_df,
        "open_digit_probs": norm_open,
        "close_digit_probs": norm_close,
        "open_probs": norm_open,
        "close_probs": norm_close,
        "jodi_probs": [float(p) for p in jodi_probs],
        "cut_cluster_density_scores": [float(s) for s in cut_cluster_density_scores],
        "top_10_jodis": top_jodis_list,
        "top_1_jodi": top_1_jodi,
        "family_bracket": family_bracket,
        "otc_digits": preds.get("otc_digits", []),
        "otc_pairs": preds.get("otc_pairs", []),
        "otc_pass_prob": preds.get("otc_pass_prob", 0.0),
        "patti_predictions": preds.get("patti_predictions", {}),
        "by_day": by_day_forecasts,
    }


def print_prediction_report(res: dict):
    print("\n" + "=" * 75)
    print("       KALYAN MATKA QUANTITATIVE FORECAST & UPCOMING PREDICTION")
    print("=" * 75)
    print(f"Target Draw Date:     {res['target_draw']['date']} ({res['target_draw']['day']})")
    print(f"Latest Known Draw:    {res['latest_draw']['date']} ({res['latest_draw']['day']}) -> Jodi: {res['latest_draw']['jodi']}")
    print(f"Model Stack:          Hybrid Ensemble (Markov + Momentum + Gap Hazard + Line)")
    print("-" * 75)

    otc_str = ", ".join(str(d) for d in res.get("otc_digits", []))
    otc_pairs_str = " & ".join(f"({p[0]}-{p[1]})" for p in res.get("otc_pairs", []))
    otc_prob = res.get("otc_pass_prob", 0.0) * 100
    print(f"\n[OTC] HIGH-CONFIDENCE 4-ANK OTC (OPEN-TO-CLOSE):")
    print(f"  -> Recommended Anks  : [ {otc_str} ]")
    print(f"  -> Harmonic Cut Pairs : {otc_pairs_str}")
    print(f"  -> Modeled Pass Prob  : {otc_prob:.1f}%")

    print("\n[JODI] TOP RECOMMENDED JODI NUMBERS (HIGH CONFIDENCE CANDIDATES):")
    print(tabulate(res["top_jodis"].head(5), headers="keys", tablefmt="github", showindex=False))

    print("\n[CANDIDATES] EXTENDED CANDIDATES (RANKS 6 - 10):")
    print(tabulate(res["top_jodis"].tail(5), headers="keys", tablefmt="github", showindex=False))

    print(f"\n[FAMILY] RECOMMENDED FAMILY / CUT BRACKET FOR TOP JODI '{res['top_1_jodi']}':")
    print("  -> Associated Pairs: " + ", ".join(f"[{j}]" for j in res["family_bracket"]))

    print("\n[OPEN] PREDICTED SINGLE OPEN DIGIT (ANK) PROBABILITY DISTRIBUTION:")
    print(tabulate(res["open_digits"].head(5), headers="keys", tablefmt="github", showindex=False))

    print("\n[CLOSE] PREDICTED SINGLE CLOSE DIGIT (ANK) PROBABILITY DISTRIBUTION:")
    print(tabulate(res["close_digits"].head(5), headers="keys", tablefmt="github", showindex=False))

    print("\n[PATTI] RECOMMENDED 3-DIGIT PATTI / PANEL FORECASTS (FOR TOP OTC ANKS):")
    patti_preds = res.get("patti_predictions", {})
    patti_rows = []
    for ank in res.get("otc_digits", []):
        info = patti_preds.get(str(ank), {})
        sp_str = ", ".join(info.get("sp", [])[:3])
        dp_str = ", ".join(info.get("dp", [])[:2])
        patti_rows.append({"Ank": ank, "Single Patti (SP)": sp_str, "Double Patti (DP)": dp_str})
    if patti_rows:
        print(tabulate(patti_rows, headers="keys", tablefmt="github", showindex=False))

    print("\n" + "=" * 75)
    print("              CRITICAL MATHEMATICAL RISK & EXPECTANCY NOTICE")
    print("=" * 75)
    print("1. House Edge: Standard Jodi payout is 90:1 on 100 possible outcomes (-10.0% EV).")
    print("2. Empirical Independence: Historical autocorrelation shows draws are statistically")
    print("   independent. While statistical models identify historical patterns, lottery")
    print("   mechanisms possess intrinsic randomness and cannot guarantee wins.")
    print("3. Bankroll Protection: Never wager more than 1% to 2% of total capital on any draw.")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kalyan Matka Upcoming Draw Predictor")
    parser.add_argument("csv_pos", nargs="?", default=None, help="Optional historical CSV file path")
    parser.add_argument("--csv", default=None, help="Optional historical CSV file path")
    parser.add_argument("--export-json", action="store_true", help="Sync prediction artifacts to disk")
    parser.add_argument("--audit", action="store_true", help="Audit previous predictions against actual outcomes and update performance_log.json")
    parser.add_argument("--update-all", action="store_true", help="Automated pipeline: audit, recalibrate, predict, and 3-way sync all artifacts")
    args = parser.parse_args()

    input_csv = args.csv or args.csv_pos
    resolved_csv = resolve_csv_path(input_csv)

    if args.audit or args.update_all:
        audit_log = audit_and_update_performance_log(resolved_csv)
        summary = audit_log.get("summary", {})
        params = audit_log.get("current_hyperparameters", {})
        print("\n" + "=" * 75)
        print("      KALYAN MATKA PERFORMANCE AUDIT & CLOSED-LOOP CALIBRATION")
        print("=" * 75)
        print(f"Total Historical Draws Logged : {summary.get('total_draws_logged', 0)}")
        print(f"Top 10 Jodi Hit Rate          : {summary.get('top10_hit_rate', 0.0) * 100:.2f}%")
        print(f"Family Cut Hit Rate           : {summary.get('family_cut_hit_rate', 0.0) * 100:.2f}%")
        print(f"4-Ank OTC Hit Rate            : {summary.get('otc_hit_rate', 0.0) * 100:.2f}%")
        print(f"Last Calibration Timestamp    : {summary.get('last_updated', '')}")
        print("-" * 75)
        print("Active Calibrated Hyperparameters:")
        print(f"  lambda_cut_cluster : {params.get('lambda_cut_cluster', 0.35)}")
        print(f"  w_recency          : {params.get('w_recency', 0.52)}")
        print(f"  w_markov           : {params.get('w_markov', 0.18)}")
        print(f"  w_seasonal         : {params.get('w_seasonal', 0.15)}")
        print(f"  w_gap              : {params.get('w_gap', 0.15)}")
        print("=" * 75 + "\n")
        if args.audit and not args.update_all and not args.export_json:
            sys.exit(0)

    res = generate_upcoming_prediction(resolved_csv)
    print_prediction_report(res)

    if args.export_json or args.update_all:
        df = pd.read_csv(resolved_csv, dtype={"Jodi": str, "Open_Patti": str, "Close_Patti": str, "Raw_Entry": str})
        valid_df = df[df["Is_Valid"] == True].copy().reset_index(drop=True)
        # Build standard prediction artifact
        open_patti = str(valid_df.iloc[-1]["Open_Patti"]) if pd.notna(valid_df.iloc[-1]["Open_Patti"]) else ""
        close_patti = str(valid_df.iloc[-1]["Close_Patti"]) if pd.notna(valid_df.iloc[-1]["Close_Patti"]) else ""
        export_payload = {
            "last_developer_update": datetime.now().strftime("%Y-%m-%d"),
            "last_update_by_developer": datetime.now().strftime("%d %b %Y"),
            "latest_draw": {
                "date": str(valid_df.iloc[-1]["Date"]),
                "day": str(valid_df.iloc[-1]["Day_Of_Week"]),
                "jodi": str(valid_df.iloc[-1]["Jodi"]).zfill(2),
                "open": int(float(valid_df.iloc[-1]["Open_Digit"])),
                "close": int(float(valid_df.iloc[-1]["Close_Digit"])),
                "open_digit": int(float(valid_df.iloc[-1]["Open_Digit"])),
                "close_digit": int(float(valid_df.iloc[-1]["Close_Digit"])),
                "open_patti": open_patti,
                "close_patti": close_patti,
                "openPatti": open_patti,
                "closePatti": close_patti,
            },
            "predicted_for": {
                "date": res["target_draw"]["date"],
                "day": res["target_draw"]["day"],
            },
            "otc_recommendation": {
                "digits": res.get("otc_digits", []),
                "cut_pairs": [f"{p[0]}-{p[1]}" for p in res.get("otc_pairs", [])],
                "pass_probability": float(res.get("otc_pass_prob", 0.0)),
            },
            "top_jodis": res.get("top_jodis", pd.DataFrame()).to_dict(orient="records"),
            "open_digit_probs": res.get("open_digit_probs", []),
            "close_digit_probs": res.get("close_digit_probs", []),
            "by_day": res.get("by_day", {}),
        }
        sync_artifacts(export_payload, valid_df, sync_perf=True)
