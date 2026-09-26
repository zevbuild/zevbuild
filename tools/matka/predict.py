"""
Kalyan Matka Upcoming Draw Predictor & Analytical Forecast Engine.
Generates ranked Jodi recommendations, Single Open/Close Digits (Ank),
Family/Cut brackets, and provides transparent mathematical risk metrics.
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tabulate import tabulate
from models import EnsemblePredictor

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def get_cut_digit(digit: int) -> int:
    """Standard Matka Cut digit (adding 5 mod 10)."""
    return (digit + 5) % 10


def get_family_jodis(jodi_str: str) -> list:
    """
    Generate the 8 Matka Family / Cut Jodis for a given 2-digit number.
    Uses open digit, close digit, cut open, cut close.
    """
    o = int(jodi_str[0])
    c = int(jodi_str[1])
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


def generate_upcoming_prediction(csv_path: str = "kalyan_historical_data.csv") -> dict:
    df = pd.read_csv(csv_path, dtype={"Jodi": str, "Raw_Entry": str})
    valid_df = df[df["Is_Valid"] == True].copy().reset_index(drop=True)
    valid_df["Open_Digit"] = valid_df["Open_Digit"].astype(int)
    valid_df["Close_Digit"] = valid_df["Close_Digit"].astype(int)

    # Identify the latest known draw
    latest_row = valid_df.iloc[-1]
    latest_date_str = latest_row["Date"]
    latest_date = datetime.strptime(latest_date_str, "%Y-%m-%d")
    latest_day = latest_row["Day_Of_Week"]
    latest_jodi = str(latest_row["Jodi"]).zfill(2)
    latest_open = int(latest_row["Open_Digit"])
    latest_close = int(latest_row["Close_Digit"])

    # Determine target next draw date (skip Sunday if next is Sunday)
    next_date = latest_date + timedelta(days=1)
    if next_date.weekday() == 6:  # Sunday
        next_date += timedelta(days=1)
    
    day_map = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Mon"}
    target_day = day_map[next_date.weekday()]
    target_date_str = next_date.strftime("%Y-%m-%d")

    # Fit Ensemble Predictor on full historical data
    ensemble = EnsemblePredictor(w_markov=0.30, w_recency=0.40, w_seasonal=0.30)
    ensemble.fit(valid_df)

    # Generate probabilities for target draw
    preds = ensemble.predict(
        day_of_week=target_day,
        prev_jodi=latest_jodi,
        prev_open=latest_open,
        prev_close=latest_close
    )

    jodi_probs = preds["jodi_probs"]
    open_probs = preds["open_probs"]
    close_probs = preds["close_probs"]

    # Top Jodis
    sorted_jodi_indices = np.argsort(jodi_probs)[::-1]
    top_jodis_list = []
    for rank, idx in enumerate(sorted_jodi_indices[:10], start=1):
        j_str = f"{idx:02d}"
        p = jodi_probs[idx] * 100.0
        edge = (p / 1.0) - 1.0  # relative edge over random 1%
        top_jodis_list.append({
            "Rank": rank,
            "Jodi": j_str,
            "Probability": f"{p:.2f}%",
            "Rel_Edge": f"{'+' if edge >= 0 else ''}{edge * 100:.1f}%",
            "Bar": make_ascii_bar(p, 12),
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
    top_1_jodi = f"{sorted_jodi_indices[0]:02d}"
    family_bracket = get_family_jodis(top_1_jodi)

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
        "top_1_jodi": top_1_jodi,
        "family_bracket": family_bracket,
    }


def print_prediction_report(res: dict):
    print("\n" + "=" * 75)
    print("       KALYAN MATKA QUANTITATIVE FORECAST & UPCOMING PREDICTION")
    print("=" * 75)
    print(f"Target Draw Date:     {res['target_draw']['date']} ({res['target_draw']['day']})")
    print(f"Latest Known Draw:    {res['latest_draw']['date']} ({res['latest_draw']['day']}) -> Jodi: {res['latest_draw']['jodi']}")
    print(f"Model Stack:          Ensemble (Markov Transition + Recency Decay + Day-of-Week)")
    print("-" * 75)

    print("\n[🎯] TOP RECOMMENDED JODI NUMBERS (HIGH CONFIDENCE CANDIDATES):")
    print(tabulate(res["top_jodis"].head(5), headers="keys", tablefmt="github", showindex=False))

    print("\n[📋] EXTENDED CANDIDATES (RANKS 6 - 10):")
    print(tabulate(res["top_jodis"].tail(5), headers="keys", tablefmt="github", showindex=False))

    print(f"\n[🔄] RECOMMENDED FAMILY / CUT BRACKET FOR TOP JODI '{res['top_1_jodi']}':")
    print("  -> Associated Pairs: " + ", ".join(f"[{j}]" for j in res["family_bracket"]))

    print("\n[🔓] PREDICTED SINGLE OPEN DIGIT (ANK) PROBABILITY DISTRIBUTION:")
    print(tabulate(res["open_digits"].head(5), headers="keys", tablefmt="github", showindex=False))

    print("\n[🔒] PREDICTED SINGLE CLOSE DIGIT (ANK) PROBABILITY DISTRIBUTION:")
    print(tabulate(res["close_digits"].head(5), headers="keys", tablefmt="github", showindex=False))

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
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "kalyan_historical_data.csv"
    res = generate_upcoming_prediction(csv_file)
    print_prediction_report(res)
