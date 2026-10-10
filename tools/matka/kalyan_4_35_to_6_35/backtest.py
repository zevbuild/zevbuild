"""
Walk-Forward Historical Backtesting Engine for Kalyan Matka Prediction Models.
Simulates out-of-sample forecasting across 2,700+ historical draws with zero data leakage.
Measures 4-Ank OTC Pass Rates, Top-1, Top-3, Top-5, Top-10 Jodi hit rates,
Open/Close digit hit rates, Top Family bracket hit rates, and tracks cumulative P&L.
Supports automated parameter optimization via '--optimize'.
"""

import os
import sys
import time
import math
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from tabulate import tabulate
from collections import defaultdict, Counter

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


if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def calc_binomial_stats(k: int, n: int, p0: float):
    """
    Pure-Python one-tailed binomial hypothesis test vs random baseline (ZERO Scipy).
    Uses standard normal approximation with math.erf for p-value:
      Z = (p_hat - p0) / sqrt(p0 * (1 - p0) / n)
      p_value = 0.5 * (1.0 - erf(Z / sqrt(2)))
    """
    if n <= 0 or p0 <= 0.0 or p0 >= 1.0:
        return 0.0, 1.0
    p_hat = k / n
    se = math.sqrt(p0 * (1.0 - p0) / n)
    if se == 0:
        return 0.0, 1.0
    z = (p_hat - p0) / se
    p_val = 0.5 * (1.0 - math.erf(z / math.sqrt(2.0)))
    return z, p_val


def run_walk_forward_backtest(
    csv_path: str = None,
    warmup_draws: int = 500,
    hl_fast: float = 8.0,
    hl_med: float = 35.0,
    hl_macro: float = 120.0,
    alpha_digit: float = 0.25,
    alpha_jodi: float = 0.05,
    w_rec: float = 0.65,
    w_mk: float = 0.15,
    w_sn: float = 0.10,
    w_gp: float = 0.10,
    w_joint: float = 0.45,
    w_total: float = 0.15,
    w_direct: float = 0.25,
    w_markov_jodi: float = 0.15,
    cut_res: float = 0.12,
) -> dict:
    target_csv = resolve_csv_path(csv_path)
    df = pd.read_csv(
        target_csv,
        dtype={"Jodi": str, "Open_Patti": str, "Close_Patti": str, "Raw_Entry": str},
    )
    valid_df = df[df["Is_Valid"] == True].copy().reset_index(drop=True)
    n_total = len(valid_df)

    if n_total <= warmup_draws:
        raise ValueError(f"Not enough draws ({n_total}) for warmup window ({warmup_draws}).")

    jodis = valid_df["Jodi"].apply(lambda x: int(float(x))).values
    opens = valid_df["Open_Digit"].apply(lambda x: int(float(x))).values
    closes = valid_df["Close_Digit"].apply(lambda x: int(float(x))).values
    days = valid_df["Day_Of_Week"].values
    dates = valid_df["Date"].values

    decay_fast = np.log(2.0) / hl_fast
    decay_med = np.log(2.0) / hl_med
    decay_macro = np.log(2.0) / hl_macro
    dfast = np.exp(-decay_fast)
    dmed = np.exp(-decay_med)
    dmacro = np.exp(-decay_macro)

    decay_markov = np.log(2.0) / 300.0
    dmarkov = np.exp(-decay_markov)

    decay_total = np.log(2.0) / 35.0
    dtotal = np.exp(-decay_total)

    # Incremental state structures
    open_trans = np.zeros((10, 10), dtype=float)
    close_trans = np.zeros((10, 10), dtype=float)
    cross_trans = np.zeros((10, 10), dtype=float)
    intra_mat = np.zeros((10, 10), dtype=float)
    jodi_trans = np.zeros((100, 100), dtype=float)

    day_open_counts = defaultdict(lambda: np.zeros(10, dtype=float))
    day_close_counts = defaultdict(lambda: np.zeros(10, dtype=float))

    rec_open_fast = np.zeros(10, dtype=float)
    rec_open_med = np.zeros(10, dtype=float)
    rec_open_macro = np.zeros(10, dtype=float)

    rec_close_fast = np.zeros(10, dtype=float)
    rec_close_med = np.zeros(10, dtype=float)
    rec_close_macro = np.zeros(10, dtype=float)

    rec_jodi_fast = np.zeros(100, dtype=float)
    rec_jodi_med = np.zeros(100, dtype=float)
    rec_jodi_macro = np.zeros(100, dtype=float)

    rec_total = np.zeros(10, dtype=float)

    open_gaps = np.zeros(10, dtype=float)
    close_gaps = np.zeros(10, dtype=float)
    total_gaps = np.zeros(10, dtype=float)

    # Warmup initialization
    print(f"[INFO] Initializing warmup window ({warmup_draws} draws)...")
    for i in range(warmup_draws):
        j = jodis[i]
        o = opens[i]
        c = closes[i]
        d = days[i]
        t_mod = (o + c) % 10

        day_open_counts[d][o] += 1.0
        day_close_counts[d][c] += 1.0
        intra_mat[o, c] += 1.0

        if i > 0:
            open_trans[opens[i - 1], o] += 1.0
            close_trans[closes[i - 1], c] += 1.0
            cross_trans[closes[i - 1], o] += 1.0
            jodi_trans[jodis[i - 1], j] += 1.0

        rec_open_fast *= dfast; rec_open_med *= dmed; rec_open_macro *= dmacro
        rec_close_fast *= dfast; rec_close_med *= dmed; rec_close_macro *= dmacro
        rec_jodi_fast *= dfast; rec_jodi_med *= dmed; rec_jodi_macro *= dmacro
        rec_total *= dtotal
        jodi_trans *= dmarkov

        rec_open_fast[o] += 1.0; rec_open_med[o] += 1.0; rec_open_macro[o] += 1.0
        rec_close_fast[c] += 1.0; rec_close_med[c] += 1.0; rec_close_macro[c] += 1.0
        rec_jodi_fast[j] += 1.0; rec_jodi_med[j] += 1.0; rec_jodi_macro[j] += 1.0
        rec_total[t_mod] += 1.0

        open_gaps += 1.0; open_gaps[o] = 0.0
        close_gaps += 1.0; close_gaps[c] = 0.0
        total_gaps += 1.0; total_gaps[t_mod] = 0.0

    # Backtesting statistics
    test_draws = n_total - warmup_draws
    print(f"[INFO] Running walk-forward out-of-sample backtest across {test_draws} draws...")
    start_time = time.time()

    otc_hits = 0
    top1_open_hits = 0
    top2_open_hits = 0
    top1_close_hits = 0
    top2_close_hits = 0

    top1_jodi_hits = 0
    top3_jodi_hits = 0
    top5_jodi_hits = 0
    top10_jodi_hits = 0
    top1_family_hits = 0

    # Precision & Ranking accumulators
    mrr_sum = 0.0
    brier_sum = 0.0

    # Financial & Drawdown tracking
    pnl_top1 = 0.0
    peak_top1 = 0.0
    max_dd_top1 = 0.0
    under_top1 = 0
    max_under_top1 = 0
    loss_streak_top1 = 0
    max_loss_streak_top1 = 0
    gross_prof_top1 = 0.0
    gross_loss_top1 = 0.0

    pnl_top5 = 0.0
    peak_top5 = 0.0
    max_dd_top5 = 0.0
    under_top5 = 0
    max_under_top5 = 0
    loss_streak_top5 = 0
    max_loss_streak_top5 = 0
    gross_prof_top5 = 0.0
    gross_loss_top5 = 0.0

    pnl_open = 0.0
    peak_open = 0.0
    max_dd_open = 0.0
    under_open = 0
    max_under_open = 0
    loss_streak_open = 0
    max_loss_streak_open = 0
    gross_prof_open = 0.0
    gross_loss_open = 0.0

    # Temporal regimes breakdown
    regimes = {
        "2014-2018": {"draws": 0, "otc_hits": 0, "open_hits": 0, "top5_hits": 0, "pnl_top5": 0.0},
        "2019-2022": {"draws": 0, "otc_hits": 0, "open_hits": 0, "top5_hits": 0, "pnl_top5": 0.0},
        "2023-2026": {"draws": 0, "otc_hits": 0, "open_hits": 0, "top5_hits": 0, "pnl_top5": 0.0},
        "Other": {"draws": 0, "otc_hits": 0, "open_hits": 0, "top5_hits": 0, "pnl_top5": 0.0},
    }

    # Weekday seasonality breakdown
    weekdays = {
        d: {"draws": 0, "otc_hits": 0, "open_hits": 0, "top1_hits": 0, "top5_hits": 0}
        for d in ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
    }

    for t in range(warmup_draws, n_total):
        prev_o = opens[t - 1]
        prev_c = closes[t - 1]
        prev_j = jodis[t - 1]
        d = days[t]
        act_j = jodis[t]
        act_o = opens[t]
        act_c = closes[t]

        # 1. Open Probabilities
        m_o1 = (open_trans[prev_o] + alpha_digit) / (open_trans[prev_o].sum() + 10 * alpha_digit)
        m_o2 = (cross_trans[prev_c] + alpha_digit) / (cross_trans[prev_c].sum() + 10 * alpha_digit)
        p_mk_o = 0.60 * m_o1 + 0.40 * m_o2

        r_f_o = (rec_open_fast + alpha_digit) / (rec_open_fast.sum() + 10 * alpha_digit)
        r_m_o = (rec_open_med + alpha_digit) / (rec_open_med.sum() + 10 * alpha_digit)
        r_mac_o = (rec_open_macro + alpha_digit) / (rec_open_macro.sum() + 10 * alpha_digit)
        p_rc_o = 0.50 * r_f_o + 0.35 * r_m_o + 0.15 * r_mac_o

        s_o = (day_open_counts[d] + alpha_digit) / (day_open_counts[d].sum() + 10 * alpha_digit)
        haz_o = np.maximum(1.0 + np.tanh((open_gaps - 10.0) / 4.0) * 0.40, 0.05)
        p_gp_o = haz_o / haz_o.sum()

        blend_o = w_mk * p_mk_o + w_rec * p_rc_o + w_sn * s_o + w_gp * p_gp_o
        open_probs = (1.0 - cut_res) * blend_o + cut_res * np.roll(blend_o, 5)
        open_probs /= open_probs.sum()

        # 2. Close Probabilities
        m_c = (close_trans[prev_c] + alpha_digit) / (close_trans[prev_c].sum() + 10 * alpha_digit)
        r_f_c = (rec_close_fast + alpha_digit) / (rec_close_fast.sum() + 10 * alpha_digit)
        r_m_c = (rec_close_med + alpha_digit) / (rec_close_med.sum() + 10 * alpha_digit)
        r_mac_c = (rec_close_macro + alpha_digit) / (rec_close_macro.sum() + 10 * alpha_digit)
        p_rc_c = 0.50 * r_f_c + 0.35 * r_m_c + 0.15 * r_mac_c

        s_c = (day_close_counts[d] + alpha_digit) / (day_close_counts[d].sum() + 10 * alpha_digit)
        haz_c = np.maximum(1.0 + np.tanh((close_gaps - 10.0) / 4.0) * 0.40, 0.05)
        p_gp_c = haz_c / haz_c.sum()

        blend_c = w_mk * m_c + w_rec * p_rc_c + w_sn * s_c + w_gp * p_gp_c
        close_probs = (1.0 - cut_res) * blend_c + cut_res * np.roll(blend_c, 5)
        close_probs /= close_probs.sum()

        # 3. 4-Ank OTC (Dual Cut Pairs)
        comb_ank = 0.55 * open_probs + 0.45 * close_probs
        rk_ank = np.argsort(comb_ank)[::-1]
        d1 = int(rk_ank[0])
        cut1 = (d1 + 5) % 10
        d2 = None
        for x in rk_ank[1:]:
            d_cand = int(x)
            if d_cand != d1 and d_cand != cut1:
                d2 = d_cand
                break
        if d2 is None:
            d2 = (d1 + 1) % 10
        cut2 = (d2 + 5) % 10
        otc = {d1, cut1, d2, cut2}

        is_otc_hit = (act_o in otc) or (act_c in otc)
        if is_otc_hit:
            otc_hits += 1

        # Open/Close Ank Rankings & Brier calibration
        rk_open = np.argsort(open_probs)[::-1]
        rk_close = np.argsort(close_probs)[::-1]

        is_top1_open = (act_o == rk_open[0])
        is_top2_open = (act_o in rk_open[:2])
        is_top1_close = (act_c == rk_close[0])
        is_top2_close = (act_c in rk_close[:2])

        if is_top1_open: top1_open_hits += 1
        if is_top2_open: top2_open_hits += 1
        if is_top1_close: top1_close_hits += 1
        if is_top2_close: top2_close_hits += 1

        # Brier score on Top-1 Open Ank prediction
        pred_top1_open = int(rk_open[0])
        prob_pred_open = float(open_probs[pred_top1_open])
        actual_open_binary = 1.0 if is_top1_open else 0.0
        brier_sum += (prob_pred_open - actual_open_binary) ** 2

        # 4. Joint Jodi Synthesis
        row_sums = intra_mat.sum(axis=1, keepdims=True)
        p_c_given_o = (intra_mat + 0.5) / (row_sums + 5.0)

        p_joint = np.zeros((10, 10), dtype=float)
        for oi in range(10):
            for ci in range(10):
                p_joint[oi, ci] = open_probs[oi] * (0.60 * close_probs[ci] + 0.40 * p_c_given_o[oi, ci])
        p_joint /= p_joint.sum()

        r_tot = (rec_total + 0.5) / (rec_total.sum() + 5.0)
        haz_tot = np.maximum(1.0 + np.tanh((total_gaps - 10.0) / 4.0) * 0.40, 0.05)
        p_tot = 0.70 * r_tot + 0.30 * (haz_tot / haz_tot.sum())
        p_tot /= p_tot.sum()

        p_total_matrix = np.zeros((10, 10), dtype=float)
        for oi in range(10):
            for ci in range(10):
                p_total_matrix[oi, ci] = p_tot[(oi + ci) % 10] / 10.0
        p_total_matrix /= p_total_matrix.sum()

        # Tri-Horizon Jodi Recency
        r_f_j = (rec_jodi_fast + alpha_jodi) / (rec_jodi_fast.sum() + 100 * alpha_jodi)
        r_m_j = (rec_jodi_med + alpha_jodi) / (rec_jodi_med.sum() + 100 * alpha_jodi)
        r_mac_j = (rec_jodi_macro + alpha_jodi) / (rec_jodi_macro.sum() + 100 * alpha_jodi)
        p_rec_j = (0.50 * r_f_j + 0.35 * r_m_j + 0.15 * r_mac_j).reshape((10, 10))
        p_rec_j /= p_rec_j.sum()

        # Markov Jodi Transitions
        c_j_mk = jodi_trans[prev_j]
        p_j_mk = (c_j_mk + alpha_jodi) / (c_j_mk.sum() + 100 * alpha_jodi)
        p_j_mk = (p_j_mk / p_j_mk.sum()).reshape((10, 10))

        # Synthesis
        jodi_matrix = (
            w_joint * p_joint
            + w_total * p_total_matrix
            + w_direct * p_rec_j
            + w_markov_jodi * p_j_mk
        )
        jodi_matrix /= jodi_matrix.sum()

        # 2D Harmonic Cut Resonance Diffusion across 3 cut states
        cut_2d_open = np.roll(jodi_matrix, 5, axis=0)
        cut_2d_close = np.roll(jodi_matrix, 5, axis=1)
        cut_2d_both = np.roll(jodi_matrix, (5, 5), axis=(0, 1))
        jodi_matrix = (1.0 - cut_res) * jodi_matrix + (cut_res / 3.0) * (cut_2d_open + cut_2d_close + cut_2d_both)
        jodi_matrix /= jodi_matrix.sum()

        jodi_prob = jodi_matrix.flatten()
        ranked_jodis = np.argsort(jodi_prob)[::-1]

        # Ranking & Mean Reciprocal Rank (MRR)
        rank_act_j = int(np.where(ranked_jodis == act_j)[0][0]) + 1
        mrr_sum += 1.0 / rank_act_j

        is_top1_jodi = (rank_act_j == 1)
        is_top3_jodi = (rank_act_j <= 3)
        is_top5_jodi = (rank_act_j <= 5)
        is_top10_jodi = (rank_act_j <= 10)

        if is_top1_jodi: top1_jodi_hits += 1
        if is_top3_jodi: top3_jodi_hits += 1
        if is_top5_jodi: top5_jodi_hits += 1
        if is_top10_jodi: top10_jodi_hits += 1

        # Top 1 family
        top_pick = ranked_jodis[0]
        top_o = top_pick // 10; top_c = top_pick % 10
        cut_to = (top_o + 5) % 10; cut_tc = (top_c + 5) % 10
        fam_pairs = {
            top_o * 10 + top_c, top_o * 10 + cut_tc,
            cut_to * 10 + top_c, cut_to * 10 + cut_tc,
            top_c * 10 + top_o, cut_tc * 10 + top_o,
            top_c * 10 + cut_to, cut_tc * 10 + cut_to,
        }
        if act_j in fam_pairs:
            top1_family_hits += 1

        # Financial P&L, Drawdown, Underwater Duration & Streaks
        # 1. Top-1 Jodi
        d_p1 = 89.0 if is_top1_jodi else -1.0
        pnl_top1 += d_p1
        if d_p1 > 0: gross_prof_top1 += d_p1
        else: gross_loss_top1 += abs(d_p1)
        if pnl_top1 > peak_top1: peak_top1 = pnl_top1
        dd1 = peak_top1 - pnl_top1
        if dd1 > max_dd_top1: max_dd_top1 = dd1
        if dd1 > 0: under_top1 += 1
        else: under_top1 = 0
        if under_top1 > max_under_top1: max_under_top1 = under_top1
        if d_p1 < 0: loss_streak_top1 += 1
        else: loss_streak_top1 = 0
        if loss_streak_top1 > max_loss_streak_top1: max_loss_streak_top1 = loss_streak_top1

        # 2. Top-5 Jodis
        d_p5 = (90.0 - 5.0) if is_top5_jodi else -5.0
        pnl_top5 += d_p5
        if d_p5 > 0: gross_prof_top5 += d_p5
        else: gross_loss_top5 += abs(d_p5)
        if pnl_top5 > peak_top5: peak_top5 = pnl_top5
        dd5 = peak_top5 - pnl_top5
        if dd5 > max_dd_top5: max_dd_top5 = dd5
        if dd5 > 0: under_top5 += 1
        else: under_top5 = 0
        if under_top5 > max_under_top5: max_under_top5 = under_top5
        if d_p5 < 0: loss_streak_top5 += 1
        else: loss_streak_top5 = 0
        if loss_streak_top5 > max_loss_streak_top5: max_loss_streak_top5 = loss_streak_top5

        # 3. Top-1 Open Ank (True equity curve tracking replacing N/A)
        d_po = 8.0 if is_top1_open else -1.0
        pnl_open += d_po
        if d_po > 0: gross_prof_open += d_po
        else: gross_loss_open += abs(d_po)
        if pnl_open > peak_open: peak_open = pnl_open
        ddo = peak_open - pnl_open
        if ddo > max_dd_open: max_dd_open = ddo
        if ddo > 0: under_open += 1
        else: under_open = 0
        if under_open > max_under_open: max_under_open = under_open
        if d_po < 0: loss_streak_open += 1
        else: loss_streak_open = 0
        if loss_streak_open > max_loss_streak_open: max_loss_streak_open = loss_streak_open

        # Temporal Regime breakdown
        date_str = str(dates[t])
        yr = int(date_str[:4]) if len(date_str) >= 4 and date_str[:4].isdigit() else 2026
        if 2014 <= yr <= 2018: reg_key = "2014-2018"
        elif 2019 <= yr <= 2022: reg_key = "2019-2022"
        elif 2023 <= yr <= 2026: reg_key = "2023-2026"
        else: reg_key = "Other"

        regimes[reg_key]["draws"] += 1
        if is_otc_hit: regimes[reg_key]["otc_hits"] += 1
        if is_top1_open: regimes[reg_key]["open_hits"] += 1
        if is_top5_jodi: regimes[reg_key]["top5_hits"] += 1
        regimes[reg_key]["pnl_top5"] += d_p5

        # Weekday seasonality breakdown
        if d in weekdays:
            weekdays[d]["draws"] += 1
            if is_otc_hit: weekdays[d]["otc_hits"] += 1
            if is_top1_open: weekdays[d]["open_hits"] += 1
            if is_top1_jodi: weekdays[d]["top1_hits"] += 1
            if is_top5_jodi: weekdays[d]["top5_hits"] += 1

        # Online step update
        day_open_counts[d][act_o] += 1.0
        day_close_counts[d][act_c] += 1.0
        intra_mat[act_o, act_c] += 1.0

        open_trans[prev_o, act_o] += 1.0
        close_trans[prev_c, act_c] += 1.0
        cross_trans[prev_c, act_o] += 1.0
        jodi_trans[prev_j, act_j] += 1.0

        rec_open_fast *= dfast; rec_open_med *= dmed; rec_open_macro *= dmacro
        rec_close_fast *= dfast; rec_close_med *= dmed; rec_close_macro *= dmacro
        rec_jodi_fast *= dfast; rec_jodi_med *= dmed; rec_jodi_macro *= dmacro
        rec_total *= dtotal
        jodi_trans *= dmarkov

        rec_open_fast[act_o] += 1.0; rec_open_med[act_o] += 1.0; rec_open_macro[act_o] += 1.0
        rec_close_fast[act_c] += 1.0; rec_close_med[act_c] += 1.0; rec_close_macro[act_c] += 1.0
        rec_jodi_fast[act_j] += 1.0; rec_jodi_med[act_j] += 1.0; rec_jodi_macro[act_j] += 1.0
        rec_total[(act_o + act_c) % 10] += 1.0

        open_gaps += 1.0; open_gaps[act_o] = 0.0
        close_gaps += 1.0; close_gaps[act_c] = 0.0
        total_gaps += 1.0; total_gaps[(act_o + act_c) % 10] = 0.0

    elapsed = time.time() - start_time
    print(f"[INFO] Backtest completed in {elapsed:.2f} seconds.")

    # 1. Predictive Accuracy Table with Binomial Z-score and p-value
    accuracy_specs = [
        ("4-Ank OTC (Dual Cut Pair)", otc_hits, 0.64),
        ("Top-1 Jodi", top1_jodi_hits, 0.01),
        ("Top-3 Jodis", top3_jodi_hits, 0.03),
        ("Top-5 Jodis", top5_jodi_hits, 0.05),
        ("Top-10 Jodis", top10_jodi_hits, 0.10),
        ("Top-1 Jodi Cut-Family (8)", top1_family_hits, 0.08),
        ("Top-1 Open Ank", top1_open_hits, 0.10),
        ("Top-2 Open Anks", top2_open_hits, 0.20),
        ("Top-1 Close Ank", top1_close_hits, 0.10),
        ("Top-2 Close Anks", top2_close_hits, 0.20),
    ]

    metrics = []
    for label, hits, p0 in accuracy_specs:
        rate = hits / test_draws
        edge = (rate / p0 - 1.0) * 100.0
        z, p_val = calc_binomial_stats(hits, test_draws, p0)
        p_val_str = "< 0.0001" if p_val < 0.0001 else f"{p_val:.4f}"
        metrics.append({
            "Prediction Target": label,
            "Observed Hit Rate": f"{rate * 100:.2f}% ({hits}/{test_draws})",
            "Theoretical Baseline": f"{p0 * 100:.2f}%",
            "Relative Edge": f"{edge:+.1f}%",
            "Z-Score": f"{z:+.2f}",
            "p-value": p_val_str,
        })

    # 2. Jodi Ranking & Precision@K Table
    mrr = mrr_sum / test_draws
    mrr_baseline = sum(1.0 / r for r in range(1, 101)) / 100.0  # ~0.05187
    precision_table = [
        {
            "Ranking Metric": "Precision@1 (Top-1 Jodi)",
            "Observed Value": f"{top1_jodi_hits / test_draws * 100:.2f}% ({top1_jodi_hits}/{test_draws})",
            "Random Baseline": "1.00%",
            "Relative Edge": f"{(top1_jodi_hits / test_draws / 0.01 - 1) * 100:+.1f}%",
        },
        {
            "Ranking Metric": "Precision@3 (Top-3 Jodis)",
            "Observed Value": f"{top3_jodi_hits / test_draws * 100:.2f}% ({top3_jodi_hits}/{test_draws})",
            "Random Baseline": "3.00%",
            "Relative Edge": f"{(top3_jodi_hits / test_draws / 0.03 - 1) * 100:+.1f}%",
        },
        {
            "Ranking Metric": "Precision@5 (Top-5 Jodis)",
            "Observed Value": f"{top5_jodi_hits / test_draws * 100:.2f}% ({top5_jodi_hits}/{test_draws})",
            "Random Baseline": "5.00%",
            "Relative Edge": f"{(top5_jodi_hits / test_draws / 0.05 - 1) * 100:+.1f}%",
        },
        {
            "Ranking Metric": "Precision@10 (Top-10 Jodis)",
            "Observed Value": f"{top10_jodi_hits / test_draws * 100:.2f}% ({top10_jodi_hits}/{test_draws})",
            "Random Baseline": "10.00%",
            "Relative Edge": f"{(top10_jodi_hits / test_draws / 0.10 - 1) * 100:+.1f}%",
        },
        {
            "Ranking Metric": "Mean Reciprocal Rank (MRR)",
            "Observed Value": f"{mrr:.4f} ({mrr * 100:.2f}%)",
            "Random Baseline": f"{mrr_baseline:.4f} ({mrr_baseline * 100:.2f}%)",
            "Relative Edge": f"{(mrr / mrr_baseline - 1) * 100:+.1f}%",
        },
    ]

    # 3. Probability Calibration (Brier Score)
    brier_score = brier_sum / test_draws
    brier_baseline = 0.0900  # 0.1*(0.9)^2 + 0.9*(0.1)^2
    brier_calib = "Calibrated" if brier_score <= brier_baseline else "Sub-calibrated"
    calibration_table = [
        {
            "Calibration Target": "Top-1 Open Ank",
            "Observed Brier Score": f"{brier_score:.4f}",
            "Random Reference": f"{brier_baseline:.4f}",
            "Calibration Assessment": f"{brier_calib} ({((brier_baseline - brier_score) / brier_baseline) * 100:+.1f}% error reduction)",
        }
    ]

    # 4. Financial & Drawdown Risk Metrics Table
    pf_top1 = (gross_prof_top1 / gross_loss_top1) if gross_loss_top1 > 0 else 0.0
    pf_top5 = (gross_prof_top5 / gross_loss_top5) if gross_loss_top5 > 0 else 0.0
    pf_open = (gross_prof_open / gross_loss_open) if gross_loss_open > 0 else 0.0

    financials = [
        {
            "Betting Strategy": "Top-1 Jodi (1 unit/draw)",
            "Total Wagered": f"{test_draws} units",
            "Net P&L": f"{pnl_top1:+.1f} units",
            "ROI": f"{pnl_top1 / test_draws * 100:.2f}%",
            "Max Drawdown": f"{max_dd_top1:.1f} units",
            "Max DD Duration": f"{max_under_top1} draws",
            "Max Loss Streak": f"{max_loss_streak_top1} draws",
            "Profit Factor": f"{pf_top1:.2f}",
        },
        {
            "Betting Strategy": "Top-5 Jodis (5 units/draw)",
            "Total Wagered": f"{test_draws * 5} units",
            "Net P&L": f"{pnl_top5:+.1f} units",
            "ROI": f"{pnl_top5 / (test_draws * 5) * 100:.2f}%",
            "Max Drawdown": f"{max_dd_top5:.1f} units",
            "Max DD Duration": f"{max_under_top5} draws",
            "Max Loss Streak": f"{max_loss_streak_top5} draws",
            "Profit Factor": f"{pf_top5:.2f}",
        },
        {
            "Betting Strategy": "Top-1 Open Ank (1 unit/draw)",
            "Total Wagered": f"{test_draws} units",
            "Net P&L": f"{pnl_open:+.1f} units",
            "ROI": f"{pnl_open / test_draws * 100:.2f}%",
            "Max Drawdown": f"{max_dd_open:.1f} units",
            "Max DD Duration": f"{max_under_open} draws",
            "Max Loss Streak": f"{max_loss_streak_open} draws",
            "Profit Factor": f"{pf_open:.2f}",
        },
    ]

    # 5. Temporal Regime Table
    regime_rows = []
    for reg_name in ["2014-2018", "2019-2022", "2023-2026", "Other"]:
        r_info = regimes[reg_name]
        if r_info["draws"] > 0:
            n_d = r_info["draws"]
            regime_rows.append({
                "Regime Epoch": reg_name,
                "Test Draws": n_d,
                "4-Ank OTC Hit Rate": f"{r_info['otc_hits'] / n_d * 100:.2f}%",
                "Top-1 Open Hit Rate": f"{r_info['open_hits'] / n_d * 100:.2f}%",
                "Top-5 Jodi Hit Rate": f"{r_info['top5_hits'] / n_d * 100:.2f}%",
                "Top-5 Net P&L": f"{r_info['pnl_top5']:+.1f} units",
                "Top-5 ROI": f"{r_info['pnl_top5'] / (n_d * 5) * 100:.2f}%",
            })

    # 6. Weekday Seasonality Table
    weekday_rows = []
    for d_name in ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]:
        w_info = weekdays[d_name]
        if w_info["draws"] > 0:
            n_w = w_info["draws"]
            weekday_rows.append({
                "Day of Week": d_name,
                "Draws": n_w,
                "4-Ank OTC Hit Rate": f"{w_info['otc_hits'] / n_w * 100:.2f}%",
                "Top-1 Open Hit Rate": f"{w_info['open_hits'] / n_w * 100:.2f}%",
                "Top-1 Jodi Hit Rate": f"{w_info['top1_hits'] / n_w * 100:.2f}%",
                "Top-5 Jodi Hit Rate": f"{w_info['top5_hits'] / n_w * 100:.2f}%",
            })

    # Print comprehensive backtest report
    print("\n" + "=" * 80)
    print("      KALYAN MATKA OUT-OF-SAMPLE WALK-FORWARD BACKTEST & STATISTICAL AUDIT")
    print("=" * 80)
    print(f"Warmup Training Window:     {warmup_draws} draws")
    print(f"Out-of-Sample Test Window:  {test_draws} draws ({dates[warmup_draws]} to {dates[-1]})")
    print("Evaluation Methodology:     Strict walk-forward (Predict draw t using 0 to t-1)")
    print("-" * 80)

    print("\n[ACCURACY] PREDICTIVE ACCURACY VS THEORETICAL RANDOM BASELINE (WITH BINOMIAL SIGNIFICANCE):")
    print(tabulate(metrics, headers="keys", tablefmt="github"))

    print("\n[RANKING] JODI RANKING PRECISION & MEAN RECIPROCAL RANK (MRR):")
    print(tabulate(precision_table, headers="keys", tablefmt="github"))

    print("\n[CALIBRATION] PROBABILITY CALIBRATION (BRIER SCORE):")
    print(tabulate(calibration_table, headers="keys", tablefmt="github"))

    print("\n[FINANCIALS] FINANCIAL EXPECTED VALUE & DRAWDOWN RISK METRICS:")
    print(tabulate(financials, headers="keys", tablefmt="github"))

    if regime_rows:
        print("\n[REGIMES] TEMPORAL REGIME BREAKDOWN (MULTI-YEAR EPOCHS):")
        print(tabulate(regime_rows, headers="keys", tablefmt="github"))

    if weekday_rows:
        print("\n[SEASONALITY] WEEKDAY SEASONALITY PERFORMANCE BREAKDOWN (MON-SAT):")
        print(tabulate(weekday_rows, headers="keys", tablefmt="github"))

    print("\n" + "=" * 80)
    print("               DATA ENGINEERING CONCLUSION & INSIGHTS")
    print("=" * 80)
    print("1. Hit-Rate Analysis: Model heuristics improve hit rates relative to random")
    print("   chance by identifying persistent empirical patterns.")
    print("2. The House Edge Reality: Because bookmakers retain a -10% house margin,")
    print("   no betting strategy on arbitrary lottery draws can deliver guaranteed profits.")
    print("3. Responsible Gaming: Treat predictions as mathematical probability estimates,")
    print("   never as guaranteed outcomes.")
    print("=" * 80 + "\n")

    return {
        "test_draws": test_draws,
        "otc_pass_rate": otc_hits / test_draws,
        "top1_jodi_hit_rate": top1_jodi_hits / test_draws,
        "top3_jodi_hit_rate": top3_jodi_hits / test_draws,
        "top5_jodi_hit_rate": top5_jodi_hits / test_draws,
        "top10_jodi_hit_rate": top10_jodi_hits / test_draws,
        "top1_open_hit_rate": top1_open_hits / test_draws,
        "top2_open_hit_rate": top2_open_hits / test_draws,
        "top1_close_hit_rate": top1_close_hits / test_draws,
        "top2_close_hit_rate": top2_close_hits / test_draws,
        "pnl_top1": pnl_top1,
        "pnl_top5": pnl_top5,
        "pnl_open": pnl_open,
        "mrr": mrr,
        "brier_score": brier_score,
        "max_dd_top1": max_dd_top1,
        "max_dd_top5": max_dd_top5,
        "max_dd_open": max_dd_open,
        "profit_factor_top1": pf_top1,
        "profit_factor_top5": pf_top5,
        "profit_factor_open": pf_open,
    }


def run_parameter_optimization(csv_path: str = None):
    """Automated grid search calibration to discover highest-accuracy weights."""
    print("[INFO] Running Automated Parameter Calibration...")
    target_csv = resolve_csv_path(csv_path)
    df = pd.read_csv(target_csv, dtype={"Jodi": str, "Open_Patti": str, "Close_Patti": str})
    valid_df = df[df["Is_Valid"] == True].copy().reset_index(drop=True)
    opens = valid_df["Open_Digit"].apply(lambda x: int(float(x))).values
    jodis = valid_df["Jodi"].apply(lambda x: int(float(x))).values
    n = len(valid_df)
    warmup = 500

    best_score = -9999.0
    best_params = {}

    for hl in [6.0, 8.0, 12.0]:
        for alpha in [0.15, 0.25, 0.40]:
            decay = np.log(2.0) / hl
            dfactor = np.exp(-decay)
            rec_o = np.zeros(10)
            rec_j = np.zeros(100)

            for i in range(warmup):
                rec_o *= dfactor; rec_o[opens[i]] += 1.0
                rec_j *= dfactor; rec_j[jodis[i]] += 1.0

            hits_o = 0
            hits_j = 0
            test_n = n - warmup
            for t in range(warmup, n):
                po = (rec_o + alpha) / (rec_o.sum() + 10 * alpha)
                pj = (rec_j + 0.05) / (rec_j.sum() + 100 * 0.05)
                if opens[t] == np.argmax(po): hits_o += 1
                if jodis[t] in np.argsort(pj)[::-1][:5]: hits_j += 1
                rec_o *= dfactor; rec_o[opens[t]] += 1.0
                rec_j *= dfactor; rec_j[jodis[t]] += 1.0

            score = (hits_o / test_n * 100) + (hits_j / test_n * 100)
            if score > best_score:
                best_score = score
                best_params = {
                    "hl_fast": hl,
                    "alpha_digit": alpha,
                    "open_hit_rate": hits_o / test_n * 100,
                    "top5_jodi_hit_rate": hits_j / test_n * 100,
                }

    print(f"[OPTIMIZER RESULT] Optimal Parameters Found:")
    print(f"  - Fast Momentum Half-Life: {best_params['hl_fast']} draws")
    print(f"  - Dirichlet Smoothing Alpha: {best_params['alpha_digit']}")
    print(f"  - Top-1 Open Hit Rate: {best_params['open_hit_rate']:.2f}% (vs 10.0% random)")
    print(f"  - Top-5 Jodi Hit Rate: {best_params['top5_jodi_hit_rate']:.2f}% (vs 5.0% random)")
    return best_params


def print_backtest_report(res=None):
    """Compatibility helper: backtest results are formatted directly during execution."""
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kalyan Walk-Forward Backtester & Optimizer")
    parser.add_argument("csv_pos", nargs="?", default=None, help="Optional historical CSV file path")
    parser.add_argument("--csv", default=None, help="Optional historical CSV file path")
    parser.add_argument("--optimize", action="store_true", help="Run automated hyperparameter optimization")
    args = parser.parse_args()

    input_csv = args.csv or args.csv_pos
    resolved_csv = resolve_csv_path(input_csv)

    if args.optimize:
        run_parameter_optimization(csv_path=resolved_csv)
    else:
        run_walk_forward_backtest(csv_path=resolved_csv)
