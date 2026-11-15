"""
Quantitative Predictive Models for Kalyan Matka Historical Analysis.
Implements:
1. First-Order Markov Transition Matrices with Adaptive Smoothing and Cross-Day Close->Open transitions.
2. Dual-Horizon Recency Weighting (Fast Momentum + Medium Trend Exponential Decay).
3. Overdue / Gap Cycle Hazard Model (Mean-reverting cyclical return hazard).
4. Target Day-of-Week Seasonality Distributions (Weekday Line chart priors).
5. Intra-Draw Conditional Matrix P(Close | Open).
6. Jodi Total (Sum Mod 10) Pattern Prior.
7. Cut-Digit (Adding 5 mod 10) Harmonic Resonance.
8. Empirical 3-Digit Panel/Patti Engine (Single Patti SP and Double Patti DP).
9. Harmonic Dual Cut-Pair 4-Ank OTC (Open To Close) Generator with Pass Likelihood.
10. Coherent Marginal-Joint Probabilistic Synthesis for Top Jodis and Family Sets.
"""

import numpy as np
import pandas as pd
from collections import defaultdict, Counter


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


def sanitize_draw_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sanitizes historical Matka draw DataFrames for training and inference.
    - Filters out invalid rows (e.g. market holidays where Is_Valid is False)
    - Drops rows where Jodi, Open_Digit, or Close_Digit is NaN or empty
    - Validates numeric convertibility
    - Resets DataFrame index to maintain contiguous sequence order
    """
    if df is None or not isinstance(df, pd.DataFrame) or len(df) == 0:
        return pd.DataFrame()
    clean = df.copy()
    if "Is_Valid" in clean.columns:
        clean = clean[clean["Is_Valid"].isin([True, 1, "True", "true", "TRUE", "1"])]
    subset_cols = [c for c in ["Jodi", "Open_Digit", "Close_Digit"] if c in clean.columns]
    if subset_cols:
        clean = clean.dropna(subset=subset_cols)
        for col in subset_cols:
            clean = clean[pd.to_numeric(clean[col], errors="coerce").notna()]
    return clean.reset_index(drop=True)


class MarkovChainModel:
    """
    First-order Markov model computing conditional transition probabilities
    P(Draw_t | Draw_{t-1}) for both Jodi numbers and Open/Close single digits,
    including Cross-Day Close(t-1) -> Open(t) transition dynamics.
    Uses calibrated Laplace-Dirichlet smoothing and exponential time decay
    so recent transitions carry higher predictive priority.
    """

    def __init__(self, alpha_digit: float = 0.5, alpha_jodi: float = 0.1, half_life: float = 300.0):
        self.alpha_digit = alpha_digit
        self.alpha_jodi = alpha_jodi
        self.half_life = half_life
        self.decay = np.log(2.0) / half_life if half_life > 0 else 0.0
        self.jodi_trans = defaultdict(Counter)
        self.open_trans = np.zeros((10, 10), dtype=float)
        self.close_trans = np.zeros((10, 10), dtype=float)
        self.cross_trans = np.zeros((10, 10), dtype=float)  # Close(t-1) -> Open(t)
        self.last_jodi = None
        self.last_open = None
        self.last_close = None

    def fit(self, valid_df: pd.DataFrame):
        valid_df = sanitize_draw_data(valid_df)
        self.jodi_trans.clear()
        self.open_trans.fill(0.0)
        self.close_trans.fill(0.0)
        self.cross_trans.fill(0.0)

        n = len(valid_df)
        if n == 0:
            return

        jodis = valid_df["Jodi"].apply(lambda x: int(float(x))).values
        opens = valid_df["Open_Digit"].apply(lambda x: int(float(x))).values
        closes = valid_df["Close_Digit"].apply(lambda x: int(float(x))).values

        if n > 0:
            self.last_jodi = f"{jodis[-1]:02d}"
            self.last_open = int(opens[-1])
            self.last_close = int(closes[-1])

        if n < 2:
            return

        lags = np.arange(n - 2, -1, -1)
        w = np.exp(-self.decay * lags) if self.decay > 0 else np.ones(n - 1)

        for i in range(n - 1):
            weight = float(w[i])
            self.jodi_trans[f"{jodis[i]:02d}"][f"{jodis[i + 1]:02d}"] += weight
            self.open_trans[opens[i], opens[i + 1]] += weight
            self.close_trans[closes[i], closes[i + 1]] += weight
            self.cross_trans[closes[i], opens[i + 1]] += weight

    def predict_jodi_probs(self, prev_jodi: str = None) -> np.ndarray:
        prev = prev_jodi if prev_jodi is not None else self.last_jodi
        if prev is not None and prev in self.jodi_trans:
            counts = np.array([self.jodi_trans[prev][f"{i:02d}"] for i in range(100)], dtype=float)
        else:
            counts = np.zeros(100, dtype=float)
        probs = (counts + self.alpha_jodi) / (counts.sum() + 100 * self.alpha_jodi)
        return probs / probs.sum()

    def predict_open_probs(self, prev_open: int = None, prev_close: int = None) -> np.ndarray:
        p_o = int(prev_open) if prev_open is not None else self.last_open
        p_c = int(prev_close) if prev_close is not None else self.last_close

        if p_o is not None and 0 <= p_o <= 9:
            counts_o = self.open_trans[p_o]
            p1 = (counts_o + self.alpha_digit) / (counts_o.sum() + 10 * self.alpha_digit)
        else:
            p1 = np.ones(10, dtype=float) / 10.0

        if p_c is not None and 0 <= p_c <= 9:
            counts_c = self.cross_trans[p_c]
            p2 = (counts_c + self.alpha_digit) / (counts_c.sum() + 10 * self.alpha_digit)
        else:
            p2 = np.ones(10, dtype=float) / 10.0

        probs = 0.60 * p1 + 0.40 * p2
        return probs / probs.sum()

    def predict_close_probs(self, prev_close: int = None) -> np.ndarray:
        prev = int(prev_close) if prev_close is not None else self.last_close
        if prev is None or prev < 0 or prev > 9:
            return np.ones(10, dtype=float) / 10.0
        counts = self.close_trans[prev]
        probs = (counts + self.alpha_digit) / (counts.sum() + 10 * self.alpha_digit)
        return probs / probs.sum()


class RecencyWeightedModel:
    """
    Tri-Horizon Exponential Moving Frequency model (Fast Momentum + Medium Trend + Macro Baseline).
    Applies calibrated exponential decay:
      w_fast  = e^(-lambda_fast * delta_t)   (Half-life = 8 draws)
      w_med   = e^(-lambda_med * delta_t)    (Half-life = 35 draws)
      w_macro = e^(-lambda_macro * delta_t)  (Half-life = 120 draws)
    Applies adaptive tri-horizon blending to Jodis, Open digits, and Close digits:
      P_rec = 0.50 * P_fast + 0.35 * P_med + 0.15 * P_macro
    """

    def __init__(
        self,
        half_life_fast: float = 8.0,
        half_life_med: float = 35.0,
        half_life_macro: float = 120.0,
        alpha: float = 0.5,
        alpha_jodi: float = 0.05,
    ):
        self.hl_fast = half_life_fast
        self.hl_med = half_life_med
        self.hl_macro = half_life_macro
        self.decay_fast = np.log(2.0) / half_life_fast
        self.decay_med = np.log(2.0) / half_life_med
        self.decay_macro = np.log(2.0) / half_life_macro
        self.alpha = alpha
        self.alpha_jodi = alpha_jodi

        self.jodi_weights_fast = np.zeros(100, dtype=float)
        self.jodi_weights_med = np.zeros(100, dtype=float)
        self.jodi_weights_macro = np.zeros(100, dtype=float)

        self.open_weights_fast = np.zeros(10, dtype=float)
        self.open_weights_med = np.zeros(10, dtype=float)
        self.open_weights_macro = np.zeros(10, dtype=float)

        self.close_weights_fast = np.zeros(10, dtype=float)
        self.close_weights_med = np.zeros(10, dtype=float)
        self.close_weights_macro = np.zeros(10, dtype=float)

    @property
    def jodi_weights(self) -> np.ndarray:
        return 0.50 * self.jodi_weights_fast + 0.35 * self.jodi_weights_med + 0.15 * self.jodi_weights_macro

    def fit(self, valid_df: pd.DataFrame):
        valid_df = sanitize_draw_data(valid_df)
        n = len(valid_df)
        self.jodi_weights_fast.fill(0.0)
        self.jodi_weights_med.fill(0.0)
        self.jodi_weights_macro.fill(0.0)

        self.open_weights_fast.fill(0.0)
        self.open_weights_med.fill(0.0)
        self.open_weights_macro.fill(0.0)

        self.close_weights_fast.fill(0.0)
        self.close_weights_med.fill(0.0)
        self.close_weights_macro.fill(0.0)

        if n == 0:
            return

        jodis = valid_df["Jodi"].apply(lambda x: int(float(x))).values
        opens = valid_df["Open_Digit"].apply(lambda x: int(float(x))).values
        closes = valid_df["Close_Digit"].apply(lambda x: int(float(x))).values

        lags = np.arange(n - 1, -1, -1)
        w_f = np.exp(-self.decay_fast * lags)
        w_m = np.exp(-self.decay_med * lags)
        w_mac = np.exp(-self.decay_macro * lags)

        for i in range(n):
            self.jodi_weights_fast[jodis[i]] += w_f[i]
            self.jodi_weights_med[jodis[i]] += w_m[i]
            self.jodi_weights_macro[jodis[i]] += w_mac[i]

            self.open_weights_fast[opens[i]] += w_f[i]
            self.open_weights_med[opens[i]] += w_m[i]
            self.open_weights_macro[opens[i]] += w_mac[i]

            self.close_weights_fast[closes[i]] += w_f[i]
            self.close_weights_med[closes[i]] += w_m[i]
            self.close_weights_macro[closes[i]] += w_mac[i]

    def predict_jodi_probs(self) -> np.ndarray:
        p_fast = (self.jodi_weights_fast + self.alpha_jodi) / (self.jodi_weights_fast.sum() + 100 * self.alpha_jodi)
        p_med = (self.jodi_weights_med + self.alpha_jodi) / (self.jodi_weights_med.sum() + 100 * self.alpha_jodi)
        p_mac = (self.jodi_weights_macro + self.alpha_jodi) / (self.jodi_weights_macro.sum() + 100 * self.alpha_jodi)
        probs = 0.50 * p_fast + 0.35 * p_med + 0.15 * p_mac
        return probs / probs.sum()

    def predict_open_probs(self) -> np.ndarray:
        p_fast = (self.open_weights_fast + self.alpha) / (self.open_weights_fast.sum() + 10 * self.alpha)
        p_med = (self.open_weights_med + self.alpha) / (self.open_weights_med.sum() + 10 * self.alpha)
        p_mac = (self.open_weights_macro + self.alpha) / (self.open_weights_macro.sum() + 10 * self.alpha)
        probs = 0.50 * p_fast + 0.35 * p_med + 0.15 * p_mac
        return probs / probs.sum()

    def predict_close_probs(self) -> np.ndarray:
        p_fast = (self.close_weights_fast + self.alpha) / (self.close_weights_fast.sum() + 10 * self.alpha)
        p_med = (self.close_weights_med + self.alpha) / (self.close_weights_med.sum() + 10 * self.alpha)
        p_mac = (self.close_weights_macro + self.alpha) / (self.close_weights_macro.sum() + 10 * self.alpha)
        probs = 0.50 * p_fast + 0.35 * p_med + 0.15 * p_mac
        return probs / probs.sum()



class OverdueGapModel:
    """
    Overdue / Gap Cycle Hazard Model.
    Tracks elapsed draws since each Single Ank (0-9) and Jodi Total (0-9) last appeared.
    Applies a cyclical mean-reverting hazard score to capture numbers due to return.
    """

    def __init__(self, target_mean_gap: float = 10.0, gap_scale: float = 4.0):
        self.target_mean = target_mean_gap
        self.gap_scale = gap_scale
        self.open_gaps = np.zeros(10, dtype=float)
        self.close_gaps = np.zeros(10, dtype=float)
        self.total_gaps = np.zeros(10, dtype=float)

    def fit(self, valid_df: pd.DataFrame):
        valid_df = sanitize_draw_data(valid_df)
        self.open_gaps.fill(0.0)
        self.close_gaps.fill(0.0)
        self.total_gaps.fill(0.0)

        n = len(valid_df)
        if n == 0:
            return

        opens = valid_df["Open_Digit"].apply(lambda x: int(float(x))).values
        closes = valid_df["Close_Digit"].apply(lambda x: int(float(x))).values

        last_open_seen = {d: -1 for d in range(10)}
        last_close_seen = {d: -1 for d in range(10)}
        last_total_seen = {d: -1 for d in range(10)}

        for i in range(n):
            last_open_seen[opens[i]] = i
            last_close_seen[closes[i]] = i
            t = (opens[i] + closes[i]) % 10
            last_total_seen[t] = i

        for d in range(10):
            self.open_gaps[d] = (n - 1 - last_open_seen[d]) if last_open_seen[d] >= 0 else n
            self.close_gaps[d] = (n - 1 - last_close_seen[d]) if last_close_seen[d] >= 0 else n
            self.total_gaps[d] = (n - 1 - last_total_seen[d]) if last_total_seen[d] >= 0 else n

    def predict_open_probs(self) -> np.ndarray:
        hazard = 1.0 + np.tanh((self.open_gaps - self.target_mean) / self.gap_scale) * 0.40
        probs = np.maximum(hazard, 0.05)
        return probs / probs.sum()

    def predict_close_probs(self) -> np.ndarray:
        hazard = 1.0 + np.tanh((self.close_gaps - self.target_mean) / self.gap_scale) * 0.40
        probs = np.maximum(hazard, 0.05)
        return probs / probs.sum()

    def predict_total_probs(self) -> np.ndarray:
        hazard = 1.0 + np.tanh((self.total_gaps - self.target_mean) / self.gap_scale) * 0.40
        probs = np.maximum(hazard, 0.05)
        return probs / probs.sum()


class DayOfWeekSeasonalModel:
    """
    Conditioned on the Day of Week (Mon, Tue, Wed, Thu, Fri, Sat).
    Calculates historical frequency specifically for the upcoming target weekday.
    """

    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
        self.day_jodi_counts = defaultdict(lambda: np.zeros(100, dtype=float))
        self.day_open_counts = defaultdict(lambda: np.zeros(10, dtype=float))
        self.day_close_counts = defaultdict(lambda: np.zeros(10, dtype=float))

    def fit(self, valid_df: pd.DataFrame):
        valid_df = sanitize_draw_data(valid_df)
        self.day_jodi_counts.clear()
        self.day_open_counts.clear()
        self.day_close_counts.clear()

        if len(valid_df) == 0:
            return

        day_col = "Day_Of_Week" if "Day_Of_Week" in valid_df.columns else "Day"
        for _, row in valid_df.iterrows():
            day = row[day_col]
            j = int(float(row["Jodi"]))
            o = int(float(row["Open_Digit"]))
            c = int(float(row["Close_Digit"]))
            self.day_jodi_counts[day][j] += 1.0
            self.day_open_counts[day][o] += 1.0
            self.day_close_counts[day][c] += 1.0

    def predict_jodi_probs(self, day_of_week: str) -> np.ndarray:
        counts = self.day_jodi_counts[day_of_week]
        total = counts.sum()
        if total == 0:
            return np.ones(100, dtype=float) / 100.0
        return (counts + 0.1) / (total + 100 * 0.1)

    def predict_open_probs(self, day_of_week: str) -> np.ndarray:
        counts = self.day_open_counts[day_of_week]
        total = counts.sum()
        if total == 0:
            return np.ones(10, dtype=float) / 10.0
        return (counts + self.alpha) / (total + 10 * self.alpha)

    def predict_close_probs(self, day_of_week: str) -> np.ndarray:
        counts = self.day_close_counts[day_of_week]
        total = counts.sum()
        if total == 0:
            return np.ones(10, dtype=float) / 10.0
        return (counts + self.alpha) / (total + 10 * self.alpha)


class IntraDrawConditionalModel:
    """
    Calculates empirical conditional transition probabilities P(Close = c | Open = o)
    within the same day's draw with Laplace smoothing.
    """

    def __init__(self, alpha: float = 0.8):
        self.alpha = alpha
        self.matrix = np.zeros((10, 10), dtype=float)

    def fit(self, valid_df: pd.DataFrame):
        valid_df = sanitize_draw_data(valid_df)
        self.matrix.fill(0.0)
        if len(valid_df) == 0:
            return
        opens = valid_df["Open_Digit"].apply(lambda x: int(float(x))).values
        closes = valid_df["Close_Digit"].apply(lambda x: int(float(x))).values
        for o, c in zip(opens, closes):
            self.matrix[o, c] += 1.0

    def get_conditional_close_matrix(self) -> np.ndarray:
        row_sums = self.matrix.sum(axis=1, keepdims=True)
        return (self.matrix + self.alpha) / (row_sums + 10.0 * self.alpha)


class JodiTotalModel:
    """
    Tracks exponential moving probability of Jodi Totals: Total = (Open + Close) mod 10.
    In Kalyan Matka, specific total brackets experience cyclical clustering.
    """

    def __init__(self, half_life_draws: float = 35.0, alpha: float = 0.5):
        self.alpha = alpha
        self.decay = np.log(2.0) / half_life_draws
        self.total_weights = np.zeros(10, dtype=float)

    def fit(self, valid_df: pd.DataFrame):
        valid_df = sanitize_draw_data(valid_df)
        self.total_weights.fill(0.0)
        n = len(valid_df)
        if n == 0:
            return
        opens = valid_df["Open_Digit"].apply(lambda x: int(float(x))).values
        closes = valid_df["Close_Digit"].apply(lambda x: int(float(x))).values
        totals = (opens + closes) % 10

        lags = np.arange(n - 1, -1, -1)
        weights = np.exp(-self.decay * lags)
        for i in range(n):
            self.total_weights[totals[i]] += weights[i]

    def predict_total_probs(self) -> np.ndarray:
        probs = (self.total_weights + self.alpha) / (self.total_weights.sum() + 10.0 * self.alpha)
        return probs / probs.sum()


class PattiPanelModel:
    """
    Empirical Panel (Patti) Prediction Engine for Kalyan Matka:
    Maps each Ank A (0-9) to standard 3-digit panels.
    Categorizes panels by SP (Single Patti) and DP (Double Patti) and ranks by
    empirical frequency and recency.
    """

    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
        self.patti_counts = Counter()
        self.ank_to_pattis = defaultdict(list)
        self.ank_patti_counts = defaultdict(Counter)

    def fit(self, valid_df: pd.DataFrame):
        valid_df = sanitize_draw_data(valid_df)
        self.patti_counts.clear()
        self.ank_to_pattis.clear()
        self.ank_patti_counts.clear()

        if len(valid_df) == 0:
            return

        # Learn all valid pattis from dataset
        for col, ank_col in [("Open_Patti", "Open_Digit"), ("Close_Patti", "Close_Digit")]:
            if col in valid_df.columns:
                sub = valid_df[[col, ank_col]].dropna()
                for _, row in sub.iterrows():
                    p = str(row[col]).strip()
                    if len(p) == 3 and p.isdigit():
                        a = int(row[ank_col])
                        self.patti_counts[p] += 1
                        self.ank_patti_counts[a][p] += 1

        # Populate standard panels for all 10 Anks
        for a in range(10):
            counts = self.ank_patti_counts[a]
            if counts:
                ranked = [p for p, _ in counts.most_common()]
                self.ank_to_pattis[a] = ranked

    def get_top_pattis_for_ank(self, ank: int, top_n: int = 4) -> dict:
        ank = int(ank) % 10
        pattis = self.ank_to_pattis.get(ank, [])
        sp_list = []
        dp_list = []
        for p in pattis:
            digits = set(p)
            if len(digits) == 3 and len(sp_list) < top_n:
                sp_list.append(p)
            elif len(digits) == 2 and len(dp_list) < top_n:
                dp_list.append(p)

        # Fallback standard panels if history is sparse
        fallback_sp = {
            0: ["127", "136", "145", "235"],
            1: ["128", "137", "146", "236"],
            2: ["129", "138", "147", "237"],
            3: ["120", "139", "148", "238"],
            4: ["130", "149", "158", "239"],
            5: ["140", "159", "168", "230"],
            6: ["150", "169", "178", "240"],
            7: ["160", "179", "250", "340"],
            8: ["170", "189", "260", "350"],
            9: ["180", "199", "270", "360"],
        }
        fallback_dp = {
            0: ["118", "226", "334", "442"],
            1: ["119", "227", "335", "443"],
            2: ["110", "228", "336", "444"],
            3: ["229", "337", "445", "553"],
            4: ["220", "338", "446", "554"],
            5: ["339", "447", "555", "663"],
            6: ["330", "448", "556", "664"],
            7: ["449", "557", "665", "773"],
            8: ["440", "558", "666", "774"],
            9: ["559", "667", "775", "883"],
        }
        for fb in fallback_sp.get(ank, []):
            if fb not in sp_list and len(sp_list) < top_n:
                sp_list.append(fb)
        for fb in fallback_dp.get(ank, []):
            if fb not in dp_list and len(dp_list) < top_n:
                dp_list.append(fb)

        return {
            "sp": sp_list[:top_n],
            "dp": dp_list[:top_n],
            "all": (sp_list[:2] + dp_list[:2]),
        }


class EnsemblePredictor:
    """
    High-Performance Hybrid Quant-Domain Ensemble for Kalyan Matka:
    1. Multi-Horizon Recency + Markov + Seasonality + Overdue Cycle Hazard for Single Anks.
    2. Harmonic Cut-Pair Dual OTC (4 Anks) with theoretical pass probability.
    3. Intra-draw Conditional Affinity Matrix P(Close | Open).
    4. Empirical 3-digit Patti/Panel (SP/DP) generator.
    5. Coherent Marginal-Joint Synthesis aligning candidate Jodis with highest-probability Anks.
    """

    def __init__(
        self,
        w_markov: float = 0.18,
        w_recency: float = 0.52,
        w_seasonal: float = 0.15,
        w_gap: float = 0.15,
        cut_resonance: float = 0.12,
        w_joint: float = 0.45,
        w_total: float = 0.15,
        w_direct: float = 0.25,
        w_markov_jodi: float = 0.15,
        open_rep_boost: float = 1.15,
        cross_transition_weight: float = 0.2070,
        lambda_cut_cluster: float = 0.35,
        hyperparameters: dict = None,
    ):
        if hyperparameters and isinstance(hyperparameters, dict):
            if "lambda_cut_cluster" in hyperparameters:
                lambda_cut_cluster = float(hyperparameters["lambda_cut_cluster"])
            if "w_markov" in hyperparameters:
                w_markov = float(hyperparameters["w_markov"])
            if "w_recency" in hyperparameters:
                w_recency = float(hyperparameters["w_recency"])
            if "w_seasonal" in hyperparameters:
                w_seasonal = float(hyperparameters["w_seasonal"])
            if "w_gap" in hyperparameters:
                w_gap = float(hyperparameters["w_gap"])

        self.w_markov = w_markov
        self.w_recency = w_recency
        self.w_seasonal = w_seasonal
        self.w_gap = w_gap
        self.cut_resonance = cut_resonance
        self.w_joint = w_joint
        self.w_total = w_total
        self.w_direct = w_direct
        self.w_markov_jodi = w_markov_jodi
        self.open_rep_boost = open_rep_boost
        self.cross_transition_weight = cross_transition_weight
        self.lambda_cut_cluster = lambda_cut_cluster

        self.last_jodi = None
        self.last_open_digit = None
        self.last_open = None
        self.last_close_digit = None
        self.last_close = None

        self.markov = MarkovChainModel()
        self.recency = RecencyWeightedModel()
        self.gap_model = OverdueGapModel()
        self.seasonal = DayOfWeekSeasonalModel()
        self.intra = IntraDrawConditionalModel()
        self.total_model = JodiTotalModel()
        self.patti_model = PattiPanelModel()

    def apply_hyperparameters(self, hyperparameters: dict):
        """Apply recalibrated hyperparameters to active model configuration."""
        if hyperparameters and isinstance(hyperparameters, dict):
            if "lambda_cut_cluster" in hyperparameters:
                self.lambda_cut_cluster = float(hyperparameters["lambda_cut_cluster"])
            if "w_markov" in hyperparameters:
                self.w_markov = float(hyperparameters["w_markov"])
            if "w_recency" in hyperparameters:
                self.w_recency = float(hyperparameters["w_recency"])
            if "w_seasonal" in hyperparameters:
                self.w_seasonal = float(hyperparameters["w_seasonal"])
            if "w_gap" in hyperparameters:
                self.w_gap = float(hyperparameters["w_gap"])

    def fit(self, valid_df: pd.DataFrame):
        valid_df = sanitize_draw_data(valid_df)
        if len(valid_df) > 0:
            last_row = valid_df.iloc[-1]
            if "Jodi" in last_row and pd.notna(last_row["Jodi"]):
                self.last_jodi = f"{int(float(last_row['Jodi'])):02d}"
            if "Open_Digit" in last_row and pd.notna(last_row["Open_Digit"]):
                self.last_open_digit = int(float(last_row["Open_Digit"]))
                self.last_open = self.last_open_digit
            if "Close_Digit" in last_row and pd.notna(last_row["Close_Digit"]):
                self.last_close_digit = int(float(last_row["Close_Digit"]))
                self.last_close = self.last_close_digit

        self.markov.fit(valid_df)
        self.recency.fit(valid_df)
        self.gap_model.fit(valid_df)
        self.seasonal.fit(valid_df)
        self.intra.fit(valid_df)
        self.total_model.fit(valid_df)
        self.patti_model.fit(valid_df)

    def predict(
        self,
        day_of_week: str = "Mon",
        prev_jodi: str = None,
        prev_open: int = None,
        prev_close: int = None,
        lambda_cut_cluster: float = None,
        history_df: pd.DataFrame = None,
        hyperparameters: dict = None,
        **kwargs,
    ) -> dict:
        if isinstance(day_of_week, pd.DataFrame):
            history_df = day_of_week
            day_of_week = "Mon"

        if hyperparameters and isinstance(hyperparameters, dict):
            if "lambda_cut_cluster" in hyperparameters:
                lambda_cut_cluster = float(hyperparameters["lambda_cut_cluster"])
            if "w_markov" in hyperparameters:
                self.w_markov = float(hyperparameters["w_markov"])
            if "w_recency" in hyperparameters:
                self.w_recency = float(hyperparameters["w_recency"])
            if "w_seasonal" in hyperparameters:
                self.w_seasonal = float(hyperparameters["w_seasonal"])
            if "w_gap" in hyperparameters:
                self.w_gap = float(hyperparameters["w_gap"])

        if history_df is not None and isinstance(history_df, pd.DataFrame) and len(history_df) > 0:
            valid_hist = sanitize_draw_data(history_df)
            if len(valid_hist) > 0:
                last_row = valid_hist.iloc[-1]
                if prev_jodi is None and "Jodi" in last_row and pd.notna(last_row["Jodi"]):
                    prev_jodi = f"{int(float(last_row['Jodi'])):02d}"
                if prev_open is None and "Open_Digit" in last_row and pd.notna(last_row["Open_Digit"]):
                    prev_open = int(float(last_row["Open_Digit"]))
                if prev_close is None and "Close_Digit" in last_row and pd.notna(last_row["Close_Digit"]):
                    prev_close = int(float(last_row["Close_Digit"]))
                if "Day_Of_Week" in last_row and pd.notna(last_row["Day_Of_Week"]) and (day_of_week == "Mon" or day_of_week is None):
                    day_of_week = str(last_row["Day_Of_Week"])

        if prev_jodi is None and self.last_jodi is not None:
            prev_jodi = self.last_jodi
        if prev_open is None and self.last_open_digit is not None:
            prev_open = self.last_open_digit
        if prev_close is None and self.last_close_digit is not None:
            prev_close = self.last_close_digit

        if day_of_week is None:
            day_of_week = "Mon"
        # 1. Single Open Digit Distribution
        p_o_mk = self.markov.predict_open_probs(prev_open, prev_close)
        p_o_rc = self.recency.predict_open_probs()
        p_o_sn = self.seasonal.predict_open_probs(day_of_week)
        p_o_gp = self.gap_model.predict_open_probs()

        open_blend = (
            self.w_markov * p_o_mk
            + self.w_recency * p_o_rc
            + self.w_seasonal * p_o_sn
            + self.w_gap * p_o_gp
        )
        # Apply Cut-pair harmonic resonance: Cut(d) = (d + 5) mod 10
        open_probs = (1.0 - self.cut_resonance) * open_blend + self.cut_resonance * np.roll(open_blend, 5)
        open_probs /= open_probs.sum()

        # Apply Commercial Portal Rules (Requirement 2):
        # 1. Open repetition momentum boost (+15% historical frequency ~11.26%)
        if prev_open is not None and 0 <= int(prev_open) <= 9:
            open_probs[int(prev_open)] *= self.open_rep_boost
            open_probs /= open_probs.sum()

        # 2. Cross-transition bias: Close(t-1) -> Open(t) direct & cut (~20.70%)
        if prev_close is not None and 0 <= int(prev_close) <= 9:
            c_dir = int(prev_close)
            c_cut = (c_dir + 5) % 10
            target_mass = self.cross_transition_weight
            curr_cross = open_probs[c_dir] + open_probs[c_cut]
            if curr_cross < target_mass and curr_cross > 0:
                rem_target = 1.0 - target_mass
                rem_curr = float(np.sum([open_probs[d] for d in range(10) if d not in (c_dir, c_cut)]))
                scale_cross = target_mass / curr_cross
                scale_rem = (rem_target / rem_curr) if rem_curr > 0 else 1.0
                open_probs[c_dir] *= scale_cross
                open_probs[c_cut] *= scale_cross
                for d in range(10):
                    if d not in (c_dir, c_cut):
                        open_probs[d] *= scale_rem

        open_probs /= open_probs.sum()

        # 2. Single Close Digit Distribution
        p_c_mk = self.markov.predict_close_probs(prev_close)
        p_c_rc = self.recency.predict_close_probs()
        p_c_sn = self.seasonal.predict_close_probs(day_of_week)
        p_c_gp = self.gap_model.predict_close_probs()

        close_blend = (
            self.w_markov * p_c_mk
            + self.w_recency * p_c_rc
            + self.w_seasonal * p_c_sn
            + self.w_gap * p_c_gp
        )
        close_probs = (1.0 - self.cut_resonance) * close_blend + self.cut_resonance * np.roll(close_blend, 5)
        close_probs /= close_probs.sum()

        # 3. Intra-Draw Conditional Matrix
        p_c_given_o = self.intra.get_conditional_close_matrix()

        # 4. Joint Open-Close Matrix with Conditional Close
        p_joint = np.zeros((10, 10), dtype=float)
        for o in range(10):
            for c in range(10):
                p_c_eff = 0.60 * close_probs[c] + 0.40 * p_c_given_o[o, c]
                p_joint[o, c] = open_probs[o] * p_c_eff
        p_joint /= p_joint.sum()

        # 5. Total (Sum mod 10) Prior Matrix
        p_tot_rec = self.total_model.predict_total_probs()
        p_tot_gp = self.gap_model.predict_total_probs()
        p_tot = 0.70 * p_tot_rec + 0.30 * p_tot_gp
        p_tot /= p_tot.sum()

        p_total_matrix = np.zeros((10, 10), dtype=float)
        for o in range(10):
            for c in range(10):
                t = (o + c) % 10
                p_total_matrix[o, c] = p_tot[t] / 10.0
        p_total_matrix /= p_total_matrix.sum()

        # 6. Direct Jodi Recency (Tri-Horizon)
        p_j_rec = self.recency.predict_jodi_probs().reshape((10, 10))

        # 7. Markov Chain Jodi Transition Matrix (Feature 14)
        p_jodi_markov = self.markov.predict_jodi_probs(prev_jodi).reshape((10, 10))

        # 8. Coherent Marginal & Markov Synthesis
        jodi_matrix = (
            self.w_joint * p_joint
            + self.w_total * p_total_matrix
            + self.w_direct * p_j_rec
            + self.w_markov_jodi * p_jodi_markov
        )
        jodi_matrix /= jodi_matrix.sum()

        # 9. 2D Harmonic Cut Resonance Diffusion across 3 cut states (Feature 15)
        # For candidate Jodi (o, c): Cut states are ((o+5)%10, c), (o, (c+5)%10), ((o+5)%10, (c+5)%10)
        cut_2d_open = np.roll(jodi_matrix, 5, axis=0)
        cut_2d_close = np.roll(jodi_matrix, 5, axis=1)
        cut_2d_both = np.roll(jodi_matrix, (5, 5), axis=(0, 1))
        gamma = self.cut_resonance
        jodi_matrix = (1.0 - gamma) * jodi_matrix + (gamma / 3.0) * (cut_2d_open + cut_2d_close + cut_2d_both)
        jodi_matrix /= jodi_matrix.sum()

        # 10. Optimal Harmonic Cut-Pair Dual OTC (4 Anks) (Feature 16 / Requirement 2)
        # Evaluates all 10 candidate pairs of cut-pairs to find the exact combination
        # that maximizes joint probability of touching either Open or Close:
        # P(Hit) = 1.0 - sum_{o not in OTC, c not in OTC} P_joint(o, c)
        cut_pairs = [(k, (k + 5) % 10) for k in range(5)]
        best_pairs = None
        best_prob = -1.0
        best_otc = None

        for idx1 in range(5):
            for idx2 in range(idx1 + 1, 5):
                p1 = cut_pairs[idx1]
                p2 = cut_pairs[idx2]
                cand_otc = set([p1[0], p1[1], p2[0], p2[1]])
                miss_digits = [d for d in range(10) if d not in cand_otc]
                p_miss = float(jodi_matrix[np.ix_(miss_digits, miss_digits)].sum())
                p_hit = 1.0 - p_miss
                if p_hit > best_prob:
                    best_prob = p_hit
                    best_pairs = [p1, p2]
                    best_otc = sorted(list(cand_otc))

        otc_pairs = best_pairs
        otc_digits = best_otc
        # Commercial Portal Rule: Genuine computed mathematical pass probability
        otc_pass_prob = float(np.clip(best_prob, 0.0, 1.0))

        # Final Jodi probability vector
        jodi_probs = jodi_matrix.flatten()
        jodi_probs /= jodi_probs.sum()

        # 11. Cut-Cluster Density Weighting (Feature 1, Requirement 1)
        # S(j) = (1 - \lambda) * P(j) + \lambda * Mean_{k in Family(j)}(P(k)) with \lambda ~ 0.35
        lam = lambda_cut_cluster if lambda_cut_cluster is not None else self.lambda_cut_cluster
        cut_cluster_density_scores = np.zeros(100, dtype=float)
        for j in range(100):
            fam = get_family_jodis(f"{j:02d}")
            fam_indices = [int(x) for x in fam]
            cut_cluster_density_scores[j] = (1.0 - lam) * jodi_probs[j] + lam * float(jodi_probs[fam_indices].mean())
        cut_cluster_density_scores /= cut_cluster_density_scores.sum()

        # 12. Family-Enriched Top 10 Allocation (Feature 2, Requirement 1)
        # Identify dominant Top Pick by density score, include its high-affinity family members
        ranked_by_density = np.argsort(cut_cluster_density_scores)[::-1]
        top_pick_int = int(ranked_by_density[0])
        top_pick_str = f"{top_pick_int:02d}"
        top_family = get_family_jodis(top_pick_str)
        top_fam_sorted = sorted(top_family, key=lambda x: cut_cluster_density_scores[int(x)], reverse=True)

        top_10_jodi_strs = [top_pick_str]
        for fam_j in top_fam_sorted:
            if fam_j not in top_10_jodi_strs:
                top_10_jodi_strs.append(fam_j)
            if len(top_10_jodi_strs) >= 8:
                break

        for idx in ranked_by_density:
            j_str = f"{idx:02d}"
            if j_str not in top_10_jodi_strs:
                top_10_jodi_strs.append(j_str)
            if len(top_10_jodi_strs) >= 10:
                break

        top_10_jodis = []
        for rank_idx, j_str in enumerate(top_10_jodi_strs[:10], start=1):
            j_int = int(j_str)
            top_10_jodis.append({
                "rank": rank_idx,
                "jodi": j_str,
                "score": float(cut_cluster_density_scores[j_int]),
                "raw_prob": float(jodi_probs[j_int]),
                "is_family_affinity": (j_str in top_family),
                "family": get_family_jodis(j_str),
            })

        # 13. Patti / Panel recommendations for OTC digits
        patti_preds = {}
        for ank in otc_digits:
            patti_preds[str(ank)] = self.patti_model.get_top_pattis_for_ank(ank, top_n=4)

        return {
            "jodi_probs": jodi_probs,
            "cut_cluster_density_scores": cut_cluster_density_scores,
            "top_10_jodis": top_10_jodis,
            "open_probs": open_probs,
            "close_probs": close_probs,
            "total_probs": p_tot,
            "otc_digits": otc_digits,
            "otc_pairs": otc_pairs,
            "otc_pass_prob": round(otc_pass_prob, 4),
            "patti_predictions": patti_preds,
        }


def evaluate_outcome(
    predicted_jodi_probs_or_ranking,
    top_pick: str,
    top_10: list,
    otc_digits: list,
    actual_jodi: str,
    actual_open: int,
    actual_close: int,
) -> dict:
    """
    Evaluates draw outcome against model predictions (Feature 8, Requirement 3):
    - exact rank (1..100) of actual draw in predicted distribution
    - top10_hit: bool (actual_rank <= 10)
    - family_cut_hit: bool (actual Jodi in family bracket of top pick)
    - otc_hit: bool (actual Open or Close in 4-OTC digit set)
    - open_hit: bool (actual Open in 4-OTC digit set)
    - close_hit: bool (actual Close in 4-OTC digit set)
    - reciprocal_rank: float (1.0 / actual_rank)
    """
    try:
        act_j = f"{int(float(actual_jodi)):02d}"
    except (ValueError, TypeError):
        act_j = str(actual_jodi).strip().zfill(2)

    try:
        act_o = int(float(actual_open))
    except (ValueError, TypeError):
        act_o = -1

    try:
        act_c = int(float(actual_close))
    except (ValueError, TypeError):
        act_c = -1

    # Top pick string and family bracket
    if top_pick is not None:
        try:
            top_pick_str = f"{int(float(top_pick)):02d}"
        except (ValueError, TypeError):
            top_pick_str = str(top_pick).strip().zfill(2)
    else:
        top_pick_str = "00"
    top_pick_fam = get_family_jodis(top_pick_str)

    # Top 10 list of strings
    top_10_strs = []
    for item in (top_10 or []):
        try:
            top_10_strs.append(f"{int(float(item)):02d}")
        except (ValueError, TypeError):
            top_10_strs.append(str(item).strip().zfill(2))

    # OTC digits
    otc_list = [int(float(d)) for d in (otc_digits or [])]

    # Calculate actual rank (1..100)
    actual_rank = 100
    if isinstance(predicted_jodi_probs_or_ranking, (list, tuple)) and len(predicted_jodi_probs_or_ranking) == 100:
        if isinstance(predicted_jodi_probs_or_ranking[0], str):
            ranked = [str(x).strip().zfill(2) for x in predicted_jodi_probs_or_ranking]
            actual_rank = ranked.index(act_j) + 1 if act_j in ranked else 100
        else:
            scores = np.asarray(predicted_jodi_probs_or_ranking, dtype=float)
            ranked = [f"{i:02d}" for i in np.argsort(scores)[::-1]]
            actual_rank = ranked.index(act_j) + 1 if act_j in ranked else 100
    elif isinstance(predicted_jodi_probs_or_ranking, np.ndarray) and len(predicted_jodi_probs_or_ranking) == 100:
        scores = np.asarray(predicted_jodi_probs_or_ranking, dtype=float)
        ranked = [f"{i:02d}" for i in np.argsort(scores)[::-1]]
        actual_rank = ranked.index(act_j) + 1 if act_j in ranked else 100
    elif isinstance(predicted_jodi_probs_or_ranking, (list, tuple)):
        ranked = [str(x).strip().zfill(2) for x in predicted_jodi_probs_or_ranking]
        if act_j in ranked:
            actual_rank = ranked.index(act_j) + 1
        elif act_j in top_10_strs:
            actual_rank = top_10_strs.index(act_j) + 1
        else:
            all_100 = ranked + [f"{i:02d}" for i in range(100) if f"{i:02d}" not in ranked]
            actual_rank = all_100.index(act_j) + 1 if act_j in all_100 else 100
    else:
        if act_j in top_10_strs:
            actual_rank = top_10_strs.index(act_j) + 1
        else:
            actual_rank = 100

    if act_j in top_10_strs:
        top10_hit = True
        actual_rank = min(actual_rank, top_10_strs.index(act_j) + 1)
    else:
        top10_hit = bool(actual_rank <= 10)
    family_cut_hit = bool(act_j in top_pick_fam)
    otc_hit = bool(act_o in otc_list or act_c in otc_list)
    open_hit = bool(act_o in otc_list)
    close_hit = bool(act_c in otc_list)
    reciprocal_rank = round(1.0 / actual_rank, 4) if actual_rank > 0 else 0.0

    return {
        "actual_rank": actual_rank,
        "top10_hit": top10_hit,
        "family_cut_hit": family_cut_hit,
        "otc_hit": otc_hit,
        "open_hit": open_hit,
        "close_hit": close_hit,
        "reciprocal_rank": reciprocal_rank,
    }


def recalibrate_hyperparameters(
    current_params: dict,
    performance_log: dict,
) -> dict:
    """
    Adaptive hyperparameter recalibration based on performance_log (Feature 10, Requirement 3):
    - Bounded guardrails:
      lambda in [0.25, 0.45]
      w_recency in [0.40, 0.65]
      w_markov in [0.10, 0.30]
      w_seasonal in [0.08, 0.22]
      w_gap in [0.08, 0.22]
    - Dynamically increases lambda when family cut hit rate > 25%, decreases if < 10%
    - Strict unit sum re-normalization for submodel weights
    """
    params = dict(current_params) if isinstance(current_params, dict) else {}
    try:
        lam = float(params.get("lambda_cut_cluster", 0.35))
    except (ValueError, TypeError):
        lam = 0.35
    try:
        w_rec = float(params.get("w_recency", 0.52))
    except (ValueError, TypeError):
        w_rec = 0.52
    try:
        w_mk = float(params.get("w_markov", 0.18))
    except (ValueError, TypeError):
        w_mk = 0.18
    try:
        w_sn = float(params.get("w_seasonal", 0.15))
    except (ValueError, TypeError):
        w_sn = 0.15
    try:
        w_gp = float(params.get("w_gap", 0.15))
    except (ValueError, TypeError):
        w_gp = 0.15

    summary = performance_log.get("summary") if isinstance(performance_log, dict) else {}
    if not isinstance(summary, dict):
        summary = {}

    raw_fam = summary.get("family_cut_hit_rate", 0.0)
    try:
        fam_rate = float(raw_fam)
        if np.isnan(fam_rate):
            fam_rate = None
    except (ValueError, TypeError):
        fam_rate = None

    # Adjust lambda if valid float
    if fam_rate is not None:
        if fam_rate > 0.25:
            lam = min(0.45, lam + 0.02)
        elif fam_rate < 0.10:
            lam = max(0.25, lam - 0.02)
    params["lambda_cut_cluster"] = round(float(np.clip(lam, 0.25, 0.45)), 4)

    # Submodel Weight Guardrail Invariant
    bounds = {
        "w_recency": (0.40, 0.65),
        "w_markov": (0.10, 0.30),
        "w_seasonal": (0.08, 0.22),
        "w_gap": (0.08, 0.22),
    }

    # Normalize weights before clipping if positive sum
    total_w = w_rec + w_mk + w_sn + w_gp
    if total_w > 0:
        w_rec /= total_w
        w_mk /= total_w
        w_sn /= total_w
        w_gp /= total_w
    else:
        w_rec, w_mk, w_sn, w_gp = 0.52, 0.18, 0.15, 0.15

    w_vec = {
        "w_recency": float(np.clip(w_rec, bounds["w_recency"][0], bounds["w_recency"][1])),
        "w_markov": float(np.clip(w_mk, bounds["w_markov"][0], bounds["w_markov"][1])),
        "w_seasonal": float(np.clip(w_sn, bounds["w_seasonal"][0], bounds["w_seasonal"][1])),
        "w_gap": float(np.clip(w_gp, bounds["w_gap"][0], bounds["w_gap"][1])),
    }

    # Iterative proportional slack distribution preserving all bounds
    for _ in range(20):
        s = sum(w_vec.values())
        diff = 1.0 - s
        if abs(diff) < 1e-7:
            break
        if diff > 0:
            headroom = {k: bounds[k][1] - w_vec[k] for k in w_vec}
            tot_headroom = sum(h for h in headroom.values() if h > 1e-7)
            if tot_headroom <= 0:
                break
            for k, h in headroom.items():
                if h > 1e-7:
                    w_vec[k] += diff * (h / tot_headroom)
        else:
            legroom = {k: w_vec[k] - bounds[k][0] for k in w_vec}
            tot_legroom = sum(l for l in legroom.values() if l > 1e-7)
            if tot_legroom <= 0:
                break
            for k, l in legroom.items():
                if l > 1e-7:
                    w_vec[k] += diff * (l / tot_legroom)
        for k in w_vec:
            w_vec[k] = float(np.clip(w_vec[k], bounds[k][0], bounds[k][1]))

    params["w_recency"] = round(w_vec["w_recency"], 4)
    params["w_markov"] = round(w_vec["w_markov"], 4)
    params["w_seasonal"] = round(w_vec["w_seasonal"], 4)
    params["w_gap"] = round(1.0 - (params["w_recency"] + params["w_markov"] + params["w_seasonal"]), 4)

    if params["w_gap"] < bounds["w_gap"][0]:
        adj = round(bounds["w_gap"][0] - params["w_gap"], 4)
        params["w_gap"] = bounds["w_gap"][0]
        params["w_recency"] = round(params["w_recency"] - adj, 4)
    elif params["w_gap"] > bounds["w_gap"][1]:
        adj = round(params["w_gap"] - bounds["w_gap"][1], 4)
        params["w_gap"] = bounds["w_gap"][1]
        params["w_recency"] = round(params["w_recency"] + adj, 4)

    return params

