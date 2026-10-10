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
        self.jodi_trans.clear()
        self.open_trans.fill(0.0)
        self.close_trans.fill(0.0)
        self.cross_trans.fill(0.0)

        n = len(valid_df)
        if n == 0:
            return

        jodis = valid_df["Jodi"].astype(int).values
        opens = valid_df["Open_Digit"].astype(int).values
        closes = valid_df["Close_Digit"].astype(int).values

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

        jodis = valid_df["Jodi"].astype(int).values
        opens = valid_df["Open_Digit"].astype(int).values
        closes = valid_df["Close_Digit"].astype(int).values

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
        self.open_gaps.fill(0.0)
        self.close_gaps.fill(0.0)
        self.total_gaps.fill(0.0)

        opens = valid_df["Open_Digit"].astype(int).values
        closes = valid_df["Close_Digit"].astype(int).values
        n = len(valid_df)

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
        self.day_jodi_counts.clear()
        self.day_open_counts.clear()
        self.day_close_counts.clear()

        for _, row in valid_df.iterrows():
            day = row["Day_Of_Week"]
            j = int(row["Jodi"])
            o = int(row["Open_Digit"])
            c = int(row["Close_Digit"])
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
        self.matrix.fill(0.0)
        opens = valid_df["Open_Digit"].astype(int).values
        closes = valid_df["Close_Digit"].astype(int).values
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
        self.total_weights.fill(0.0)
        n = len(valid_df)
        opens = valid_df["Open_Digit"].astype(int).values
        closes = valid_df["Close_Digit"].astype(int).values
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
        self.patti_counts.clear()
        self.ank_to_pattis.clear()
        self.ank_patti_counts.clear()

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
    ):
        self.w_markov = w_markov
        self.w_recency = w_recency
        self.w_seasonal = w_seasonal
        self.w_gap = w_gap
        self.cut_resonance = cut_resonance
        self.w_joint = w_joint
        self.w_total = w_total
        self.w_direct = w_direct
        self.w_markov_jodi = w_markov_jodi

        self.markov = MarkovChainModel()
        self.recency = RecencyWeightedModel()
        self.gap_model = OverdueGapModel()
        self.seasonal = DayOfWeekSeasonalModel()
        self.intra = IntraDrawConditionalModel()
        self.total_model = JodiTotalModel()
        self.patti_model = PattiPanelModel()

    def fit(self, valid_df: pd.DataFrame):
        self.markov.fit(valid_df)
        self.recency.fit(valid_df)
        self.gap_model.fit(valid_df)
        self.seasonal.fit(valid_df)
        self.intra.fit(valid_df)
        self.total_model.fit(valid_df)
        self.patti_model.fit(valid_df)

    def predict(
        self,
        day_of_week: str,
        prev_jodi: str = None,
        prev_open: int = None,
        prev_close: int = None,
    ) -> dict:
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

        # 3. Harmonic Cut-Pair Dual OTC (4 Anks)
        # Combined marginal strength across Open and Close
        comb_ank_probs = 0.55 * open_probs + 0.45 * close_probs
        ranked_digits = np.argsort(comb_ank_probs)[::-1]

        d1 = int(ranked_digits[0])
        cut1 = (d1 + 5) % 10

        # Find second primary digit not identical to d1 or cut1
        d2 = None
        for d in ranked_digits[1:]:
            d_int = int(d)
            if d_int != d1 and d_int != cut1:
                d2 = d_int
                break
        if d2 is None:
            d2 = (d1 + 1) % 10
        cut2 = (d2 + 5) % 10

        otc_pairs = [(d1, cut1), (d2, cut2)]
        otc_digits = sorted([d1, cut1, d2, cut2])

        # 4. Intra-Draw Conditional Matrix
        p_c_given_o = self.intra.get_conditional_close_matrix()

        # 5. Joint Open-Close Matrix with Conditional Close
        p_joint = np.zeros((10, 10), dtype=float)
        for o in range(10):
            for c in range(10):
                p_c_eff = 0.60 * close_probs[c] + 0.40 * p_c_given_o[o, c]
                p_joint[o, c] = open_probs[o] * p_c_eff
        p_joint /= p_joint.sum()

        # 6. Total (Sum mod 10) Prior Matrix
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

        # 7. Direct Jodi Recency (Tri-Horizon)
        p_j_rec = self.recency.predict_jodi_probs().reshape((10, 10))

        # 8. Markov Chain Jodi Transition Matrix (Feature 14)
        p_jodi_markov = self.markov.predict_jodi_probs(prev_jodi).reshape((10, 10))

        # 9. Coherent Marginal & Markov Synthesis
        jodi_matrix = (
            self.w_joint * p_joint
            + self.w_total * p_total_matrix
            + self.w_direct * p_j_rec
            + self.w_markov_jodi * p_jodi_markov
        )
        jodi_matrix /= jodi_matrix.sum()

        # 10. 2D Harmonic Cut Resonance Diffusion across 3 cut states (Feature 15)
        # For candidate Jodi (o, c): Cut states are ((o+5)%10, c), (o, (c+5)%10), ((o+5)%10, (c+5)%10)
        cut_2d_open = np.roll(jodi_matrix, 5, axis=0)
        cut_2d_close = np.roll(jodi_matrix, 5, axis=1)
        cut_2d_both = np.roll(jodi_matrix, (5, 5), axis=(0, 1))
        gamma = self.cut_resonance
        jodi_matrix = (1.0 - gamma) * jodi_matrix + (gamma / 3.0) * (cut_2d_open + cut_2d_close + cut_2d_both)
        jodi_matrix /= jodi_matrix.sum()

        # 11. Exact OTC Joint Pass Probability Formulation (Feature 16)
        # P(OTC Hit) = 1.0 - sum_{o not in OTC, c not in OTC} P_joint(o, c)
        non_otc = [d for d in range(10) if d not in otc_digits]
        p_miss = float(jodi_matrix[np.ix_(non_otc, non_otc)].sum())
        otc_pass_prob = float(np.clip(1.0 - p_miss, 0.0, 1.0))

        # Final Jodi probability vector
        jodi_probs = jodi_matrix.flatten()
        jodi_probs /= jodi_probs.sum()

        # 12. Patti / Panel recommendations for OTC digits
        patti_preds = {}
        for ank in otc_digits:
            patti_preds[str(ank)] = self.patti_model.get_top_pattis_for_ank(ank, top_n=4)

        return {
            "jodi_probs": jodi_probs,
            "open_probs": open_probs,
            "close_probs": close_probs,
            "total_probs": p_tot,
            "otc_digits": otc_digits,
            "otc_pairs": otc_pairs,
            "otc_pass_prob": round(otc_pass_prob, 4),
            "patti_predictions": patti_preds,
        }
