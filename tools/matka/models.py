"""
Quantitative Predictive Models for Kalyan Matka Historical Analysis.
Implements:
1. First-Order Markov Transition Matrices with Adaptive Smoothing
2. Dual-Horizon Recency Weighting (Fast Momentum + Medium Trend Exponential Decay)
3. Target Day-of-Week Seasonality Distributions
4. Intra-Draw Conditional Matrix P(Close | Open)
5. Jodi Total (Sum Mod 10) Pattern Prior
6. Cut-Digit (Adding 5 mod 10) Harmonic Resonance
7. Coherent Marginal-Joint Probabilistic Synthesis (Top Jodis aligned with Open/Close Anks)
"""

import numpy as np
import pandas as pd
from collections import defaultdict, Counter


class MarkovChainModel:
    """
    First-order Markov model computing conditional transition probabilities
    P(Draw_t | Draw_{t-1}) for both Jodi numbers and Open/Close single digits.
    Uses calibrated Laplace-Dirichlet smoothing to avoid zero-probability traps.
    """

    def __init__(self, alpha_digit: float = 0.6, alpha_jodi: float = 0.1):
        self.alpha_digit = alpha_digit
        self.alpha_jodi = alpha_jodi
        self.jodi_trans = defaultdict(Counter)
        self.open_trans = np.zeros((10, 10), dtype=float)
        self.close_trans = np.zeros((10, 10), dtype=float)
        self.last_jodi = None
        self.last_open = None
        self.last_close = None

    def fit(self, valid_df: pd.DataFrame):
        self.jodi_trans.clear()
        self.open_trans.fill(0.0)
        self.close_trans.fill(0.0)

        jodis = valid_df["Jodi"].astype(int).values
        opens = valid_df["Open_Digit"].astype(int).values
        closes = valid_df["Close_Digit"].astype(int).values
        n = len(valid_df)

        for i in range(n - 1):
            self.jodi_trans[f"{jodis[i]:02d}"][f"{jodis[i + 1]:02d}"] += 1
            self.open_trans[opens[i], opens[i + 1]] += 1.0
            self.close_trans[closes[i], closes[i + 1]] += 1.0

        if n > 0:
            self.last_jodi = f"{jodis[-1]:02d}"
            self.last_open = int(opens[-1])
            self.last_close = int(closes[-1])

    def predict_jodi_probs(self, prev_jodi: str = None) -> np.ndarray:
        prev = prev_jodi if prev_jodi is not None else self.last_jodi
        counts = np.array([self.jodi_trans[prev][f"{i:02d}"] for i in range(100)], dtype=float)
        probs = (counts + self.alpha_jodi) / (counts.sum() + 100 * self.alpha_jodi)
        return probs

    def predict_open_probs(self, prev_open: int = None) -> np.ndarray:
        prev = int(prev_open) if prev_open is not None else self.last_open
        if prev is None or prev < 0 or prev > 9:
            return np.ones(10, dtype=float) / 10.0
        counts = self.open_trans[prev]
        probs = (counts + self.alpha_digit) / (counts.sum() + 10 * self.alpha_digit)
        return probs

    def predict_close_probs(self, prev_close: int = None) -> np.ndarray:
        prev = int(prev_close) if prev_close is not None else self.last_close
        if prev is None or prev < 0 or prev > 9:
            return np.ones(10, dtype=float) / 10.0
        counts = self.close_trans[prev]
        probs = (counts + self.alpha_digit) / (counts.sum() + 10 * self.alpha_digit)
        return probs


class RecencyWeightedModel:
    """
    Dual-Horizon Exponential Moving Frequency model (Fast Momentum + Medium Trend).
    Applies exponential decay:
      w_fast = e^(-lambda_fast * delta_t)  (Half-life ~ 10 draws)
      w_med  = e^(-lambda_med * delta_t)   (Half-life ~ 40 draws)
    """

    def __init__(self, half_life_fast: float = 10.0, half_life_med: float = 40.0, alpha: float = 0.6):
        self.hl_fast = half_life_fast
        self.hl_med = half_life_med
        self.decay_fast = np.log(2.0) / half_life_fast
        self.decay_med = np.log(2.0) / half_life_med
        self.alpha = alpha

        self.jodi_weights = np.zeros(100, dtype=float)
        self.open_weights_fast = np.zeros(10, dtype=float)
        self.open_weights_med = np.zeros(10, dtype=float)
        self.close_weights_fast = np.zeros(10, dtype=float)
        self.close_weights_med = np.zeros(10, dtype=float)

    def fit(self, valid_df: pd.DataFrame):
        n = len(valid_df)
        self.jodi_weights.fill(0.0)
        self.open_weights_fast.fill(0.0)
        self.open_weights_med.fill(0.0)
        self.close_weights_fast.fill(0.0)
        self.close_weights_med.fill(0.0)

        jodis = valid_df["Jodi"].astype(int).values
        opens = valid_df["Open_Digit"].astype(int).values
        closes = valid_df["Close_Digit"].astype(int).values

        lags = np.arange(n - 1, -1, -1)
        w_f = np.exp(-self.decay_fast * lags)
        w_m = np.exp(-self.decay_med * lags)

        for i in range(n):
            self.jodi_weights[jodis[i]] += w_m[i]
            self.open_weights_fast[opens[i]] += w_f[i]
            self.open_weights_med[opens[i]] += w_m[i]
            self.close_weights_fast[closes[i]] += w_f[i]
            self.close_weights_med[closes[i]] += w_m[i]

    def predict_jodi_probs(self) -> np.ndarray:
        probs = (self.jodi_weights + 0.1) / (self.jodi_weights.sum() + 100 * 0.1)
        return probs

    def predict_open_probs(self) -> np.ndarray:
        p_fast = (self.open_weights_fast + self.alpha) / (self.open_weights_fast.sum() + 10 * self.alpha)
        p_med = (self.open_weights_med + self.alpha) / (self.open_weights_med.sum() + 10 * self.alpha)
        probs = 0.60 * p_fast + 0.40 * p_med
        return probs / probs.sum()

    def predict_close_probs(self) -> np.ndarray:
        p_fast = (self.close_weights_fast + self.alpha) / (self.close_weights_fast.sum() + 10 * self.alpha)
        p_med = (self.close_weights_med + self.alpha) / (self.close_weights_med.sum() + 10 * self.alpha)
        probs = 0.60 * p_fast + 0.40 * p_med
        return probs / probs.sum()


class DayOfWeekSeasonalModel:
    """
    Conditioned on the Day of Week (Mon, Tue, Wed, Thu, Fri, Sat).
    Calculates historical frequency specifically for the upcoming target weekday.
    """

    def __init__(self, alpha: float = 0.6):
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

    def __init__(self, half_life_draws: float = 40.0, alpha: float = 0.5):
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


class EnsemblePredictor:
    """
    State-of-the-art Hybrid Ensemble Predictor for Kalyan Matka:
    1. Multi-Horizon Recency + Markov + Seasonality with Cut-Pair Harmonic Resonance for Single Anks.
    2. Intra-draw Conditional Affinity Matrix P(Close | Open).
    3. Jodi Total (Sum mod 10) prior filter.
    4. Coherent Marginal-Joint Synthesis: P(J = o, c) aligns candidate Jodis with highest-probability Anks.
    """

    def __init__(
        self,
        w_markov: float = 0.20,
        w_recency: float = 0.65,
        w_seasonal: float = 0.15,
        cut_resonance: float = 0.14,
        w_joint: float = 0.55,
        w_total: float = 0.20,
        w_direct: float = 0.25,
    ):
        self.w_markov = w_markov
        self.w_recency = w_recency
        self.w_seasonal = w_seasonal
        self.cut_resonance = cut_resonance
        self.w_joint = w_joint
        self.w_total = w_total
        self.w_direct = w_direct

        self.markov = MarkovChainModel()
        self.recency = RecencyWeightedModel()
        self.seasonal = DayOfWeekSeasonalModel()
        self.intra = IntraDrawConditionalModel()
        self.total_model = JodiTotalModel()

    def fit(self, valid_df: pd.DataFrame):
        self.markov.fit(valid_df)
        self.recency.fit(valid_df)
        self.seasonal.fit(valid_df)
        self.intra.fit(valid_df)
        self.total_model.fit(valid_df)

    def predict(
        self,
        day_of_week: str,
        prev_jodi: str = None,
        prev_open: int = None,
        prev_close: int = None,
    ) -> dict:
        # 1. Single Open Digit Distribution
        p_o_mk = self.markov.predict_open_probs(prev_open)
        p_o_rc = self.recency.predict_open_probs()
        p_o_sn = self.seasonal.predict_open_probs(day_of_week)
        open_blend = (
            self.w_markov * p_o_mk
            + self.w_recency * p_o_rc
            + self.w_seasonal * p_o_sn
        )
        # Apply Cut-pair resonance: Cut(d) = (d + 5) mod 10
        open_probs = (1.0 - self.cut_resonance) * open_blend + self.cut_resonance * np.roll(open_blend, 5)
        open_probs /= open_probs.sum()

        # 2. Single Close Digit Distribution
        p_c_mk = self.markov.predict_close_probs(prev_close)
        p_c_rc = self.recency.predict_close_probs()
        p_c_sn = self.seasonal.predict_close_probs(day_of_week)
        close_blend = (
            self.w_markov * p_c_mk
            + self.w_recency * p_c_rc
            + self.w_seasonal * p_c_sn
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

        # 5. Total (Sum mod 10) Prior Matrix
        p_tot = self.total_model.predict_total_probs()
        p_total_matrix = np.zeros((10, 10), dtype=float)
        for o in range(10):
            for c in range(10):
                t = (o + c) % 10
                p_total_matrix[o, c] = p_tot[t] / 10.0

        # 6. Direct Jodi Recency & Markov Matrix
        p_j_rec = self.recency.predict_jodi_probs().reshape((10, 10))

        # 7. Coherent Marginal Synthesis
        jodi_matrix = (
            self.w_joint * p_joint
            + self.w_total * p_total_matrix
            + self.w_direct * p_j_rec
        )
        jodi_probs = jodi_matrix.flatten()
        jodi_probs /= jodi_probs.sum()

        return {
            "jodi_probs": jodi_probs,
            "open_probs": open_probs,
            "close_probs": close_probs,
            "total_probs": p_tot,
        }
