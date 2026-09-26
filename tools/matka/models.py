"""
Quantitative Predictive Models for Kalyan Matka Historical Analysis.
Implements Markov Chain transitions, Recency-Weighted Exponential Decay (Hot/Cold),
Day-of-Week Seasonality, and an Ensemble Calibrated Probabilistic Predictor.
"""

import numpy as np
import pandas as pd
from collections import defaultdict, Counter


class MarkovChainModel:
    """
    First-order Markov model computing conditional transition probabilities
    P(Draw_t | Draw_{t-1}) for both Jodi numbers and Open/Close single digits.
    Uses Laplace smoothing to avoid zero-probability traps.
    """

    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha  # smoothing parameter
        self.jodi_trans = defaultdict(Counter)
        self.open_trans = defaultdict(Counter)
        self.close_trans = defaultdict(Counter)
        self.last_jodi = None
        self.last_open = None
        self.last_close = None

    def fit(self, valid_df: pd.DataFrame):
        self.jodi_trans.clear()
        self.open_trans.clear()
        self.close_trans.clear()

        jodis = valid_df["Jodi"].astype(str).str.zfill(2).tolist()
        opens = valid_df["Open_Digit"].astype(int).tolist()
        closes = valid_df["Close_Digit"].astype(int).tolist()

        for i in range(len(jodis) - 1):
            self.jodi_trans[jodis[i]][jodis[i + 1]] += 1
            self.open_trans[opens[i]][opens[i + 1]] += 1
            self.close_trans[closes[i]][closes[i + 1]] += 1

        if len(jodis) > 0:
            self.last_jodi = jodis[-1]
            self.last_open = opens[-1]
            self.last_close = closes[-1]

    def predict_jodi_probs(self, prev_jodi: str = None) -> np.ndarray:
        prev = prev_jodi if prev_jodi is not None else self.last_jodi
        counts = np.array([self.jodi_trans[prev][f"{i:02d}"] for i in range(100)], dtype=float)
        # Laplace smoothed probabilities
        probs = (counts + self.alpha) / (counts.sum() + 100 * self.alpha)
        return probs

    def predict_open_probs(self, prev_open: int = None) -> np.ndarray:
        prev = prev_open if prev_open is not None else self.last_open
        counts = np.array([self.open_trans[prev][i] for i in range(10)], dtype=float)
        probs = (counts + self.alpha) / (counts.sum() + 10 * self.alpha)
        return probs

    def predict_close_probs(self, prev_close: int = None) -> np.ndarray:
        prev = prev_close if prev_close is not None else self.last_close
        counts = np.array([self.close_trans[prev][i] for i in range(10)], dtype=float)
        probs = (counts + self.alpha) / (counts.sum() + 10 * self.alpha)
        return probs


class RecencyWeightedModel:
    """
    Exponential Moving Frequency model (Hot/Cold Numbers).
    Applies exponential decay w_i = e^(-lambda * delta_t) to weight recent draws
    significantly higher than historical draws.
    """

    def __init__(self, half_life_draws: float = 60.0, alpha: float = 0.5):
        self.half_life = half_life_draws
        self.decay_rate = np.log(2.0) / half_life_draws
        self.alpha = alpha
        self.jodi_weights = np.zeros(100, dtype=float)
        self.open_weights = np.zeros(10, dtype=float)
        self.close_weights = np.zeros(10, dtype=float)

    def fit(self, valid_df: pd.DataFrame):
        n = len(valid_df)
        self.jodi_weights.fill(0.0)
        self.open_weights.fill(0.0)
        self.close_weights.fill(0.0)

        jodis = valid_df["Jodi"].astype(int).values
        opens = valid_df["Open_Digit"].astype(int).values
        closes = valid_df["Close_Digit"].astype(int).values

        # Compute weights for all draws up to latest (lag 0)
        lags = np.arange(n - 1, -1, -1)
        weights = np.exp(-self.decay_rate * lags)

        for i in range(n):
            w = weights[i]
            self.jodi_weights[jodis[i]] += w
            self.open_weights[opens[i]] += w
            self.close_weights[closes[i]] += w

    def predict_jodi_probs(self) -> np.ndarray:
        probs = (self.jodi_weights + self.alpha) / (self.jodi_weights.sum() + 100 * self.alpha)
        return probs

    def predict_open_probs(self) -> np.ndarray:
        probs = (self.open_weights + self.alpha) / (self.open_weights.sum() + 10 * self.alpha)
        return probs

    def predict_close_probs(self) -> np.ndarray:
        probs = (self.close_weights + self.alpha) / (self.close_weights.sum() + 10 * self.alpha)
        return probs


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
            return np.ones(100) / 100.0
        return (counts + self.alpha) / (total + 100 * self.alpha)

    def predict_open_probs(self, day_of_week: str) -> np.ndarray:
        counts = self.day_open_counts[day_of_week]
        total = counts.sum()
        if total == 0:
            return np.ones(10) / 10.0
        return (counts + self.alpha) / (total + 10 * self.alpha)

    def predict_close_probs(self, day_of_week: str) -> np.ndarray:
        counts = self.day_close_counts[day_of_week]
        total = counts.sum()
        if total == 0:
            return np.ones(10) / 10.0
        return (counts + self.alpha) / (total + 10 * self.alpha)


class EnsemblePredictor:
    """
    Ensemble Mixture Model combining:
    1. Markov Chain Transition Likelihood (Weight: 0.30)
    2. Recency-Weighted Exponential Frequency (Weight: 0.40)
    3. Day-of-Week Seasonality (Weight: 0.30)
    Generates joint probability distribution over all 100 Jodis and single digits.
    """

    def __init__(self, w_markov: float = 0.30, w_recency: float = 0.40, w_seasonal: float = 0.30):
        self.w_markov = w_markov
        self.w_recency = w_recency
        self.w_seasonal = w_seasonal
        self.markov = MarkovChainModel()
        self.recency = RecencyWeightedModel()
        self.seasonal = DayOfWeekSeasonalModel()

    def fit(self, valid_df: pd.DataFrame):
        self.markov.fit(valid_df)
        self.recency.fit(valid_df)
        self.seasonal.fit(valid_df)

    def predict(self, day_of_week: str, prev_jodi: str = None, prev_open: int = None, prev_close: int = None) -> dict:
        # Jodi probabilities
        p_jodi_markov = self.markov.predict_jodi_probs(prev_jodi)
        p_jodi_recency = self.recency.predict_jodi_probs()
        p_jodi_seasonal = self.seasonal.predict_jodi_probs(day_of_week)
        jodi_probs = (
            self.w_markov * p_jodi_markov
            + self.w_recency * p_jodi_recency
            + self.w_seasonal * p_jodi_seasonal
        )
        jodi_probs /= jodi_probs.sum()

        # Open Digit probabilities
        p_open_markov = self.markov.predict_open_probs(prev_open)
        p_open_recency = self.recency.predict_open_probs()
        p_open_seasonal = self.seasonal.predict_open_probs(day_of_week)
        open_probs = (
            self.w_markov * p_open_markov
            + self.w_recency * p_open_recency
            + self.w_seasonal * p_open_seasonal
        )
        open_probs /= open_probs.sum()

        # Close Digit probabilities
        p_close_markov = self.markov.predict_close_probs(prev_close)
        p_close_recency = self.recency.predict_close_probs()
        p_close_seasonal = self.seasonal.predict_close_probs(day_of_week)
        close_probs = (
            self.w_markov * p_close_markov
            + self.w_recency * p_close_recency
            + self.w_seasonal * p_close_seasonal
        )
        close_probs /= close_probs.sum()

        return {
            "jodi_probs": jodi_probs,
            "open_probs": open_probs,
            "close_probs": close_probs,
        }
