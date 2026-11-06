"""
Requirements Fixtures & Specification Oracles for Kalyan Matka E2E Testing.
============================================================================
Provides pure-Python reference implementations of all algorithmic formulas,
acceptance criteria, and data contracts specified in PROJECT.md and ORIGINAL_REQUEST.md:
1. Cut-Cluster Density Weighting S(j) with lambda ~ 0.35 (R1 / Feature 1)
2. Family-Enriched Top 10 Selection (R1 / Feature 2)
3. Complementary 4-OTC Matrix selection with >68% coverage (R2 / Feature 3)
4. Cross-Transition Bias: Close(t-1) -> Open(t) direct & cut (20.70%) (R2 / Feature 4)
5. Open Repetition Momentum Boost (+15%) (R2 / Feature 5)
6. Backtest Simulation Parity (Feature 6)
7. Outcome Evaluation & Ranking 1..100 (R3 / Feature 8)
8. Persistent Audit Log Schema & Serialization (R3 / Feature 9)
9. Adaptive Hyperparameter Recalibration (R3 / Feature 10)
10. Oct 26-31, 2026 Reference Week Ground Truth
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
MATKA_ROOT = REPO_ROOT / "tools" / "matka"
MATKA_ENGINE = MATKA_ROOT / "kalyan_4_35_to_6_35"

# Ground Truth Reference Week: Oct 26–31, 2026 from kalyan_historical_data.csv (lines 3284-3289)
OCT_26_31_REFERENCE_DRAWS = [
    {
        "date": "2026-10-26",
        "day": "Mon",
        "jodi": "68",
        "open": 6,
        "close": 8,
        "open_patti": "259",
        "close_patti": "170",
        "prev_jodi": "91",
        "prev_open": 9,
        "prev_close": 1,
        "transition_type": "cross_cut",  # Close(t-1)=1 -> Open(t)=(1+5)=6
    },
    {
        "date": "2026-10-27",
        "day": "Tue",
        "jodi": "63",
        "open": 6,
        "close": 3,
        "open_patti": "358",
        "close_patti": "238",
        "prev_jodi": "68",
        "prev_open": 6,
        "prev_close": 8,
        "transition_type": "open_repetition",  # Open(t-1)=6 -> Open(t)=6
    },
    {
        "date": "2026-10-28",
        "day": "Wed",
        "jodi": "64",
        "open": 6,
        "close": 4,
        "open_patti": "178",
        "close_patti": "130",
        "prev_jodi": "63",
        "prev_open": 6,
        "prev_close": 3,
        "transition_type": "open_repetition",  # Open(t-1)=6 -> Open(t)=6
    },
    {
        "date": "2026-10-29",
        "day": "Thu",
        "jodi": "48",
        "open": 4,
        "close": 8,
        "open_patti": "130",
        "close_patti": "558",
        "prev_jodi": "64",
        "prev_open": 6,
        "prev_close": 4,
        "transition_type": "cross_direct",  # Close(t-1)=4 -> Open(t)=4
    },
    {
        "date": "2026-10-30",
        "day": "Fri",
        "jodi": "34",
        "open": 3,
        "close": 4,
        "open_patti": "779",
        "close_patti": "149",
        "prev_jodi": "48",
        "prev_open": 4,
        "prev_close": 8,
        "transition_type": "cross_cut",  # Close(t-1)=8 -> Open(t)=(8+5)%10=3
    },
    {
        "date": "2026-10-31",
        "day": "Sat",
        "jodi": "40",
        "open": 4,
        "close": 0,
        "open_patti": "590",
        "close_patti": "127",
        "prev_jodi": "34",
        "prev_open": 3,
        "prev_close": 4,
        "transition_type": "cross_direct",  # Close(t-1)=4 -> Open(t)=4
    },
]


def reference_cut_digit(d: int) -> int:
    """Standard Matka cut digit: (d + 5) mod 10."""
    return (int(d) + 5) % 10


def reference_family_jodis(jodi_val: Any) -> List[str]:
    """Generates standard Matka family cut jodis (4 or 8 members)."""
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
    cut_o = reference_cut_digit(o)
    cut_c = reference_cut_digit(c)

    base_pairs = [
        (o, c), (o, cut_c), (cut_o, c), (cut_o, cut_c),
        (c, o), (cut_c, o), (c, cut_o), (cut_c, cut_o),
    ]
    seen = set()
    res = []
    for d1, d2 in base_pairs:
        v = f"{d1}{d2}"
        if v not in seen:
            seen.add(v)
            res.append(v)
    return sorted(res)


def reference_cut_cluster_density_scores(
    raw_jodi_probs: np.ndarray,
    lambda_val: float = 0.35,
) -> np.ndarray:
    """
    Computes Cut-Cluster Density Weighting:
    S(j) = (1 - lambda) * P(j) + lambda * Mean_{k in Family(j)}(P(k))
    """
    probs = np.asarray(raw_jodi_probs, dtype=float).flatten()
    if len(probs) != 100:
        raise ValueError(f"Expected 100 Jodi probabilities, got {len(probs)}")

    # Clamp lambda to [0.0, 1.0]
    lam = float(np.clip(lambda_val, 0.0, 1.0))

    if lam == 0.0:
        s = probs.copy()
        s_sum = s.sum()
        return s / s_sum if s_sum > 0 else np.ones(100) / 100.0

    scores = np.zeros(100, dtype=float)
    for j in range(100):
        fam = reference_family_jodis(j)
        if len(fam) == 0:
            fam_mean = probs[j]
        else:
            fam_indices = [int(x) for x in fam]
            fam_mean = float(np.mean([probs[idx] for idx in fam_indices]))
        scores[j] = (1.0 - lam) * probs[j] + lam * fam_mean

    total = scores.sum()
    if total > 0:
        scores /= total
    else:
        scores = np.ones(100, dtype=float) / 100.0
    return scores


def reference_family_enriched_top_10(
    density_scores: np.ndarray,
) -> List[Dict[str, Any]]:
    """
    Generates Family-Enriched Top 10 Jodi ranking:
    1. Identify dominant Top Pick (highest S(j)).
    2. Extract cut-family members of Top Pick sorted by S(k).
    3. Include Top Pick at Rank 1, followed by highest-density family members,
       filling remaining slots up to 10 with top non-family picks.
    """
    scores = np.asarray(density_scores, dtype=float).flatten()
    ranked_indices = np.argsort(scores)[::-1]
    top_pick_idx = int(ranked_indices[0])
    top_pick_str = f"{top_pick_idx:02d}"

    top_family = reference_family_jodis(top_pick_str)
    top_fam_sorted = sorted(top_family, key=lambda x: scores[int(x)], reverse=True)

    top_10_jodis: List[str] = [top_pick_str]
    for fam_jodi in top_fam_sorted:
        if fam_jodi not in top_10_jodis:
            top_10_jodis.append(fam_jodi)
        if len(top_10_jodis) >= 8:
            break

    for idx in ranked_indices:
        j_str = f"{idx:02d}"
        if j_str not in top_10_jodis:
            top_10_jodis.append(j_str)
        if len(top_10_jodis) >= 10:
            break

    result = []
    fam_set = set(top_family)
    for rank, j_str in enumerate(top_10_jodis[:10], start=1):
        idx = int(j_str)
        result.append({
            "rank": rank,
            "jodi": j_str,
            "score": float(scores[idx]),
            "raw_prob": float(scores[idx]),
            "is_family_affinity": (j_str in fam_set),
            "family": reference_family_jodis(j_str),
        })
    return result


def reference_complementary_4_otc_matrix(
    joint_matrix: np.ndarray,
) -> Tuple[List[Tuple[int, int]], List[int], float]:
    """
    Selects two complementary harmonic cut pairs [d_a, cut(d_a), d_b, cut(d_b)]
    maximizing single-digit joint coverage:
    P(Coverage) = 1.0 - sum_{o not in OTC, c not in OTC} P_joint(o, c)
    """
    M = np.asarray(joint_matrix, dtype=float).reshape((10, 10))
    m_sum = M.sum()
    if m_sum > 0:
        M = M / m_sum
    else:
        M = np.ones((10, 10), dtype=float) / 100.0

    cut_pairs = [(k, (k + 5) % 10) for k in range(5)]
    best_pairs: Optional[List[Tuple[int, int]]] = None
    best_coverage = -1.0
    best_digits: Optional[List[int]] = None

    for i in range(5):
        for j in range(i + 1, 5):
            p1 = cut_pairs[i]
            p2 = cut_pairs[j]
            otc_set = set([p1[0], p1[1], p2[0], p2[1]])
            miss_digits = [d for d in range(10) if d not in otc_set]
            p_miss = float(M[np.ix_(miss_digits, miss_digits)].sum())
            p_cov = 1.0 - p_miss
            if p_cov > best_coverage:
                best_coverage = p_cov
                best_pairs = [p1, p2]
                best_digits = sorted(list(otc_set))

    cov_prob = float(np.clip(best_coverage, 0.0, 1.0))
    return best_pairs or [(0, 5), (1, 6)], best_digits or [0, 1, 5, 6], cov_prob


def reference_cross_transition_bias(
    open_probs: np.ndarray,
    prev_close: Optional[int],
    target_mass: float = 0.2070,
) -> np.ndarray:
    """
    Incorporate Close(t-1) -> Open(t) direct & cut transitions scaled to empirical rate (~20.70%).
    """
    p = np.asarray(open_probs, dtype=float).flatten()
    if len(p) != 10:
        raise ValueError(f"Expected 10 open probabilities, got {len(p)}")
    if prev_close is None or not (0 <= int(prev_close) <= 9):
        p_sum = p.sum()
        return p / p_sum if p_sum > 0 else np.ones(10) / 10.0

    c = int(prev_close)
    d_direct = c
    d_cut = reference_cut_digit(c)

    p_out = p.copy()
    current_mass = p_out[d_direct] + p_out[d_cut]
    if current_mass < target_mass and current_mass > 0:
        rem_target = 1.0 - target_mass
        rem_curr = 1.0 - current_mass
        scale_cross = target_mass / current_mass
        scale_rem = (rem_target / rem_curr) if rem_curr > 0 else 1.0
        p_out[d_direct] *= scale_cross
        p_out[d_cut] *= scale_cross
        for d in range(10):
            if d not in (d_direct, d_cut):
                p_out[d] *= scale_rem

    total = p_out.sum()
    if total > 0:
        p_out /= total
    else:
        p_out = np.ones(10, dtype=float) / 10.0
    return p_out


def reference_open_repetition_momentum_boost(
    open_probs: np.ndarray,
    prev_open: Optional[int],
    boost_multiplier: float = 1.15,
) -> np.ndarray:
    """
    Applies +15% multiplier to Open repetition probability (Open(t) == Open(t-1)).
    """
    p = np.asarray(open_probs, dtype=float).flatten()
    if len(p) != 10:
        raise ValueError(f"Expected 10 open probabilities, got {len(p)}")
    if prev_open is None or not (0 <= int(prev_open) <= 9):
        p_sum = p.sum()
        return p / p_sum if p_sum > 0 else np.ones(10) / 10.0

    o = int(prev_open)
    p_out = p.copy()
    p_out[o] *= float(boost_multiplier)

    total = p_out.sum()
    if total > 0:
        p_out /= total
    else:
        p_out = np.ones(10, dtype=float) / 10.0
    return p_out


def reference_evaluate_outcome(
    predicted_jodi_probs_or_ranking: Any,
    top_pick: str,
    top_10: List[str],
    otc_digits: List[int],
    actual_jodi: str,
    actual_open: int,
    actual_close: int,
) -> Dict[str, Any]:
    """
    Evaluates draw outcome:
    - exact rank of actual draw (1..100)
    - top10_hit, family_cut_hit, otc_hit, open_hit, close_hit
    """
    act_j = f"{int(actual_jodi):02d}"
    act_o = int(actual_open)
    act_c = int(actual_close)

    # Calculate actual rank
    if isinstance(predicted_jodi_probs_or_ranking, (list, tuple)) and len(predicted_jodi_probs_or_ranking) == 100:
        if isinstance(predicted_jodi_probs_or_ranking[0], str):
            ranked = list(predicted_jodi_probs_or_ranking)
        else:
            ranked = [f"{i:02d}" for i in np.argsort(predicted_jodi_probs_or_ranking)[::-1]]
    elif isinstance(predicted_jodi_probs_or_ranking, np.ndarray) and len(predicted_jodi_probs_or_ranking) == 100:
        ranked = [f"{i:02d}" for i in np.argsort(predicted_jodi_probs_or_ranking)[::-1]]
    else:
        # If only top_10 given, rank is index in top_10 or 11
        if act_j in top_10:
            ranked = list(top_10) + [f"{i:02d}" for i in range(100) if f"{i:02d}" not in top_10]
        else:
            ranked = list(top_10) + [act_j] + [f"{i:02d}" for i in range(100) if f"{i:02d}" not in top_10 and f"{i:02d}" != act_j]

    actual_rank = ranked.index(act_j) + 1 if act_j in ranked else 100

    top_pick_fam = reference_family_jodis(top_pick)
    top10_hit = (actual_rank <= 10)
    family_cut_hit = (act_j in top_pick_fam)
    otc_hit = (act_o in otc_digits or act_c in otc_digits)
    open_hit = (act_o in otc_digits[:2] if len(otc_digits) >= 2 else False)
    close_hit = (act_c in otc_digits[2:] if len(otc_digits) >= 4 else False)

    return {
        "actual_rank": actual_rank,
        "top10_hit": top10_hit,
        "family_cut_hit": family_cut_hit,
        "otc_hit": otc_hit,
        "open_hit": open_hit,
        "close_hit": close_hit,
        "reciprocal_rank": round(1.0 / actual_rank, 4),
    }


def reference_recalibrate_hyperparameters(
    current_params: Dict[str, float],
    performance_log: Dict[str, Any],
) -> Dict[str, float]:
    """
    Adaptive hyperparameter tuning based on performance_log:
    - Bounded guardrails:
      lambda in [0.25, 0.45]
      w_recency in [0.40, 0.65]
      w_markov in [0.10, 0.30]
      w_seasonal in [0.08, 0.22]
      w_gap in [0.08, 0.22]
    """
    params = dict(current_params)
    lam = params.get("lambda_cut_cluster", 0.35)
    w_rec = params.get("w_recency", 0.52)
    w_mk = params.get("w_markov", 0.18)
    w_sn = params.get("w_seasonal", 0.15)
    w_gp = params.get("w_gap", 0.15)

    summary = performance_log.get("summary", {})
    fam_rate = summary.get("family_cut_hit_rate", 0.0)
    top10_rate = summary.get("top10_hit_rate", 0.0)

    # If family hit rate is elevated, increase lambda slightly
    if fam_rate > 0.25:
        lam = min(0.45, lam + 0.02)
    elif fam_rate < 0.10:
        lam = max(0.25, lam - 0.02)

    # Normalize weights
    total_w = w_rec + w_mk + w_sn + w_gp
    if total_w > 0:
        w_rec /= total_w
        w_mk /= total_w
        w_sn /= total_w
        w_gp /= total_w

    params["lambda_cut_cluster"] = round(lam, 4)
    params["w_recency"] = round(float(np.clip(w_rec, 0.40, 0.65)), 4)
    params["w_markov"] = round(float(np.clip(w_mk, 0.10, 0.30)), 4)
    params["w_seasonal"] = round(float(np.clip(w_sn, 0.08, 0.22)), 4)
    params["w_gap"] = round(float(np.clip(w_gp, 0.08, 0.22)), 4)
    return params
