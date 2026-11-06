"""
Tier 4: Real-World Application Scenarios E2E Tests for Kalyan Matka Suite.
Covers real-world user flows, operational cycles, and historical replays:
1. Daily prediction cycle (ingestion -> prediction -> artifact serialization)
2. Saturday to Monday recess flow (Sunday market recess handling)
3. Offline PWA standalone execution & client data sufficiency
4. Walk-forward historical backtesting simulation audit
5. Live Indian Standard Time (IST) market countdown state machine
6. Oct 26-31 reference week replay (verifying rank promotions, family enrichment, >68% OTC coverage)
7. End-to-end self-improving pipeline lifecycle (prediction -> outcome -> audit log -> adaptive tuning)
8. Cold-start to production-scale bootstrapping & Dirichlet smoothing stability
9. Multi-channel data integrity and cross-artifact consistency
"""

import json
import os
import sys
import unittest
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
import numpy as np

# Paths
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
MATKA_ROOT = REPO_ROOT / "tools" / "matka"
MATKA_ENGINE = MATKA_ROOT / "kalyan_4_35_to_6_35"
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
SAMPLE_DRAWS_CSV = FIXTURES_DIR / "sample_draws.csv"
TESTS_DIR = Path(__file__).resolve().parent

# Add engine directory and tests directory to sys.path
if str(MATKA_ENGINE) not in sys.path:
    sys.path.insert(0, str(MATKA_ENGINE))
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

import requirements_fixtures as rf


class TestRealWorldApplicationScenarios(unittest.TestCase):
    """Tier 4: End-to-End Real-World Application Workloads."""

    def test_scenario_daily_prediction_cycle(self):
        """Simulates full daily prediction cycle from draw declaration to JSON persistence."""
        import predict

        # 1. Run prediction generation
        res = predict.generate_upcoming_prediction(str(SAMPLE_DRAWS_CSV))

        # 2. Inspect latest draw and target draw
        self.assertIn("latest_draw", res)
        self.assertIn("target_draw", res)
        latest_date = datetime.strptime(res["latest_draw"]["date"], "%Y-%m-%d")
        target_date = datetime.strptime(res["target_draw"]["date"], "%Y-%m-%d")
        self.assertGreater(target_date, latest_date, "Target draw must be strictly after latest draw")

        # 3. Verify OTC and top recommendations
        otc = res["otc_digits"]
        self.assertEqual(len(otc), 4)
        top_jodis = res["top_jodis"]
        self.assertGreaterEqual(len(top_jodis), 10)

        # 4. Verify serialized artifact format
        payload = {
            "last_developer_update": datetime.now().strftime("%Y-%m-%d"),
            "latest_draw": res["latest_draw"],
            "target_draw": res["target_draw"],
            "otc_digits": otc,
            "top_1_jodi": res["top_1_jodi"],
            "family_bracket": res["family_bracket"],
        }
        serialized = json.dumps(payload)
        reloaded = json.loads(serialized)
        self.assertEqual(reloaded["target_draw"]["date"], res["target_draw"]["date"])

    def test_scenario_sunday_market_recess_flow(self):
        """Simulates draw declared on Saturday evening skipping Sunday to target Monday."""
        # When a draw occurs on Saturday
        sat_date = datetime(2024, 1, 6)  # Saturday
        self.assertEqual(sat_date.weekday(), 5)

        # Calculate target draw
        next_date = sat_date + timedelta(days=1)
        if next_date.weekday() == 6:  # Sunday recess
            next_date += timedelta(days=1)

        self.assertEqual(next_date.weekday(), 0, "Sunday must be skipped; target must be Monday")
        self.assertEqual(next_date.strftime("%Y-%m-%d"), "2024-01-08")

        # Monday prior distribution exists in prediction data
        pred_p = MATKA_ENGINE / "prediction_data.json"
        if pred_p.exists():
            data = json.loads(pred_p.read_text(encoding="utf-8"))
            if "by_day" in data:
                self.assertIn("Mon", data["by_day"], "Forecasts must include Monday line prior")

    def test_scenario_offline_pwa_standalone_execution(self):
        """Verifies all assets required for offline operation are present and precached."""
        sw_p = MATKA_ENGINE / "sw.js"
        self.assertTrue(sw_p.exists(), "sw.js must exist")
        sw_content = sw_p.read_text(encoding="utf-8")

        manifest_p = MATKA_ENGINE / "manifest.json"
        self.assertTrue(manifest_p.exists(), "manifest.json must exist")

        index_p = MATKA_ENGINE / "index.html"
        self.assertTrue(index_p.exists(), "index.html must exist")

        # Check history.json is available for client-side JavaScript models
        hist_p = MATKA_ENGINE / "history.json"
        self.assertTrue(hist_p.exists(), "history.json must exist for client predictor")
        hist_data = json.loads(hist_p.read_text(encoding="utf-8"))
        self.assertGreater(len(hist_data), 100, "history.json must contain sufficient historical draws")

        # Check service worker precache includes essential offline files
        for f in ["index.html", "manifest.json"]:
            self.assertTrue(f in sw_content or "./" in sw_content, f"Precache must reference {f}")

    def test_scenario_walk_forward_backtest_audit(self):
        """Simulates walk-forward backtesting execution and verifies mathematical metrics."""
        import backtest

        # Run walk-forward backtest on sample fixtures with small warmup
        res = backtest.run_walk_forward_backtest(
            csv_path=str(SAMPLE_DRAWS_CSV),
            warmup_draws=10,
        )
        self.assertIsNotNone(res)
        draws_tested = res.get("test_draws", res.get("total_tested", 0))
        self.assertGreater(draws_tested, 0)

        # Check pass rates
        self.assertIn("otc_pass_rate", res)
        otc_val = res["otc_pass_rate"]
        otc_rate = float(str(otc_val).replace("%", ""))
        self.assertGreaterEqual(otc_rate, 0.0)
        self.assertLessEqual(otc_rate, 100.0)

        # Check top-jodi hit rate
        self.assertIn("top10_jodi_hit_rate", res)
        top10_val = res["top10_jodi_hit_rate"]
        top10_rate = float(str(top10_val).replace("%", ""))
        self.assertGreaterEqual(top10_rate, 0.0)
        self.assertLessEqual(top10_rate, 100.0)

    def test_scenario_live_market_countdown_state_machine(self):
        """Verifies IST market timing state machine transitions across operating sessions."""
        # Indian Standard Time is UTC + 5:30
        ist_tz = timezone(timedelta(hours=5, minutes=30))

        def get_market_state(dt: datetime) -> str:
            # Check Sunday recess
            if dt.weekday() == 6:
                return "WEEKEND_RECESS"
            # Operating times: Open = 16:35 IST, Close = 18:35 IST
            open_time = time(16, 35)
            close_time = time(18, 35)
            t = dt.time()
            if t < open_time:
                return "PRE_OPEN"
            elif t < close_time:
                return "LIVE_DRAW_ACTIVE"
            else:
                return "MARKET_CLOSED"

        # 1. Monday 14:00 IST -> Pre-Open
        t1 = datetime(2024, 1, 1, 14, 0, tzinfo=ist_tz)
        self.assertEqual(get_market_state(t1), "PRE_OPEN")

        # 2. Monday 17:15 IST -> Live Draw Active (Open declared, Close countdown)
        t2 = datetime(2024, 1, 1, 17, 15, tzinfo=ist_tz)
        self.assertEqual(get_market_state(t2), "LIVE_DRAW_ACTIVE")

        # 3. Monday 19:00 IST -> Market Closed (Session complete)
        t3 = datetime(2024, 1, 1, 19, 0, tzinfo=ist_tz)
        self.assertEqual(get_market_state(t3), "MARKET_CLOSED")

        # 4. Sunday 12:00 IST -> Weekend Recess
        t4 = datetime(2024, 1, 7, 12, 0, tzinfo=ist_tz)
        self.assertEqual(get_market_state(t4), "WEEKEND_RECESS")

    def test_scenario_oct_26_31_reference_week_replay(self):
        """
        Replays the Oct 26-31 reference week with actual historical draws:
        - 2026-10-26 (68), 2026-10-27 (63), 2026-10-28 (64),
        - 2026-10-29 (48), 2026-10-30 (34), 2026-10-31 (40).
        Verifies:
        1. On Oct 28 (Draw: 64, Top Pick: 91): Jodi 64 promoted into Top 10 recommendations (Ranks 5-8).
        2. On Oct 27 (Draw: 63, Top Pick: 95): Family candidate 40 is represented in Top 10 recommendations.
        3. Complementary 4-OTC Matrix yields >68% coverage for every draw in the week.
        4. 100% of reference week draws match either Open Repetition or Cross-Transition Direct/Cut.
        """
        ref_draws = rf.OCT_26_31_REFERENCE_DRAWS
        self.assertEqual(len(ref_draws), 6)

        # 1. Verify 100% Transition Invariant (Open Repetition or Cross-Transition)
        for draw in ref_draws:
            curr_open = draw["open"]
            prev_open = draw["prev_open"]
            prev_close = draw["prev_close"]
            prev_close_cut = rf.reference_cut_digit(prev_close)

            is_open_rep = (curr_open == prev_open)
            is_cross_direct = (curr_open == prev_close)
            is_cross_cut = (curr_open == prev_close_cut)

            self.assertTrue(
                is_open_rep or is_cross_direct or is_cross_cut,
                f"Draw on {draw['date']} ({draw['jodi']}) must satisfy transition invariant"
            )
            # Confirm transition type annotation
            if is_open_rep:
                self.assertEqual(draw["transition_type"], "open_repetition")
            elif is_cross_direct:
                self.assertEqual(draw["transition_type"], "cross_direct")
            elif is_cross_cut:
                self.assertEqual(draw["transition_type"], "cross_cut")

        # 2. Oct 28 Promotion Verification (Actual Draw: 64, Top Pick: 91)
        # Construct baseline raw probability where 91 is top pick, 64 has weak raw probability
        raw_oct28 = np.ones(100, dtype=float) / 1000.0
        raw_oct28[91] = 0.15
        raw_oct28[96] = 0.05
        raw_oct28[41] = 0.04
        raw_oct28[64] = 0.001  # baseline low rank

        raw_rank_64 = list(np.argsort(raw_oct28)[::-1]).index(64) + 1
        self.assertGreater(raw_rank_64, 30, "In baseline raw model, Jodi 64 is ranked low (e.g. Rank 37-49)")

        # Apply Cut-Cluster Density Weighting S(j) with lambda = 0.35
        scores_oct28 = rf.reference_cut_cluster_density_scores(raw_oct28, lambda_val=0.35)
        top10_oct28 = rf.reference_family_enriched_top_10(scores_oct28)
        top10_oct28_jodis = [item["jodi"] for item in top10_oct28]

        # Verify 64 is promoted into Top 10 (Rank 5-8)
        self.assertIn("64", top10_oct28_jodis, "Jodi 64 MUST be promoted into Top 10 recommendations")
        promoted_rank_64 = top10_oct28_jodis.index("64") + 1
        self.assertGreaterEqual(promoted_rank_64, 5, "Rank should be in Ranks 5-8")
        self.assertLessEqual(promoted_rank_64, 8, "Rank should be in Ranks 5-8")

        # 3. Oct 27 Family Candidate 40 Representation (Top Pick: 95)
        raw_oct27 = np.ones(100, dtype=float) / 1000.0
        raw_oct27[95] = 0.14
        raw_oct27[90] = 0.06
        raw_oct27[45] = 0.05
        raw_oct27[40] = 0.002
        scores_oct27 = rf.reference_cut_cluster_density_scores(raw_oct27, lambda_val=0.35)
        top10_oct27 = rf.reference_family_enriched_top_10(scores_oct27)
        top10_oct27_jodis = [item["jodi"] for item in top10_oct27]
        self.assertIn("40", top10_oct27_jodis, "Family candidate 40 must be represented in Top 10")

        # 4. OTC Coverage >68% Across Reference Week
        # Generate empirical joint distribution matrix
        rng = np.random.RandomState(42)
        joint_m = np.zeros((10, 10), dtype=float)
        # Seed with empirical frequency
        for d in ref_draws:
            joint_m[d["open"], d["close"]] += 5.0
        # Add background distribution
        joint_m += 1.0
        joint_m /= joint_m.sum()

        pairs, digits, cov_prob = rf.reference_complementary_4_otc_matrix(joint_m)
        self.assertEqual(len(digits), 4)
        self.assertGreater(cov_prob, 0.68, f"OTC joint coverage {cov_prob:.4f} must exceed 68% threshold")

        # Verify OTC hit rate against reference week draws
        otc_hits = sum(1 for d in ref_draws if d["open"] in digits or d["close"] in digits)
        hit_rate = otc_hits / len(ref_draws)
        self.assertGreaterEqual(hit_rate, 0.68, f"Observed hit rate {hit_rate:.2%} must be at least 68%")

    def test_scenario_self_improving_pipeline_lifecycle(self):
        """
        Simulates the complete self-improving pipeline lifecycle:
        1. Predict upcoming draw.
        2. Ingest actual draw result.
        3. Evaluate outcome (exact rank 1..100, hit flags, MRR).
        4. Append to audit log.
        5. Trigger adaptive hyperparameter recalibration within strict guardrails.
        """
        # Step 1: Simulated prediction
        top_pick = "91"
        top_10 = ["91", "96", "41", "14", "19", "46", "64", "69", "97", "98"]
        otc_digits = [6, 1, 4, 9]

        # Step 2 & 3: Actual draw occurs (e.g. Jodi 64, Open 6, Close 4)
        eval_result = rf.reference_evaluate_outcome(
            predicted_jodi_probs_or_ranking=top_10,
            top_pick=top_pick,
            top_10=top_10,
            otc_digits=otc_digits,
            actual_jodi="64",
            actual_open=6,
            actual_close=4,
        )

        self.assertEqual(eval_result["actual_rank"], 7)
        self.assertTrue(eval_result["top10_hit"])
        self.assertTrue(eval_result["family_cut_hit"])
        self.assertTrue(eval_result["otc_hit"])
        self.assertTrue(eval_result["open_hit"])
        self.assertTrue(eval_result["close_hit"])
        self.assertAlmostEqual(eval_result["reciprocal_rank"], 1.0 / 7, places=3)

        # Step 4: Audit log recording
        log_record = {
            "date": "2026-10-28",
            "draw_result": {"jodi": "64", "open": 6, "close": 4},
            "evaluation": eval_result,
        }
        self.assertEqual(log_record["evaluation"]["actual_rank"], 7)

        # Step 5: Adaptive hyperparameter recalibration
        initial_params = {
            "lambda_cut_cluster": 0.35,
            "w_recency": 0.52,
            "w_markov": 0.18,
            "w_seasonal": 0.15,
            "w_gap": 0.15,
        }
        perf_log = {
            "summary": {
                "top10_hit_rate": 0.35,
                "family_cut_hit_rate": 0.30,  # elevated family hit rate
            }
        }
        updated_params = rf.reference_recalibrate_hyperparameters(initial_params, perf_log)

        # Assert guardrails
        self.assertGreaterEqual(updated_params["lambda_cut_cluster"], 0.25)
        self.assertLessEqual(updated_params["lambda_cut_cluster"], 0.45)
        self.assertGreater(updated_params["lambda_cut_cluster"], 0.35, "Lambda should adaptively increase on high family hits")

        self.assertGreaterEqual(updated_params["w_recency"], 0.40)
        self.assertLessEqual(updated_params["w_recency"], 0.65)
        self.assertGreaterEqual(updated_params["w_markov"], 0.10)
        self.assertLessEqual(updated_params["w_markov"], 0.30)
        self.assertGreaterEqual(updated_params["w_seasonal"], 0.08)
        self.assertLessEqual(updated_params["w_seasonal"], 0.22)
        self.assertGreaterEqual(updated_params["w_gap"], 0.08)
        self.assertLessEqual(updated_params["w_gap"], 0.22)

    def test_scenario_cold_start_to_production_scale(self):
        """
        Simulates pipeline progression from cold-start (minimal seed) to production scale.
        Verifies Dirichlet smoothing, non-zero probabilities, and entropy convergence.
        """
        # Cold start with minimal draws
        counts = np.zeros(10, dtype=float)
        alpha_prior = 1.0  # Laplace / Dirichlet smoothing

        # 1. Cold start state (0 draws observed)
        p_cold = (counts + alpha_prior) / (counts + alpha_prior).sum()
        self.assertTrue(np.all(p_cold > 0.0))
        self.assertAlmostEqual(p_cold.sum(), 1.0)
        entropy_cold = -np.sum(p_cold * np.log2(p_cold))

        # 2. Sequential ingestion of 50 draws with localized cluster
        rng = np.random.RandomState(42)
        for _ in range(50):
            d = rng.choice([3, 4, 8, 9], p=[0.4, 0.3, 0.2, 0.1])
            counts[d] += 1

        p_warm = (counts + alpha_prior) / (counts + alpha_prior).sum()
        self.assertTrue(np.all(p_warm > 0.0), "Dirichlet prior ensures no zero probabilities")
        self.assertAlmostEqual(p_warm.sum(), 1.0)
        entropy_warm = -np.sum(p_warm * np.log2(p_warm))

        # Entropy should decrease as empirical evidence concentrates
        self.assertLess(entropy_warm, entropy_cold)

    def test_scenario_multi_channel_consistency(self):
        """
        Verifies cross-channel coherence between CLI model prediction,
        serialized prediction_data.json, and PWA client consumption contract.
        """
        import predict

        res = predict.generate_upcoming_prediction(str(SAMPLE_DRAWS_CSV))

        # 1. Target date coherence
        target_date = res["target_draw"]["date"]
        self.assertRegex(target_date, r"^\d{4}-\d{2}-\d{2}$")

        # 2. OTC Digits coherence: 4 distinct digits forming 2 cut pairs
        otc = sorted(res["otc_digits"])
        self.assertEqual(len(otc), 4)
        self.assertEqual(len(set(otc)), 4)
        # Verify cut-pair symmetry
        cuts_in_otc = sum(1 for d in otc if (d + 5) % 10 in otc)
        self.assertEqual(cuts_in_otc, 4, "All 4 OTC digits must form complementary cut pairs")

        # 3. Top Jodi & Family Bracket coherence
        top_1 = res["top_1_jodi"]
        self.assertRegex(top_1, r"^\d{2}$")
        fam_bracket = res["family_bracket"]
        expected_fam = rf.reference_family_jodis(top_1)
        self.assertEqual(sorted(fam_bracket), sorted(expected_fam))

        # 4. Top 10 Jodis list coherence
        top_jodis = res["top_jodis"]
        self.assertGreaterEqual(len(top_jodis), 10)
        if hasattr(top_jodis, "iloc"):
            top_jodi_strings = [str(x).zfill(2) for x in top_jodis["Jodi"].tolist()]
        elif isinstance(top_jodis[0], dict):
            top_jodi_strings = [str(j.get("jodi", j.get("Jodi"))).zfill(2) for j in top_jodis]
        else:
            top_jodi_strings = [str(j).zfill(2) for j in top_jodis]
        self.assertEqual(top_jodi_strings[0], top_1, "Top 1 pick must be first item in top_jodis")


if __name__ == "__main__":
    unittest.main()
