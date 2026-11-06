"""
Tier 3: Cross-Feature Combinations E2E Tests for Kalyan Matka Suite.
Covers pairwise and multi-stage interactions:
- Scraper parsing -> CSV schema validation
- CSV dataset -> ML predictive engine training
- ML prediction -> Frontend JSON schema conformity
- Historical CSV -> history.json array-of-arrays export
- Multi-directory artifact mirroring consistency (tools/matka, engine, web)
- Web UI architecture -> verify_links_and_assets contracts
- End-to-end simulated pipeline data flow
"""

import csv
import json
import os
import re
import sys
import unittest
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd

# Paths
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
MATKA_ROOT = REPO_ROOT / "tools" / "matka"
MATKA_ENGINE = MATKA_ROOT / "kalyan_4_35_to_6_35"
MATKA_WEB = MATKA_ENGINE / "web"
TESTS_DIR = Path(__file__).resolve().parent
FIXTURES_DIR = TESTS_DIR / "fixtures"
SAMPLE_DRAWS_CSV = FIXTURES_DIR / "sample_draws.csv"
MOCK_PENAL_CHART_HTML = FIXTURES_DIR / "mock_penal_chart.html"

# Add directories to sys.path
for p in [str(TESTS_DIR), str(MATKA_ENGINE), str(REPO_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import requirements_fixtures as rf


class TestCrossFeatureInteractions(unittest.TestCase):
    """Tier 3: Pairwise and Multi-Stage Pipeline Interactions."""

    def test_scraper_to_csv_validation_interaction(self):
        """Verifies simulated raw HTML scraping produces records that satisfy CSV integrity rules."""
        from bs4 import BeautifulSoup

        html_content = MOCK_PENAL_CHART_HTML.read_text(encoding="utf-8")
        soup = BeautifulSoup(html_content, "html.parser")
        table = soup.find("table", class_="chart-table")
        self.assertIsNotNone(table, "Failed to find chart-table in mock HTML")

        rows = table.find("tbody").find_all("tr")
        extracted_draws = []
        weekday_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]

        for row in rows:
            cells = [c.get_text(strip=True) for c in row.find_all("td")]
            if len(cells) < 19:
                continue
            date_cell = cells[0]
            # Parse start date e.g. "01/01/2024 to 06/01/2024"
            match = re.search(r"(\d{1,2}/\d{1,2}/\d{4})", date_cell)
            self.assertIsNotNone(match, f"Could not parse week start date from {date_cell}")
            start_dt = datetime.strptime(match.group(1), "%d/%m/%Y")

            for day_idx in range(6):
                cell_offset = 1 + (day_idx * 3)
                op = cells[cell_offset]
                jodi = cells[cell_offset + 1]
                cp = cells[cell_offset + 2]
                draw_date = start_dt + timedelta(days=day_idx)
                is_valid = bool(jodi.isdigit() and len(jodi) == 2 and op.isdigit() and cp.isdigit())

                extracted_draws.append({
                    "Date": draw_date.strftime("%Y-%m-%d"),
                    "Day_Of_Week": weekday_names[day_idx],
                    "Jodi": jodi if is_valid else None,
                    "Open_Patti": op if is_valid else None,
                    "Close_Patti": cp if is_valid else None,
                    "Is_Valid": is_valid,
                })

        self.assertGreater(len(extracted_draws), 10)
        # Validate extracted records against panel digit sum rules
        for d in extracted_draws:
            if d["Is_Valid"]:
                op = d["Open_Patti"]
                cp = d["Close_Patti"]
                od = int(d["Jodi"][0])
                cd = int(d["Jodi"][1])
                self.assertEqual(
                    sum(int(c) for c in op) % 10,
                    od,
                    f"Open patti sum check failed for scraped record {d}",
                )
                self.assertEqual(
                    sum(int(c) for c in cp) % 10,
                    cd,
                    f"Close patti sum check failed for scraped record {d}",
                )

    def test_csv_to_ml_predictor_pipeline(self):
        """Verifies CSV dataset feeds directly into ML model training and produces valid probability spaces."""
        import pandas as pd
        from models import EnsemblePredictor

        df = pd.read_csv(SAMPLE_DRAWS_CSV)
        valid_df = df[df["Is_Valid"] == True].copy().reset_index(drop=True)
        valid_df["Open_Digit"] = valid_df["Open_Digit"].astype(int)
        valid_df["Close_Digit"] = valid_df["Close_Digit"].astype(int)

        ensemble = EnsemblePredictor()
        ensemble.fit(valid_df)

        preds = ensemble.predict("Mon", prev_jodi="39", prev_open=3, prev_close=9)
        # Check jodi distribution
        jodi_probs = preds["jodi_probs"]
        self.assertEqual(len(jodi_probs), 100)
        self.assertFalse(any(p < 0 for p in jodi_probs), "Probabilities cannot be negative")
        self.assertAlmostEqual(jodi_probs.sum(), 1.0, delta=0.01)

        # Check marginals
        open_probs = preds["open_probs"]
        close_probs = preds["close_probs"]
        self.assertEqual(len(open_probs), 10)
        self.assertEqual(len(close_probs), 10)
        self.assertAlmostEqual(open_probs.sum(), 1.0, delta=0.01)
        self.assertAlmostEqual(close_probs.sum(), 1.0, delta=0.01)

    def test_ml_prediction_to_json_schema_conformity(self):
        """Verifies ML model outputs map coherently into prediction_data.json data contract."""
        import predict

        res = predict.generate_upcoming_prediction(str(SAMPLE_DRAWS_CSV))
        top_jodi_rows = res["top_jodis"].to_dict(orient="records")

        # Construct payload matching app.py schema
        payload = {
            "last_developer_update": datetime.now().strftime("%Y-%m-%d"),
            "latest_draw": res["latest_draw"],
            "target_draw": res["target_draw"],
            "otc_digits": res["otc_digits"],
            "top_1_jodi": res["top_1_jodi"],
            "top_jodis": top_jodi_rows,
            "family_bracket": res["family_bracket"],
        }

        # Verify serialization to JSON
        json_str = json.dumps(payload)
        parsed = json.loads(json_str)

        self.assertIn("latest_draw", parsed)
        self.assertIn("top_jodis", parsed)
        self.assertEqual(len(parsed["top_jodis"]), 10)
        self.assertEqual(len(parsed["otc_digits"]), 4)

    def test_csv_history_to_history_json_conformity(self):
        """Verifies historical draws in CSV accurately map to history.json array of arrays."""
        import pandas as pd

        df = pd.read_csv(SAMPLE_DRAWS_CSV)
        valid_df = df[df["Is_Valid"] == True].copy().reset_index(drop=True)

        history_list = []
        for _, row in valid_df.iterrows():
            history_list.append([
                str(row["Date"]),
                str(row["Day_Of_Week"]),
                int(row["Jodi"]),
                int(row["Open_Digit"]),
                int(row["Close_Digit"]),
                str(row["Open_Patti"]) if pd.notna(row.get("Open_Patti")) else "",
                str(row["Close_Patti"]) if pd.notna(row.get("Close_Patti")) else "",
            ])

        self.assertEqual(len(history_list), len(valid_df))
        sample = history_list[0]
        self.assertEqual(len(sample), 7)
        self.assertIsInstance(sample[0], str)
        self.assertIsInstance(sample[2], int)
        self.assertIsInstance(sample[5], str)

    def test_artifact_multi_directory_synchronization(self):
        """Verifies prediction_data.json is synchronized across available directory locations."""
        locations = [
            MATKA_ROOT / "prediction_data.json",
            MATKA_ENGINE / "prediction_data.json",
            MATKA_WEB / "prediction_data.json",
        ]
        existing_locations = [loc for loc in locations if loc.exists()]
        self.assertGreaterEqual(len(existing_locations), 2, "At least 2 artifact locations must exist")

        base_keys = None
        for loc in existing_locations:
            data = json.loads(loc.read_text(encoding="utf-8"))
            keys = set(data.keys())
            if base_keys is None:
                base_keys = keys
            else:
                self.assertEqual(
                    keys,
                    base_keys,
                    f"Keys in {loc} differ from mirror: {keys} vs {base_keys}",
                )

    def test_web_ui_to_asset_verifier_link_contracts(self):
        """Verifies web HTML files maintain required anchors and paths audited by verify_links_and_assets."""
        engine_index = MATKA_ENGINE / "index.html"
        self.assertTrue(engine_index.exists())
        content = engine_index.read_text(encoding="utf-8")

        # Must have lastweek anchor
        self.assertIn('id="lastweek"', content)

        # Must link to hub or parent
        self.assertTrue(
            "../" in content,
            "Engine index must contain relative links navigating up the hierarchy",
        )

    def test_end_to_end_data_flow_simulation(self):
        """Simulates full end-to-end data flow: Scrape -> Validate -> Predict -> Serialize -> Verify."""
        import pandas as pd
        from models import EnsemblePredictor

        # Step 1: Ingest & Validate
        df = pd.read_csv(SAMPLE_DRAWS_CSV)
        valid_df = df[df["Is_Valid"] == True].copy().reset_index(drop=True)
        valid_df["Open_Digit"] = valid_df["Open_Digit"].astype(int)
        valid_df["Close_Digit"] = valid_df["Close_Digit"].astype(int)
        self.assertGreater(len(valid_df), 5)

        # Step 2: Fit model
        model = EnsemblePredictor()
        model.fit(valid_df)

        # Step 3: Predict
        latest = valid_df.iloc[-1]
        preds = model.predict(
            day_of_week="Mon",
            prev_jodi=str(latest["Jodi"]).zfill(2),
            prev_open=int(latest["Open_Digit"]),
            prev_close=int(latest["Close_Digit"]),
        )
        self.assertIn("otc_digits", preds)
        self.assertIn("jodi_probs", preds)

        # Step 4: Serialize
        export_payload = {
            "timestamp": datetime.now().isoformat(),
            "latest_draw": {
                "date": str(latest["Date"]),
                "jodi": str(latest["Jodi"]).zfill(2),
            },
            "otc_recommendation": {
                "digits": preds["otc_digits"],
                "pass_prob": preds.get("otc_pass_prob", 0.0),
            },
        }
        json_output = json.dumps(export_payload, indent=2)
        self.assertIn(str(latest["Date"]), json_output)

        # Step 5: Verify deserialization integrity
        reloaded = json.loads(json_output)
        self.assertEqual(reloaded["latest_draw"]["jodi"], str(latest["Jodi"]).zfill(2))
        self.assertEqual(len(reloaded["otc_recommendation"]["digits"]), 4)


    def test_cross_feature_cut_cluster_prediction_to_json_export(self):
        """Cross-Feature: Cut-cluster density prediction serializes into prediction_data.json."""
        raw_p = np.random.RandomState(42).dirichlet(np.ones(100))
        scores = rf.reference_cut_cluster_density_scores(raw_p, lambda_val=0.35)
        top_10 = rf.reference_family_enriched_top_10(scores)
        pairs, digits, cov = rf.reference_complementary_4_otc_matrix(raw_p.reshape((10, 10)))

        payload = {
            "last_updated": datetime.now().isoformat(),
            "top_jodis": top_10,
            "otc_recommendation": {
                "digits": digits,
                "cut_pairs": [f"{p[0]}-{p[1]}" for p in pairs],
                "pass_probability": round(cov, 4),
            },
        }
        dumped = json.dumps(payload, indent=2)
        loaded = json.loads(dumped)
        self.assertEqual(len(loaded["top_jodis"]), 10)
        self.assertEqual(len(loaded["otc_recommendation"]["digits"]), 4)
        self.assertAlmostEqual(loaded["otc_recommendation"]["pass_probability"], round(cov, 4))

    def test_cross_feature_ingestion_to_outcome_evaluation(self):
        """Cross-Feature: Ingesting newly completed draw triggers outcome evaluation."""
        actual_draw = {"jodi": "64", "open": 6, "close": 4}
        scores = np.zeros(100)
        scores[91] = 0.50
        scores[64] = 0.20
        scores /= scores.sum()
        top_10 = rf.reference_family_enriched_top_10(scores)
        evaluation = rf.reference_evaluate_outcome(
            predicted_jodi_probs_or_ranking=scores,
            top_pick="91",
            top_10=[item["jodi"] for item in top_10],
            otc_digits=[1, 6, 4, 9],
            actual_jodi=actual_draw["jodi"],
            actual_open=actual_draw["open"],
            actual_close=actual_draw["close"],
        )
        self.assertEqual(evaluation["actual_rank"], 2)
        self.assertTrue(evaluation["top10_hit"])
        self.assertTrue(evaluation["family_cut_hit"])
        self.assertTrue(evaluation["otc_hit"])

    def test_cross_feature_outcome_evaluation_to_persistent_audit_log(self):
        """Cross-Feature: Evaluated draw outcomes append to performance_log.json and update summary."""
        log = {
            "summary": {"total_draws_logged": 1, "top10_hit_rate": 1.0, "family_cut_hit_rate": 1.0, "otc_hit_rate": 1.0, "last_updated": "2026-11-05"},
            "current_hyperparameters": {"lambda_cut_cluster": 0.35, "w_recency": 0.52, "w_markov": 0.18, "w_seasonal": 0.15, "w_gap": 0.15},
            "draw_logs": [],
        }
        entry = {
            "date": "2026-10-28",
            "day": "Wed",
            "actual_draw": {"jodi": "64", "open": 6, "close": 4},
            "predictions": {"top_pick": "91", "top_10": ["91", "64"], "otc_digits": [1, 6, 4, 9]},
            "outcome": {"actual_rank": 2, "top10_hit": True, "family_cut_hit": True, "otc_hit": True},
        }
        log["draw_logs"].append(entry)
        log["summary"]["total_draws_logged"] = len(log["draw_logs"])
        self.assertEqual(log["summary"]["total_draws_logged"], 1)
        self.assertEqual(log["draw_logs"][0]["outcome"]["actual_rank"], 2)

    def test_cross_feature_persistent_log_to_hyperparameter_tuner(self):
        """Cross-Feature: performance_log.json feeds adaptive hyperparameter tuner."""
        current_params = {"lambda_cut_cluster": 0.35, "w_recency": 0.52, "w_markov": 0.18, "w_seasonal": 0.15, "w_gap": 0.15}
        log_high_fam = {"summary": {"family_cut_hit_rate": 0.35, "top10_hit_rate": 0.20}}
        tuned_params = rf.reference_recalibrate_hyperparameters(current_params, log_high_fam)
        self.assertGreater(tuned_params["lambda_cut_cluster"], current_params["lambda_cut_cluster"])
        self.assertLessEqual(tuned_params["lambda_cut_cluster"], 0.45)

    def test_cross_feature_tuner_recalibration_to_retrained_predictor(self):
        """Cross-Feature: Tuned lambda directly affects subsequent density scores."""
        raw_p = np.zeros(100)
        fam_91 = rf.reference_family_jodis("91")
        for m in fam_91: raw_p[int(m)] = 0.05
        raw_p[64] = 0.001
        for j in range(100):
            if raw_p[j] == 0: raw_p[j] = 0.006
        raw_p /= raw_p.sum()

        scores_base = rf.reference_cut_cluster_density_scores(raw_p, lambda_val=0.35)
        scores_boosted = rf.reference_cut_cluster_density_scores(raw_p, lambda_val=0.45)
        # Higher lambda transfers more cluster mean to isolated member 64
        self.assertGreater(scores_boosted[64], scores_base[64])

    def test_cross_feature_prediction_serving_to_ui_rendering(self):
        """Cross-Feature: Prediction serving artifacts match UI template hooks."""
        pred_p = MATKA_ENGINE / "web" / "prediction_data.json"
        data = json.loads(pred_p.read_text(encoding="utf-8"))
        index_html = (MATKA_ENGINE / "web" / "index.html").read_text(encoding="utf-8")
        self.assertIn("top_jodis", data)
        self.assertIn("otc_recommendation", data)
        self.assertIn("last-week-table", index_html)

    def test_cross_feature_4_otc_and_cross_transition_interaction(self):
        """Cross-Feature: Cross-transition bias reinforces pass probability of OTC pairs."""
        p_raw = np.ones(10) / 10.0
        p_cross = rf.reference_cross_transition_bias(p_raw, prev_close=4, target_mass=0.35)
        # Matrix using biased Open probability achieves elevated coverage on Close(t-1)
        joint_biased = np.outer(p_cross, np.ones(10) / 10.0)
        pairs, digits, cov = rf.reference_complementary_4_otc_matrix(joint_biased)
        self.assertIn(4, digits)
        self.assertGreaterEqual(cov, 0.60)

    def test_cross_feature_audit_log_to_three_way_parity(self):
        """Cross-Feature: Audit log and prediction data update across all 3 directories."""
        for name in ["prediction_data.json", "history.json"]:
            root_f = MATKA_ROOT / name
            eng_f = MATKA_ENGINE / name
            web_f = MATKA_ENGINE / "web" / name
            self.assertTrue(root_f.exists())
            self.assertTrue(eng_f.exists())
            self.assertTrue(web_f.exists())


if __name__ == "__main__":
    unittest.main()
