# Kalyan Matka Quantitative Analytics & Predictive Suite — Test Infrastructure

## Overview & Architecture

The Kalyan Matka Test Infrastructure is a comprehensive, requirement-driven, opaque-box E2E test harness built exclusively with the Python standard library (`unittest`, `pathlib`, `json`, `csv`, `subprocess`, `re`, `datetime`). It requires **zero external pip or npm dependencies** and runs without `scipy`.

The test suite validates the full lifecycle of the Kalyan Matka platform: data ingestion and scraping, multi-point CSV dataset integrity, quantitative predictive models (Markov transitions, tri-horizon recency, harmonic cut resonance), JSON schema adherence, web asset presentation (Zev Glass 2.0), link and anchor integrity, and offline PWA service worker resilience.

```
tools/matka/tests/
├── __init__.py
├── run_e2e_tests.py                 # Primary runner (CLI flags: --tier, --verbose, --report)
├── fixtures/                        # Clean mock/test fixtures
│   ├── __init__.py
│   ├── sample_draws.csv             # Valid multi-year historical draw fixture
│   ├── corrupted_draws.csv          # Corrupted draw fixture for boundary testing
│   └── mock_penal_chart.html        # Mock raw HTML table structure
├── test_tier1_feature_coverage.py   # Tier 1: Core Feature Coverage (>=5 tests/feature)
├── test_tier2_boundary_corner.py    # Tier 2: Boundary & Corner Cases (>=5 tests/feature)
├── test_tier3_cross_feature.py      # Tier 3: Cross-Feature Combinations & Pipelines
└── test_tier4_real_world.py         # Tier 4: Real-World Scenarios & Workload Flows
```

---

## 4-Tier Test Taxonomy

### Tier 1: Core Feature Coverage
Validates the fundamental interfaces and expected primary behaviors across all subsystem layers:
- **CLI Commands (`TestCLICommands`)**: Execution of `predict.py`, `backtest.py`, `sync_and_push.py` with arguments, stdout formatting, and path portability.
- **Predictive Outputs (`TestPredictiveOutputs`)**: Structural integrity of Top-10 Jodis, unit-sum marginal probability vectors for Open and Close digits (10 values summing to 100%), exact 4-Ank OTC sets, harmonic cut pairs, family brackets, and Single/Double Patti panel forecasts.
- **Historical CSV Verification (`TestCSVVerification`)**: Compliance with the 11-column interface schema, date monotonicity, panel digit sum rule (`sum(patti) % 10 == digit`), Jodi consistency (`f"{Open}{Close}"`), Gregorian calendar weekday alignment, and Patti type classification (SP, DP, TP).
- **Web UI Assets (`TestWebAssets`)**: HTML5 validation, viewport meta tags, dark glassmorphic DOM structures, penal chart pages, favicon SVG, and PWA icons.
- **Link Health (`TestLinkHealth`)**: Verification against `verify_links_and_assets.py --local-only`, preservation of `#lastweek` DOM anchor in `kalyan_4_35_to_6_35/index.html`, Matka hub back-links (`../../index.html`), and breadcrumb hierarchies.
- **Schema Compliance (`TestSchemaCompliance`)**: Structure of `prediction_data.json` (`latest_draw`, `by_day` covering Mon-Sat), `history.json` array-of-arrays representation, and mirror consistency.
- **Offline PWA Readiness (`TestOfflinePWAReadiness`)**: Manifest declarations (`name`, `start_url`, `display: standalone`, `icons`), service worker lifecycle listeners (`install`, `activate`, `fetch`), precache manifest, and zero external tracking telemetry.

### Tier 2: Boundary & Corner Cases
Stress-tests the analytical engine under extreme, corrupted, and edge conditions:
- **Edge Dates & Leap Years (`TestEdgeDatesAndLeapYears`)**: Quadrennial leap day parsing (`2016-02-29`, `2020-02-29`, `2024-02-29`), non-leap year rejection (`2023-02-29`), year boundary rollover (Dec 31 -> Jan 01), 28/29/30/31-day month boundaries, Saturday draw skipping Sunday (+2 days) to Monday, and earliest historical draw (`2012-12-31`).
- **Market Holidays & Incomplete Draws (`TestMarketHolidaysAndIncompleteDraws`)**: Handling of holiday records (`Is_Valid == False`, `**`, `---`), model training resilience when blank draws exist, multi-day consecutive market recess, and dataset sanity.
- **Patti Edge Combinations (`TestPattiEdgeCombinations`)**: Verification of all 10 Triple Patti combinations (`000` through `999`), representative Double Patti combinations across all digits 0-9 (`118`, `227`, `339`, `448`, etc.), Single Patti combinations (`127`, `128`, `149`, `159`, etc.), and cyclic modulo 10 sums.
- **Invalid CLI Flags & Resilience (`TestInvalidCLIFlagsAndArguments`)**: Graceful non-zero exit and error reporting for unrecognized flags, non-existent CSV paths, 0-byte CSV files, and negative numeric parameters.
- **Missing Files Fallback (`TestMissingFilesFallbackAndRelocation`)**: Scraper cache fallback during network outages, model training on minimal synthetic dataframes, resilience against missing optional columns, and working directory independence.
- **Leap Years & Multi-Year Temporal Intervals (`TestLeapYearsAndTemporalBoundaries`)**: Century leap rules (2000 leap vs 1900 non-leap), leap vs non-leap Feb 28 transitions, and full Mon-Sat operating weekday coverage.

### Tier 3: Cross-Feature Combinations
Audits cross-module contracts and multi-step data transformations:
- **Scraper ↔ CSV Validation**: Scraped raw table cells from `mock_penal_chart.html` parse into records strictly satisfying CSV validation contracts.
- **CSV Dataset ↔ ML Predictor**: Sanitized CSV records feed into `EnsemblePredictor.fit()` without NaN, Inf, or divide-by-zero errors.
- **ML Prediction ↔ JSON Schema Conformity**: Predictive engine output serializes directly into the frontend `prediction_data.json` schema.
- **CSV ↔ History JSON Export**: Historical draw rows map losslessly into the `history.json` array-of-arrays representation.
- **Three-Way Artifact Mirroring**: Validates identical keys and structure across root `tools/matka/`, engine `kalyan_4_35_to_6_35/`, and local server `web/`.
- **Web UI ↔ Repository Verifier**: Asserts web assets pass repository-wide static asset and link checks.
- **End-to-End Pipeline Simulation**: Ingest mock HTML -> Validate CSV -> Fit model -> Compute predictions -> Export JSON -> Verify output.

### Tier 4: Real-World Workload Scenarios
Validates end-to-end user journeys and system state flows:
- **Daily Prediction Cycle**: Simulates complete daily draw announcement, model retraining, upcoming draw prediction, and JSON persistence.
- **Saturday to Monday Recess Flow**: Ingestion of a Saturday draw correctly bypasses Sunday and prepares predictions for Monday using Monday line prior distributions.
- **Offline PWA Standalone Operation**: Verifies all required assets (`index.html`, `manifest.json`, `favicon.svg`, `prediction_data.json`, `history.json`) are cached and capable of powering client-side predictions without internet connectivity.
- **Walk-Forward Historical Backtesting Simulation**: Runs out-of-sample backtest simulation, verifying rolling window progression, hit rates (0% to 100%), and financial metric calculations without lookahead bias.
- **Live IST Market Countdown State Machine**: State-machine validation for Indian Standard Time (UTC+5:30):
  - Pre-Open (< 16:35 IST)
  - Live Draw Active (16:35 - 18:35 IST)
  - Market Closed (>= 18:35 IST)
  - Weekend Recess (Sunday all day)

---

## Feature Inventory Mapping

| Feature # | Feature Name | Primary Milestone | Test Module | Test Method(s) |
|---|---|---|---|---|
| F1 | Zev Glass 2.0 Obsidian Aesthetic | M3 | `test_tier1_feature_coverage.py` | `test_root_matka_index_html_structure`, `test_engine_index_html_structure` |
| F2 | Live IST Market Countdown Engine | M3 | `test_tier4_real_world.py` | `test_scenario_live_market_countdown_state_machine` |
| F3 | Zero Third-Party Telemetry | M3 | `test_tier1_feature_coverage.py` | `test_no_external_telemetry_links` |
| F4 | Offline PWA Resilience & Caching | M3 | `test_tier1_feature_coverage.py`, `test_tier4_real_world.py` | `test_manifest_json_required_fields`, `test_sw_js_lifecycle_events`, `test_scenario_offline_pwa_standalone_execution` |
| F5 | Modernized Historical Penal Chart | M3 | `test_tier1_feature_coverage.py` | `test_penal_chart_html_files_exist` |
| F6 | Anchor `#lastweek` & Hub Links | M3 | `test_tier1_feature_coverage.py`, `test_tier3_cross_feature.py` | `test_anchor_lastweek_preserved_in_engine_index`, `test_matka_hub_back_links_exist`, `test_engine_breadcrumbs_exist` |
| F7 | CLI Path Portability & Root Exec | M1 | `test_tier1_feature_coverage.py` | `test_cli_root_path_portability_predict`, `test_cli_root_path_portability_backtest` |
| F8 | Raw Cache Decoupling in Scraper | M1 | `test_tier2_boundary_corner.py`, `test_tier3_cross_feature.py` | `test_scraper_cache_fallback_when_network_fails`, `test_scraper_to_csv_validation_interaction` |
| F9 | Scraper Network Resilience & Retries | M1 | `test_tier2_boundary_corner.py` | `test_scraper_cache_fallback_when_network_fails` |
| F10 | REPO_ROOT Path Calculation Fix | M1 | `test_tier1_feature_coverage.py` | `test_cli_sync_and_push_help_or_dryrun` |
| F11 | Historical CSV Integrity & Deduplication | M1 | `test_tier1_feature_coverage.py`, `test_tier2_boundary_corner.py` | `test_csv_panel_digit_sum_rule`, `test_csv_jodi_digit_consistency`, `test_csv_date_monotonicity`, `test_csv_weekday_accuracy` |
| F12 | Three-Way Artifact Mirroring | M1 | `test_tier1_feature_coverage.py`, `test_tier3_cross_feature.py` | `test_multi_location_schema_consistency`, `test_artifact_multi_directory_synchronization` |
| F13 | Tri-Horizon Recency Modeling | M2 | `test_tier1_feature_coverage.py` | `test_top_jodis_length_and_structure` |
| F14 | Markov Chain Jodi Transition Integration | M2 | `test_tier1_feature_coverage.py`, `test_tier3_cross_feature.py` | `test_open_close_digit_distributions_sum_to_unit`, `test_csv_to_ml_predictor_pipeline` |
| F15 | 2D Harmonic Cut Resonance Diffusion | M2 | `test_tier1_feature_coverage.py` | `test_harmonic_cut_pairs_mathematical_property` |
| F16 | Exact OTC Joint Pass Likelihood | M2 | `test_tier1_feature_coverage.py` | `test_otc_ank_recommendation_exact_four_digits` |
| F17 | Advanced Backtesting Metrics | M2 | `test_tier1_feature_coverage.py`, `test_tier4_real_world.py` | `test_cli_backtest_help`, `test_scenario_walk_forward_backtest_audit` |
| F18 | Temporal Regime Validation | M2 | `test_tier2_boundary_corner.py` | `test_leap_years_and_temporal_boundaries` |
| F19 | JSON Schema Validation & Export CLI | M2 | `test_tier1_feature_coverage.py`, `test_tier3_cross_feature.py` | `test_prediction_data_json_root_keys`, `test_ml_prediction_to_json_schema_conformity` |
| F20 | Opaque-Box E2E Test Suite | M4 | `run_e2e_tests.py` | Full Test Harness & 4-Tier Orchestration |
| F21 | Final 100% E2E Pass & Hardening | M5 | `run_e2e_tests.py` | End-to-End Suite Clean Execution |

---

## Execution & Pass/Fail Semantics

### Standard Execution Command
From the repository root (`zevbuild`):
```bash
python tools/matka/tests/run_e2e_tests.py
```

### Targeted Execution
```bash
# Run specific tiers (e.g. Tier 1 and Tier 3)
python tools/matka/tests/run_e2e_tests.py --tier 1,3

# Run with verbose test-by-test output
python tools/matka/tests/run_e2e_tests.py --verbose

# Save machine-readable audit report
python tools/matka/tests/run_e2e_tests.py --report
```

### Pass/Fail Semantics
- **Exit Code 0 (Success)**: All tests pass cleanly. Tests targeting uncompleted features from future milestones (e.g. M1 CLI path portability, M1 raw CSV deduplication) detect the pending state and issue an explicit `SKIP` with the exact feature and milestone annotation. A skip does NOT fail the suite.
- **Exit Code 1 (Failure)**: Any test assertion fails, or an unhandled exception/error is encountered during execution.
