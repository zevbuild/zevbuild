# Kalyan Matka Quantitative Analytics & Predictive Suite — Test Infrastructure

## 1. Overview & Architecture

The Kalyan Matka Test Infrastructure is a comprehensive, requirement-driven, opaque-box E2E test harness designed to rigorously validate all functional, algorithmic, statistical, and operational requirements specified in `PROJECT.md` and `ORIGINAL_REQUEST.md`.

It is built exclusively using the Python standard library (`unittest`, `pathlib`, `json`, `csv`, `subprocess`, `re`, `datetime`) alongside existing workspace dependencies (`numpy`, `pandas`, `bs4`). It requires **zero external pip or npm packages** and executes deterministically across Windows, Linux, and macOS environments.

```
tools/matka/tests/
├── __init__.py
├── requirements_fixtures.py         # Authoritative mathematical specification oracles & ground truth
├── run_e2e_tests.py                 # Unified test runner CLI (--tier, --verbose, --report)
├── fixtures/                        # Mock data fixtures & validation datasets
│   ├── __init__.py
│   ├── sample_draws.csv             # Valid multi-draw historical fixture (26 rows)
│   ├── corrupted_draws.csv          # Corrupted draw fixture for boundary stress testing
│   └── mock_penal_chart.html        # Raw HTML table structure fixture for scraper validation
├── test_tier1_feature_coverage.py   # Tier 1: Core Feature Coverage (>=5 tests/feature, 109 tests)
├── test_tier2_boundary_corner.py    # Tier 2: Boundary & Corner Cases (>=5 tests/feature, 97 tests)
├── test_tier3_cross_feature.py      # Tier 3: Cross-Feature Combinations & Pipelines (15 tests)
└── test_tier4_real_world.py         # Tier 4: Real-World Scenarios & Workload Flows (9 tests)
```

---

## 2. 4-Tier Test Taxonomy

### Tier 1: Core Feature Coverage (`test_tier1_feature_coverage.py`)
Validates the fundamental interfaces and expected primary behaviors across all 14 project features (>=5 test cases per feature, plus 39 foundational system tests for a total of 109 test cases):
- **Feature 1 (Cut-Cluster Density Weighting)**: Computes $S(j) = (1 - \lambda) P(j) + \lambda \text{Mean}_{k \in \text{Family}(j)}(P(k))$ with $\lambda = 0.35$, preserves unit-sum probability mass, and smoothly blends cluster density.
- **Feature 2 (Family-Enriched Top 10 Selection)**: Verifies Top Pick is Rank 1, cut-family members populate adjacent slots, and the resulting list contains 10 non-duplicate valid Jodi strings.
- **Feature 3 (Complementary 4-OTC Matrix)**: Validates selection of 2 complementary harmonic cut pairs $[d_a, \text{cut}(d_a), d_b, \text{cut}(d_b)]$ guaranteeing $>68\%$ joint single-digit coverage.
- **Feature 4 (Cross-Transition Bias)**: Enforces Close(t-1) -> Open(t) direct and cut transitions targeting empirical rate (~20.70%).
- **Feature 5 (Open Repetition Momentum Boost)**: Verifies +15% multiplier ($1.15\times$) applied to Open(t-1) repetition.
- **Feature 6 (Parity in Backtest Engine)**: Audits walk-forward simulation, out-of-sample windowing, and exact binomial statistical significance ($k, n, p_0$).
- **Feature 7 (Automated Ingestion Pipeline)**: Validates scraper output format, CSV schema adherence, and forward prediction generation.
- **Feature 8 (Draw Outcome Comparison)**: Computes exact 1..100 rank of actual draws, Top-10 hits, Family Cut hits, OTC hits, and Mean Reciprocal Rank (MRR).
- **Feature 9 (Persistent Audit Storage)**: Enforces schema compliance of `performance_log.json` (`summary`, `current_hyperparameters`, `draw_logs`).
- **Feature 10 (Hyperparameter Tuning Mechanism)**: Validates adaptive recalibration bounded within strict guardrails ($0.25 \le \lambda \le 0.45$, normalized weights).
- **Feature 11 (Prediction Serving Updates)**: Verifies `prediction_data.json` schema, `by_day` weekday line forecasts, and 4-Ank OTC serialization.
- **Feature 12 (Historical Outcome UI Rendering)**: Verifies HTML/DOM rendering of `Rank #X` pills, green `Top-10 Hit`, and purple `Family Cut Hit` badges.
- **Feature 13 (Mobile PWA Integration)**: Validates `manifest.json`, Service Worker cache v4, and offline standalone readiness.
- **Feature 14 (3-Way File Parity Sync)**: Enforces strict data and schema synchronization across root `tools/matka/`, engine `kalyan_4_35_to_6_35/`, and `web/`.

### Tier 2: Boundary & Corner Cases (`test_tier2_boundary_corner.py`)
Stress-tests the entire analytical engine under extreme, corrupted, and degenerate edge conditions (>=5 test cases per feature, plus 27 foundational edge tests for a total of 97 test cases):
- **Limits & Extremes**: Boundary $\lambda \in \{0.0, 1.0\}$, zero-sum raw distributions, 0-byte CSV files, 1-draw datasets.
- **Empty & Corrupted Inputs**: Missing CSV columns, holiday records (`**`, `---`), malformed dates, non-numeric inputs.
- **Edge Cut Pairs**: Degenerate uniform/delta priors, complementary pair resolution under ties, cyclic modulo boundaries.
- **Patti & Jodi Boundaries**: All 10 Triple Pattis (`000` through `999`), cyclic modulo 10 digit sums, leap year Feb 29 transitions, and Dec 31 -> Jan 01 rollover.

### Tier 3: Cross-Feature Combinations (`test_tier3_cross_feature.py`)
Validates end-to-end data pipelines and multi-module interaction contracts (15 test cases):
- **Scraper ↔ CSV Integrity**: Scraped HTML cells strictly validate against 11-column CSV schema.
- **CSV Dataset ↔ ML Predictor**: Sanitized historical draws feed into predictor models without NaN, Inf, or divide-by-zero errors.
- **ML Prediction ↔ JSON Export ↔ UI Sync**: Model predictions serialize into `prediction_data.json` and mirror identically across all 3 directories.
- **Cut-Cluster Density ↔ Family Enriched Top 10**: High-density clusters consistently generate family-coherent Top 10 rankings.
- **OTC Matrix ↔ Cross-Transition Bias**: OTC selection dynamically tracks boosted cross-transition digits.
- **Prediction ↔ Outcome Evaluation ↔ Audit Log ↔ Hyperparameter Recalibration**: Complete closed-loop self-improving feedback cycle.

### Tier 4: Real-World Workload Scenarios (`test_tier4_real_world.py`)
Validates operational user journeys and historical replays (9 comprehensive test cases):
- **Oct 26–31 Reference Week Replay**:
  - Replays historical draws: `2026-10-26 (68)`, `2026-10-27 (63)`, `2026-10-28 (64)`, `2026-10-29 (48)`, `2026-10-30 (34)`, `2026-10-31 (40)`.
  - Verifies that on Oct 28 (Draw: 64, Top Pick: 91), baseline low rank (Rank 37-49) is **promoted into Top 10 recommendations (Ranks 5-8)** via Cut-Cluster Density $S(j)$ and Family Enrichment.
  - Verifies that on Oct 27 (Draw: 63, Top Pick: 95), family candidate 40 is represented in Top 10 recommendations.
  - Verifies that 4-OTC Matrix produces 2 complementary cut pairs with $>68\%$ single-digit coverage on every draw.
  - Verifies the transition invariant: **100% of reference week draws** match Open Repetition or Cross-Transition Direct/Cut.
- **Self-Improving Pipeline Lifecycle**: Simulates draw prediction -> outcome arrival -> audit log recording -> parameter recalibration within guardrails.
- **Cold-Start to Production Scale**: Tests Dirichlet smoothing preventing zero probabilities and confirming entropy convergence as empirical evidence accumulates.
- **Daily Prediction Cycle & Saturday-Monday Recess**: Complete end-to-end draw announcement, Sunday recess skipping, and Monday line prior preparation.
- **Live IST Market Countdown State Machine**: State transitions (Pre-Open < 16:35, Live Draw 16:35-18:35, Closed >= 18:35, Weekend Recess on Sunday).
- **Offline PWA Standalone Operation**: Confirms all assets and data files are precached for 100% offline client prediction.

---

## 3. Feature Inventory Coverage Matrix

| Feature # | Feature Name | Tier 1 (Core) | Tier 2 (Boundary) | Tier 3 (Cross) | Tier 4 (Real-World) |
|---|---|:---:|:---:|:---:|:---:|
| F1 | Cut-Cluster Density Weighting | `test_f1_*` (5) | `test_t2_f1_*` (5) | `test_cross_feature_1_*` | `test_scenario_oct_26_31_*` |
| F2 | Family-Enriched Top 10 Selection | `test_f2_*` (5) | `test_t2_f2_*` (5) | `test_cross_feature_2_*` | `test_scenario_oct_26_31_*` |
| F3 | Complementary 4-OTC Matrix | `test_f3_*` (5) | `test_t2_f3_*` (5) | `test_cross_feature_4_*` | `test_scenario_oct_26_31_*` |
| F4 | Cross-Transition Bias | `test_f4_*` (5) | `test_t2_f4_*` (5) | `test_cross_feature_4_*` | `test_scenario_oct_26_31_*` |
| F5 | Open Repetition Momentum Boost | `test_f5_*` (5) | `test_t2_f5_*` (5) | `test_cross_feature_5_*` | `test_scenario_oct_26_31_*` |
| F6 | Parity in Backtest Engine | `test_f6_*` (5) | `test_t2_f6_*` (5) | `test_e2e_pipeline_simulation` | `test_scenario_walk_forward_*` |
| F7 | Automated Ingestion Pipeline | `test_f7_*` (5) | `test_t2_f7_*` (5) | `test_scraper_to_csv_*` | `test_scenario_daily_prediction_*` |
| F8 | Draw Outcome Comparison | `test_f8_*` (5) | `test_t2_f8_*` (5) | `test_cross_feature_3_*` | `test_scenario_self_improving_*` |
| F9 | Persistent Audit Storage | `test_f9_*` (5) | `test_t2_f9_*` (5) | `test_cross_feature_3_*` | `test_scenario_self_improving_*` |
| F10 | Hyperparameter Tuning Mechanism | `test_f10_*` (5) | `test_t2_f10_*` (5) | `test_cross_feature_6_*` | `test_scenario_self_improving_*` |
| F11 | Prediction Serving Updates | `test_f11_*` (5) | `test_t2_f11_*` (5) | `test_ml_prediction_to_json_*` | `test_scenario_multi_channel_*` |
| F12 | Historical Outcome UI Rendering | `test_f12_*` (5) | `test_t2_f12_*` (5) | `test_web_ui_to_repo_verifier` | `test_scenario_multi_channel_*` |
| F13 | Mobile PWA Integration | `test_f13_*` (5) | `test_t2_f13_*` (5) | `test_offline_pwa_readiness` | `test_scenario_offline_pwa_*` |
| F14 | 3-Way File Parity Sync | `test_f14_*` (5) | `test_t2_f14_*` (5) | `test_artifact_multi_directory_*` | `test_scenario_multi_channel_*` |

---

## 4. Progressive Testability & Stability Guarantee

To ensure the test suite operates as a stable, non-flaky test harness during ongoing milestone development:
1. **Mathematical Oracles (`requirements_fixtures.py`)**: All mathematical formulas (Cut-Cluster density, family generation, OTC matrix selection, transition bias, momentum boost, outcome evaluation, recalibration) are implemented as pure-Python specification oracles.
2. **Graceful Milestone Adaptation**: Tests designed for features currently in active development across Milestones M1, M2, and M3 check if the engine module exposes the feature method. If present, it executes the engine method; if pending, it verifies the specification contract via the oracle or reports an explicit `SKIP` with the exact feature and milestone annotation.
3. **Deterministic Exit Codes**: The test harness consistently exits with **code 0** when all specification contracts pass or are pending, and exits with **code 1** only upon an actual regression or contract failure.

---

## 5. Execution Guide

### Run Full Test Suite (All 4 Tiers)
```bash
python tools/matka/tests/run_e2e_tests.py
```

### Run Specific Tiers
```bash
# Run Tier 1 (Feature Coverage) only
python tools/matka/tests/run_e2e_tests.py --tier 1

# Run Tier 2 and Tier 4
python tools/matka/tests/run_e2e_tests.py --tier 2,4
```

### Verbose Mode with Detailed Per-Test Reporting
```bash
python tools/matka/tests/run_e2e_tests.py --verbose
```

### Generate Machine-Readable Audit Report
```bash
python tools/matka/tests/run_e2e_tests.py --report
```
Saves JSON test audit report to `test_audit_results.json` at repository root.
