# Project: Kalyan Matka Quantitative Analytics & Predictive Suite Full-Stack Upgrade

## Architecture
The Kalyan Matka suite is a zero-build-step, offline-resilient quantitative prediction, backtesting, and visualization platform. The architecture comprises four core layers:
1. **Data Ingestion & Integrity Layer (`scraper.py`, `sync_and_push.py`)**: Fetches historical and live draw results from remote sources, decouples raw scrape caching (`kalyan_penal_chart_raw.html`), performs strict multi-point CSV validation (monotonic dates, no duplicates, panel digit sum rule), and safely orchestrates synchronization.
2. **Quantitative Engine Layer (`models.py`, `predict.py`, `backtest.py`)**: Mathematical prediction models using Dirichlet-smoothed Markov chain transitions, tri-horizon exponential recency (Fast 8d, Medium 35d, Macro 120d), 2D harmonic cut resonance diffusion, and exact OTC joint pass probabilities. Backtest engine provides walk-forward validation with Precision@K, MRR, Brier scores, drawdown duration, profit factor, and temporal regime breakdown.
3. **Data Artifact Layer (`prediction_data.json`, `history.json`, `kalyan_historical_data.csv`)**: Normalized, atomic data interchange format shared identically across root `tools/matka/`, subfolder `tools/matka/kalyan_4_35_to_6_35/`, and web server `tools/matka/kalyan_4_35_to_6_35/web/`.
4. **Presentation & PWA Layer (`index.html`, `dashboard.html`, `kalyan_penal_chart.html`, `sw.js`)**: Pure HTML5, Tailwind CDN, and vanilla JS styled with Zev Glass 2.0 obsidian aesthetics. Features live Indian Standard Time (IST, UTC+5:30) market countdowns for Kalyan Open (4:35 PM IST) and Kalyan Close (6:35 PM IST), interactive visualization cards, offline Service Worker precaching, zero external telemetry, and full compatibility with `verify_links_and_assets.py`.

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | Zev Glass 2.0 Obsidian Aesthetic | Modernize all web pages (`tools/matka/index.html`, `kalyan_4_35_to_6_35/index.html`, `dashboard.html`, `kalyan_penal_chart.html`) with obsidian dark surfaces (`#06080d`, `#0b0f19`), ambient radial glow, glass cards, and high-contrast typography | M3 | R1, Survey |
| 2 | Live IST Market Countdown Engine | State-machine countdown for Kalyan Open (4:35 PM IST) and Kalyan Close (6:35 PM IST) with Pre-Open, Live Draw Active, Market Closed, and Weekend Recess states | M3 | R1, Survey |
| 3 | Zero Third-Party Telemetry & Privacy Leak Removal | Eliminate external CORS proxies (`allorigins.win`, `corsproxy.io`, `codetabs.com`) and scraped Google AMP tags / external images | M3 | R1, Survey |
| 4 | Offline PWA Resilience & Service Worker Cache Expansion | Expand `sw.js` precache to include `kalyan_penal_chart.html`, `prediction_data.json`, `history.json`, and Tailwind CDN script for full offline operation | M3 | R1, Survey |
| 5 | Modernized Historical Penal Chart | Redesign `kalyan_penal_chart.html` into a dark obsidian glassmorphic historical matrix with year filtering and breadcrumbs, eliminating raw AMP scrapes | M3 | R1, Survey |
| 6 | Cross-Page Anchor & Hub Link Preservation | Maintain `#lastweek` DOM ID in `kalyan_4_35_to_6_35/index.html` and 3-level breadcrumbs (`../../../index.html` -> `../../index.html` -> `../index.html`) for 100% HEALTHY verification | M3 | R1, R3, Survey |
| 7 | CLI Path Portability & Root Execution | Enable running `python tools/matka/kalyan_4_35_to_6_35/predict.py` and `backtest.py` from repo root via dynamic `Path(__file__).resolve().parent` CSV resolution | M1 | R2, R3, Survey |
| 8 | Raw Cache Decoupling in Scraper | Decouple `LOCAL_CACHE_HTML` in `scraper.py` to `kalyan_penal_chart_raw.html` so live scrapes never overwrite user-facing web interfaces | M1 | R3, Survey |
| 9 | Scraper Network Resilience & Table Validation | Add 3-attempt exponential backoff retries and HTML table structure validation (`chart-table`) in `scraper.py` | M1 | R3, Survey |
| 10 | REPO_ROOT Path Calculation Fix in Sync Pipeline | Fix 3-level parent directory calculation in `sync_and_push.py` so Git commands execute from actual repo root | M1 | R3, Survey |
| 11 | Historical CSV Integrity Verification & Deduplication | Implement multi-point CSV validation (monotonic dates, no duplicate dates, Jodi digit consistency, panel digit sum rule `sum(patti) % 10 == digit`) and deduplicate 30 legacy dates | M1 | R3, Survey |
| 12 | Three-Way Artifact Mirroring | Synchronize generated `prediction_data.json`, `history.json`, and `kalyan_historical_data.csv` atomically across root `tools/matka/`, subfolder `kalyan_4_35_to_6_35/`, and `web/` | M1 | R2, R3, Survey |
| 13 | Tri-Horizon Recency Modeling | Implement calibrated 3-horizon exponential recency (Fast 8d, Medium 35d, Macro 120d) applied to Open Anks, Close Anks, and Jodis in `models.py` | M2 | R2, Survey |
| 14 | Markov Chain Jodi Transition Integration | Incorporate 100x100 Jodi Markov transition probability distribution into `EnsemblePredictor.predict()` `jodi_matrix` synthesis with time decay | M2 | R2, Survey |
| 15 | 2D Harmonic Cut Resonance Diffusion | Extend cut resonance across 2D joint Jodi distribution space for candidate cut pairs $((o+5)\%10, c)$, $(o, (c+5)\%10)$, and $((o+5)\%10, (c+5)\%10)$ | M2 | R2, Survey |
| 16 | Exact OTC Joint Pass Likelihood Formula | Replace arbitrary 0.9 heuristic with exact complementary joint probability: $1.0 - \sum_{o \notin OTC} \sum_{c \notin OTC} P_{joint}(o, c)$ | M2 | R2, Survey |
| 17 | Advanced Backtesting Statistical Metrics | Expand `backtest.py` with Precision@K (1, 3, 5, 10), MRR, Brier calibration, Max Drawdown Duration, Max Consecutive Losses, Profit Factor, and true Open Ank equity tracking | M2 | R2, Survey |
| 18 | Temporal Regime & Seasonality Validation | Multi-year historical regime breakdown (2014-2018, 2019-2022, 2023-2026), weekday performance validation, and one-tailed binomial Z-score tests in `backtest.py` (zero scipy) | M2 | R2, Survey |
| 19 | JSON Schema Validation & Direct Export CLI | Add `--export-json` mode in `predict.py` with strict schema validation against frontend data bindings | M2 | R2, Survey |
| 20 | Requirement-Driven Opaque-Box E2E Test Suite | Build comprehensive E2E test suite covering Tiers 1-4 (feature coverage, boundary cases, cross-feature interactions, real-world workloads) publishing `TEST_READY.md` | M4 | AC, Pattern |
| 21 | Final 100% E2E Pass & Adversarial Hardening | Pass 100% of E2E test suite + Challenger-led Tier 5 adversarial coverage hardening | M5 | AC, Pattern |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Scraper, Historical Data Pipeline & Path Resolution | Features 7, 8, 9, 10, 11, 12: Scraper retry & cache decoupling, CSV deduplication & integrity validator, `sync_and_push.py` REPO_ROOT fix, root vs subfolder sync, dynamic CLI path resolution | none | DONE (Gate PASS: path defense, scraper isolation, CSV 3,270 records cleaned, 3-way sync verified) |
| M2 | Quantitative Engine & Predictive Algorithms | Features 13, 14, 15, 16, 17, 18, 19: Tri-horizon recency, Markov Jodi transition integration, 2D harmonic cut resonance, exact OTC probability, expanded backtest metrics (Precision@K, MRR, Drawdown, Z-score), schema export | M1 | DONE (Models & backtest enhanced, verified exit code 0, 78/78 tests passed) |
| M3 | Web UI, Visualizations & Offline PWA Modernization | Features 1, 2, 3, 4, 5, 6: Zev Glass 2.0 obsidian styling, IST market countdown (4:35/6:35 PM), telemetry purge, offline PWA cache expansion, penal chart modernization, anchor `#lastweek` retention | M1, M2 | IN_PROGRESS (worker_m3_1) |
| M4 | E2E Testing Track: Comprehensive Test Suite | Feature 20: Design opaque-box test runner and test cases covering Tiers 1-4, publish `TEST_READY.md` | none (Parallel Track) | DONE (78 tests: Tiers 1-4 implemented in tools/matka/tests/, TEST_READY.md published) |
| M5 | Final Milestone: 100% E2E Pass & Adversarial Hardening | Feature 21: Phase 1: Pass 100% E2E tests across CLI, web, links, schemas. Phase 2: Challenger-led Tier 5 white-box adversarial stress tests | M1, M2, M3, M4 | PLANNED |

## Interface Contracts
### Scraper / Pipeline ↔ Quantitative Engine (`scraper.py` / `sync_and_push.py` ↔ `models.py` / `predict.py`)
- **CSV Data Source**: `kalyan_historical_data.csv`
  - Columns: `Date` (YYYY-MM-DD), `Day_Of_Week` (Mon..Sat), `Jodi` (2-digit str '00'..'99'), `Open_Digit` (int 0..9), `Close_Digit` (int 0..9), `Open_Patti` (3-digit str), `Close_Patti` (3-digit str), `Open_Patti_Type` ('SP'|'DP'|'TP'), `Close_Patti_Type` ('SP'|'DP'|'TP'), `Is_Valid` (bool 'True'|'False'), `Raw_Entry` (str).
  - Ordering: Monotonically increasing by Date. Deduplicated (1 row per Date).
  - Validation: `sum(map(int, patti)) % 10 == digit` for all valid records.
- **Path Resolution Contract**:
  - All scripts (`predict.py`, `backtest.py`, `scraper.py`, `sync_and_push.py`, `app.py`) must resolve `kalyan_historical_data.csv` using:
    `Path(__file__).resolve().parent / "kalyan_historical_data.csv"` with fallback to `Path(__file__).resolve().parent.parent / "kalyan_historical_data.csv"` and CLI argument.

### Quantitative Engine ↔ Frontend Web UI (`predict.py` / `app.py` ↔ `index.html` / `dashboard.html`)
- **`prediction_data.json` Schema**:
  - Keys:
    - `"latest_draw"`: `{"date": str, "day": str, "jodi": str, "open_digit": int, "close_digit": int, "open_patti": str, "close_patti": str}`
    - `"predicted_for"`: `{"date": str, "day": str}`
    - `"otc_recommendation"`: `{"digits": [int, int, int, int], "cut_pairs": [str, str], "pass_probability": float}`
    - `"top_jodis"`: `[{"jodi": str, "prob": float, "confidence": float, ...}]` (length >= 10)
    - `"open_digit_probs"`: `[float, ...]` (length 10, sums to 1.0)
    - `"close_digit_probs"`: `[float, ...]` (length 10, sums to 1.0)
    - `"by_day"`: dictionary mapped by day name `"Mon"`, `"Tue"`, `"Wed"`, `"Thu"`, `"Fri"`, `"Sat"` containing day-specific distributions.
- **`history.json` Schema**:
  - Array of arrays: `[[Date: str, Day: str, Jodi: int, Open_Digit: int, Close_Digit: int, Open_Patti: str, Close_Patti: str], ...]`
  - Preserves exact array indexing for frontend `BrowserEnsemblePredictor` consumption.

### Presentation Layer ↔ Repository Link Verifier (`index.html` ↔ `verify_links_and_assets.py`)
- Target `tools/matka/kalyan_4_35_to_6_35/index.html` MUST contain `<... id="lastweek" ...>` element to satisfy Tier 2 cross-page anchor audit.
- Target `tools/matka/index.html` MUST contain back-links `<a href="../../index.html">` and `<a href="../index.html">` for Tier 3 hub hierarchy compliance.
- Target `tools/matka/kalyan_4_35_to_6_35/index.html` MUST contain breadcrumbs resolving to Studio (`../../../index.html`), Tools (`../../index.html`), and Matka Hub (`../index.html`).

## Code Layout
```
tools/matka/
├── index.html                           # Root Matka Suite launchpad & directory (Zev Glass 2.0)
├── kalyan_penal_chart.html               # Root mirror of modernized penal chart
├── prediction_data.json                 # Mirrored prediction artifact
├── history.json                         # Mirrored historical draws artifact
├── kalyan_historical_data.csv           # Mirrored historical CSV dataset
├── favicon.svg                          # Suite favicon
└── kalyan_4_35_to_6_35/                 # Primary Kalyan engine & tools directory
    ├── index.html                       # Primary interactive predictor UI (Zev Glass 2.0)
    ├── dashboard.html                   # Synchronized dashboard UI
    ├── kalyan_penal_chart.html          # Modernized obsidian historical penal chart
    ├── kalyan_penal_chart_raw.html      # Decoupled raw scraper HTML cache (safe from overwriting UI)
    ├── manifest.json                    # PWA Web App Manifest
    ├── sw.js                            # PWA Service Worker with offline asset precache
    ├── kalyan_historical_data.csv       # Primary 3,300+ draw historical dataset
    ├── prediction_data.json             # Generated prediction output
    ├── history.json                     # Generated historical output
    ├── models.py                        # Quantitative mathematical models (Markov, Recency, Resonance)
    ├── predict.py                       # Prediction orchestration & JSON export CLI
    ├── backtest.py                      # Walk-forward backtest simulation & advanced metrics
    ├── scraper.py                       # Network scraper with backoff retries & CSV validator
    ├── sync_and_push.py                 # Multi-directory sync & pipeline runner
    ├── app.py                           # Local server & unified prediction generator
    ├── main.py                          # Terminal runner
    └── web/                             # Embedded local web mirror
        ├── index.html                   # Synchronized web mirror
        ├── kalyan_penal_chart.html      # Synchronized web penal chart
        ├── manifest.json                # Web manifest
        ├── sw.js                        # Web service worker
        ├── prediction_data.json         # Web prediction json
        └── history.json                 # Web history json
```
