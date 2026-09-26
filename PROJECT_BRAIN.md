# 🧠 Zevbuild Project Brain

> **Purpose:** This document is the single source of truth for any new AI agent or developer to understand the entire Zevbuild repository at a glance. Read this first before touching any code.

---

## 📌 What is Zevbuild?

**Zevbuild Studio** is an independent software engineering studio founded in 2024, based in **Goa, India**. It builds privacy-first, zero-telemetry, offline-capable digital tools and publishes them as MIT open source.

- **Website:** [zevbuild.pages.dev](https://zevbuild.pages.dev)
- **GitHub Org:** [github.com/zevbuild](https://github.com/zevbuild)
- **Email:** zevbuildstudio@gmail.com
- **License:** MIT — Copyright © 2026 Zevbuild Studio

### Core Design Principles

| Principle | What it means in practice |
|---|---|
| **Zero-Knowledge** | No plaintext or passwords leave the user's device. Crypto runs client-side (browser / Android) |
| **Offline-First** | Designed for air-gapped / disaster scenarios using local caching & Bluetooth mesh |
| **Zero Telemetry** | No trackers, analytics beacons, or behavioral monitoring. Ever. |
| **Open Standards** | NIST-approved primitives only: AES-256-GCM, PBKDF2-SHA512, SHA-512, Lamport Timestamps |

---

## 🗂️ Repository Layout

```
zevbuild/                             ← Root: org portal & multi-tool monorepo
├── index.html                        ← Main company website (1,091 lines, self-contained SPA)
├── 404.html                          ← Custom 404 error page
├── og-image.svg                      ← Open Graph social share image
├── robots.txt                        ← SEO crawl rules (blocks /.git/ and /functions/)
├── sitemap.xml                       ← SEO sitemap
├── README.md                         ← Public-facing project README
├── PROJECT_BRAIN.md                  ← YOU ARE HERE
│
├── functions/                        ← Cloudflare Pages serverless edge functions
│   └── api/
│       ├── live-kalyan.js            ← Edge worker: live Kalyan draw fetcher & parser
│       └── fetch-and-predict.js      ← Edge worker stub (minimal, delegates to live-kalyan)
│
└── tools/                            ← All sub-products live here
    ├── index.html                    ← Tools index/landing redirect page
    │
    ├── satta-matka-tools/            ← Kalyan Matka Predictive Analytics Engine
    │   ├── main.py                   ← CLI orchestrator (entry point)
    │   ├── app.py                    ← Threaded HTTP web server + REST API
    │   ├── scraper.py                ← HTML scraper & data sanitizer
    │   ├── eda.py                    ← Exploratory frequency analysis
    │   ├── models.py                 ← Statistical models (Markov, Recency, Seasonal, Ensemble)
    │   ├── predict.py                ← Forecast generator & family bracket builder
    │   ├── backtest.py               ← Walk-forward out-of-sample backtesting engine
    │   ├── requirements.txt          ← Python deps: requests, beautifulsoup4, pandas, tabulate
    │   ├── start_web.bat             ← Windows one-click launcher
    │   ├── kalyan_historical_data.csv← Normalized dataset (~3,282 rows, 2012–2026)
    │   ├── kalyan_penal_chart.html   ← Local HTML cache (avoids re-scraping every run)
    │   ├── history.json              ← Cached history data for the web UI
    │   ├── prediction_data.json      ← Baseline snapshot of latest predictions
    │   ├── dashboard.html            ← Standalone HTML dashboard (mirrors web/index.html)
    │   ├── index.html                ← Duplicate/alias of dashboard.html
    │   └── web/                      ← Web server's static root (served by app.py)
    │       ├── index.html            ← The browser UI served at http://127.0.0.1:8080
    │       ├── history.json          ← History data for the UI
    │       └── prediction_data.json  ← Cached prediction data for the UI
    │
    ├── sp-ms-downloader/             ← Spotify Playlist Batch Downloader + RetroWave Player
    │   ├── index.html                ← Offline RetroWave audio player (drag-and-drop MP3)
    │   ├── spotify_download_script.txt ← Browser console script for spotifymp3.com
    │   ├── website.txt               ← Target website URL reference
    │   └── README.md                 ← Step-by-step usage guide
    │
    └── top-10-free-video-downloaders-in-india/
        └── index.html                ← Static SEO article/landing page (1,684 bytes)
    │
    └── v_yt/                         ← YouTube Video Downloader (yt-dlp via Cobalt API)
        ├── index.html                ← Downloader UI (vanilla HTML + Tailwind + vanilla JS)
        └── README.md                 ← Usage guide & API reference
```

---

## 🛠️ Products Deep-Dive

### 1. Company Main Portal (`index.html`)

A **fully self-contained single-page app** — no build step, no framework, no CDN dependency at runtime.

- **Size:** ~68 KB, 1,091 lines
- **Tech:** Vanilla HTML + Tailwind CSS (CDN) + vanilla JS
- **SEO:** Full JSON-LD structured data (Organization schema), Open Graph, Twitter Cards, geo-targeting (IN-GA Goa)
- **Deployed at:** `https://zevbuild.pages.dev/` via GitHub Pages
- **Notable:** `theme-color: #8b5cf6` (purple branding), 600,000-iteration PBKDF2 referenced in product copy

---

### 2. Kalyan Matka Predictive Analytics Engine (`tools/satta-matka-tools/`)

The most complex tool in this repo. A full **data engineering + ML + web server system**.

#### Architecture

```
dpbossx.net ──► scraper.py ──► kalyan_historical_data.csv
                                        │
               ┌────────────────────────┼────────────────────┐
               ▼                        ▼                    ▼
            eda.py                  models.py           backtest.py
          (frequency)          (Markov+Recency         (walk-forward
                                +Seasonal+Ensemble)     out-of-sample)
                                        │
                                    predict.py
                                        │
                                     app.py (HTTP :8080)
                                        │
                               web/index.html (browser UI)
```

#### File Responsibilities

| File | Role |
|---|---|
| `main.py` | CLI entry point. Routes `--web`, `--predict`, `--backtest`, `--eda`, `--preview`, `--export` flags |
| `scraper.py` | Fetches `dpbossx.net/kalyan-penal-chart.php`. Caches to `kalyan_penal_chart.html`. Parses weekly table (19 cols = 1 date + 6 days × 3). Outputs normalized DataFrame |
| `models.py` | 4 classes: `MarkovChainModel`, `RecencyWeightedModel`, `DayOfWeekSeasonalModel`, `EnsemblePredictor` |
| `predict.py` | Generates 6-day forward forecasts; builds family/cut bracket jodi sets (+5 mod 10) |
| `backtest.py` | Strict walk-forward backtesting on 2,680 out-of-sample draws |
| `eda.py` | Frequency analytics: Top/Bottom Jodis, Open/Close digit distributions |
| `app.py` | Threaded HTTP server (`ThreadingMixIn + HTTPServer`) on `127.0.0.1:8080`. REST: `GET /api/status`, `GET|POST /api/fetch-and-predict` |

#### Predictive Model Architecture

The `EnsemblePredictor` blends three sub-models:

| Model | Weight | Algorithm |
|---|---|---|
| `MarkovChainModel` | **0.30** | First-order Markov: P(Draw_t | Draw_{t-1}). Laplace smoothing (alpha=0.5) |
| `RecencyWeightedModel` | **0.40** | Exponential decay w_i = e^(-lambda * delta_t), 60-draw half-life |
| `DayOfWeekSeasonalModel` | **0.30** | Historical frequency conditioned on day-of-week (Mon–Sat) |

All three output probability arrays over **100 Jodis (00–99)** and **10 single digits (0–9)** for Open and Close independently.

#### Dataset Schema (`kalyan_historical_data.csv`)

| Column | Type | Description |
|---|---|---|
| `Date` | `YYYY-MM-DD` | Draw date |
| `Day_Of_Week` | `Mon`–`Sat` | Day abbreviation |
| `Jodi` | str (2-char) | Winning result e.g. `"07"` |
| `Open_Digit` | Int64 (nullable) | First digit of Jodi (0–9) |
| `Close_Digit` | Int64 (nullable) | Second digit of Jodi (0–9) |
| `Is_Valid` | bool | False for holidays / missing draws |
| `Raw_Entry` | str | Original scraped value before sanitization |

#### Backtest Results (2,680 out-of-sample draws)
- **Top-5 Jodi Hit Rate:** 5.11% (vs 5.00% random baseline — slight positive edge)
- **Top-1 Close Ank Hit Rate:** 9.96%
- **House Edge Reality:** 90:1 payout on 100 outcomes → long-run ROI ≈ **-7.99%**

#### CLI Reference
```bash
python main.py              # Full pipeline: scrape → CSV → EDA → predict
python main.py --web        # Launch browser UI at http://127.0.0.1:8080
python main.py --predict    # Print forecast to terminal
python main.py --backtest   # Run walk-forward backtest
python main.py --eda        # Frequency analysis from existing CSV
python main.py --preview    # Preview first 20 rows (no disk write)
python main.py --export     # Scrape and export CSV only
```

#### Python Dependencies
```
requests>=2.31.0
beautifulsoup4>=4.12.0
pandas>=2.2.0
tabulate>=0.9.0
```
> **Note:** `numpy` is used in `models.py` but absent from `requirements.txt` — it arrives as a transitive dep of pandas.

---

### 3. Spotify Playlist Downloader (`tools/sp-ms-downloader/`)

A **non-code utility toolkit** — no backend, no build step.

- **`spotify_download_script.txt`** — Browser console JS that auto-iterates all songs on spotifymp3.com and clicks download with a 10-second throttle per song (to evade Cloudflare bot detection).
- **`index.html`** — Offline "RetroWave" audio player. Drag-and-drop MP3 files, real-time audio visualizer. 100% local browser app.
- **Workflow:** Firefox → spotifymp3.com → paste script → batch download → drag MP3s to RetroWave player.

---

### 4. SEO Content Page (`tools/top-10-free-video-downloaders-in-india/`)

A minimal **static article page** (1,684 bytes). SEO content targeting "top 10 free video downloaders in India" queries. No dynamic content.

---

### 5. v_yt — YouTube Video Downloader (`tools/v_yt/`)

A **browser-based video downloader** powered by yt-dlp via the Cobalt API.

#### Architecture

```
Browser (index.html)
        │  GET /api/yt-download?url=...&format=...
        ▼
Cloudflare Worker (functions/api/yt-download.js)
        │  POST https://api.cobalt.tools/
        ▼
   Cobalt API (open-source yt-dlp wrapper)
        │  { status: "redirect", url: "..." }
        ▼
  Direct CDN download URL ──► returned to browser ──► user downloads file
```

#### Supported Formats

| Format Code | Label | Description |
|---|---|---|
| `mp4-720` | MP4 720p | Standard HD video (default) |
| `mp4-1080` | MP4 1080p | Full HD video |
| `mp3` | MP3 Audio | Audio-only extraction |

#### File Responsibilities

| File | Role |
|---|---|
| `tools/v_yt/index.html` | Static UI: URL input, format selector, fetch button, results panel |
| `tools/v_yt/README.md` | Usage docs, API reference, legal notice |
| `functions/api/yt-download.js` | CF Worker: validates URL, maps format → Cobalt params, proxies response |

#### API Endpoint
`GET /api/yt-download?url=<encoded-url>&format=<format-code>`

---

## ☁️ Cloudflare Edge Functions (`functions/api/`)

Auto-deployed by Cloudflare Pages on push to main. Run as **serverless edge workers globally**.

### `live-kalyan.js` — Live Kalyan Result Fetcher
- **Endpoint:** `https://zevbuild.pages.dev/api/live-kalyan`
- **Purpose:** Server-side proxy to `dpbossx.net/kalyan-chart.php` — bypasses browser CORS restrictions.
- **Response Schema:**
```json
{
  "status": "success",
  "timestamp": "2026-09-16T06:14:00.000Z",
  "raw_result": "560-15-249",
  "draw": {
    "open_panna": "560",
    "open_digit": 1,
    "close_digit": 5,
    "close_panna": "249",
    "jodi": "15",
    "status": "FULL_JODI_DECLARED"
  }
}
```
- **Draw status values:** `FULL_JODI_DECLARED`, `OPEN_DECLARED`, `UNKNOWN`
- **Parsing:** Regex on raw string. Full: `NNN-NN-NNN`, partial: `NNN-N[N]`

### `fetch-and-predict.js`
Stub file (46 bytes) — delegates to live-kalyan.

### `yt-download.js` — YouTube Download Link Extractor
- **Endpoint:** `https://zevbuild.pages.dev/api/yt-download`
- **Purpose:** Server-side proxy to the Cobalt API (yt-dlp wrapper). Extracts direct download links for YouTube videos without running yt-dlp in the browser.
- **Query params:** `url` (YouTube URL), `format` (`mp4-720`, `mp4-1080`, `mp3`)
- **Response Schema:**
```json
{
  "status": "ok",
  "format": "mp4-720",
  "downloads": [
    { "label": "MP4 720p", "url": "https://...googlevideo.com/...", "filename": "title.mp4" }
  ]
}
```
- **Supported URL formats:** `youtube.com/watch?v=`, `youtu.be/`, `/shorts/`, `/live/`, `/embed/`
- **Error codes propagated from Cobalt:** `content.video.private`, `content.video.age`, `content.video.live`, `fetch.fail`, `fetch.rate`, etc.

---

## 🚀 Deployment Infrastructure

| Layer | Technology | URL |
|---|---|---|
| Main site | **Cloudflare Pages** | `https://zevbuild.pages.dev/` |
| Serverless Edge API | **Cloudflare Pages Functions** | `https://zevbuild.pages.dev/api/...` |
| Kalyan predictor (local) | **Python stdlib HTTP server** | `http://127.0.0.1:8080` |
| ZevSafe (separate repo) | Cloudflare Pages | `https://zevsafe.pages.dev` |
| CollegeBus (separate repo) | Cloudflare Pages | `https://collegebus.pages.dev` |

---

## 🔗 External Products (Separate Repos — Referenced Here)

| Product | Description | Stack |
|---|---|---|
| **ZevSafe** | In-browser zero-knowledge vault encryption | WebCrypto API · AES-256-GCM · PBKDF2-SHA512 (600k iter) · PWA |
| **ZevSync** | Offline Bluetooth P2P mesh sync for Android | Kotlin 2.0+ · Jetpack Compose M3 · Vector Clocks · CAS |
| **CollegeBus** | Real-time transit & multi-hop schedule tracker PWA | Vanilla JS · Service Worker · Edge CDN |

---

## 🤖 AI Agent Working Guide

### Editing the main portal (`index.html`)
- Self-contained — no build step. Edit directly.
- Tailwind CSS via CDN — use utility classes, no separate CSS files.
- JSON-LD structured data in `<head>` must stay consistent with product details.
- All SEO tags (OG, Twitter, canonical) are in `<head>`.

### Editing the Kalyan Analytics Engine
- **Entry point for all work:** `main.py`
- **Model logic lives exclusively in:** `models.py` — do not scatter model code into other files
- **Web server static root:** `tools/satta-matka-tools/web/` (NOT the tool root)
- **CSV + HTML cache files** are generated artifacts — be mindful before committing large updates
- `app.py` uses `ThreadingMixIn` — API handlers are concurrent; be careful with shared mutable state

### Editing Cloudflare Functions
- Files in `functions/api/` are auto-deployed on push — changes go live immediately
- Use Cloudflare Workers APIs (`fetch`, `Response`, `context.request`) — no Node.js APIs
- CORS headers are set manually in every response — **do not remove them**
- `cf: { cacheTtl: 0, cacheEverything: false }` is intentional to prevent stale draw data

### Conventions

| Convention | Details |
|---|---|
| Python version | Python 3.14 (per README); 3.10+ safe in practice |
| JS style | ESM modules (`export async function`), no bundler |
| HTML style | Vanilla HTML, Tailwind via CDN, no frameworks |
| Indentation | 4 spaces (Python), 2 spaces (HTML/JS) |
| No build tools | No webpack, vite, or npm — zero dependency toolchain |

---

## 📊 Full Data Flow

```
[dpbossx.net] ──scrape──► scraper.py ──► DataFrame ──► kalyan_historical_data.csv
                                                │
                                           models.py
                                        EnsemblePredictor.fit()
                                                │
                     ┌──────────────────────────┼──────────────────────────┐
                     ▼                          ▼                          ▼
            predict.py                    backtest.py                  eda.py
            (6-day forecast)        (2,680 OOS draws)           (frequency stats)
                     │
                  app.py
               (HTTP :8080)
                     │
          ┌──────────┴────────────┐
          ▼                       ▼
  web/index.html        /api/fetch-and-predict
  (browser UI)          (JSON REST endpoint)
                                  ▲
                        [Cloudflare Edge]
                        functions/api/live-kalyan.js
                        (live draw proxy, bypasses CORS)
```

---

## ❓ Known Gaps & Awareness Items

> Not bugs — just things a new contributor should be aware of.

- `numpy` is used in `models.py` but missing from `requirements.txt` (arrives via pandas)
- `fetch-and-predict.js` is only 46 bytes — appears to be a placeholder stub
- `dashboard.html` and `index.html` at the tool root look like duplicates of `web/index.html`
- `kalyan_penal_chart.html` (317 KB) and `kalyan_historical_data.csv` (101 KB) are committed artifacts — consider `.gitignore`-ing these generated files
- No automated unit tests exist for any Python modules
- ZevSafe and ZevSync live in **separate GitHub repos** (not present in this directory)
