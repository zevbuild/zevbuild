# Kalyan Realtime Auto-Predictor & Web Engine

An automated data engineering and predictive modeling platform for Kalyan Matka historical records (2012–2026). It features an **interactive real-time Web Browser Application** that auto-fetches the latest chart updates from `dpbossx.net`, retrains statistical models in real time, and broadcasts upcoming predictions directly in your browser.

---

## 🌐 Real-Time Web Browser System

To launch the web interface:

```bash
# Option 1: Using Python CLI
python main.py --web

# Option 2: Using the app server directly
python app.py

# Option 3: Double-click start_web.bat (Windows)
```

The app will start the server and automatically launch your default browser at:
👉 **`http://127.0.0.1:8080`**

### Web Dashboard Features
* **⚡ Live Auto-Fetch**: Automatically polls the source site on load and at customizable intervals (e.g. 45s countdown timer).
* **🔄 Instant Manual Sync**: "Sync Now" button triggers real-time HTML re-scraping, dataset synchronization, and model recalibration.
* **📅 Multi-Day Forecast**: Interactive tabs for Monday, Tuesday, Wednesday, Thursday, Friday, and Saturday.
* **🎯 Top 10 Jodis Table**: Displays probability, relative edge over random (1.00%), visual confidence gauges, and Cut Family sets.
* **🔓 Open & Close Anks (0–9)**: Single-digit probability breakdown for opening and closing draws.
* **🧩 Family / Cut Brackets**: Automatic generation of Matka bracket family pairings (+5 mod 10).
* **🛡️ Risk & Bankroll Guardrail**: Mathematical disclosure regarding the -10% bookmaker house edge and bankroll sizing guidance.

---

## 📁 Repository Structure

```
.
├── app.py                     # Realtime Threaded HTTP Web Server and REST API
├── web/
│   ├── index.html             # Full-featured interactive web browser dashboard
│   └── prediction_data.json   # Cached baseline data snapshot
├── start_web.bat              # One-click Windows batch launcher
├── requirements.txt           # Python dependencies (requests, beautifulsoup4, pandas, tabulate)
├── scraper.py                 # Core scraping, HTML parsing, and data sanitization
├── eda.py                     # Frequency analytics, Top/Bottom Jodis, Open/Close digit distribution
├── models.py                  # Markov Chain, Recency Decay, Seasonality & Ensemble Models
├── predict.py                 # Upcoming draw forecast, Family/Cut bracket generator
├── backtest.py                # Strict walk-forward out-of-sample backtesting engine
├── main.py                    # Unified CLI orchestrator (--web, --predict, --backtest, --preview)
└── kalyan_historical_data.csv # Output normalized historical dataset (3,282 rows)
```

---

## 🚀 CLI Commands

| Command | Action |
|---|---|
| `python main.py --web` | Starts the web server and opens the browser interface |
| `python main.py --predict` | Prints upcoming draw predictions to the terminal |
| `python main.py --backtest` | Executes out-of-sample walk-forward backtest across 2,680 historical draws |
| `python main.py --eda` | Runs exploratory frequency analysis on `kalyan_historical_data.csv` |
| `python main.py --preview` | Previews the first 20 parsed rows without writing to disk |
| `python main.py` | Full pipeline: scrape, export CSV, run EDA, and generate forecast |

---

## 🧠 Predictive Model Architecture (`models.py`)

- **Markov Transition Matrix ($w=0.30$)**: Computes $P(\text{Draw}_t \mid \text{Draw}_{t-1})$ state transition probabilities with Laplace smoothing.
- **Recency-Weighted Frequency ($w=0.40$)**: Exponential half-life decay ($w_i = e^{-\lambda \Delta t}$, 60-day half-life) capturing hot streaks.
- **Day-of-Week Seasonality ($w=0.30$)**: Historical weekday prior distributions (e.g. Monday vs Saturday biases).
- **Calibrated Ensemble**: Blends all model outputs into normalized probability distributions across 100 Jodis and 10 digits.

---

## 📈 Backtest Evidence (2,680 Out-of-Sample Draws)

* **Top-5 Jodi Hit Rate**: **5.11%** (137 hits / 2,680 test draws, exceeding the 5.00% random baseline).
* **Top-1 Close Ank Hit Rate**: **9.96%** (267 hits / 2,680 test draws).
* **House Edge Reality**: Bookmakers pay 90:1 on 100 possible outcomes (-10.0% expected value), resulting in a long-term ROI of -7.99% for Top-5 Jodis in backtesting.
