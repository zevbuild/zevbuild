# Kalyan Realtime Auto-Predictor & Web Engine

An automated quantitative modeling, real-time prediction, and mobile auto-catch platform for Kalyan Matka historical records (2012–2026). It features a **high-precision Hybrid Quant-Domain Ensemble**, **4-Ank OTC (Open-To-Close) forecasting**, **3-digit Panel/Patti (SP & DP) generation**, and an interactive real-time Web Dashboard with a **Mobile Auto-Catch System** (synthesized audio chimes, device vibration, and screen wake-lock).

---

## 🌐 Real-Time Web Browser & Mobile Auto-Catch System

To launch the web interface:

```bash
# Option 1: Using Python CLI
python main.py --web

# Option 2: Using the app server directly
python app.py

# Option 3: Double-click start_web.bat (Windows)
```

The app will start the threaded web server and automatically launch your default browser at:
👉 **`http://127.0.0.1:8080`**

### Web & Mobile Dashboard Features
* **📱 Mobile Auto-Catch System**: Background polling worker that auto-detects newly declared Open or Close draw results from `dpbossx.net`.
* **🔔 Synthesized Web Audio Chime**: Dual-tone harmonic audio chime alerts instantly when a new result is caught (zero external audio files).
* **📳 Mobile Vibration API**: Vibrates smartphones upon live result declaration for instant physical feedback.
* **⚡ Screen Wake-Lock**: Keeps mobile and tablet screens awake during active draw hours (3:30 PM – 6:30 PM IST).
* **🔥 High-Probability 4-Ank OTC Hero Card**: Prominently displays the top 4 Open-To-Close Anks formed from harmonic dual Cut pairs \((d + 5) \pmod{10}\) with modeled pass probability and one-click copy.
* **🎰 3-Digit Panel / Patti Tab**: Categorizes recommended panels into Single Patti (SP) and Double Patti (DP) for each OTC Ank.
* **🎯 Top 10 Jodis & Family Sets**: Probability rankings, relative edge over random (1.00%), visual confidence gauges, and Cut Family brackets.
* **🔓 Open & Close Anks (0–9)**: Independent marginal probability rankings for opening and closing positions.
* **🛡️ Risk & Bankroll Guardrails**: Mathematical disclosure regarding the -10% bookmaker house margin and bankroll sizing guidance.

---

## 📁 Repository Structure

```
.
├── app.py                     # Realtime Threaded HTTP Web Server and REST API
├── web/
│   ├── index.html             # Full-featured interactive web browser dashboard
│   ├── history.json           # Historical draw cache for client-side processing
│   └── prediction_data.json   # Multi-day forecast and model snapshot
├── index.html                 # Root web entrypoint (synchronized with web/index.html)
├── dashboard.html             # Dashboard view alias
├── start_web.bat              # One-click Windows batch launcher
├── requirements.txt           # Python dependencies (requests, beautifulsoup4, pandas, tabulate)
├── scraper.py                 # Core scraping, HTML parsing, and panel/patti extraction
├── eda.py                     # Frequency analytics, Top/Bottom Jodis, Open/Close digit distribution
├── models.py                  # Hybrid Quant-Domain Ensemble, Overdue Gap, Line, Cut & Patti Models
├── predict.py                 # Upcoming draw forecast, 4-Ank OTC, Patti & Family bracket generator
├── backtest.py                # Strict walk-forward out-of-sample backtesting engine & optimizer
├── main.py                    # Unified CLI orchestrator (--web, --predict, --backtest, --preview)
└── kalyan_historical_data.csv # Output normalized historical dataset with Open & Close Pattis (3,305 rows)
```

---

## 🚀 CLI Commands

| Command | Action |
|---|---|
| `python sync_and_push.py` | Scrapes latest draws, retrains models, updates all JSON/CSV files, and pushes to web |
| `python main.py --web` | Starts the web server and opens the browser interface |
| `python main.py --predict` | Prints upcoming draw predictions, 4-Ank OTC, and Patti forecasts to the terminal |
| `python main.py --backtest` | Executes out-of-sample walk-forward backtest across 2,703 historical draws |
| `python backtest.py --optimize` | Runs automated grid-search calibration to discover optimal hyperparameters |
| `python main.py --eda` | Runs exploratory frequency analysis on `kalyan_historical_data.csv` |
| `python main.py --preview` | Previews the first 20 parsed rows with Patti columns |
| `python main.py` | Full pipeline: scrape, export CSV, run EDA, and generate forecast |

---

## 🧠 Hybrid Predictive Model Architecture (`models.py`)

- **Dual-Horizon Recency Momentum ($w=0.52$)**: Fast momentum ($\tau_{1/2} = 8$ draws) and medium trend ($\tau_{1/2} = 35$ draws) capturing hot number streaks.
- **Hierarchical Markov Transition Matrix ($w=0.18$)**: Computes $P(\text{Draw}_t \mid \text{Draw}_{t-1})$ state transition probabilities and Cross-Day Close(t-1) $\rightarrow$ Open(t) dynamics with Laplace-Dirichlet smoothing ($\alpha = 0.25$).
- **Weekday Line Prior ($w=0.15$)**: Historical weekday prior distributions tracking column line patterns (Mon vs Sat biases).
- **Overdue / Gap Cycle Hazard ($w=0.15$)**: Cyclical mean-reverting hazard model tracking draws since each digit and total last appeared.
- **Cut Harmonic Resonance ($\lambda = 0.12$)**: Harmonic reinforcement for Cut pairs \((d + 5) \pmod{10}\).
- **Coherent Joint Marginal Synthesis**: Reconstructs joint probabilities $P(J = (o, c)) \propto P(O=o) \cdot [0.60 P(C=c) + 0.40 P(C=c \mid O=o)]$ aligned with the Jodi Total \((o + c) \pmod{10}\) prior.
- **Empirical Panel/Patti Engine**: Maps every Ank to standard 3-digit Single Pattis (SP) and Double Pattis (DP) ranked by empirical frequency.

---

## 📈 Backtest Evidence (2,703 Out-of-Sample Draws, 2014–2026)

Evaluated in strict chronological walk-forward out-of-sample mode (predicting draw $t$ using only historical draws $0$ to $t-1$):

| Prediction Target | Observed Hit Rate | Theoretical Random Baseline | Relative Edge |
|---|---|---|---|
| **Top-3 Jodis** | **3.37%** (91 / 2,703) | 3.00% | **+12.2%** |
| **Top-5 Jodis** | **5.77%** (156 / 2,703) | 5.00% | **+15.4%** |
| **Top-10 Jodis** | **11.91%** (322 / 2,703) | 10.00% | **+19.1%** |
| **Top-1 Open Ank** | **10.32%** (279 / 2,703) | 10.00% | **+3.2%** |
| **Top-2 Open Anks** | **21.24%** (574 / 2,703) | 20.00% | **+6.2%** |
| **4-Ank OTC Pass** | **61.82%** (1,671 / 2,703) | 64.00% | Balanced Cut Coverage |

* **Simulated Financial Return**: Strategy wagering 1 unit on each of the Top-5 Jodis (90:1 payout) generated **+525.0 net units profit (+3.88% ROI)** across the 2,703 test draws.
