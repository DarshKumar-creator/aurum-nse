# AURUM NSE — Real-Time Signal Terminal

Live NSE signal terminal for 15 large-cap Indian equities. Full-stack FastAPI + single-file HTML frontend, deployed on Render (Singapore region).

**Live:** https://aurum-nse.onrender.com

---

## Architecture

```
regime_engine.py   → 3-state market classifier (TRENDING_BULL / SIDEWAYS / TRANSITION)
signal_engine.py   → BUY signal engine (SL=1.2×ATR, TP=2.5×ATR adaptive)
sell_engine.py     → SELL detectors (SELL_ENABLED=False — pending 90-day real-data calibration)
risk_manager.py    → Position sizing (HIGH=1.5×, MEDIUM=1.0×, LOW=0.5×, early_warning=0.5×)
evaluation_engine.py → Walk-forward backtester (CLI)
main.py            → FastAPI (9 endpoints), Twelve Data, Gemini Oracle
aurum_standalone.html → Black/gold terminal UI (single file, no build step)
```

## Stock Universe (15)

| Sector | Symbols |
|--------|---------|
| IT | TCS, INFY, WIPRO, LTIM |
| Banking | HDFCBANK, ICICIBANK, AXISBANK, BAJFINANCE |
| Auto | TATAMOTORS, MARUTI |
| Infra | RELIANCE, LT, ADANIENT |
| Pharma | SUNPHARMA |
| FMCG | NESTLEIND |

**Banking note:** Weakest sector at -0.024R historical expectancy. Signals valid, lower conviction.

## Eval Results (v18 — synthetic GBM, 15 stocks, 1200 days, seed=42)

```
Verdict:    9/9 READY
E/trade:    +0.1436R
Dir acc:    67.8%
Win rate:   47.8%
Streak:     10 max consecutive losses
Windows:    9/15 positive
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Frontend HTML |
| GET | `/health` | Health check |
| GET | `/universe` | 15 stocks with sector + warnings |
| POST | `/analyze` | Full signal analysis (returns candles, SL/TP, gates, sector) |
| POST | `/oracle` | Gemini AI commentary (beginner/expert mode) |
| GET | `/chart/{symbol}` | OHLCV data (252 bars, 5-min cache) |
| GET | `/price/{symbol}` | Live price + change (30s cache) |
| GET | `/indices` | NIFTY 50, BANK NIFTY, IT, Midcap, VIX |
| GET | `/news/{symbol}` | News sentiment (3-day rolling window) |
| GET | `/expectancy` | Rolling expectancy from cached signals |

## Deployment

### Render (production)

1. Push to GitHub: `git add -A && git commit -m "v18" && git push`
2. Render auto-deploys from `main` branch
3. Set env vars in Render dashboard:
   - `TWELVE_DATA_KEY` — [twelvedata.com](https://twelvedata.com) free tier
   - `GEMINI_API_KEY` — [aistudio.google.com](https://aistudio.google.com) free
   - `NEWS_API_KEY` — [newsapi.org](https://newsapi.org) optional

### Local

```bash
cp .env.example .env   # fill in your keys
bash start.sh
# → http://localhost:8000
```

## Rate Limits

Twelve Data free tier: 8 calls/minute. The backend enforces an 8-second minimum interval between calls via `asyncio.Semaphore`. Batch analysis of all 15 stocks takes ~2 minutes on cold start — this is expected.

## Evaluation

```bash
python3 evaluation_engine.py              # run eval (takes ~30s)
python3 evaluation_engine.py show all     # full results
python3 evaluation_engine.py show verdict # pass/fail checks only
```

## Known Limitations

- Backtested on synthetic GBM data — real NSE results will differ
- SELL engine disabled — architecture correct, unvalidated on real data
- Banking sector historically weaker (flagged in every signal response)
- Cold start on Render free tier: 30–60s delay on first request
- 8 calls/min TD rate limit means full batch takes ~2 min

## v18 Changes vs v17

- Fixed 24 crash paths (NameError, AttributeError, missing endpoints)
- GET / now registered at module level (was 404 on Render)
- Charts now show real NSE OHLCV data (was always synthetic)
- SL/TP sourced from signal engine (includes adaptive TP for compressed vol)
- risk_manager wired into position sizing
- Sector system: sector + sector_warning on every analyze response
- TD rate limiter: prevents 429 on batch analysis
- Banking warning surfaced in UI
- 15 stocks in eval (was 12 — BAJFINANCE, ADANIENT, AXISBANK missing)
- eval _sim corrected to actual R calculation (was hardcoded +1.2R for TP)
