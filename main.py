"""AURUM Backend v18 — multi-timeframe hedge-fund scoring"""
import asyncio, logging, os
from pathlib import Path
from time import time

import httpx, numpy as np, pandas as pd, uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from regime_engine import classify_regime, regime_to_dict
from signal_engine import generate_buy_signal, _SCORE_CACHE

log = logging.getLogger("aurum")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
HERE = Path(__file__).parent.resolve()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
TWELVE_DATA_KEY = os.getenv("TWELVE_DATA_KEY", "")
TD_BASE = "https://api.twelvedata.com"
_TD_SEM = None
_TD_LAST_CALL: float = 0.0
_TD_MIN_INTERVAL = 8.0

_CACHE: dict = {}
_MAX_CACHE = 200

def _get(k, ttl):
    e = _CACHE.get(k)
    return e[0] if e and (time() - e[1]) < ttl else None

def _set(k, v):
    if len(_CACHE) >= _MAX_CACHE:
        now = time()
        expired = [ek for ek,(ev,et) in _CACHE.items() if now-et > 600]
        for ek in expired: del _CACHE[ek]
        if len(_CACHE) >= _MAX_CACHE:
            oldest = min(_CACHE, key=lambda k: _CACHE[k][1])
            del _CACHE[oldest]
    _CACHE[k] = (v, time())

app = FastAPI(title="AURUM", version="v18")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class AnalyzeReq(BaseModel):
    symbol: str
    horizon: str = "short"

class OracleReq(BaseModel):
    symbol: str
    analysis: dict
    mode: str = "expert"

_TD_SYMS = {
    "RELIANCE","TCS","INFY","HDFCBANK","ICICIBANK","WIPRO",
    "TATAMOTORS","SUNPHARMA","BAJFINANCE","ADANIENT",
    "NESTLEIND","MARUTI","AXISBANK","LTIM","LT",
}

_SECTOR = {
    "RELIANCE":"Infra","LT":"Infra","ADANIENT":"Infra",
    "TCS":"IT","INFY":"IT","WIPRO":"IT","LTIM":"IT",
    "HDFCBANK":"Banking","ICICIBANK":"Banking","AXISBANK":"Banking","BAJFINANCE":"Banking",
    "TATAMOTORS":"Auto","MARUTI":"Auto",
    "SUNPHARMA":"Pharma",
    "NESTLEIND":"FMCG",
}

_SECTOR_WARN = {"Banking": "Banking sector shows lower historical expectancy. Signal valid — reduce position size or tighten SL."}
_COMPANY = {
    "RELIANCE":"Reliance Industries","TCS":"Tata Consultancy Services",
    "INFY":"Infosys","HDFCBANK":"HDFC Bank","ICICIBANK":"ICICI Bank",
    "WIPRO":"Wipro","TATAMOTORS":"Tata Motors","SUNPHARMA":"Sun Pharma",
    "BAJFINANCE":"Bajaj Finance","ADANIENT":"Adani Enterprises",
    "NESTLEIND":"Nestle India","MARUTI":"Maruti Suzuki",
    "AXISBANK":"Axis Bank","LTIM":"LTIMindtree","LT":"Larsen Toubro",
}

def _td_sym(sym): return sym.upper() + ":NSE"

async def _td_fetch_history(symbol: str, outputsize: int = 500) -> pd.DataFrame:
    """Fetch daily OHLCV from Twelve Data."""
    global _TD_SEM, _TD_LAST_CALL
    if _TD_SEM is None:
        _TD_SEM = asyncio.Semaphore(1)
    async with _TD_SEM:
        gap = _TD_MIN_INTERVAL - (time() - _TD_LAST_CALL)
        if gap > 0:
            await asyncio.sleep(gap)
        _TD_LAST_CALL = time()
    url = (f"{TD_BASE}/time_series?symbol={_td_sym(symbol)}"
           f"&interval=1day&outputsize={outputsize}&order=ASC&apikey={TWELVE_DATA_KEY}")
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(url)
    if r.status_code != 200:
        raise ValueError(f"DATA_INVALID: Twelve Data HTTP {r.status_code}")
    j = r.json()
    if j.get("status") == "error":
        raise ValueError(f"DATA_INVALID: {j.get('message','')}")
    values = j.get("values", [])
    if not values:
        raise ValueError(f"DATA_INVALID: No data returned")
    rows = []
    for v in values:
        try:
            rows.append({
                "Date": pd.Timestamp(v["datetime"]),
                "Open": float(v["open"]),
                "High": float(v["high"]),
                "Low": float(v["low"]),
                "Close": float(v["close"]),
                "Volume": int(v.get("volume", 0) or 0),
            })
        except (KeyError, ValueError):
            continue
    if not rows:
        raise ValueError(f"DATA_INVALID: Could not parse data")
    df = pd.DataFrame(rows).set_index("Date").sort_index()
    return df

async def _td_fetch_quote(symbol: str) -> dict:
    """Fetch latest quote."""
    url = f"{TD_BASE}/quote?symbol={_td_sym(symbol)}&apikey={TWELVE_DATA_KEY}"
    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.get(url)
    if r.status_code != 200:
        raise ValueError(f"Quote HTTP {r.status_code}")
    j = r.json()
    if j.get("status") == "error":
        raise ValueError(f"Quote error: {j.get('message','')}")
    price = float(j.get("close", 0))
    prev = float(j.get("previous_close", price))
    chg = round(price - prev, 2)
    pct = round((chg / prev) * 100, 2) if prev else 0.0
    return {"price": price, "change": chg, "change_pct": pct}

async def _analyse_async(symbol: str, horizon: str = "short") -> dict:
    """Core analysis engine."""
    cached = _get(f"analysis:{symbol}", 60)
    if cached:
        log.info("[cache] %s", symbol)
        return cached
    
    _SCORE_CACHE.clear()
    log.info("[twelvedata] %s", symbol)
    
    try:
        df = await _td_fetch_history(symbol, outputsize=500)
    except Exception as e:
        log.error("[fetch] %s: %s", symbol, e)
        raise
    
    if df is None or df.empty or len(df) < 60:
        raise ValueError(f"DATA_INVALID: Insufficient data")
    
    regime_result = classify_regime(df.tail(252))
    buy_result = generate_buy_signal(df, regime_result, symbol=symbol, horizon=horizon)
    
    df_long = df.tail(252)
    close_1y = df_long["Close"].dropna()
    price = float(round(close_1y.iloc[-1], 2))
    
    result = {
        "symbol": symbol,
        "price": price,
        "score": buy_result.score,
        "verdict": buy_result.signal,
        "confidence_label": buy_result.confidence,
        "stop_loss": buy_result.stop_loss,
        "take_profit": buy_result.take_profit,
        "rr": buy_result.risk_reward,
        "sector": _SECTOR.get(symbol, "Other"),
        "candles": [
            {"t": int(row.name.timestamp()*1000),
             "o": round(float(row.Open),2),
             "h": round(float(row.High),2),
             "l": round(float(row.Low),2),
             "c": round(float(row.Close),2),
             "v": int(row.Volume)}
            for _, row in df.tail(252).iterrows()
        ],
    }
    
    _set(f"analysis:{symbol}", result)
    return result

@app.get("/health")
async def health():
    return {"status": "ok", "version": "v18"}

@app.get("/")
async def serve_frontend():
    return FileResponse(HERE / "aurum_standalone.html", media_type="text/html")

@app.post("/analyze")
async def analyze(req: AnalyzeReq):
    sym = req.symbol.strip().upper()
    if not sym:
        raise HTTPException(400, "symbol required")
    try:
        return await _analyse_async(sym, req.horizon)
    except ValueError as e:
        raise HTTPException(422, str(e))
    except Exception as e:
        log.error("/analyze %s: %s", sym, e)
        raise HTTPException(500, str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"\n⚡ AURUM v18 → http://localhost:{port}")
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)