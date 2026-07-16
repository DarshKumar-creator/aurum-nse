"""AURUM Backend v12 — multi-timeframe hedge-fund scoring"""
import asyncio, logging, os
from pathlib import Path
from time import time

import httpx, numpy as np, pandas as pd, uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

# ── v17 layered signal engine (regime → buy/sell → risk) ───────────────────
from regime_engine import classify_regime, regime_to_dict
from signal_engine import generate_buy_signal, _SCORE_CACHE

log = logging.getLogger("aurum")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
HERE = Path(__file__).parent.resolve()

GEMINI_API_KEY    = os.getenv("GEMINI_API_KEY", "")
NEWS_API_KEY      = os.getenv("NEWS_API_KEY", "")
TWELVE_DATA_KEY   = os.getenv("TWELVE_DATA_KEY", "")           # required — set via env
TD_BASE           = "https://api.twelvedata.com"
_TD_SEM = None   # asyncio.Semaphore — init on first use (needs running loop)
_TD_LAST_CALL: float = 0.0   # timestamp of last TD API call
_TD_MIN_INTERVAL = 8.0       # seconds between TD calls (8/min free tier)

# ─── CACHE ────────────────────────────────────────────────────────────────────
_CACHE: dict = {}
_MAX_CACHE = 200   # evict oldest when exceeded
def _get(k, ttl):
    e = _CACHE.get(k)
    return e[0] if e and (time() - e[1]) < ttl else None
def _set(k, v):
    if len(_CACHE) >= _MAX_CACHE:          # evict expired first, then oldest
        now = time()
        expired = [ek for ek,(ev,et) in _CACHE.items() if now-et > 600]
        for ek in expired: del _CACHE[ek]
        if len(_CACHE) >= _MAX_CACHE:      # still full — drop oldest
            oldest = min(_CACHE, key=lambda k: _CACHE[k][1])
            del _CACHE[oldest]
    _CACHE[k] = (v, time())

# ─── SYMBOL MAP ───────────────────────────────────────────────────────────────
# Twelve Data uses SYMBOL:NSE format for NSE stocks
_TD_SYMS = {
    "RELIANCE","TCS","INFY","HDFCBANK","ICICIBANK","WIPRO",
    "TATAMOTORS","SUNPHARMA","BAJFINANCE","ADANIENT",
    "NESTLEIND","MARUTI","AXISBANK","LTIM","LT",
}
# ─── SECTOR MAP + WARNINGS ────────────────────────────────────────────────────
_SECTOR = {
    "RELIANCE":"Infra","LT":"Infra","ADANIENT":"Infra",
    "TCS":"IT","INFY":"IT","WIPRO":"IT","LTIM":"IT",
    "HDFCBANK":"Banking","ICICIBANK":"Banking","AXISBANK":"Banking","BAJFINANCE":"Banking",
    "TATAMOTORS":"Auto","MARUTI":"Auto",
    "SUNPHARMA":"Pharma",
    "NESTLEIND":"FMCG",
}
# Banking sector eval expectancy: -0.024R — valid signals, lower conviction
_SECTOR_WARN = {"Banking": "Banking sector shows lower historical expectancy (-0.024R). "
                            "Signal valid — reduce position size or tighten SL."}


def _td_sym(sym): return sym.upper() + ":NSE"

async def _td_fetch_history(symbol: str, outputsize: int = 500) -> pd.DataFrame:
    """Fetch daily OHLCV from Twelve Data, return as DataFrame sorted oldest→newest."""
    global _TD_SEM, _TD_LAST_CALL
    if _TD_SEM is None:
        _TD_SEM = asyncio.Semaphore(1)   # one TD call at a time
    async with _TD_SEM:
        gap = _TD_MIN_INTERVAL - (time() - _TD_LAST_CALL)
        if gap > 0:
            await asyncio.sleep(gap)
        _TD_LAST_CALL = time()
    url = (f"{TD_BASE}/time_series?symbol={_td_sym(symbol)}"
           f"&interval=1day&outputsize={outputsize}&order=ASC&apikey={TWELVE_DATA_KEY}")
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(url)
    if r.status_code == 429:
        raise ValueError(f"DATA_INVALID: Twelve Data rate limit for {symbol}")
    if r.status_code != 200:
        raise ValueError(f"DATA_INVALID: Twelve Data HTTP {r.status_code} for {symbol}")
    j = r.json()
    if j.get("status") == "error" or j.get("code") == 429:
        raise ValueError(f"DATA_INVALID: Twelve Data error for {symbol}: {j.get('message','')}")
    values = j.get("values", [])
    if not values:
        raise ValueError(f"DATA_INVALID: Twelve Data returned no data for {symbol}")
    rows = []
    for v in values:
        try:
            rows.append({
                "Date":   pd.Timestamp(v["datetime"]),
                "Open":   float(v["open"]),
                "High":   float(v["high"]),
                "Low":    float(v["low"]),
                "Close":  float(v["close"]),
                "Volume": int(v.get("volume", 0) or 0),
            })
        except (KeyError, ValueError):
            continue
    if not rows:
        raise ValueError(f"DATA_INVALID: could not parse Twelve Data response for {symbol}")
    df = pd.DataFrame(rows).set_index("Date").sort_index()
    return df

async def _td_fetch_quote(symbol: str) -> dict:
    """Fetch latest quote from Twelve Data."""
    url = (f"{TD_BASE}/quote?symbol={_td_sym(symbol)}&apikey={TWELVE_DATA_KEY}")
    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.get(url)
    if r.status_code == 429:
        raise ValueError(f"Quote rate limit for {symbol} — retry in 60s")
    if r.status_code != 200:
        raise ValueError(f"Quote HTTP {r.status_code} for {symbol}")
    j = r.json()
    if j.get("code") == 429 or j.get("status") == "error":
        raise ValueError(f"Quote error for {symbol}: {j.get('message','')}")
    price = float(j.get("close", 0))
    prev  = float(j.get("previous_close", price))
    chg   = round(price - prev, 2)
    pct   = round((chg / prev) * 100, 2) if prev else 0.0
    return {"price": price, "change": chg, "change_pct": pct}

# ─────────────────────────────────────────────────────────────────────────────
# INDICATORS — all throw on bad data, zero silent fallbacks
# ─────────────────────────────────────────────────────────────────────────────
def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def _rsi(close: pd.Series, n: int = 14) -> float:
    need = n + 5
    if len(close) < need:
        raise ValueError(f"RSI({n}) needs {need} bars, got {len(close)}")
    d  = close.diff()
    ag = d.clip(lower=0).ewm(alpha=1/n, min_periods=n, adjust=False).mean()
    al = (-d).clip(lower=0).ewm(alpha=1/n, min_periods=n, adjust=False).mean()
    rs = ag / al.replace(0, np.nan)
    v  = (100 - 100 / (1 + rs)).iloc[-1]
    if np.isnan(v):
        raise ValueError(f"RSI({n}) is NaN")
    return float(round(v, 2))

def _macd(close: pd.Series, fast=12, slow=26, sig=9):
    need = slow + sig + 2
    if len(close) < need:
        raise ValueError(f"MACD needs {need} bars, got {len(close)}")
    line   = _ema(close, fast) - _ema(close, slow)
    signal = _ema(line, sig)
    mv, sv = line.iloc[-1], signal.iloc[-1]
    if np.isnan(mv) or np.isnan(sv):
        raise ValueError("MACD is NaN")
    return float(round(mv, 4)), float(round(sv, 4))

def _atr(h: pd.Series, l: pd.Series, c: pd.Series, n: int = 14) -> float:
    if len(c) < n + 2:
        raise ValueError(f"ATR needs {n+2} bars, got {len(c)}")
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    v  = tr.ewm(alpha=1/n, min_periods=n, adjust=False).mean().iloc[-1]
    if np.isnan(v):
        raise ValueError("ATR is NaN")
    return float(round(v, 2))


def _adx(h: pd.Series, l: pd.Series, c: pd.Series, n: int = 14) -> float:
    """
    ADX via Wilder RMA smoothing — user's implementation (cleaner np.maximum TR).
    Returns 0–100:  <18 weak/sideways · 18–25 developing · >25 confirmed trend · >40 strong
    """
    if len(c) < n * 2 + 5:
        return 0.0

    # True Range — user's np.maximum approach (avoids pd.concat overhead)
    pc   = c.shift(1)
    tr1  = h - l
    tr2  = (h - pc).abs()
    tr3  = (l - pc).abs()
    tr   = pd.Series(np.maximum(tr1, np.maximum(tr2, tr3)), index=c.index)

    # Directional movement — user's .diff() approach
    plus_dm_raw  = h.diff()
    minus_dm_raw = -l.diff()
    plus_dm  = np.where((plus_dm_raw  > minus_dm_raw) & (plus_dm_raw  > 0), plus_dm_raw,  0.0)
    minus_dm = np.where((minus_dm_raw > plus_dm_raw)  & (minus_dm_raw > 0), minus_dm_raw, 0.0)

    # Wilder (RMA) smoothing
    atr_s      = tr.ewm(alpha=1/n, adjust=False).mean()
    plus_di    = 100 * pd.Series(plus_dm,  index=c.index).ewm(alpha=1/n, adjust=False).mean() / atr_s.replace(0, np.nan)
    minus_di   = 100 * pd.Series(minus_dm, index=c.index).ewm(alpha=1/n, adjust=False).mean() / atr_s.replace(0, np.nan)

    dx  = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
    adx = dx.ewm(alpha=1/n, adjust=False).mean()

    val = adx.iloc[-1]
    return float(round(val, 2)) if pd.notna(val) else 0.0


def _is_sideways(
    h: pd.Series,
    l: pd.Series,
    c: pd.Series,
    ema50:  float,
    ema200: float,
    n: int = 14,
) -> tuple[bool, dict]:
    """
    Multi-condition SIDEWAYS detector — user's conditions:
      1. ADX < 18          — no directional strength (Wilder RMA)
      2. EMA50/200 spread < 1%   — both MAs compressed, no structural trend
      3. ATR% < 1.2%       — volatility contracted (user's condition, cleaner than range %)

    ALL THREE must hold (AND logic) — reduces false positives vs any single filter.
    Returns (is_sideways: bool, diagnostics: dict)
    """
    px = float(c.iloc[-1]) if len(c) > 0 else 1.0

    # 1. ADX
    adx_val  = _adx(h, l, c, n)
    cond_adx = adx_val < 18.0

    # 2. EMA spread (user's ema50/200 pair — long-term structural compression)
    ema_dist = abs(ema50 - ema200) / ema200 if ema200 > 0 else 0.0
    cond_ema = ema_dist < 0.01

    # 3. ATR% — relative to the stock's own history, not a universal absolute threshold.
    # Universal 1.2% wrongly blocked NESTLEIND/SUNPHARMA (inherently ~0.9–1.1% ATR)
    # even during genuine trends. Stock-relative check: is current ATR contracted
    # vs its own 60-day rolling average? If yes → volatility is compressed → sideways.
    pc  = c.shift(1)
    tr  = pd.Series(np.maximum(h-l, np.maximum((h-pc).abs(), (l-pc).abs())), index=c.index)
    atr_series = tr.ewm(alpha=1/n, adjust=False).mean()
    atr_now    = float(atr_series.iloc[-1])
    atr_60d    = float(atr_series.iloc[-60:].mean()) if len(atr_series) >= 60 else atr_now
    atr_pct    = atr_now / px if px > 0 else 0.0
    # Sideways if current ATR is below 80% of own 60-day average (contracted volatility)
    atr_relative_ratio = atr_now / atr_60d if atr_60d > 0 else 1.0
    cond_atr   = atr_relative_ratio < 0.80

    sideways = cond_adx and cond_ema and cond_atr

    trend_strength = (
        "strong"   if adx_val > 25 else
        "moderate" if adx_val > 18 else
        "weak"
    )

    diag = {
        # ADX block
        "adx_value":           round(adx_val,            2),
        "trend_strength":      trend_strength,
        # Regime block
        "ema_spread":          round(ema_dist,            6),
        "atr_pct":             round(atr_pct,             6),
        "atr_relative_ratio":  round(atr_relative_ratio,  3),  # <0.80 → contracted
        # Individual gate conditions
        "cond_adx":            cond_adx,
        "cond_ema":            cond_ema,
        "cond_atr":            cond_atr,
    }
    return sideways, diag



def _centered_score(
    rsi: float, macd: float, signal: float,
    price: float, ema_fast: float, ema_slow: float,
    close: "pd.Series", n_mom: int, atr: float
) -> float:
    """
    Centered scoring model (0-100) that spreads across the full range.
    Starts at 50 and adds/subtracts based on signal strength.
    Strong signals move score 20-30 pts; weak/conflicting → stays near 50.
    """
    score = 50.0

    # ── RSI: oversold → bullish, overbought → bearish ───────────────────────
    if rsi < 30:
        score += 20.0                       # strong oversold — bounce potential
    elif rsi > 70:
        score -= 20.0                       # overbought — mean reversion risk
    else:
        score += (50.0 - rsi) * 0.35        # RSI=60 → -3.5, RSI=40 → +3.5

    # ── MACD: magnitude matters, not just direction ──────────────────────────
    macd_delta = macd - signal
    # Normalise: typical NSE MACD deltas are 0-20 units vs price; cap at 15pt swing
    macd_pct   = macd_delta / price * 100 if price > 0 else 0.0
    macd_pts   = max(-15.0, min(15.0, macd_pct * 50))
    score += macd_pts

    # ── EMA position: distance from EMA matters ──────────────────────────────
    if ema_fast > 0:
        ema_dist_pct = (price - ema_fast) / ema_fast * 100
        # Far above EMA = overextended; far below = value; near = neutral
        ema_pts = max(-12.0, min(12.0, ema_dist_pct * 1.5))
        score += ema_pts

    # ── EMA50 vs EMA200 trend (golden/death cross) ───────────────────────────
    if ema_slow > 0 and ema_fast > 0:
        cross_pct = (ema_fast - ema_slow) / ema_slow * 100
        cross_pts = max(-10.0, min(10.0, cross_pct * 2.0))
        score += cross_pts

    # ── Momentum: recent price change ────────────────────────────────────────
    n_close = len(close)
    if n_close > n_mom:
        prev  = float(close.iloc[-(n_mom + 1)])
        curr  = float(close.iloc[-1])
        mom   = (curr - prev) / prev if prev > 0 else 0.0
        if mom > 0.02:
            score += 10.0
        elif mom < -0.02:
            score -= 10.0
        else:
            score += mom * 400          # linear in between: -8 to +8
        # Sideways detection — flat = low conviction
        if abs(mom) < 0.005:
            score -= 5.0
    else:
        mom = 0.0

    # ── Volatility penalty ────────────────────────────────────────────────────
    if price > 0:
        atr_ratio = atr / price
        if atr_ratio > 0.03:
            score -= 8.0
        elif atr_ratio > 0.02:
            score -= 4.0

    # ── Anti-bias: prevent overbought BUY spam / oversold SELL spam ──────────
    if score > 65 and rsi > 65:
        score -= 5.0
    if score < 35 and rsi < 35:
        score += 5.0

    return round(float(np.clip(score, 0.0, 100.0)), 1)


def _compute_tf_score(df_slice: pd.DataFrame, tf: str, global_atr: float = 0.0) -> dict:
    """
    Centered scoring model per timeframe. Returns score 0-100 with full spread.
    """
    close  = df_slice["Close"].dropna()
    high   = df_slice["High"].dropna()
    low    = df_slice["Low"].dropna()
    n      = len(close)

    if n < 20:
        raise ValueError(f"DATA_INVALID: {tf} timeframe needs 20 bars, got {n}")

    price = float(close.iloc[-1])

    if tf == "short":
        rsi9  = _rsi(close, 9)
        ema20 = float(_ema(close, 20).iloc[-1])
        mv, sv = _macd(close)
        n_mom = 5
        ema_slow = ema20   # no long EMA available on 44 bars

        if not np.isfinite(rsi9) or not np.isfinite(ema20):
            raise ValueError(f"DATA_INVALID: NaN in short-term indicators")

        atr_val = global_atr if global_atr > 0 else float(_atr(high, low, close))
        score = _centered_score(rsi9, mv, sv, price, ema20, ema_slow, close, n_mom, atr_val)

        return {"score": score, "rsi": rsi9, "ema": ema20, "macd": mv, "macd_signal": sv}

    elif tf == "medium":
        if n < 37:
            raise ValueError(f"DATA_INVALID: medium needs 37 bars, got {n}")
        rsi14  = _rsi(close, 14)
        ema50  = float(_ema(close, min(50, n)).iloc[-1])
        ema20  = float(_ema(close, 20).iloc[-1])
        mv, sv = _macd(close)
        n_mom  = 10

        if not np.isfinite(rsi14) or not np.isfinite(ema50):
            raise ValueError(f"DATA_INVALID: NaN in medium-term indicators")

        atr_val = global_atr if global_atr > 0 else float(_atr(high, low, close))
        score = _centered_score(rsi14, mv, sv, price, ema20, ema50, close, n_mom, atr_val)

        return {"score": score, "rsi": rsi14, "ema": ema50, "macd": mv, "macd_signal": sv}

    else:  # long
        if n < 50:
            raise ValueError(f"DATA_INVALID: long needs 50 bars, got {n}")
        rsi14  = _rsi(close, 14)
        ema50  = float(_ema(close, 50).iloc[-1])
        ema200 = float(_ema(close, 200).iloc[-1]) if n >= 200 else ema50
        mv, sv = _macd(close)
        n_mom  = 20

        if not np.isfinite(rsi14) or not np.isfinite(ema50):
            raise ValueError(f"DATA_INVALID: NaN in long-term indicators")

        atr_val = global_atr if global_atr > 0 else float(_atr(high, low, close))
        score = _centered_score(rsi14, mv, sv, price, ema50, ema200, close, n_mom, atr_val)

        return {
            "score":       score,
            "rsi":         rsi14,
            "ema50":       ema50,
            "ema200":      ema200,
            "macd":        mv,
            "macd_signal": sv,
        }


# ─────────────────────────────────────────────────────────────────────────────
# REGIME DETECTION
# ─────────────────────────────────────────────────────────────────────────────
def _detect_regime(close: pd.Series) -> str:
    """
    TRENDING_BULL  — price above EMA50 and EMA50 itself is rising
    TRENDING_BEAR  — price below EMA50 and EMA50 itself is falling
    SIDEWAYS       — everything else
    """
    if len(close) < 22:
        return "SIDEWAYS"
    price = float(close.iloc[-1])
    ema50_series = _ema(close, 50)
    ema50_now    = float(ema50_series.iloc[-1])
    ema50_20ago  = float(ema50_series.iloc[-21])
    ema_rising   = ema50_now > ema50_20ago

    if price > ema50_now and ema_rising:
        return "TRENDING_BULL"
    elif price < ema50_now and not ema_rising:
        return "TRENDING_BEAR"
    else:
        return "SIDEWAYS"

# ─────────────────────────────────────────────────────────────────────────────
# EXPERT BAR HELPERS  (FIX: these were called in _analyse_async but never defined)
# Each returns a value in [-1, +1] for use in bar calculations.
# ─────────────────────────────────────────────────────────────────────────────
def _score_rsi(rsi: float) -> float:
    """RSI → [-1, +1]. <30 = strong bull, >70 = strong bear, else linear."""
    if rsi < 30:   return  1.0
    elif rsi > 70: return -1.0
    return (50.0 - rsi) / 20.0   # 0 at 50, ±1 at 30/70

def _score_macd(macd: float, signal: float) -> float:
    """MACD vs Signal → [-1, +1].  Positive when macd > signal."""
    delta = macd - signal
    if delta == 0: return 0.0
    return float(np.clip(delta / (abs(signal) + 1e-9), -1.0, 1.0))

def _score_ema(price: float, ema50: float) -> float:
    """Price vs EMA50 → [-1, +1]. Above EMA = bullish."""
    if ema50 <= 0: return 0.0
    dist_pct = (price - ema50) / ema50 * 100
    return float(np.clip(dist_pct / 5.0, -1.0, 1.0))   # ±5% → ±1

def _score_trend(ema50: float, ema200: float) -> float:
    """EMA50 vs EMA200 golden/death cross → [-1, +1]."""
    if ema200 <= 0: return 0.0
    cross_pct = (ema50 - ema200) / ema200 * 100
    return float(np.clip(cross_pct / 5.0, -1.0, 1.0))  # ±5% → ±1

def _score_volume(vol_series: "pd.Series") -> float:
    """Recent volume vs 20-day avg → [-1, +1].  Above avg = bullish."""
    if len(vol_series) < 21: return 0.0
    recent = float(vol_series.iloc[-1])
    avg    = float(vol_series.iloc[-20:].mean())
    if avg <= 0: return 0.0
    ratio = (recent - avg) / avg     # >0 means above avg
    return float(np.clip(ratio / 1.5, -1.0, 1.0))  # 1.5× avg → +1

# ─────────────────────────────────────────────────────────────────────────────
# RISK LEVELS
# ─────────────────────────────────────────────────────────────────────────────
def _risk(price: float, atr: float, verdict: str):
    """
    Risk levels — backtest-validated configuration.
    TP = 1.2×ATR, SL = 1.0×ATR  (1.2:1 R:R)
    Previous: TP=2×ATR, SL=1.5×ATR — too far for 10-day window, won only 38% of trades.
    At 1.2×ATR TP with 60%+ directional accuracy: E ≈ +0.12R per trade (positive).
    """
    if verdict == "HOLD" or atr <= 0:
        return 0.0, 0.0, 0.0
    if "BUY" in verdict:
        sl = round(price - 1.0 * atr, 2)
        tp = round(price + 1.2 * atr, 2)
    else:
        sl = round(price + 1.0 * atr, 2)
        tp = round(price - 1.2 * atr, 2)
    rr = round(abs(tp - price) / abs(price - sl), 2) if abs(price - sl) > 0 else 0.0
    return sl, tp, rr


# ─────────────────────────────────────────────────────────────────────────────
# NEWS SENTIMENT ENGINE
# Keyword-based VADER-style scorer — no extra pip install required.
# Returns sentiment in [-1, +1]. Fails to 0.0 silently.
# ─────────────────────────────────────────────────────────────────────────────

_POS = {
    "surge","surges","surged","rally","rallied","rallies","jump","jumped","jumps",
    "gain","gains","gained","rise","rises","rose","soar","soared","soars",
    "beat","beats","record","strong","stronger","bullish","upgrade","upgraded",
    "outperform","buy","positive","growth","profit","profits","revenue","boost",
    "boosted","expansion","expand","expanding","highest","record-high","upbeat",
    "breakout","momentum","robust","beat expectations","raises guidance",
}
_NEG = {
    "fall","falls","fell","drop","drops","dropped","plunge","plunged","plunges",
    "decline","declines","declined","slip","slips","slipped","crash","crashes",
    "loss","losses","weak","weaker","bearish","downgrade","downgraded",
    "underperform","sell","negative","miss","misses","missed","disappoint",
    "disappoints","disappointed","cut","cuts","concern","concerns","warning",
    "warns","warned","slowdown","slowdowns","lowest","record-low","downbeat",
    "investigation","probe","fraud","scandal","regulatory","penalty","fine",
    "miss expectations","lowers guidance",
}

def _score_text(text: str) -> float:
    """Score a single text string → [-1, +1]."""
    words = text.lower().split()
    pos = sum(1 for w in words if w.strip(".,!?;:'\"") in _POS)
    neg = sum(1 for w in words if w.strip(".,!?;:'\"") in _NEG)
    total = pos + neg
    if total == 0:
        return 0.0
    return max(-1.0, min(1.0, (pos - neg) / total))

# Map NSE symbol → company name for better news search
_COMPANY = {
    "RELIANCE":"Reliance Industries","TCS":"Tata Consultancy Services",
    "INFY":"Infosys","HDFCBANK":"HDFC Bank","ICICIBANK":"ICICI Bank",
    "WIPRO":"Wipro","TATAMOTORS":"Tata Motors","SUNPHARMA":"Sun Pharma",
    "BAJFINANCE":"Bajaj Finance","ADANIENT":"Adani Enterprises",
    "NESTLEIND":"Nestle India","MARUTI":"Maruti Suzuki",
    "AXISBANK":"Axis Bank","LTIM":"LTIMindtree","LT":"Larsen Toubro",
}

async def _fetch_news_sentiment(symbol: str) -> dict:
    """Fetch latest news and return sentiment metrics.
    Falls back to neutral (0.0) if API key missing or request fails.
    Result cached 300s."""
    neutral = {"news_sentiment": 0.0, "news_strength": 0, "news_bias": "NEUTRAL"}
    if not NEWS_API_KEY:
        return neutral

    cache_key = f"news_sent:{symbol}"
    cached = _get(cache_key, 300)
    if cached:
        return cached

    company = _COMPANY.get(symbol, symbol)
    query   = f"{company} NSE stock India"
    url     = (f"https://newsapi.org/v2/everything?q={query}"
               f"&sortBy=publishedAt&pageSize=8&language=en"
               f"&from={(lambda: (__import__('datetime').date.today() - __import__('datetime').timedelta(days=3)).isoformat())()}"
               f"&apiKey={NEWS_API_KEY}")

    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            r = await client.get(url)
        if r.status_code != 200:
            return neutral

        articles = r.json().get("articles", [])
        scores   = []
        for a in articles:
            if not a.get("title") or "[Removed]" in a.get("title",""):
                continue
            text  = (a.get("title","") + " " + (a.get("description") or "")).strip()
            score = _score_text(text)
            scores.append(score)

        if not scores:
            return neutral

        avg = round(sum(scores) / len(scores), 3)
        avg = max(-1.0, min(1.0, avg))
        bias = "POSITIVE" if avg > 0.15 else ("NEGATIVE" if avg < -0.15 else "NEUTRAL")

        result = {"news_sentiment": avg, "news_strength": len(scores), "news_bias": bias}
        _set(cache_key, result)
        return result

    except Exception as e:
        log.warning("[news_sent] %s: %s", symbol, e)
        return neutral


async def _fetch_market_sentiment() -> float:
    """Fetch NIFTY 50 news sentiment for macro bias. Returns [-1,+1]."""
    if not NEWS_API_KEY:
        return 0.0
    cached = _get("market_sent", 300)
    if cached is not None:
        return cached

    url = (f"https://newsapi.org/v2/everything?q=Nifty+50+NSE+India+market"
           f"&sortBy=publishedAt&pageSize=5&language=en&apiKey={NEWS_API_KEY}")
    try:
        async with httpx.AsyncClient(timeout=6.0) as client:
            r = await client.get(url)
        articles = r.json().get("articles", []) if r.status_code == 200 else []
        scores   = [_score_text((a.get("title","")+" "+(a.get("description") or "")).strip())
                    for a in articles if a.get("title") and "[Removed]" not in a.get("title","")]
        val = round(sum(scores)/len(scores), 3) if scores else 0.0
        val = max(-1.0, min(1.0, val))
        _set("market_sent", val)
        return val
    except Exception:
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# CORE ANALYSIS — multi-timeframe hedge-fund model
# Single Twelve Data call (500 daily bars), sliced into 3 timeframes.
# ─────────────────────────────────────────────────────────────────────────────
async def _analyse_async(symbol: str, horizon: str = "short") -> dict:
    cached = _get(f"analysis:{symbol}", 60)
    if cached:
        log.info("[cache] %s", symbol)
        return cached

    _SCORE_CACHE.clear()   # prevent stale cross-request score bleed
    log.info("[twelvedata] %s", symbol)

    # Fire TD history, stock news, and market news in parallel — all I/O overlaps.
    # News is capped at 3 s so the critical path (TD + scoring) is never held hostage.
    _neutral = {"news_sentiment": 0.0, "news_strength": 0, "news_bias": "NEUTRAL"}

    async def _safe_news():
        try:
            return await asyncio.wait_for(_fetch_news_sentiment(symbol), timeout=3.0)
        except Exception:
            return _neutral

    async def _safe_market():
        try:
            return await asyncio.wait_for(_fetch_market_sentiment(), timeout=3.0)
        except Exception:
            return 0.0

    try:
        df, news_data, market_sent = await asyncio.gather(
            _td_fetch_history(symbol, outputsize=500),
            _safe_news(),
            _safe_market(),
        )
    except Exception as _gather_err:
        log.warning("[gather] TD failed: %s — re-raising", _gather_err)
        raise

    if df is None or df.empty:
        raise ValueError(f"DATA_INVALID: Twelve Data returned empty for {symbol}")
    if len(df) < 60:
        raise ValueError(f"DATA_INVALID: only {len(df)} bars for {symbol}, need ≥60")

    # ── v17 Engine: regime → BUY signal → risk ──────────────────────────────
    regime_result = classify_regime(df.tail(252))
    buy_result    = generate_buy_signal(df, regime_result, symbol=symbol, horizon=horizon)

    final_score  = buy_result.score
    verdict      = buy_result.signal
    conf_label   = buy_result.confidence
    regime       = regime_result.state

    # ── Raw references for overlays + display ─────────────────────────────────
    df_long   = df.tail(252)
    close_1y  = df_long["Close"].dropna()
    high_1y   = df_long["High"].dropna()
    low_1y    = df_long["Low"].dropna()
    vol_1y    = df_long["Volume"].dropna()
    price     = float(round(close_1y.iloc[-1], 2))
    atr       = _atr(high_1y, low_1y, close_1y)
    ema50     = float(_ema(close_1y, 50).iloc[-1])
    ema200    = float(_ema(close_1y, min(200, len(close_1y))).iloc[-1])

    # ── News overlay ──────────────────────────────────────────────────────────
    news_sentiment = news_data["news_sentiment"]
    news_strength  = news_data["news_strength"]
    news_bias      = news_data["news_bias"]
    final_score    = round(max(0.0, min(100.0, final_score + news_sentiment * 15)), 1)

    if   final_score > 75: verdict = "STRONG BUY"
    elif final_score > 63: verdict = "BUY"
    elif final_score < 25: verdict = "STRONG SELL"
    elif final_score < 37: verdict = "SELL"
    else:                  verdict = "HOLD"

    conf_num = {"HIGH":75.0,"MEDIUM":50.0,"LOW":25.0}.get(conf_label,25.0)
    bull_v = verdict in ("BUY","STRONG BUY"); bear_v = verdict in ("SELL","STRONG SELL")
    if   news_sentiment > 0.4  and bull_v:  conf_num = min(100.0, conf_num+10)
    elif news_sentiment < -0.4 and bear_v:  conf_num = min(100.0, conf_num+10)
    elif (news_sentiment > 0.3 and bear_v) or (news_sentiment < -0.3 and bull_v):
        conf_num = max(0.0, conf_num-15)
    conf_num = round(conf_num,1)
    if   conf_num > 70: conf_label = "HIGH"
    elif conf_num > 40: conf_label = "MEDIUM"
    else:               conf_label = "LOW"

    # ── Market sentiment overlay ───────────────────────────────────────────────
    if   market_sent > 0.15 and bull_v: final_score = min(100.0, final_score+5)
    elif market_sent<-0.15 and bear_v:  final_score = min(100.0, final_score+5)
    elif market_sent > 0.15 and bear_v: final_score = max(0.0,   final_score-5)
    elif market_sent<-0.15 and bull_v:  final_score = max(0.0,   final_score-5)
    final_score = round(final_score, 1)

    # ── Risk levels — taken directly from signal engine (includes adaptive TP) ─
    # buy_result already computed: SL=1.2×ATR, TP=2.5×ATR (1.8× in compressed vol)
    sl_price = buy_result.stop_loss  if buy_result.signal != "HOLD" else 0.0
    tp_price = buy_result.take_profit if buy_result.signal != "HOLD" else 0.0
    rr       = buy_result.risk_reward if buy_result.signal != "HOLD" else 0.0

    # ── V17 Deployed: confidence-based position sizing ────────────────────────
    # Circuit breakers run in the background; this surfaces sizing to the UI
    # early_warning: TRANSITION state or low-confidence tradeable regime → 0.5× sizing
    _regime_early_warn = (regime_result.state == "TRANSITION" or
                          (regime_result.confidence < 0.70 and regime_result.tradeable))
    _conf_mult   = {"HIGH": 1.5, "MEDIUM": 1.0, "LOW": 0.5}.get(conf_label, 1.0)
    _pos_mult    = round(_conf_mult * (0.5 if _regime_early_warn else 1.0), 3)
    # Effective risk per trade in R units (1R = base account risk %)
    pos_size_r   = _pos_mult

    # ── Gate fields from engine ────────────────────────────────────────────────
    gate_tf_aligned  = regime_result.strength in ("STRONG","MODERATE")
    gate_regime_ok   = regime_result.state not in ("SIDEWAYS","TRANSITION")
    _vol_20avg       = float(vol_1y.tail(20).mean()) if len(vol_1y)>=20 else 0.0
    gate_volume_ok   = float(vol_1y.iloc[-1]) >= _vol_20avg * 0.70 if _vol_20avg > 0 else True
    gate_no_sideways = regime_result.state not in ("SIDEWAYS","TRANSITION")
    gates_passed     = sum([gate_tf_aligned,gate_regime_ok,gate_volume_ok,gate_no_sideways])
    pre_gate_verdict = buy_result.signal

    # ── MACD for tech_bar (computed here so tech_bar can use real values) ──
    _mc,_ms    = _macd(close_1y)
    macd_main  = round(_mc,4); msig_main = round(_ms,4)

    # ── Expert bars for UI progress bars ──────────────────────────────────────
    comps     = buy_result.components
    rsi_main  = comps.get("rsi",50.0)
    macd_bull = comps.get("macd_bull", False)
    tech_bar  = round(max(0,min(100, 50+_score_rsi(rsi_main)*12.5+_score_macd(macd_main,msig_main)*8.0)), 1)
    trend_bar = round(max(0,min(100, 50+_score_ema(price,ema50)*20+_score_trend(ema50,ema200)*15)), 1)
    vol_bar   = round(max(0,min(100, 50+_score_volume(vol_1y)*16.67)), 1)

    # ── Legacy vars: derived from v17 engine outputs (result dict key compat) ──
    chg        = round(float(close_1y.iloc[-1])-float(close_1y.iloc[-2]),2) if len(close_1y)>=2 else 0.0
    pct        = round(chg/float(close_1y.iloc[-2])*100,2) if len(close_1y)>=2 and float(close_1y.iloc[-2])>0 else 0.0
    open_px    = float(round(df_long["Open"].dropna().iloc[-1],2))
    high_px    = float(round(df_long["High"].dropna().iloc[-1],2))
    low_px     = float(round(df_long["Low"].dropna().iloc[-1],2))
    h52        = float(round(high_1y.max(),2))
    l52        = float(round(low_1y.min(),2))
    short_score  = round(buy_result.score,1)
    medium_score = round(buy_result.score,1)
    long_score   = round(buy_result.score,1)
    weighted     = round(buy_result.score,1)
    vola_ratio   = regime_result.atr_ratio
    vola_penalty = round(max(0.0,(vola_ratio-1.0)*20),1) if vola_ratio>1 else 0.0
    conf_score   = conf_num
    _sideways    = regime_result.state == "SIDEWAYS"
    _sw_diag     = {"adx_value":regime_result.adx,"trend_strength":regime_result.strength,
                    "ema_spread":regime_result.ema_spread_pct,
                    "atr_pct":round(atr/price*100,3) if price>0 else 0.0,
                    "cond_adx":regime_result.adx>=25.0,"cond_ema":abs(regime_result.ema_spread_pct)>0.5,
                    "cond_atr":regime_result.atr_ratio<1.4}
    sl = sl_price; tp = tp_price

    result = {
        # identification
        "symbol":       symbol,
        "sector":       _SECTOR.get(symbol,"Other"),
        "sector_warning": _SECTOR_WARN.get(_SECTOR.get(symbol,""),""),
        "price":        price,
        "change":       chg,
        "change_pct":   pct,
        "open":         open_px,
        "high":         high_px,
        "low":          low_px,
        "h52":          h52,
        "l52":          l52,

        # main indicators (1y / long timeframe)
        "rsi":          rsi_main,
        "macd":         macd_main,
        "macd_signal":  msig_main,
        "macd_hist":    round(macd_main - msig_main, 4),
        "ema50":        ema50,
        "ema200":       ema200,
        "atr":          atr,

        # multi-timeframe scores (new in v10)
        "short_score":        short_score,
        "medium_score":       medium_score,
        "long_score":         long_score,
        "weighted_score":     weighted,
        "regime":             regime,
        "volatility_ratio":   vola_ratio,
        "volatility_penalty": vola_penalty,

        # final scoring
        "score":            final_score,
        "verdict":          verdict,
        "confidence":       conf_score,
        "confidence_label": conf_label,
        "confidence_score": conf_score,

        # risk
        "stop_loss":    sl,
        "take_profit":  tp,
        "rr":           rr,

        # news sentiment (new in v12)
        "news_sentiment":  news_sentiment,
        "news_strength":   news_strength,
        "news_bias":       news_bias,
        "market_sentiment": round(market_sent, 3),

        # signal quality gates
        "gate_tf_aligned":    gate_tf_aligned,
        "gate_regime_ok":     gate_regime_ok,
        "gate_volume_ok":     gate_volume_ok,
        "gate_no_sideways":   gate_no_sideways,
        "gates_passed":       gates_passed,
        # ADX diagnostics — matches result["adx"] shape from user spec
        "adx": {
            "value":          _sw_diag["adx_value"],
            "trend_strength": _sw_diag["trend_strength"],
        },
        # Regime diagnostics — matches result["regime"] shape from user spec
        "regime_diag": {
            "sideways_gate":       _sideways,
            "ema_spread":          _sw_diag["ema_spread"],
            "atr_pct":             _sw_diag["atr_pct"],
            "atr_relative_ratio":  regime_result.atr_ratio,
            "cond_adx":            _sw_diag["cond_adx"],
            "cond_ema":            _sw_diag["cond_ema"],
            "cond_atr":            _sw_diag["cond_atr"],
        },
        "pre_gate_verdict":   pre_gate_verdict,
        # v17 engine diagnostics
        "engine_regime":   regime_result.state,
        "regime_confidence": regime_result.confidence,   # 0.0–1.0 — used by oracle ctx
        "pos_size_r":      pos_size_r,       # effective position size multiplier
        "early_warning":   _regime_early_warn,  # True = 0.5x size applied
        "conf_mult":       _conf_mult,         # HIGH=1.5, MEDIUM=1.0, LOW=0.5
        "engine_horizon":  horizon,
        "regime_detail":   regime_to_dict(regime_result),
        "entry_detail":    buy_result.components,

        # legacy keys (frontend renderOracleDecision uses these)
        "action":              verdict,
        "composite_score":     final_score,
        "bullish_probability": round(final_score, 1),
        "bearish_probability": round(100 - final_score, 1),
        "expert_scores": {
            "technical": tech_bar,
            "volume":    vol_bar,
            "trend":     trend_bar,
        },
        "entry":       price,
        "analysis_id": None,
        "ai_status":   "none",
        # ── Real OHLCV for chart rendering (last 252 bars) ────────────────
        "candles": [
            {"t": int(row.name.timestamp()*1000),
             "o": round(float(row.Open),2),  "h": round(float(row.High),2),
             "l": round(float(row.Low),2),   "c": round(float(row.Close),2),
             "v": int(row.Volume)}
            for _, row in df.tail(252).iterrows()
        ],
    }

    log.info(
        "[%s] short=%.1f med=%.1f long=%.1f weighted=%.1f "
        "regime=%s vola_pen=%.2f final=%.1f verdict=%s conf=%.1f(%s)",
        symbol,
        short_score, medium_score, long_score, weighted,
        regime, vola_penalty, final_score, verdict, conf_score, conf_label,
    )

    _set(f"analysis:{symbol}", result)
    return result

# ─────────────────────────────────────────────────────────────────────────────
# ORACLE (unchanged from v9, just adds regime + timeframe data to prompt)
# ─────────────────────────────────────────────────────────────────────────────
def _parse_json(raw: str) -> dict:
    """Robustly extract JSON from Gemini response — strips markdown fences, finds first {...}."""
    import json, re
    # Strip markdown code fences
    text = re.sub(r"```(?:json)?", "", raw, flags=re.IGNORECASE).strip()
    # Find first balanced {...} block
    start = text.find("{")
    if start == -1:
        raise ValueError(f"No JSON object in Gemini response: {text[:120]!r}")
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{": depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i+1])
                except json.JSONDecodeError as e:
                    raise ValueError(f"Malformed JSON from Gemini: {e}")
    raise ValueError(f"Unterminated JSON in Gemini response: {text[:120]!r}")


async def _call_gemini(system, user):
    """Gemini 1.5 Flash — free, no credit card required."""
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not set on server")
    prompt  = f"{system}\n\n{user}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.7, "maxOutputTokens": 500, "topP": 0.9},
    }
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}")
    async with httpx.AsyncClient(timeout=18.0) as client:
        r = await client.post(url, json=payload, headers={"content-type": "application/json"})
    if r.status_code != 200:
        raise ValueError(f"Gemini API {r.status_code}: {r.text[:200]}")
    candidates = r.json().get("candidates", [])
    if not candidates:
        raise ValueError("Gemini returned no candidates")
    parts = candidates[0].get("content", {}).get("parts", [])
    return "".join(p.get("text", "") for p in parts)



async def get_price(symbol: str):
    sym = symbol.strip().upper()
    cached = _get(f"price:{sym}", 30)
    if cached: return cached
    try:
        result_q = await _td_fetch_quote(sym)
    except Exception as e:
        raise HTTPException(500, str(e))
    result = {"symbol": sym, **result_q}
    _set(f"price:{sym}", result)
    return result


@app.get("/price/{symbol}")
async def price(symbol: str):
    sym = symbol.strip().upper()
    if sym not in _TD_SYMS:
        raise HTTPException(400, f"Unknown symbol: {sym}")
    cached = _get(f"price:{sym}", 30)
    if cached:
        return cached
    try:
        df = await _td_fetch_history(sym, outputsize=5)
        if df.empty or len(df) < 2:
            raise ValueError("insufficient data")
        price = float(round(df["Close"].iloc[-1], 2))
        prev  = float(df["Close"].iloc[-2])
        chg   = round(price - prev, 2)
        pct   = round(chg / prev * 100, 2) if prev > 0 else 0.0
        result = {"symbol": sym, "price": price, "change": chg, "change_pct": pct}
        _set(f"price:{sym}", result)
        return result
    except Exception as e:
        raise HTTPException(500, f"Price fetch failed: {e}")


@app.get("/chart/{symbol}")
async def chart_data(symbol: str, bars: int = 252):
    sym = symbol.strip().upper()
    if sym not in _TD_SYMS:
        raise HTTPException(400, f"Unknown symbol: {sym}")
    cache_key = f"chart:{sym}:{bars}"
    cached = _get(cache_key, 300)
    if cached:
        return cached
    try:
        df = await _td_fetch_history(sym, outputsize=max(bars, 252))
        candles = [
            {"t": int(row.name.timestamp()*1000),
             "o": round(float(row.Open),2),  "h": round(float(row.High),2),
             "l": round(float(row.Low),2),   "c": round(float(row.Close),2),
             "v": int(row.Volume)}
            for _, row in df.tail(bars).iterrows()
        ]
        result = {"symbol": sym, "bars": len(candles), "candles": candles}
        _set(cache_key, result)
        return result
    except Exception as e:
        raise HTTPException(500, f"Chart fetch failed: {e}")


@app.get("/health")
async def health():
    return {"status":"ok","version":"v18","stocks":len(_TD_SYMS)}


@app.get("/universe")
async def universe():
    stocks = []
    for sym in sorted(_TD_SYMS):
        stocks.append({"symbol":sym,"sector":_SECTOR.get(sym,"Other"),
                        "company":_COMPANY.get(sym,sym),
                        "sector_warning":_SECTOR_WARN.get(_SECTOR.get(sym,""),"")})
    return {"stocks":stocks,"count":len(stocks)}

@app.post("/analyze")
async def analyze(req: AnalyzeReq):
    sym = req.symbol.strip().upper()
    if not sym:
        raise HTTPException(400, "symbol required")
    try:
        return await _analyse_async(sym)
    except ValueError as e:
        raise HTTPException(422, str(e))
    except Exception as e:
        log.error("/analyze %s: %s", sym, e)
        raise HTTPException(500, f"Analysis failed: {e}")

@app.post("/oracle")
async def oracle(req: OracleReq):
    sym  = req.symbol.strip().upper()
    mode = req.mode
    cached = _get(f"oracle:{sym}:{mode}", 300)
    if cached:
        log.info("[cache] oracle %s", sym)
        return cached

    a  = req.analysis
    es = a.get("expert_scores", {})

    # Include multi-timeframe + regime data in the oracle prompt
    rsi      = a.get("rsi", 50)
    regime   = a.get("regime", "UNKNOWN")
    rsi_lbl  = "oversold" if rsi < 40 else ("overbought" if rsi > 60 else "neutral")
    macd_dir = "bullish" if a.get("macd", 0) > a.get("macd_signal", 0) else "bearish"
    t_sc  = str(int(es.get("technical", 50)))
    v_sc  = str(int(es.get("volume", 50)))
    q_sc  = str(int(es.get("trend", 50)))
    c_sc  = str(round(a.get("composite_score", a.get("score", 50)), 1))

    system = "NSE India analyst. Return ONLY compact JSON, no markdown, no text outside braces."

    if mode == "beginner":
        ctx = (f"Stock {sym} NSE. Price Rs{a.get('price',0)}, RSI {rsi} ({rsi_lbl}), "
               f"MACD {macd_dir}, regime {regime} (conf {a.get('regime_confidence',0):.0%}), "
               f"signal score={a.get('score',50):.0f}/100, conf={a.get('confidence_label','LOW')}, "
               f"sector={a.get('sector','')}, verdict={a.get('verdict','HOLD')}."
               + (f" NOTE: {a['sector_warning']}" if a.get('sector_warning') else ""))
        tmpl = ('{{"trend":"","momentum":"","risk":"",'
                '"simple_explanation":"","key_insight":"","confidence":"High|Med|Low",'
                '"catalysts":[""],"risks":[""],'
                f'"expert_scores":{{"technical":{t_sc},"volume":{v_sc},"quant":{q_sc}}},'
                f'"composite_score":{c_sc}}}')
    else:
        ctx = (f"NSE {sym}: Rs{a.get('price',0)}, RSI {rsi} ({rsi_lbl}), MACD {macd_dir}, "
               f"ATR {a.get('atr',0):.2f}, 52W Rs{a.get('l52',0)}-Rs{a.get('h52',0)}, "
               f"regime {regime} (conf {a.get('regime_confidence',0):.0%}), "
               f"sector {a.get('sector','')}, signal score={a.get('score',50):.0f}/100, "
               f"v17_verdict={a.get('verdict','HOLD')} conf={a.get('confidence_label','')}, "
               f"sl=Rs{a.get('stop_loss',0)}, tp=Rs{a.get('take_profit',0)}."
               + (f" SECTOR ALERT: {a['sector_warning']}" if a.get('sector_warning') else ""))
        tmpl = ('{{"trend":"","valuation":"","momentum":"","risk":"",'
                '"key_insight":"","verdict":"Buy|Hold|Avoid","verdict_reason":"",'
                '"confidence":"High|Med|Low",'
                '"expert_views":{{"technical":"","volume":"","quant":""}},'
                f'"expert_scores":{{"technical":{t_sc},"volume":{v_sc},"quant":{q_sc}}},'
                f'"composite_score":{c_sc},"catalysts":[""],"risks":[""],"target_1y":0}}')
    user = ctx + " Return ONLY this JSON: " + tmpl

    try:
        raw    = await _call_gemini(system, user)
        result = _parse_json(raw)
    except Exception as e:
        log.error("/oracle %s: %s", sym, e)
        raise HTTPException(500, f"Oracle failed: {e}")

    _set(f"oracle:{sym}:{mode}", result)
    return result

@app.get("/indices")
async def get_indices():
    """Batch-fetch live NSE index quotes: Nifty 50, Bank Nifty, IT, Midcap, VIX.
    Cached 60 s — cheap single TD call via batch /quote endpoint."""
    cached = _get("indices", 60)
    if cached:
        return cached

    # Twelve Data batch quote — one call for all symbols
    _INDEX_MAP = [
        ("NIFTY:NSE",          "NIFTY 50"),
        ("BANKNIFTY:NSE",      "NIFTY BANK"),
        ("CNXIT:NSE",          "NIFTY IT"),
        ("NIFTYMIDCAP100:NSE", "NIFTY MIDCAP"),
        ("INDIAVIX:NSE",       "INDIA VIX"),
    ]
    syms_param = ",".join(td for td, _ in _INDEX_MAP)
    url = f"{TD_BASE}/quote?symbol={syms_param}&apikey={TWELVE_DATA_KEY}"

    results = []
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(url)

        if r.status_code == 429:
            raise ValueError("TD rate limit on /indices")
        if r.status_code != 200:
            raise ValueError(f"TD HTTP {r.status_code}")

        data = r.json()
        # TD returns a single object when 1 symbol, list-of-objects when multiple.
        # With batch it returns a dict keyed by symbol string.
        for td_sym, label in _INDEX_MAP:
            q = data.get(td_sym, {})
            if not q or q.get("status") == "error":
                results.append({"label": label, "value": None, "change_pct": None, "error": True})
                continue
            try:
                price = float(q.get("close", 0))
                prev  = float(q.get("previous_close", price) or price)
                chg   = round(price - prev, 2)
                pct   = round((chg / prev) * 100, 2) if prev else 0.0
                # Format value: integers ≥1000 with comma, VIX as decimal
                if price >= 1000:
                    val = f"{price:,.0f}"
                else:
                    val = f"{price:.2f}"
                results.append({
                    "label":      label,
                    "value":      val,
                    "change_pct": pct,
                    "raw":        price,
                })
            except (TypeError, ValueError):
                results.append({"label": label, "value": None, "change_pct": None, "error": True})

    except Exception as e:
        log.warning("[indices] fetch failed: %s", e)
        # Return empty so frontend falls back to dashes — never crash
        return {"indices": [], "error": str(e)}

    out = {"indices": results}
    _set("indices", out)
    return out


@app.get("/news/{symbol}")
async def get_news(symbol: str):
    sym    = symbol.strip().upper()
    cached = _get(f"news:{sym}", 300)
    if cached: return cached
    if not NEWS_API_KEY:
        return {"news": [], "note": "NEWS_API_KEY not configured"}
    url = (f"https://newsapi.org/v2/everything?q={sym}+NSE+stock+India"
           f"&sortBy=publishedAt&pageSize=5&language=en&apiKey={NEWS_API_KEY}")
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.get(url)
        articles = [
            {"title": a.get("title", ""), "source": a.get("source", {}).get("name", ""),
             "url": a.get("url", ""), "time": a.get("publishedAt", "")[:10]}
            for a in resp.json().get("articles", [])[:5]
            if a.get("title") and "[Removed]" not in a.get("title", "")
        ]
    except Exception as e:
        log.error("/news %s: %s", sym, e)
        return {"news": [], "error": str(e)}
    result = {"news": articles}
    _set(f"news:{sym}", result)
    return result

@app.get("/")
async def serve_frontend():
    """Serve the AURUM frontend — same origin as API so no CORS issues."""
    return FileResponse(HERE / "aurum_standalone.html", media_type="text/html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"\n⚡ AURUM v18 → http://localhost:{port}")
    print(f"   Data: Twelve Data live NSE | Engine: v18 regime+signal pipeline")
    print(f"   Oracle:  {'ready' if GEMINI_API_KEY else 'SET GEMINI_API_KEY'}")
    print(f"   News:    {'ready' if NEWS_API_KEY else 'optional'}\n")
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False, log_level="info")


