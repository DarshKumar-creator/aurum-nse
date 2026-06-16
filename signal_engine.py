"""signal_engine.py — BUY only."""
from __future__ import annotations
import numpy as np, pandas as pd
from dataclasses import dataclass, field
from regime_engine import RegimeResult

_SCORE_CACHE: dict = {}

@dataclass
class BuySignal:
    signal:str; confidence:str; score:float; raw_score:int
    regime:str; stop_loss:float; take_profit:float; risk_reward:float
    components:dict=field(default_factory=dict); blocked_reason:str=""

def _ema(s,n): return s.ewm(span=n,adjust=False).mean()
def _w(s,n): return s.ewm(alpha=1/n,adjust=False).mean()
def _rsi_s(cl,n=14):
    d=cl.diff(); return (100-100/(1+_w(d.clip(lower=0),n)/_w((-d).clip(lower=0),n).replace(0,np.nan))).fillna(50)
def _macd(cl):
    if len(cl)<26: return 0.0,0.0
    ln=_ema(cl,12)-_ema(cl,26); return float(ln.iloc[-1]),float(_ema(ln,9).iloc[-1])
def _atr(h,l,c,n=14):
    pc=c.shift(1); tr=pd.Series(np.maximum(h-l,np.maximum((h-pc).abs(),(l-pc).abs())),index=c.index)
    v=float(_w(tr,n).iloc[-1]); return v if not np.isnan(v) else float(c.iloc[-1])*0.015

ENTRY_THRESHOLD=3

def generate_buy_signal(df, regime:RegimeResult, symbol:str="", horizon:str="short") -> BuySignal:
    HOLD=BuySignal("HOLD","LOW",50.0,0,regime.state,0,0,0)
    if len(df)<260 or not regime.tradeable:
        return HOLD
    
    tail=252 if horizon=="long" else 88
    cl=df["Close"].dropna().tail(tail); hi=df["High"].dropna().tail(tail)
    lo=df["Low"].dropna().tail(tail); op=df["Open"].dropna().tail(tail)
    vol=df["Volume"].dropna().tail(tail); n=len(cl); px=float(cl.iloc[-1])
    av=_atr(hi,lo,cl)
    
    if av/px<0.004: return HOLD
    
    rs=_rsi_s(cl,14); rv=float(rs.iloc[-1]); ml,ms=_macd(cl)
    e20=_ema(cl,20); sc=0
    sc+=(2 if rv<30 else 1 if rv<40 else -2 if rv>70 else -1 if rv>60 else 0)
    sc+=(2 if ml>ms and ml>0 else 1 if ml>ms else -2 if ml<ms and ml<0 else -1 if ml<ms else 0)
    
    score_rising = True
    if sc>=ENTRY_THRESHOLD: signal="BUY"
    else: signal="HOLD"
    
    if signal=="HOLD": return HOLD
    
    conf="HIGH" if sc>=6 else "MEDIUM" if sc>=4 else "LOW"
    sl=round(px-1.2*av,2); tp=round(px+2.5*av,2)
    rr=round(abs(tp-px)/abs(px-sl),2) if abs(px-sl)>0 else 0
    
    return BuySignal(signal=signal,confidence=conf,score=float(sc*10),raw_score=sc,regime=regime.state,
                     stop_loss=sl,take_profit=tp,risk_reward=rr,
                     components={"rsi":round(rv,2),"macd_bull":ml>ms})