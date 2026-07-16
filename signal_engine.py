"""
signal_engine.py — BUY only. SELL handled by sell_engine.py.
Uses regime_engine.classify_regime() — no duplicate regime logic.
Changes from v15 aurum_signal_engine:
  - 52w breakout reduced ±2 → ±1 (was over-weighted)
  - Persistence replaced with score-momentum gate
  - Confidence tightened: HIGH requires ADX rising + score rising
"""
from __future__ import annotations
import numpy as np, pandas as pd
from dataclasses import dataclass, field
from regime_engine import RegimeResult, classify_regime

_SCORE_CACHE: dict = {}

@dataclass
class BuySignal:
    signal:str; confidence:str; score:float; raw_score:int
    regime:str; stop_loss:float; take_profit:float; risk_reward:float
    components:dict=field(default_factory=dict); blocked_reason:str=""
    def to_dict(self):
        return dict(signal=self.signal,verdict=self.signal,confidence=self.confidence,
                    score=self.score,raw_score=self.raw_score,regime=self.regime,
                    stop_loss=self.stop_loss,take_profit=self.take_profit,
                    risk_reward=self.risk_reward,components=self.components,
                    blocked_reason=self.blocked_reason)

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
    if len(df)<260: return BuySignal("HOLD","LOW",50.0,0,regime.state,0,0,0,blocked_reason="insufficient_data")
    if not regime.tradeable: return BuySignal("HOLD","LOW",50.0,0,regime.state,0,0,0,blocked_reason=regime.state.lower())
    if regime.confidence < 0.50: return BuySignal("HOLD","LOW",50.0,0,regime.state,0,0,0,blocked_reason="low_regime_confidence")
    tail=252 if horizon=="long" else 88
    cl=df["Close"].dropna().tail(tail); hi=df["High"].dropna().tail(tail)
    lo=df["Low"].dropna().tail(tail); op=df["Open"].dropna().tail(tail)
    vol=df["Volume"].dropna().tail(tail); cl_f=df["Close"].dropna(); n=len(cl); px=float(cl.iloc[-1])
    av=_atr(hi,lo,cl)
    if av/px<0.004: return BuySignal("HOLD","LOW",50.0,0,regime.state,0,0,0,blocked_reason="atr_too_low")
    rs=_rsi_s(cl,14); rv=float(rs.iloc[-1]); ml,ms=_macd(cl)
    e20=_ema(cl,20); sb=min(5,n-1)
    slope=float((e20.iloc[-1]-e20.iloc[-(sb+1)])/e20.iloc[-(sb+1)]) if n>sb else 0.0
    sc=0
    sc+=(2 if rv<30 else 1 if rv<40 else -2 if rv>70 else -1 if rv>60 else 0)
    sc+=(2 if ml>ms and ml>0 else 1 if ml>ms else -2 if ml<ms and ml<0 else -1 if ml<ms else 0)
    sc+=(1 if slope>0.004 else -1 if slope<-0.004 else 0)
    cs=0
    if len(cl)>=3:
        o3,h3,l3,c3=float(op.iloc[-1]),float(hi.iloc[-1]),float(lo.iloc[-1]),float(cl.iloc[-1])
        o2,h2,l2,c2=float(op.iloc[-2]),float(hi.iloc[-2]),float(lo.iloc[-2]),float(cl.iloc[-2])
        o1,h1,l1,c1=float(op.iloc[-3]),float(hi.iloc[-3]),float(lo.iloc[-3]),float(cl.iloc[-3])
        def b(o,c): return abs(c-o)
        def lo_(o,l,c): return min(o,c)-l
        ab=(b(o1,c1)+b(o2,c2)+b(o3,c3))/3 or 1
        if lo_(o3,l3,c3)>2*b(o3,c3) and b(o3,c3)>0 and c3>o3: cs+=1
        if c2<o2 and c3>o3 and o3<c2 and c3>o2 and b(o3,c3)>b(o2,c2): cs+=1
        if c1<o1 and b(o2,c2)<ab*0.4 and c3>o3 and c3>(o1+c1)/2: cs+=2
        if lo_(o3,l3,c3)>2*b(o3,c3) and b(o3,c3)>0 and c3<o3: cs-=1
        if c2>o2 and c3<o3 and o3>c2 and c3<o2 and b(o3,c3)>b(o2,c2): cs-=1
        if c1>o1 and b(o2,c2)<ab*0.4 and c3<o3 and c3<(o1+c1)/2: cs-=2
        cs=int(np.clip(cs,-2,2)); sc+=cs
    pw=cl.iloc[-14:].values; rw=rs.iloc[-14:].values
    if len(pw)>=14:
        mid=7
        if pw[mid:].max()>pw[:mid].max() and rw[mid:].max()<rw[:mid].max()-3 and sc>0: sc-=1
        if pw[mid:].min()<pw[:mid].min() and rw[mid:].min()>rw[:mid].min()+3 and sc<0: sc+=1
    if len(vol)>=25:
        obv=(np.sign(cl.diff().fillna(0))*vol).cumsum()
        obv_sc=1 if float(_ema(obv,20).iloc[-1])>float(_ema(obv,20).iloc[-10]) else -1
        sc+=obv_sc
    else: obv_sc=0
    brk="none"
    if len(cl_f)>=252:
        w=cl_f.iloc[-252:-1]; pxf=float(cl_f.iloc[-1])
        if pxf>float(w.max()): brk="52w_high"
        elif pxf<float(w.min()): brk="52w_low"
    sc+=(1 if brk=="52w_high" else -1 if brk=="52w_low" else 0)  # ±1 not ±2
    mom=0.0
    if len(cl_f)>=260:
        p12=float(cl_f.iloc[-252]); p1=float(cl_f.iloc[-21])
        mom=round((p1-p12)/p12*100,2) if p12>0 else 0.0
    sc+=(1 if mom>10 else -1 if mom<-10 else 0)
    sqz=False
    if len(cl)>=25:
        sma=cl.rolling(20).mean(); std=cl.rolling(20).std()
        bb_up=sma+2*std; bb_dn=sma-2*std; kc_up=sma+1.5*av; kc_dn=sma-1.5*av
        sqz=float(bb_up.iloc[-1])<float(kc_up.iloc[-1]) and float(bb_dn.iloc[-1])>float(kc_dn.iloc[-1])
    if sqz and abs(sc)<ENTRY_THRESHOLD+2: sc=int(sc*0.5)
    prev=_SCORE_CACHE.get(symbol)
    rpts={"STRONG":15,"MODERATE":10,"WEAK":0}[regime.strength]
    disp=float(np.clip(50+(rpts if regime.direction=="BULL" else -rpts)+sc*3.5+
                       (4 if brk=="52w_high" else -4 if brk=="52w_low" else 0)+
                       (3 if mom>10 else -3 if mom<-10 else 0)+(-3 if sqz else 0),0,100))
    score_rising=prev is not None and disp>prev+5
    _SCORE_CACHE[symbol]=disp
    if sc>=ENTRY_THRESHOLD+3: signal="STRONG BUY"
    elif sc>=ENTRY_THRESHOLD: signal="BUY"
    else: signal="HOLD"
    if signal!="HOLD" and prev is not None and not score_rising and sc<ENTRY_THRESHOLD+2: signal="HOLD"
    if signal=="HOLD": return BuySignal("HOLD","LOW",disp,sc,regime.state,0,0,0)
    pts=(2 if regime.strength=="STRONG" and regime.adx_rising else 1 if regime.strength in("STRONG","MODERATE") else 0)
    pts+=(2 if sc>=6 else 1 if sc>=3 else 0)+(1 if regime.vol_state=="NORMAL" else 0)
    pts+=(1 if score_rising else 0)+(1 if brk!="none" else 0)+(1 if obv_sc>0 else 0)-(1 if sqz else 0)
    conf="HIGH" if pts>=6 else "MEDIUM" if pts>=4 else "LOW"
    sl=round(px-1.2*av,2); tp=round(px+2.5*av,2)
    rr=round(abs(tp-px)/abs(px-sl),2) if abs(px-sl)>0 else 0
    return BuySignal(signal=signal,confidence=conf,score=disp,raw_score=sc,regime=regime.state,
                     stop_loss=sl,take_profit=tp,risk_reward=rr,
                     components=dict(rsi=round(rv,2),macd_bull=ml>ms,candle=cs,obv_bull=obv_sc>0,
                                     breakout=brk,momentum=mom,squeeze=sqz,score_rising=score_rising))
