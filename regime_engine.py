"""regime_engine.py — v15"""
from __future__ import annotations
import numpy as np, pandas as pd
from dataclasses import dataclass
from typing import Literal

RegimeState = Literal["TRENDING_BULL","TRENDING_BEAR","SIDEWAYS","TRANSITION"]
ADX_STRONG=25.0; ADX_MODERATE=18.0; ADX_TRANSITION=15.0; EMA_SPREAD_MIN=0.005

@dataclass
class RegimeResult:
    state:str; direction:str; strength:str; vol_state:str
    adx:float; adx_delta:float; adx_rising:bool
    ema50:float; ema200:float; ema_spread_pct:float; ema_spread_delta:float
    atr_ratio:float; bars_stable:int; tradeable:bool; sell_eligible:bool; confidence:float

def _ema(s,n): return s.ewm(span=n,adjust=False).mean()
def _w(s,n): return s.ewm(alpha=1/n,adjust=False).mean()

def _adx_series(h,l,c,n=14):
    if len(c)<n*2+5: return pd.Series(np.zeros(len(c)),index=c.index)
    pc=c.shift(1); tr=pd.Series(np.maximum(h-l,np.maximum((h-pc).abs(),(l-pc).abs())),index=c.index)
    up=h.diff(); dn=-l.diff()
    pdm=np.where((up>dn)&(up>0),up,0.0); mdm=np.where((dn>up)&(dn>0),dn,0.0)
    ts=_w(tr,n)
    pdi=100*_w(pd.Series(pdm,index=c.index),n)/ts.replace(0,np.nan)
    mdi=100*_w(pd.Series(mdm,index=c.index),n)/ts.replace(0,np.nan)
    dx=(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan)*100
    return _w(dx.fillna(0),n)

def classify_regime(df):
    cl=df["Close"].dropna(); hi=df["High"].dropna(); lo=df["Low"].dropna(); n=len(cl)
    if n<60: return RegimeResult("SIDEWAYS","FLAT","WEAK","NORMAL",0,0,False,0,0,0,0,1,False,False,0)
    adx_s=_adx_series(hi,lo,cl); adx_now=float(adx_s.iloc[-1])
    adx_5=float(adx_s.iloc[-6]) if len(adx_s)>=6 else adx_now
    adx_delta=round(adx_now-adx_5,2); adx_rising=adx_delta>0.5
    e50=_ema(cl,50); e200=_ema(cl,min(200,n))
    spread_s=(e50-e200)/e200.replace(0,np.nan)
    spread_now=float(spread_s.iloc[-1]) if not spread_s.empty else 0.0
    e50v=float(e50.iloc[-1]); e200v=float(e200.iloc[-1])
    direction="BULL" if spread_now>EMA_SPREAD_MIN else("BEAR" if spread_now<-EMA_SPREAD_MIN else "FLAT")
    strength=("STRONG" if adx_now>=ADX_STRONG and adx_rising else
              "MODERATE" if adx_now>=ADX_STRONG or (adx_now>=ADX_MODERATE and adx_rising) else "WEAK")
    pc=cl.shift(1); tr=pd.Series(np.maximum(hi-lo,np.maximum((hi-pc).abs(),(lo-pc).abs())),index=cl.index)
    atr_s=_w(tr,14); atr_now=float(atr_s.iloc[-1]); atr_60=float(atr_s.iloc[-60:].mean()) if len(atr_s)>=60 else atr_now
    atr_r=round(atr_now/atr_60,3) if atr_60>0 else 1.0
    vol_state="HIGH" if atr_r>1.40 else("LOW" if atr_r<0.65 else "NORMAL")
    if strength=="WEAK": state,trade,sell="SIDEWAYS",False,False
    elif direction=="BULL" and strength in("STRONG","MODERATE"): state="TRENDING_BULL"; trade=True; sell=False
    elif direction=="BEAR" and strength in("STRONG","MODERATE"): state="TRENDING_BEAR"; trade=False; sell=True
    else: state,trade,sell="SIDEWAYS",False,False
    conf=min(1.0,round(0.30*(adx_now>=ADX_STRONG)+0.15*(adx_now>=ADX_MODERATE)+0.15*adx_rising+
                       0.20*(abs(spread_now)>EMA_SPREAD_MIN*2)+0.20*(state!="SIDEWAYS")+0.15*(vol_state=="NORMAL"),3))
    return RegimeResult(state=state,direction=direction,strength=strength,vol_state=vol_state,
                        adx=round(adx_now,2),adx_delta=adx_delta,adx_rising=adx_rising,
                        ema50=round(e50v,2),ema200=round(e200v,2),ema_spread_pct=round(spread_now*100,3),
                        ema_spread_delta=0,atr_ratio=atr_r,bars_stable=0,
                        tradeable=trade,sell_eligible=sell,confidence=conf)

def regime_to_dict(r):
    return {k:getattr(r,k) for k in r.__dataclass_fields__}