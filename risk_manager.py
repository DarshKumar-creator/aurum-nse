"""risk_manager.py — R-based sizing, expectancy, tail risk."""
from __future__ import annotations
import numpy as np, pandas as pd
from dataclasses import dataclass
from regime_engine import RegimeResult

@dataclass
class RiskConfig:
    max_risk_pct:float=1.0; high_conf_mult:float=1.5; med_conf_mult:float=1.0
    low_conf_mult:float=0.5; sell_risk_mult:float=0.75

@dataclass
class TradeRisk:
    stop_loss:float; take_profit:float; risk_reward:float
    position_size_r:float; tp_mult:float; sl_mult:float
    regime_adjusted:bool; atr:float; atr_pct:float
    def to_dict(self): return {k:getattr(self,k) for k in self.__dataclass_fields__}

_PARAMS={"BUY":{"sl":1.2,"tp":2.5},"STRONG BUY":{"sl":1.2,"tp":2.5},
         "SELL":{"sl":1.2,"tp":2.5},"STRONG SELL":{"sl":1.2,"tp":2.5}}

def _atr(h,l,c,n=14):
    pc=c.shift(1); tr=pd.Series(np.maximum(h-l,np.maximum((h-pc).abs(),(l-pc).abs())),index=c.index)
    v=float(tr.ewm(alpha=1/n,adjust=False).mean().iloc[-1])
    return v if not np.isnan(v) else float(c.iloc[-1])*0.015

def compute_risk(df, signal, confidence, regime:RegimeResult, config=None):
    if config is None: config=RiskConfig()
    cl=df["Close"].dropna(); hi=df["High"].dropna().tail(88); lo=df["Low"].dropna().tail(88)
    px=float(cl.iloc[-1]); av=_atr(hi,lo,cl.tail(88)); atr_pct=av/px if px>0 else 0.015
    p=_PARAMS.get(signal,{"sl":1.0,"tp":1.2}); sl_m=p["sl"]; tp_m=p["tp"]
    if "BUY" in signal: sl=round(px-sl_m*av,2); tp=round(px+tp_m*av,2)
    elif "SELL" in signal: sl=round(px+sl_m*av,2); tp=round(px-tp_m*av,2)
    else: return TradeRisk(0,0,0,0,1.2,1.0,False,av,atr_pct)
    rr=round(abs(tp-px)/abs(px-sl),2) if abs(px-sl)>0 else 0
    cm={"HIGH":config.high_conf_mult,"MEDIUM":config.med_conf_mult,"LOW":config.low_conf_mult}.get(confidence,1.0)
    ra=False
    if regime.state=="TRANSITION": cm*=0.5; ra=True
    elif regime.confidence<0.4: cm*=0.75; ra=True
    if "SELL" in signal: cm*=config.sell_risk_mult
    return TradeRisk(stop_loss=sl,take_profit=tp,risk_reward=rr,position_size_r=round(cm,3),
                     tp_mult=tp_m,sl_mult=sl_m,regime_adjusted=ra,atr=round(av,3),atr_pct=atr_pct)

def compute_expectancy(pnl_r, label=""):
    if not pnl_r: return {"n":0,"expectancy":0,"win_rate":0,"label":label}
    arr=np.array(pnl_r); w=arr[arr>0]; l=arr[arr<0]
    wr=float((arr>0).mean()*100); aw=float(w.mean()) if len(w) else 0; al=float(abs(l.mean())) if len(l) else 0
    exp=round((wr/100)*aw-((100-wr)/100)*al,4)
    eq=np.cumsum(arr); pk=np.maximum.accumulate(eq); mdd=float((eq-pk).min())
    streak=cs=0
    for p in arr: cs=(cs+1 if p<0 else 0); streak=max(streak,cs)
    return dict(label=label,n=len(arr),win_rate=round(wr,1),avg_win_r=round(aw,3),avg_loss_r=round(al,3),
                expectancy=exp,annual_r_50=round(exp*50,2),max_drawdown_r=round(mdd,3),
                max_loss_streak=streak,tail_p5=round(float(np.percentile(arr,5)) if len(arr)>=20 else arr.min(),3),
                skewness=round(float(pd.Series(arr).skew()),3))

def simulate_r_pnl(df_full, idx, signal, risk:TradeRisk, eval_bars=20, slip=0.0005, delay=1):
    ei=idx+delay
    if ei>=len(df_full)-1: return 0.0
    ep=float(df_full["Open"].iloc[ei])*(1+slip if "BUY" in signal else 1-slip)
    hit="open"
    for j in range(ei,min(ei+eval_bars,len(df_full)-1)):
        h=float(df_full["High"].iloc[j]); l=float(df_full["Low"].iloc[j])
        if "BUY" in signal:
            if l<=risk.stop_loss: hit="sl"; break
            if h>=risk.take_profit: hit="tp"; break
        else:
            if h>=risk.stop_loss: hit="sl"; break
            if l<=risk.take_profit: hit="tp"; break
    sl_dist=abs(ep-risk.stop_loss) if abs(ep-risk.stop_loss)>risk.atr*0.1 else risk.atr*1.2
    tp_dist=abs(risk.take_profit-ep)
    r_tp=round(tp_dist/sl_dist,4) if sl_dist>0 else risk.tp_mult
    if hit=="tp": return round(r_tp*risk.position_size_r,4)
    if hit=="sl": return -round(1.0*risk.position_size_r,4)
    xp=float(df_full["Close"].iloc[min(ei+eval_bars-1,len(df_full)-1)])
    xp*=(1-slip if "BUY" in signal else 1+slip)
    pnl=(xp-ep)/sl_dist if "BUY" in signal else (ep-xp)/sl_dist
    return round(float(np.clip(pnl,-1.0,r_tp))*risk.position_size_r,4)
