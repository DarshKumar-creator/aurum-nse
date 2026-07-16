"""
sell_engine.py — Dedicated SELL module. NOT inverted BUY logic.
SELL threshold=3, confidence>=0.40 (conservative — asymmetric by design).
6 independent detectors: RSI failure, MACD rollover, EMA breakdown,
OBV distribution, ATR asymmetry, bearish candle.
DISABLED in production by default (SELL_ENABLED=False in main.py).
Requires real NSE data calibration before enabling.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from dataclasses import dataclass
from regime_engine import RegimeResult

SELL_THRESHOLD=3; STRONG_SELL_THRESHOLD=5; SELL_CONFIDENCE_THRESH=0.40

@dataclass
class SellSignal:
    signal:str; confidence:str; confidence_score:float; regime_context:str
    breakdown_confirmed:bool; score:int
    rsi_failure:bool; macd_rollover:bool; ema_breakdown:bool
    obv_distribution:bool; atr_expansion_down:bool; candle_bearish:bool
    stop_loss:float; take_profit:float; risk_reward:float
    def to_dict(self):
        return dict(signal=self.signal,verdict=self.signal,confidence=self.confidence,
                    confidence_score=self.confidence_score,regime_context=self.regime_context,
                    breakdown_confirmed=self.breakdown_confirmed,score=self.score,
                    components=dict(rsi_failure=self.rsi_failure,macd_rollover=self.macd_rollover,
                                    ema_breakdown=self.ema_breakdown,obv_distribution=self.obv_distribution,
                                    atr_expansion_down=self.atr_expansion_down,candle_bearish=self.candle_bearish),
                    stop_loss=self.stop_loss,take_profit=self.take_profit,risk_reward=self.risk_reward)

def _ema(s,n): return s.ewm(span=n,adjust=False).mean()
def _w(s,n): return s.ewm(alpha=1/n,adjust=False).mean()
def _rsi_s(cl,n=14):
    d=cl.diff(); return (100-100/(1+_w(d.clip(lower=0),n)/_w((-d).clip(lower=0),n).replace(0,np.nan))).fillna(50)
def _atr(h,l,c,n=14):
    pc=c.shift(1); tr=pd.Series(np.maximum(h-l,np.maximum((h-pc).abs(),(l-pc).abs())),index=c.index)
    v=float(_w(tr,n).iloc[-1]); return v if not np.isnan(v) else float(c.iloc[-1])*0.015

def detect_rsi_failure(close):
    if len(close)<25: return False
    rs=_rsi_s(close.tail(25),14).values
    above=[i for i,v in enumerate(rs) if v>50]
    if not above or above[-1]>=len(rs)-1: return False
    returned=any(v<50 for v in rs[above[-1]:])
    return returned and float(rs[-1])<50

def detect_macd_rollover(close):
    if len(close)<30: return False
    ln=_ema(close,12)-_ema(close,26); sig=_ema(ln,9); hist=ln-sig
    hv=hist.values; lv=ln.values; sv=sig.values
    if len(hv)>=4 and all(hv[-4+i]>hv[-4+i+1] for i in range(3)) and float(hv[-1])<0: return True
    if len(lv)>=2 and float(lv[-2])>float(sv[-2]) and float(lv[-1])<float(sv[-1]) and float(lv[-1])<0: return True
    return False

def detect_ema_breakdown(close, confirm_bars=3):
    if len(close)<210: return False
    e50=_ema(close,50); e200=_ema(close,200)
    rc=close.iloc[-confirm_bars:]; re50=e50.iloc[-confirm_bars:]; re200=e200.iloc[-confirm_bars:]
    if not all(float(rc.iloc[i])<float(re50.iloc[i]) for i in range(confirm_bars)): return False
    return float(close.iloc[-1])<float(e200.iloc[-1]) or float(e50.iloc[-1])<float(e200.iloc[-1])

def detect_obv_distribution(close, volume, lookback=20):
    if len(close)<lookback+5 or len(volume)<lookback+5: return False
    cl=close.tail(lookback); vl=volume.tail(lookback)
    obv=(np.sign(cl.diff().fillna(0))*vl).cumsum()
    px_ret=(float(cl.iloc[-1])-float(cl.iloc[0]))/float(cl.iloc[0])
    obv_falling=float(obv.iloc[-1])<float(obv.iloc[0])*0.98
    return obv_falling and px_ret>-0.03

def detect_atr_expansion_down(close, high, low, lookback=10):
    if len(close)<lookback+5: return False
    cl=close.tail(lookback); hi=high.tail(lookback); lo=low.tail(lookback)
    rets=cl.pct_change().dropna(); rng=(hi-lo).tail(len(rets))
    dn=[float(rng.iloc[i]) for i,r in enumerate(rets) if r<0 and i<len(rng)]
    up=[float(rng.iloc[i]) for i,r in enumerate(rets) if r>0 and i<len(rng)]
    return len(dn)>=2 and len(up)>=2 and np.mean(dn)>np.mean(up)*1.15

def confirm_breakdown(close, high, low, volume):
    if len(close)<25: return False
    bars_lower=sum(1 for i in range(1,3) if float(close.iloc[-i])<float(close.iloc[-(i+1)]))
    if bars_lower<1: return False
    vol_avg=float(volume.tail(20).mean()) if len(volume)>=20 else float(volume.mean())
    high_vol=sum(1 for i in range(1,4) if i<=len(volume) and float(volume.iloc[-i])>vol_avg*0.80)
    if high_vol<1: return False
    dr=float(high.iloc[-1])-float(low.iloc[-1])
    if dr>0 and (float(close.iloc[-1])-float(low.iloc[-1]))/dr>0.60: return False
    return True

def _bearish_candle(op,hi,lo,cl):
    if len(cl)<3: return False
    o2,h2,l2,c2=float(op.iloc[-2]),float(hi.iloc[-2]),float(lo.iloc[-2]),float(cl.iloc[-2])
    o3,h3,l3,c3=float(op.iloc[-1]),float(hi.iloc[-1]),float(lo.iloc[-1]),float(cl.iloc[-1])
    def b(o,c): return abs(c-o)
    if c2>o2 and c3<o3 and o3>c2 and c3<o2 and b(o3,c3)>b(o2,c2): return True
    o1,c1=float(op.iloc[-3]),float(cl.iloc[-3])
    ab=(b(o1,c1)+b(o2,c2)+b(o3,c3))/3 or 1
    if c1>o1 and b(o2,c2)<ab*0.4 and c3<o3 and c3<(o1+c1)/2: return True
    return False

def validate_short_regime(regime):
    if regime.state=="TRENDING_BEAR" and regime.sell_eligible: return True,"confirmed_downtrend"
    if regime.state=="TRANSITION" and regime.sell_eligible: return True,"transition_weakening"
    return False,f"blocked_{regime.state.lower()}"

def generate_sell_signal(df, regime):
    HOLD=SellSignal("HOLD","LOW",0.0,regime.state,False,0,False,False,False,False,False,False,0,0,0)
    cl=df["Close"].dropna(); hi=df["High"].dropna(); lo=df["Low"].dropna()
    op=df["Open"].dropna(); vol=df["Volume"].dropna()
    if len(cl)<220: return HOLD
    allowed,_=validate_short_regime(regime)
    if not allowed: return HOLD
    comps=dict(
        rsi_failure=detect_rsi_failure(cl),
        macd_rollover=detect_macd_rollover(cl),
        ema_breakdown=detect_ema_breakdown(cl),
        obv_distribution=detect_obv_distribution(cl,vol),
        atr_expansion_down=detect_atr_expansion_down(cl,hi,lo),
        candle_bearish=_bearish_candle(op,hi,lo,cl))
    score=sum(1 for v in comps.values() if v)
    breakdown=confirm_breakdown(cl,hi,lo,vol)
    if score<SELL_THRESHOLD: return HOLD
    signal="STRONG SELL" if score>=STRONG_SELL_THRESHOLD else "SELL"
    pts=min(1.0,0.15*score/5+0.20*breakdown+
            0.15*(regime.state=="TRENDING_BEAR")+0.08*(regime.state=="TRANSITION")+
            0.10*(not regime.adx_rising)+0.10*comps["ema_breakdown"]+
            0.10*comps["obv_distribution"]+0.08*comps["rsi_failure"]+0.07*comps["macd_rollover"])
    pts=round(pts,3)
    if pts<SELL_CONFIDENCE_THRESH: return HOLD
    conf="HIGH" if pts>=0.75 else "MEDIUM" if pts>=0.50 else "LOW"
    px=float(cl.iloc[-1]); av=_atr(hi.tail(88),lo.tail(88),cl.tail(88))
    sl=round(px+1.2*av,2); tp=round(px-2.5*av,2)
    rr=round(abs(tp-px)/abs(px-sl),2) if abs(px-sl)>0 else 0
    return SellSignal(signal=signal,confidence=conf,confidence_score=pts,regime_context=regime.state,
                      breakdown_confirmed=breakdown,score=score,**comps,stop_loss=sl,take_profit=tp,risk_reward=rr)
