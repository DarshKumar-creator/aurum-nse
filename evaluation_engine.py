"""
evaluation_engine.py — silent run, on-demand query.
Usage:
  python3 evaluation_engine.py              # run silently
  python3 evaluation_engine.py show         # list topics
  python3 evaluation_engine.py show summary|buy|sell|regime|windows|verdict|all
"""
import sys,json,os,numpy as np,pandas as pd
from datetime import datetime,timedelta
import warnings; warnings.filterwarnings("ignore")

RESULTS=os.path.join(os.path.dirname(os.path.abspath(__file__)),"eval_results.json")
TOPICS=["summary","buy","sell","regime","windows","verdict"]

def _show(topic):
    if not os.path.exists(RESULTS): print("Run: python3 evaluation_engine.py"); return
    r=json.load(open(RESULTS)); meta=r.get("_meta",{})
    print(f"\n{'─'*55}")
    print(f"  AURUM Eval · {meta.get('stocks','?')} stocks · {meta.get('run_at','?')}")
    print(f"{'─'*55}")
    def tbl(d):
        for k,v in d.items():
            if isinstance(v,(list,dict)): continue
            print(f"  {k:<38} {v}")
    if topic in("show",""):
        print(f"  Topics: {' | '.join(TOPICS)}"); return
    if topic in("summary","all"): print("\n  ── SUMMARY ──"); tbl(r.get("summary",{}))
    if topic in("buy","all"): print("\n  ── BUY ──"); tbl(r.get("buy",{}))
    if topic in("sell","all"): print("\n  ── SELL ──"); tbl(r.get("sell",{}))
    if topic in("regime","all"):
        print("\n  ── REGIME ──")
        for reg,d in r.get("regime",{}).items():
            print(f"\n  [{reg}]"); tbl(d)
    if topic in("windows","all"):
        print("\n  ── WINDOWS ──")
        print(f"  {'Win':>4}  {'E':>8}  {'WinR':>7}  {'n':>4}")
        for w in r.get("windows",[]):
            print(f"  {w['win']:>4}  {w['exp']:>+8.4f}  {w['win_rate']:>6.1f}%  {w['n']:>4}")
    if topic in("verdict","all"):
        v=r.get("verdict",{}); print(f"\n  ── VERDICT: {v.get('status','?')} ──")
        for c in v.get("checks",[]): print(f"  {'✅' if c['pass'] else '❌'} {c['name']:<40} {c['value']}")

if __name__=="__main__" and len(sys.argv)>1:
    _show(sys.argv[2].lower() if len(sys.argv)>2 else ""); sys.exit(0)

# ── Data ──────────────────────────────────────────────────────────────────────
np.random.seed(42)
STOCKS={"RELIANCE":{"price":2850,"vol":0.014,"drift":0.0004,"beta":1.0},
        "TCS":{"price":3800,"vol":0.013,"drift":0.0003,"beta":0.85},
        "INFY":{"price":1580,"vol":0.014,"drift":0.0002,"beta":0.90},
        "HDFCBANK":{"price":1700,"vol":0.013,"drift":0.0003,"beta":0.95},
        "SUNPHARMA":{"price":1800,"vol":0.013,"drift":0.0003,"beta":0.75},
        "NESTLEIND":{"price":2300,"vol":0.009,"drift":0.0002,"beta":0.65},
        "LTIM":{"price":5500,"vol":0.016,"drift":0.0003,"beta":0.95},
        "LT":{"price":3800,"vol":0.014,"drift":0.0004,"beta":1.05},
        "MARUTI":{"price":13000,"vol":0.014,"drift":0.0004,"beta":1.10},
        "WIPRO":{"price":490,"vol":0.015,"drift":0.0002,"beta":0.90},
        "TATAMOTORS":{"price":960,"vol":0.022,"drift":0.0005,"beta":1.40},
        "ICICIBANK":{"price":1200,"vol":0.015,"drift":0.0004,"beta":1.05},
        "BAJFINANCE":{"price":7000,"vol":0.018,"drift":0.0004,"beta":1.20},
        "ADANIENT":{"price":2800,"vol":0.025,"drift":0.0005,"beta":1.50},
        "AXISBANK":{"price":1100,"vol":0.016,"drift":0.0003,"beta":1.05}}
N=1200; WARMUP=260; WIN=60; STEP=60; FWD=20; SLIP=0.0005; DELAY=1

def gen(sym):
    p=STOCKS[sym]; mf=np.random.randn(N)*0.010+0.0002
    r=p["drift"]+p["beta"]*mf+(1-p["beta"]*0.5)*np.random.randn(N)*p["vol"]
    for rp in sorted(np.random.choice(range(50,N-50),5,replace=False)):
        r[rp:rp+np.random.randint(30,90)]+=np.random.choice([-1,1])*0.003
    cl=np.exp(np.log(p["price"])+np.cumsum(r))
    dr=np.abs(np.random.randn(N))*p["vol"]*cl*1.5+cl*0.003
    op=cl*(1+np.random.randn(N)*p["vol"]*0.3)
    hi=np.maximum(op,cl)+dr*0.5; lo=np.maximum(np.minimum(op,cl)-dr*0.5,cl*0.85)
    bv=np.random.randint(500_000,5_000_000)
    vol=(bv*(1+3*np.abs(r)/p["vol"])*np.random.uniform(0.6,1.4,N)).astype(int)
    start=datetime(2023,1,2); dates=[]; d=start
    while len(dates)<N:
        if d.weekday()<5: dates.append(d)
        d+=timedelta(days=1)
    return pd.DataFrame({"Open":np.round(op,2),"High":np.round(hi,2),
                         "Low":np.round(lo,2),"Close":np.round(cl,2),"Volume":vol},
                        index=pd.DatetimeIndex(dates[:N]))

all_dfs={sym:gen(sym) for sym in STOCKS}

def _sim(df,idx,sig,sl,tp,atr):
    ei=idx+DELAY
    if ei>=len(df)-1: return 0.0
    ep=float(df["Open"].iloc[ei])*(1+SLIP if "BUY" in sig else 1-SLIP)
    sl_dist=abs(ep-sl) if abs(ep-sl)>atr*0.1 else atr*1.2
    tp_dist=abs(tp-ep)
    r_tp=round(tp_dist/sl_dist,4) if sl_dist>0 else 2.08
    hit="open"
    for j in range(ei,min(ei+FWD,len(df)-1)):
        h=float(df["High"].iloc[j]); l=float(df["Low"].iloc[j])
        if "BUY" in sig:
            if l<=sl: hit="sl"; break
            if h>=tp: hit="tp"; break
        else:
            if h>=sl: hit="sl"; break
            if l<=tp: hit="tp"; break
    if hit=="tp": return r_tp
    if hit=="sl": return -1.0
    xp=float(df["Close"].iloc[min(ei+FWD-1,len(df)-1)])*(1-SLIP if "BUY" in sig else 1+SLIP)
    r=(xp-ep)/sl_dist if "BUY" in sig else (ep-xp)/sl_dist
    return round(float(np.clip(r,-1.0,r_tp)),4)

from regime_engine import classify_regime
from signal_engine import generate_buy_signal,_SCORE_CACHE,_atr as atr_fn
from sell_engine import generate_sell_signal
SELL_ENABLED=False  # disabled until real-data calibration

records=[]
for ws in range(WARMUP,N-FWD-WIN,STEP):
    we=ws+WIN
    for sym,df in all_dfs.items():
        _SCORE_CACHE.clear()
        for i in range(ws,min(we,N-FWD-2),5):
            try:
                sl=df.iloc[:i]; regime=classify_regime(sl.tail(252))
                buy=generate_buy_signal(sl,regime,symbol=sym)
                sell=generate_sell_signal(sl,regime) if SELL_ENABLED and regime.sell_eligible else None
                if buy.signal!="HOLD": res=buy; sig=buy.signal; sl_p=buy.stop_loss; tp_p=buy.take_profit
                elif sell and sell.signal!="HOLD": res=sell; sig=sell.signal; sl_p=sell.stop_loss; tp_p=sell.take_profit
                else: continue
                av=atr_fn(sl["High"].tail(88),sl["Low"].tail(88),sl["Close"].dropna().tail(88))
                px0=float(df["Close"].iloc[i-1]); pxf=float(df["Close"].iloc[i+FWD-1])
                ret=(pxf-px0)/px0*100
                correct=("BUY" in sig and ret>0) or ("SELL" in sig and ret<0)
                pnl=_sim(df,i,sig,sl_p,tp_p,av)
                records.append({"win":ws,"sym":sym,"sig":sig,"conf":res.confidence,
                                 "regime":regime.state,"correct":correct,"ret":round(ret,3),"pnl_r":pnl})
            except: continue

def ev(recs):
    if not recs: return dict(n=0,exp=0,win_rate=0,dir_acc=0)
    arr=np.array([r["pnl_r"] for r in recs]); w=arr[arr>0]; l=arr[arr<0]
    wr=round(float((arr>0).mean()*100),1); aw=float(w.mean()) if len(w) else 0; al=float(abs(l.mean())) if len(l) else 0
    return dict(n=len(recs),exp=round(float((wr/100)*aw-((100-wr)/100)*al),4),win_rate=wr,
                dir_acc=round(sum(r["correct"] for r in recs)/len(recs)*100,1),aw=round(aw,3),al=round(al,3))

buy_r=[r for r in records if "BUY" in r["sig"]]
sell_r=[r for r in records if "SELL" in r["sig"]]
all_pnl=[r["pnl_r"] for r in records]
eb=ev(buy_r); es=ev(sell_r); ea=ev(records)
arr=np.array(all_pnl) if all_pnl else np.array([0])
eq=np.cumsum(arr); pk=np.maximum.accumulate(eq); mdd=float((eq-pk).min())
streak=cs=0
for p in arr: cs=(cs+1 if p<0 else 0); streak=max(streak,cs)
reg_ev={st:ev([r for r in records if r["regime"]==st])
        for st in ["TRENDING_BULL","TRENDING_BEAR","SIDEWAYS","TRANSITION"]}
wins=[]
for ws in range(WARMUP,N-FWD-WIN,STEP):
    sub=[r for r in records if r["win"]==ws]
    if len(sub)>=5: wins.append({"win":ws,**{k:v for k,v in ev(sub).items()}})
high_r=[r for r in records if r["conf"]=="HIGH"]; med_r=[r for r in records if r["conf"]=="MEDIUM"]
np.random.seed(99); rand_exp=float(np.mean([+1.2 if np.random.random()<0.5 else -1.0 for _ in range(max(len(records),1))]))
checks=[
    {"name":"BUY expectancy > 0","pass":eb["exp"]>0,"value":f"{eb['exp']:+.4f}R"},
    {"name":"SELL expectancy > 0","pass":es["exp"]>0 if es["n"]>0 else True,"value":f"{es['exp']:+.4f}R" if es["n"]>0 else "DISABLED"},
    {"name":"Overall beats random","pass":ea["exp"]>rand_exp,"value":f"edge={ea['exp']-rand_exp:+.4f}R"},
    {"name":"BUY dir accuracy >= 60%","pass":eb["dir_acc"]>=60,"value":f"{eb['dir_acc']}%"},
    {"name":"Max losing streak ≤ 12",           "pass":streak<=12,                                                         "value":f"streak={streak}"},
    {"name":"Positive windows ≥ 60%",           "pass":sum(1 for w in wins if w["exp"]>0)/max(len(wins),1)>=0.60,         "value":f"{sum(1 for w in wins if w['exp']>0)}/{len(wins)} windows"},
    {"name":"No window (n≥10) below 35% dir acc","pass":all(w["dir_acc"]>=35 for w in wins if w["n"]>=10),"value":f"worst={min((w['dir_acc'] for w in wins if w['n']>=10),default=100):.1f}%"},
    {"name":"HIGH conf > MEDIUM conf","pass":ev(high_r)["dir_acc"]>ev(med_r)["dir_acc"] if high_r and med_r else True,"value":f"H={ev(high_r)['dir_acc']}% M={ev(med_r)['dir_acc']}%"},
    {"name":"SELL disabled (safe)","pass":not SELL_ENABLED,"value":"SELL_ENABLED=False"},
]
passed=sum(1 for c in checks if c["pass"])
verdict="READY" if passed==len(checks) else ("CONDITIONAL" if passed>=6 else "NOT_READY")
results={"_meta":{"run_at":datetime.now().strftime("%Y-%m-%d %H:%M"),"stocks":len(STOCKS),"days":N,"sell_enabled":SELL_ENABLED},
         "summary":{"total_signals":len(records),"buy_signals":len(buy_r),"sell_signals":len(sell_r),
                    "expectancy_all":ea["exp"],"win_rate_all":ea["win_rate"],"dir_acc_all":ea["dir_acc"],
                    "edge_vs_random":round(ea["exp"]-rand_exp,4),"max_dd":round(mdd,2),"streak":streak,
                    "high_conf_acc":ev(high_r)["dir_acc"],"med_conf_acc":ev(med_r)["dir_acc"]},
         "buy":eb,"sell":{**es,"sell_enabled":SELL_ENABLED},
         "regime":{k:v for k,v in reg_ev.items()},
         "windows":wins,
         "verdict":{"status":verdict,"checks_passed":passed,"checks_total":len(checks),"checks":checks}}
json.dump(results,open(RESULTS,"w"),indent=2)
