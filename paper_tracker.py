"""
AURUM Paper Trade Tracker
=========================
Logs every signal from /analyze, tracks outcome at configurable exit bars,
computes running expectancy on real NSE data.

Usage:
    python3 paper_tracker.py log   SYMBOL SIGNAL ENTRY SL TP CONF DATE
    python3 paper_tracker.py check                        # resolve open trades
    python3 paper_tracker.py stats                        # show expectancy
    python3 paper_tracker.py show                         # open trades
"""
import sys, json, os
from datetime import datetime, date

DB = os.path.join(os.path.dirname(__file__), "paper_trades.json")

def _load():
    if os.path.exists(DB):
        with open(DB) as f: return json.load(f)
    return {"open": [], "closed": [], "meta": {"created": str(date.today())}}

def _save(db):
    with open(DB, "w") as f: json.dump(db, f, indent=2)

def cmd_log(args):
    if len(args) < 7:
        print("Usage: log SYMBOL SIGNAL ENTRY SL TP CONF [DATE]")
        return
    sym, sig, entry, sl, tp, conf = args[0], args[1], float(args[2]), float(args[3]), float(args[4]), args[5]
    dt = args[6] if len(args) > 6 else str(date.today())
    db = _load()
    trade = {"id": len(db["open"]) + len(db["closed"]) + 1,
             "symbol": sym.upper(), "signal": sig, "entry": entry,
             "sl": sl, "tp": tp, "conf": conf, "date": dt,
             "status": "open", "outcome": None, "pnl_r": None}
    db["open"].append(trade)
    _save(db)
    sl_d = abs(entry - sl)
    rr = round(abs(tp - entry) / sl_d, 2) if sl_d > 0 else 0
    print(f"✅ Logged #{trade['id']}: {sym} {sig} @ ₹{entry} | SL=₹{sl} TP=₹{tp} R:R={rr} conf={conf}")

def cmd_show(args):
    db = _load()
    if not db["open"]:
        print("No open trades.")
        return
    print(f"\n{'#':>3} {'Sym':<10} {'Sig':<10} {'Entry':>8} {'SL':>8} {'TP':>8} {'Conf':<8} {'Date'}")
    print("-" * 70)
    for t in db["open"]:
        sl_d = abs(t["entry"] - t["sl"])
        rr = round(abs(t["tp"] - t["entry"]) / sl_d, 2) if sl_d > 0 else 0
        print(f"{t['id']:>3} {t['symbol']:<10} {t['signal']:<10} "
              f"₹{t['entry']:>7.2f} ₹{t['sl']:>7.2f} ₹{t['tp']:>7.2f} "
              f"{t['conf']:<8} {t['date']}")
    print(f"\n{len(db['open'])} open trade(s)")

def cmd_close(args):
    """close ID OUTCOME EXIT_PRICE  (outcome: tp|sl|time)"""
    if len(args) < 3:
        print("Usage: close ID OUTCOME EXIT_PRICE")
        return
    tid, outcome, exit_px = int(args[0]), args[1].lower(), float(args[2])
    db = _load()
    trade = next((t for t in db["open"] if t["id"] == tid), None)
    if not trade:
        print(f"Trade #{tid} not found in open trades.")
        return
    entry = trade["entry"]; sl = trade["sl"]; tp = trade["tp"]
    sl_d = abs(entry - sl) if abs(entry - sl) > 0 else 1
    if "BUY" in trade["signal"]:
        raw_r = (exit_px - entry) / sl_d
    else:
        raw_r = (entry - exit_px) / sl_d
    pnl_r = round(float(raw_r), 4)
    trade.update({"status": "closed", "outcome": outcome,
                  "exit_price": exit_px, "pnl_r": pnl_r,
                  "close_date": str(date.today())})
    db["open"] = [t for t in db["open"] if t["id"] != tid]
    db["closed"].append(trade)
    _save(db)
    print(f"✅ Closed #{tid} {trade['symbol']}: {outcome.upper()} @ ₹{exit_px} → {pnl_r:+.3f}R")

def cmd_stats(args):
    db = _load()
    closed = db["closed"]
    if not closed:
        print("No closed trades yet."); return
    pnls = [t["pnl_r"] for t in closed if t["pnl_r"] is not None]
    if not pnls:
        print("No P&L data."); return
    n = len(pnls)
    E = round(sum(pnls) / n, 4)
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    wr = round(len(wins) / n * 100, 1)
    aw = round(sum(wins) / len(wins), 3) if wins else 0
    al = round(abs(sum(losses) / len(losses)), 3) if losses else 0
    # by confidence
    by_conf = {}
    for t in closed:
        c = t.get("conf","?"); by_conf.setdefault(c, []).append(t.get("pnl_r",0))
    # running equity
    eq = 0; peak = 0; mdd = 0
    for p in pnls:
        eq += p; peak = max(peak, eq)
        mdd = min(mdd, eq - peak)

    print(f"\n{'─'*50}")
    print(f"  AURUM Paper Trade Stats — {n} closed trades")
    print(f"{'─'*50}")
    print(f"  Expectancy:    {E:>+.4f}R per trade")
    print(f"  Win rate:      {wr}%  ({len(wins)}W / {len(losses)}L)")
    print(f"  Avg win:       +{aw}R")
    print(f"  Avg loss:      -{al}R")
    print(f"  Total P&L:     {sum(pnls):>+.2f}R")
    print(f"  Max drawdown:  {mdd:.2f}R")
    print(f"\n  By confidence:")
    for conf, ps in sorted(by_conf.items()):
        e = round(sum(ps)/len(ps),4)
        print(f"    {conf:<8} n={len(ps):>3}  E={e:>+.4f}R")
    print(f"{'─'*50}")
    if n >= 30:
        # Estimate if edge is real (2-sigma)
        import math
        std = (sum((p-E)**2 for p in pnls)/n)**0.5
        se = std / math.sqrt(n)
        z = E / se if se > 0 else 0
        sig = "✅ statistically significant (>2σ)" if abs(z)>2 else f"⚠️  z={z:.2f} — need more data"
        print(f"\n  Edge significance: {sig}")
    else:
        print(f"\n  ⚠️  Need {30-n} more trades for statistical significance (30-trade minimum)")

COMMANDS = {"log": cmd_log, "show": cmd_show, "close": cmd_close, "stats": cmd_stats}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__); sys.exit(0)
    COMMANDS[sys.argv[1]](sys.argv[2:])
