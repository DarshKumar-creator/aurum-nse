"""sell_engine.py — Dedicated SELL module. DISABLED by default."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass
class SellSignal:
    signal:str; confidence:str

def generate_sell_signal(df, regime):
    return SellSignal("HOLD", "LOW")