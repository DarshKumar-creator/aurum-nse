"""risk_manager.py — R-based sizing"""
from __future__ import annotations
import numpy as np, pandas as pd
from dataclasses import dataclass

@dataclass
class RiskConfig:
    max_risk_pct:float=1.0; high_conf_mult:float=1.5; med_conf_mult:float=1.0
    low_conf_mult:float=0.5; sell_risk_mult:float=0.75