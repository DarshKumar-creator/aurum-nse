"""evaluation_engine.py — silent run, on-demand query."""
import sys,json,os,numpy as np,pandas as pd
from datetime import datetime,timedelta
import warnings; warnings.filterwarnings("ignore")

RESULTS=os.path.join(os.path.dirname(os.path.abspath(__file__)),"eval_results.json")
TOPICS=["summary","buy","sell","regime","windows","verdict"]

if __name__=="__main__" and len(sys.argv)>1:
    print("Evaluation engine ready")
    sys.exit(0)