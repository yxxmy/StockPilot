"""时序页面服务：显式执行与会话防重，不调用Agent或报告库。"""
from hashlib import sha256
import json
from pathlib import Path
import os

from forecasting.data import fetch_snapshot,save_snapshot,load_snapshot
from forecasting.evaluation import CONFIG,evaluate_snapshot,save_experiment

ROOT=Path(__file__).resolve().parents[1]


def data_root():
    return Path(os.environ.get("STOCKPILOT_FORECAST_DIR",str(ROOT/"data/forecasting")))


def snapshots():
    found=[]
    errors=0
    for path in sorted(data_root().glob("*.json")):
        try:
            snap=load_snapshot(path)
            found.append((path,snap))
        except Exception:
            errors+=1
    return found,errors


def acquire(session,symbol,start,end,source,refresh=False):
    key=(symbol,start,end,source)
    cached=session.setdefault("forecast_acquired",{})
    if not refresh and key in cached:
        return cached[key]
    snap=fetch_snapshot(symbol,start,end,source=source)
    save_snapshot(snap,data_root())
    cached[key]=snap
    return snap


def run_once(session,snapshot,force=False):
    key=(snapshot["snapshot_id"],sha256(json.dumps(CONFIG,sort_keys=True).encode()).hexdigest())
    cached=session.setdefault("forecast_completed",{})
    if not force and key in cached and "next_observation" in cached[key]["experiment"]:
        session["forecast_result"]=cached[key]
        return False
    result=evaluate_snapshot(snapshot, include_latest=True)
    destination=save_experiment(result,data_root()/"experiments")
    bundle={"experiment":result,"directory":str(destination)}
    session["forecast_result"]=bundle
    if result["status"]!="failed": cached[key]=bundle
    return True
