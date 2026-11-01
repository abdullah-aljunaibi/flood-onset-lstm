"""Shared setup of the final critic's check scripts (the independent review of the documents of 8 October 2026, results/review_documents_2026-10-08.json).

These scripts are an AI review (see README.md). They recompute numbers that the documents quote, with the critic's own code, and write nothing into the project: every output goes to the folder OUT
(environment variable REVIEW_OUT; default <system temp folder>/documents_review). Only d9_assemble_review.py --write-into-project writes the review record into results/.
Run them from the project root with the project's environment, one thread:   ~/AI_Class/.venv/bin/python code/review/documents_2026-10-08/<script>.py
"""
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("FLOOD_THREADS", "1")                    # before flood.py is imported (forward passes only)
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True                                 # no __pycache__ in the project
HERE = Path(__file__).resolve().parent                          # <project>/code/review/documents_2026-10-08
PROJ = HERE.parents[2]                                          # <project>
RES, DATA = PROJ / "results", PROJ / "data"
sys.path.insert(0, str(PROJ / "code"))                          # flood.py, camels.py ... as the numbered scripts see them
OUT = Path(os.environ.get("REVIEW_OUT") or Path(tempfile.gettempdir()) / "documents_review")
OUT.mkdir(parents=True, exist_ok=True)

import numpy as np
import pandas as pd
import torch

torch.set_num_threads(1)


def jload(name):
    """A results file of the project by its name, e.g. jload('uncertainty') reads results/uncertainty.json."""
    with open(RES / f"{name}.json") as f:
        return json.load(f)


def jsave(name, obj):
    """The output of a check: OUT/<name> (never inside the project)."""
    with open(OUT / name, "w") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.item() if isinstance(o, np.generic) else o.tolist() if isinstance(o, np.ndarray) else str(o))


def jread(name):
    """The output of an earlier check: OUT/<name>."""
    with open(OUT / name) as f:
        return json.load(f)


# ---- the 12 basins and their clocks, from the critic's own reading of the audit file --------------------------------------------------------------
def study_basins():
    """The 12 selected basins, in the order of results/data_audit.json: {gauge_id: dict(name, lat, lon)}; read from the audit file and results/basin_audit.csv, nothing else."""
    sel = [r["gauge_id"] for r in jload("data_audit")["selected"]]
    aud = pd.read_csv(RES / "basin_audit.csv", dtype={"gauge_id": str}).set_index("gauge_id")
    return {g: dict(name=aud.loc[g, "gauge_name"], lat=float(aud.loc[g, "gauge_lat"]), lon=float(aud.loc[g, "gauge_lon"])) for g in sel}


def time_zone(info):
    """The wall clock of a gauge: America/Chicago for the gauge in Mississippi (the state is the last two letters of the USGS name), America/New_York for the other eleven (NC, VA, GA, FL)."""
    state = info["name"].strip().split()[-1].strip(".,")
    assert state in ("NC", "VA", "GA", "FL", "MS", "AL"), info["name"]
    return "America/Chicago" if state in ("MS", "AL") else "America/New_York"


def utc_offset_standard(zone):
    """Hours from UTC of the standard time of the zone (the clock of the rain-clock windows): -5 Eastern, -6 Central."""
    return {"America/New_York": -5, "America/Chicago": -6}[zone]


def high_flow_levels(D):
    """The level of each basin: the 95th percentile of the daily flow of the training years 1980-10-01 .. 2000-09-30 (the rule of the report, section 1), computed here from the flow table D[g].q_mm."""
    return {g: float(d.q_mm.loc["1980-10-01":"2000-09-30"].quantile(0.95)) for g, d in D.items()}


def test_onsets(D, thr):
    """The onsets of the test years at lead 1: DataFrame(gauge_id, target, issue) for every day d of 2005-10-01 .. 2014-09-30 with flow above the level while the day before it was at or below it."""
    rows = []
    for g, d in D.items():
        q = d.q_mm
        on = (q > thr[g]) & (q.shift(1) <= thr[g]) & (q.index >= pd.Timestamp("2005-10-01")) & (q.index <= pd.Timestamp("2014-09-30"))
        rows += [(g, t, t - pd.Timedelta(days=1)) for t in q.index[on.values]]
    return pd.DataFrame(rows, columns=["gauge_id", "target", "issue"]).sort_values(["gauge_id", "target"]).reset_index(drop=True)
