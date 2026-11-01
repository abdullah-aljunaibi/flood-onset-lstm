"""Shared setup of the independent reviewer's scripts for experiment E2 (the archived GEFSv12 reforecast rain as an LSTM input: code/gefs_to_basins.py, code/18_forecast_input.py).

These scripts are an AI review (see README.md): they re-compute the results of E2 with the reviewer's own code and write nothing into the project except this folder and the review record.
Every output goes to the folder OUT (environment variable REVIEW_OUT; default: <system temp folder>/e2_review). Nothing is trained. The raw GRIB days of f2_raw_days.py are decoded in memory and dropped.
Run them from the project root with the project's environment, one thread:   ~/AI_Class/.venv/bin/python code/review/forecast_input/<script>.py      (f2_raw_days.py: ~/AI_Class/.venv-grib/bin/python)
"""
import os, sys, json, tempfile
from pathlib import Path
os.environ.setdefault("FLOOD_THREADS", "1")                    # must be set before flood.py is imported
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True                                 # no __pycache__ in the project
HERE = Path(__file__).resolve().parent                          # <project>/code/review/forecast_input
PROJ = str(HERE.parents[2])                                     # <project>
sys.path.insert(0, PROJ + "/code")                              # flood.py, features.py, camels.py ... as the numbered scripts see them
sys.path.insert(0, str(HERE))                                   # the helpers of this folder
OUT = os.environ.get("REVIEW_OUT") or str(Path(tempfile.gettempdir()) / "e2_review")
os.makedirs(OUT, exist_ok=True)
import numpy as np, pandas as pd, torch
torch.set_num_threads(1)

SEEDS = (0, 1, 2)
KS = (1, 2, 3)
SPLIT_MAIN = {"train": ("1980-10-01", "2000-09-30"), "val": ("2000-10-01", "2005-09-30"), "test": ("2005-10-01", "2014-09-30")}
SPLIT_SAT = {"train": ("2000-10-01", "2008-09-30"), "val": ("2008-10-01", "2010-09-30"), "test": ("2010-10-01", "2014-09-30")}
SPLITS = {"main": SPLIT_MAIN, "sat": SPLIT_SAT}
# the arm of every set: the sets of the satellite-era split end in _s
SETS_E2 = ["wqaF_sub", "wqaF_clim", "wqaP", "wqF_sub", "wqaF_s", "wqa_s", "wqaF_clim_s", "wq_s", "wqaP_s", "wqaF_sub_raw"]
ARM = {**{n: ("sat" if n.endswith("_s") else "main") for n in SETS_E2}, "wq": "main", "wqa": "main", "wqf": "main", "wqaF_sub_rawpred": "main", "wqaF_sub_perfectpred": "main", "wqaP_forecastpred": "main"}
ZONE = {-5: "America/New_York", -6: "America/Chicago"}


def jload(rel):
    """A JSON file of the project, by its path from the project root."""
    with open(f"{PROJ}/{rel}") as f:
        return json.load(f)


def jsave(name, obj):
    """The output of a check: OUT/<name> (never inside the project)."""
    with open(f"{OUT}/{name}", "w") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.item() if isinstance(o, np.generic) else o.tolist() if isinstance(o, np.ndarray) else str(o))


def jread(name):
    """The output of an earlier check: OUT/<name>."""
    with open(f"{OUT}/{name}") as f:
        return json.load(f)
