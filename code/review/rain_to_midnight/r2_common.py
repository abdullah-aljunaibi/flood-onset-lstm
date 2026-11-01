"""Shared setup of the independent reviewer's scripts for experiment E1 (code/15_rain_to_midnight.py).

These scripts are an AI review (see README.md): they re-compute the results of E1 with the reviewer's own code and write nothing into the project.
Every output goes to the folder OUT (environment variable REVIEW_OUT; default: <system temp folder>/rain_to_midnight_review).
Run them from the project root with the project's environment, one thread:   ~/AI_Class/.venv/bin/python code/review/rain_to_midnight/<script>.py
"""
import os, sys, json, tempfile
from pathlib import Path
os.environ.setdefault("FLOOD_THREADS", "1")                    # must be set before flood.py is imported
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True                                 # no __pycache__ in the project
HERE = Path(__file__).resolve().parent                          # <project>/code/review/rain_to_midnight
PROJ = str(HERE.parents[2])                                     # <project>
sys.path.insert(0, PROJ + "/code")                              # flood.py, features.py, camels.py ... as the numbered scripts see them
sys.path.insert(0, str(HERE))                                   # the helpers of this folder
OUT = os.environ.get("REVIEW_OUT") or str(Path(tempfile.gettempdir()) / "rain_to_midnight_review")
os.makedirs(OUT, exist_ok=True)
import numpy as np, pandas as pd, torch
torch.set_num_threads(1)


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
