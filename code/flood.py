"""Shared engine for the flood-prediction project: cached data, the fair split, scoring, and the sequence-model trainer.

Forecast convention (every script uses it): a forecast is issued at the end of day t, knowing the weather - and, for some
models, the river flow - up to and including day t. It predicts the flow on days t+1, t+2 and t+3 (lead 1 to 3).
A sample belongs to the split part that contains its TARGET day. Scalers, flood thresholds and model weights are fitted on
the training years only; the validation years choose settings and the stopping point; the test years are only scored.
"""
import contextlib
import copy
import hashlib
import json
import os
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from camels import attributes, load_basin
from common import AQUA, AXIS, BLUE, DATA, FIG, MUTED, ORANGE, RES, ROOT, YELLOW

PERIOD = ("1980-10-01", "2014-09-30")                       # 34 water years, complete for all 12 basins (01_data_audit.py)
SPLIT = {"train": ("1980-10-01", "2000-09-30"), "val": ("2000-10-01", "2005-09-30"), "test": ("2005-10-01", "2014-09-30")}
LEADS = (1, 2, 3)
FORCING = ["prcp", "srad", "tmax", "tmin", "vp", "dayl"]    # Daymet: rain, sunlight, max/min temperature, humidity, day length
STATIC = ["log_area", "elev_mean", "slope_mean", "p_mean", "aridity", "p_seasonality", "frac_forest", "soil_depth_pelletier",
          "sand_frac", "geol_permeability"]                 # basin properties; none is computed from the flow record
HIGH_Q = 0.95                                               # "high flow" = above the level exceeded on 5% of training days

QUICK = os.environ.get("FLOOD_QUICK", "").lower() in ("1", "true", "yes")                # smoke-test mode: tiny budgets, outputs kept apart from real ones
RES_OUT, FIG_OUT = (RES / "quick", FIG / "quick") if QUICK else (RES, FIG)
PRED, DERIVED = RES_OUT / "preds", DATA / "derived"
for _p in (RES_OUT, FIG_OUT, PRED, DERIVED):
    _p.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(int(os.environ.get("FLOOD_THREADS", 1 if QUICK else 4)))     # smoke tests stay out of the way of the one real job
if QUICK and hasattr(os, "nice"):
    os.nice(10)                                             # ... and yield the CPU to it

# names and colours shared by every figure, so the same thing always looks the same
SHORT = {"02077200": "Hyco Creek, NC", "02108000": "NE Cape Fear, NC", "02128000": "Little River, NC", "02137727": "Catawba, NC",
         "02221525": "Murder Creek, GA", "02296500": "Charlie Creek, FL", "02298608": "Myakka, FL", "02349900": "Turkey Creek, GA",
         "02472000": "Leaf River, MS", "03439000": "French Broad, NC", "03463300": "South Toe, NC", "03473000": "SF Holston, VA"}
LAND_COL = {"Mountains (above 600 m)": BLUE, "Northern lowlands and Piedmont": AQUA, "Southern lowlands (south of 32.5N)": ORANGE}
LAND_SHORT = {"Mountains (above 600 m)": "Mountains", "Northern lowlands and Piedmont": "Northern lowlands", "Southern lowlands (south of 32.5N)": "Southern lowlands"}
MODEL_COL = {"persistence": MUTED, "climatology": AXIS, "linreg": YELLOW, "rf": AQUA, "lstm": BLUE, "transformer": ORANGE}
INPUT_LABEL = {"w": "past weather", "wq": "past weather + river gauge", "wf": "past weather + perfect 3-day forecast",
               "wqf": "past weather + river gauge + perfect 3-day forecast"}


# ---- data ------------------------------------------------------------------------------------------------------------
def basins(ids=None):
    """The 12 study basins (chosen by 01_data_audit.py) with the CAMELS attribute tables joined; ids=[...] gives audit candidates instead (short is empty for those not among the 12)."""
    if ids is None:
        sel = pd.DataFrame(json.loads((RES / "data_audit.json").read_text())["selected"]).set_index("gauge_id")
    else:
        aud = pd.read_csv(RES / "basin_audit.csv", dtype={"gauge_id": str}).set_index("gauge_id")
        assert len(set(ids)) == len(ids) and set(ids) <= set(aud.index), "ids must be distinct basins of results/basin_audit.csv"
        sel = aud.loc[list(ids), ["landscape"]]
    A = attributes().loc[sel.index].copy()
    A["landscape"], A["log_area"], A["short"] = sel.landscape, np.log10(A.area_gages2), pd.Series(SHORT)
    assert ids is None or A[STATIC].notna().all().all(), "every basin needs a value for each of the STATIC properties"
    return A


def study_ids():
    """The 12 study basins, in the order of results/data_audit.json (the order of load_data() and basins())."""
    return list(basins().index)


def candidate_ids():
    """The 63 south-eastern candidate basins of results/basin_audit.csv with a complete record, in the file's order: no flow day and no forcing value missing over PERIOD (the rule of 01_data_audit.py)."""
    aud = pd.read_csv(RES / "basin_audit.csv", dtype={"gauge_id": str}).set_index("gauge_id")
    ok = (aud.missing_share == 0) & (aud.forcing_missing == 0)
    assert (ok == aud.complete).all(), "results/basin_audit.csv: its complete column does not follow the rule of 01_data_audit.py"
    return list(aud.index[ok])


def load_data(source="daymet", ids=None):
    """{gauge_id: daily table over PERIOD with the FORCING columns and q_mm (flow, mm/day)}; parsed once, then cached.

    No ids: the 12 study basins, cached in camels12_<source>.pkl. ids=[...] (audit candidates, for example candidate_ids()): those basins in that order, cached apart in
    camels<n>_<source>_<digest of the sorted ids>.pkl so that the cache of the 12 is never touched; each needs flow and forcing on every day of PERIOD, with no gap."""
    f = DERIVED / (f"camels12_{source}.pkl" if ids is None else f"camels{len(ids)}_{source}_{hashlib.sha256(','.join(sorted(ids)).encode()).hexdigest()[:10]}.pkl")
    if f.exists():
        D = pd.read_pickle(f)
        return D if ids is None else {g: D[g] for g in ids}
    D = {g: load_basin(g, source).loc[PERIOD[0]:PERIOD[1], FORCING + ["q_mm"]].astype("float64") for g in (basins().index if ids is None else ids)}
    assert not any(d.isna().any().any() for d in D.values()), "the selected basins must have complete records"
    assert ids is None or (len(set(ids)) == len(ids) and all(d.index.equals(pd.date_range(*PERIOD)) for d in D.values())), "every basin needs a row for every day of PERIOD"
    tmp = f.with_suffix(f".{os.getpid()}.tmp")
    pd.to_pickle(D, tmp)
    tmp.replace(f)
    return D


def part_mask(idx, lead, part, split=SPLIT):
    """True on the issue days whose target day (issue day + lead) lies in the given split part."""
    t = idx + pd.Timedelta(days=lead)
    return np.asarray((t >= pd.Timestamp(split[part][0])) & (t <= pd.Timestamp(split[part][1])))


def frame_from_issue(d, preds):
    """Turn forecasts stored by ISSUE day into the standard prediction frame, indexed by TARGET day: obs, l1, l2, l3.

    preds: {lead: array aligned with d.index}, where element t is the forecast for day t+lead made on day t."""
    p = pd.DataFrame({"obs": d.q_mm})
    for k in LEADS:
        p[f"l{k}"] = pd.Series(np.asarray(preds[k], float), index=d.index).shift(k)
    return p


def thresholds(D, split=SPLIT):
    """High-flow level per basin: the 95th percentile of daily flow in the training years only."""
    return {g: float(d.q_mm.loc[split["train"][0]:split["train"][1]].quantile(HIGH_Q)) for g, d in D.items()}


# ---- scoring ---------------------------------------------------------------------------------------------------------
def nse(obs, sim):
    """Nash-Sutcliffe efficiency: 1 = perfect, 0 = no better than always predicting the mean of the scored days."""
    obs, sim = np.asarray(obs, float), np.asarray(sim, float)
    return float(1 - np.sum((sim - obs) ** 2) / np.sum((obs - obs.mean()) ** 2))


def kge(obs, sim):
    """Kling-Gupta efficiency (2009): combines correlation, variability ratio and bias ratio; 1 = perfect."""
    obs, sim = np.asarray(obs, float), np.asarray(sim, float)
    if np.ptp(sim) == 0:                                    # a constant forecast has no correlation
        return float("nan")
    r, a, b = np.corrcoef(obs, sim)[0, 1], sim.std() / obs.std(), sim.mean() / obs.mean()
    return float(1 - np.sqrt((r - 1) ** 2 + (a - 1) ** 2 + (b - 1) ** 2))


def contingency(obs_hi, sim_hi):
    """Warning table from two boolean arrays (was the day high / was a warning given), with the usual rates.

    pod = share of high-flow days that were warned; far = share of warnings that were false; csi = hits / (hits + misses +
    false alarms); bias = warnings given / high-flow days observed."""
    o, s = np.asarray(obs_hi, bool), np.asarray(sim_hi, bool)
    h, m, f, c = int((o & s).sum()), int((o & ~s).sum()), int((~o & s).sum()), int((~o & ~s).sum())
    return rates(dict(hits=h, misses=m, false_alarms=f, correct_neg=c))


def rates(t):
    """Add pod, far, csi and bias to a table of counts (NaN when the denominator is zero)."""
    h, m, f = t["hits"], t["misses"], t["false_alarms"]
    div = lambda a, b: a / b if b else float("nan")
    return dict(t, pod=div(h, h + m), far=div(f, h + f), csi=div(h, h + m + f), bias=div(h + f, h + m))


def flood_scores(p, thr, lead, part="test", trigger=1.0, split=SPLIT):
    """High-flow warning skill for one basin and lead, on the target days of one split part.

    p is a prediction frame (target day; obs, l1..l3). A day is high when obs > thr; a warning is given when the forecast
    exceeds trigger * thr. "onset" keeps only the forecasts issued while the river was still at or below thr (lead days
    before the target), i.e. the warnings that matter most: with trigger >= 1 persistence can never score a hit there."""
    was_low = (p.obs.shift(lead) <= thr).loc[split[part][0]:split[part][1]]
    q = p.loc[split[part][0]:split[part][1]]
    ok = (q.obs.notna() & q[f"l{lead}"].notna()).values
    o, s, was_low = (q.obs > thr).values[ok], (q[f"l{lead}"] > trigger * thr).values[ok], was_low.values[ok]
    return dict(all=contingency(o, s), onset=contingency(o[was_low], s[was_low]))


def evaluate(P, thr, part="test", trigger=1.0, split=SPLIT):
    """Score one prediction set {gauge_id: frame} on a split part: per basin and lead, plus a summary over basins.

    The summary pools the warning tables over basins (sums of hits, misses, false alarms) and gives the median, mean and
    worst NSE. "coverage" is the share of the part's calendar days that have a forecast; compare models only when it is 1.
    pbias is 100 * (forecast total - observed total) / observed total, so positive means the forecast is too high."""
    a, b = split[part]
    n_days = len(pd.date_range(a, b))
    out = {"part": part, "trigger": trigger, "basins": {}, "summary": {}}
    for g, p in P.items():
        out["basins"][g] = {}
        for k in LEADS:
            q = p.loc[a:b]
            ok = (q.obs.notna() & q[f"l{k}"].notna()).values
            o, s = q.obs.values[ok], q[f"l{k}"].values[ok]
            out["basins"][g][f"l{k}"] = dict(n=int(ok.sum()), coverage=float(ok.sum() / n_days), nse=nse(o, s), kge=kge(o, s),
                                             rmse=float(np.sqrt(np.mean((s - o) ** 2))), pbias=float(100 * (s.sum() - o.sum()) / o.sum()),
                                             **flood_scores(p, thr[g], k, part, trigger, split))
    for k in LEADS:
        rows = [out["basins"][g][f"l{k}"] for g in P]
        v = np.array([r["nse"] for r in rows])
        pool = lambda key: rates({c: int(sum(r[key][c] for r in rows)) for c in ("hits", "misses", "false_alarms", "correct_neg")})
        out["summary"][f"l{k}"] = dict(nse_median=float(np.median(v)), nse_mean=float(v.mean()), nse_min=float(v.min()), nse_max=float(v.max()),
                                       kge_median=float(np.nanmedian([r["kge"] for r in rows])), coverage_min=float(min(r["coverage"] for r in rows)),
                                       all=pool("all"), onset=pool("onset"))
    return out


# ---- prediction store ------------------------------------------------------------------------------------------------
def save_preds(name, P, meta=None):
    """Write one prediction set {gauge_id: frame by target day with obs, l1, l2, l3} to results/preds/<name>.npz."""
    g = sorted(P)
    assert all(P[b].index.equals(P[g[0]].index) for b in g), "all basins of a prediction set must share one date index"
    np.savez_compressed(PRED / f"{name}.npz", basins=np.array(g), dates=P[g[0]].index.values.astype("datetime64[D]"),
                        obs=np.stack([P[b].obs.values for b in g]), pred=np.stack([P[b][[f"l{k}" for k in LEADS]].values for b in g]),
                        meta=json.dumps(meta or {}))


def _pred_file(name):
    f = PRED / f"{name}.npz"
    return f if f.exists() or not QUICK else RES / "preds" / f"{name}.npz"      # a smoke test may read another script's real output


def have_preds(name):
    return _pred_file(name).exists()


def load_preds(name):
    """Read a prediction set written by save_preds."""
    z = np.load(_pred_file(name), allow_pickle=False)
    idx = pd.DatetimeIndex(z["dates"])
    return {str(b): pd.DataFrame({"obs": z["obs"][i], **{f"l{k}": z["pred"][i, :, j] for j, k in enumerate(LEADS)}}, index=idx)
            for i, b in enumerate(z["basins"])}


def preds_meta(name):
    return json.loads(str(np.load(_pred_file(name), allow_pickle=False)["meta"]))


def ensemble(Ps):
    """Average several prediction sets day by day (the same model trained with different random seeds)."""
    return {g: (sum(P[g] for P in Ps) / len(Ps)).assign(obs=Ps[0][g].obs) for g in Ps[0]}       # obs is kept exact, not averaged


# ---- sequence model --------------------------------------------------------------------------------------------------
@dataclass
class Cfg:
    dyn: tuple = tuple(FORCING)   # daily input columns, read up to and including the issue day
    use_flow: bool = False        # also read the observed flow up to the issue day (needs a river gauge)
    future: bool = False          # also read the dyn columns of the three target days (a perfect weather forecast)
    static: tuple = ()            # basin properties repeated on every day (for one model over many basins)
    hidden: int = 64
    dropout: float = 0.2
    lr: float = 2e-3
    batch: int = 32               # training sequences per update
    warmup: int = 365             # days the model reads before its first scored forecast
    block: int = 365              # scored days per sequence, so a forecast has seen between warmup and warmup+block days
    updates: int = 2000           # upper limit on gradient updates
    eval_every: int = 100         # validation check every this many updates
    patience: int = 6             # stop after this many checks without a better validation score
    clip: float = 1.0
    seed: int = 0


class LSTMNet(nn.Module):
    """One LSTM layer read day by day; at every day a linear head gives the flow of the next three days."""

    def __init__(self, n_in, cfg):
        super().__init__()
        self.lstm, self.drop, self.head = nn.LSTM(n_in, cfg.hidden, batch_first=True), nn.Dropout(cfg.dropout), nn.Linear(cfg.hidden, len(LEADS))
        with torch.no_grad():
            self.lstm.bias_hh_l0[cfg.hidden:2 * cfg.hidden].fill_(3.0)          # open the forget gate: start by remembering

    def forward(self, x):                                                        # [sequence, day, feature] -> [sequence, day, lead]
        return self.head(self.drop(self.lstm(x)[0]))


def _design(D, ids, cfg, sc, static_df):
    """X [basin, day, feature] scaled inputs known on each issue day; Y [basin, day, lead] scaled flow on day + lead."""
    X, Y = [], []
    for g in ids:
        d = D[g]
        base = (d[list(cfg.dyn)].values - sc["dyn_mu"]) / sc["dyn_sd"]
        parts = [base]
        if cfg.use_flow:
            parts.append(((np.log1p(d.q_mm.values) - sc["lq_mu"]) / sc["lq_sd"])[:, None])
        if cfg.future:
            parts += [np.vstack([base[k:], np.zeros((k, base.shape[1]))]) for k in LEADS]       # weather of day t+k (0 past the record end)
        if cfg.static:
            s = (static_df.loc[g, list(cfg.static)].values.astype(float) - sc["st_mu"]) / sc["st_sd"]
            parts.append(np.repeat(s[None], len(d), 0))
        X.append(np.hstack(parts))
        q = d.q_mm.values
        Y.append(np.stack([np.concatenate([q[k:], np.full(k, np.nan)]) for k in LEADS], 1))
    return torch.tensor(np.stack(X), dtype=torch.float32), torch.tensor((np.stack(Y) - sc["q_mu"]) / sc["q_sd"], dtype=torch.float32)


def _first_valid(X):
    """First day from which every input of every basin is present (inputs such as satellite series may start late)."""
    ok = torch.isfinite(X).all(dim=2).all(dim=0).numpy()
    lo = int(np.argmax(ok))
    assert ok[lo:].all(), "inputs have gaps after their first valid day"
    return lo


def _roll(net, X, warmup, block, lo):
    """Forecasts for every issue day from lo+warmup on, made in blocks of `block` days after `warmup` days of reading."""
    out = torch.full((X.shape[0], X.shape[1], len(LEADS)), float("nan"))
    was = net.training
    net.eval()
    with torch.no_grad():
        for c in range(lo + warmup, X.shape[1], block):
            out[:, c:c + block] = net(X[:, c - warmup:c + block])[:, warmup:]
    net.train(was)
    return out


def _to_target_day(out, sc):
    """out[basin, issue day, lead] (scaled) -> array[basin, target day, lead] in mm/day, never below zero."""
    o = np.clip(out.numpy().astype("float64") * sc["q_sd"] + sc["q_mu"], 0, None)
    p = np.full_like(o, np.nan)
    for j, k in enumerate(LEADS):
        p[:, k:, j] = o[:, :-k, j]
    return p


def _assert_causal(net, n_in, T):
    """A forecast issued on day t may read inputs up to day t only: changing later days must not change it."""
    g = torch.Generator().manual_seed(0)
    x = torch.randn(2, T, n_in, generator=g)
    x2 = x.clone()
    x2[:, T // 2:] += torch.randn(2, T - T // 2, n_in, generator=g)
    net.eval()
    with torch.no_grad():
        same = torch.allclose(net(x)[:, :T // 2], net(x2)[:, :T // 2], atol=1e-5)
    net.train()
    assert same, "the network is not causal: its output on day t changed when only later days changed (add a causal mask)"


@dataclass
class Run:
    """A trained model with everything needed to reuse it: settings, scalers, training curve and cost.

    seconds is the wall-clock time of the training loop INCLUDING the validation checks (each one rolls the whole record);
    cpu_seconds is CPU time summed over the threads. Both depend on what else the laptop is doing (smoke tests do not wait
    for the lock): compare them only between runs timed back to back on an idle machine. history[i]["load"] is the
    1-minute load average at that check (about the thread count when the job runs alone)."""
    cfg: Cfg
    sc: dict
    net: nn.Module
    ids: list
    history: list
    seconds: float
    cpu_seconds: float
    updates_run: int
    best_update: int
    best_val_nse: float
    n_params: int
    split: dict = None

    def predict(self, D, ids=None, static_df=None, warmup=None, block=None):
        """{gauge_id: frame by target day with obs, l1, l2, l3} over the whole record; works for basins not trained on."""
        ids = list(ids or self.ids)
        X, _ = _design(D, ids, self.cfg, self.sc, static_df)
        lo = _first_valid(X)
        p = _to_target_day(_roll(self.net, X, self.cfg.warmup if warmup is None else warmup, block or self.cfg.block, lo), self.sc)
        return {g: pd.DataFrame({"obs": D[g].q_mm.values, **{f"l{k}": p[i, :, j] for j, k in enumerate(LEADS)}}, index=D[g].index) for i, g in enumerate(ids)}

    def predict_context(self, D, ids=None, static_df=None, context=365, part="test", split=None, chunk=256):
        """Forecasts for one split part when the net may read only the last `context` days before each issue day."""
        ids, split = list(ids or self.ids), split or self.split or SPLIT
        X, _ = _design(D, ids, self.cfg, self.sc, static_df)
        idx = D[ids[0]].index
        need = np.flatnonzero(np.any([part_mask(idx, k, part, split) for k in LEADS], axis=0))
        need = need[need >= _first_valid(X) + context - 1]
        out = torch.full((len(ids), len(idx), len(LEADS)), float("nan"))
        was = self.net.training
        self.net.eval()
        with torch.no_grad():
            for i in range(len(ids)):
                win = X[i].unfold(0, context, 1).permute(0, 2, 1)                 # window j covers days j .. j+context-1
                for c in range(0, len(need), chunk):
                    t = torch.as_tensor(need[c:c + chunk])
                    out[i, t] = self.net(win[t - context + 1])[:, -1]
        self.net.train(was)
        p = _to_target_day(out, self.sc)
        return {g: pd.DataFrame({"obs": D[g].q_mm.values, **{f"l{k}": p[i, :, j] for j, k in enumerate(LEADS)}}, index=idx) for i, g in enumerate(ids)}

    def summary(self):
        return dict(seconds=round(self.seconds, 1), cpu_seconds=round(self.cpu_seconds, 1), updates_run=self.updates_run, best_update=self.best_update,
                    best_val_nse=round(self.best_val_nse, 4), n_params=self.n_params)


def fit(cfg, D, ids, split=SPLIT, static_df=None, make_net=None, log=None):
    """Train one sequence model on the basins `ids` and return a Run.

    Training reads random stretches of warmup+block days that end inside the training years and scores the forecasts made
    on the last `block` days whose target day is a training day. The loss is the squared error divided by (the basin's
    flow spread + 0.1 x the spread of all training basins) squared - close to dividing by each basin's flow variance, so wet
    and dry basins count nearly alike. Every eval_every updates the model is scored on the validation years
    of the same basins (mean NSE over basins and leads); the best state is kept, and training stops early when the score has
    not improved for `patience` checks. make_net(n_in, cfg) may supply another architecture with the same in/out shapes."""
    ids = list(ids)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    idx = D[ids[0]].index
    tr = np.asarray((idx >= pd.Timestamp(split["train"][0])) & (idx <= pd.Timestamp(split["train"][1])))
    dyn = np.concatenate([D[g].loc[tr, list(cfg.dyn)].values for g in ids])
    q_tr = np.concatenate([D[g].q_mm.values[tr] for g in ids])
    sc = dict(dyn_mu=np.nanmean(dyn, 0), dyn_sd=np.nanstd(dyn, 0) + 1e-9, q_mu=float(q_tr.mean()), q_sd=float(q_tr.std()),
              lq_mu=float(np.log1p(q_tr).mean()), lq_sd=float(np.log1p(q_tr).std()))
    if cfg.static:
        assert len(ids) > 1, "basin properties need more than one training basin (their spread would be zero)"
        S = static_df.loc[ids, list(cfg.static)].values.astype(float)
        sc["st_mu"], sc["st_sd"] = S.mean(0), S.std(0) + 1e-9
    X, Y = _design(D, ids, cfg, sc, static_df)
    lo, hi, W, T = _first_valid(X), int(np.flatnonzero(tr)[-1]) + 1, cfg.warmup, cfg.warmup + cfg.block
    s_lo = max(lo, int(np.flatnonzero(tr)[0]) - W - len(LEADS))     # earliest start: its first scored day still has a training target
    assert hi - s_lo >= T, "training years are shorter than one training sequence"
    M = torch.tensor(np.stack([np.concatenate([tr[k:], np.zeros(k, bool)]) for k in LEADS], 1))       # target day is a training day
    Y0 = torch.nan_to_num(Y)
    w = torch.tensor([1 / (D[g].q_mm.values[tr].std() / sc["q_sd"] + 0.1) ** 2 for g in ids], dtype=torch.float32)
    val = [part_mask(idx, 0, "val", split)] * len(LEADS)                                              # validation target days
    q = np.stack([D[g].q_mm.values for g in ids])

    net = (make_net or LSTMNet)(X.shape[2], cfg)
    _assert_causal(net, X.shape[2], T)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    best, best_state, best_u, bad, hist, run_loss = -np.inf, None, 0, 0, [], []
    t0, c0, u = time.perf_counter(), time.process_time(), 0
    ar = torch.arange(T)
    for u in range(1, cfg.updates + 1):
        b = torch.tensor(rng.integers(0, len(ids), cfg.batch))
        pos = torch.tensor(rng.integers(s_lo, hi - T + 1, cfg.batch))[:, None] + ar                  # [sequence, day]
        out = net(X[b[:, None], pos])[:, W:]
        m = M[pos[:, W:]]
        loss = (((out - Y0[b[:, None], pos[:, W:]]) ** 2) * w[b][:, None, None])[m].mean()
        opt.zero_grad()
        loss.backward()
        if cfg.clip:                                            # 0 or None switches clipping off (clipping to 0 would freeze the weights)
            nn.utils.clip_grad_norm_(net.parameters(), cfg.clip)
        opt.step()
        run_loss.append(float(loss))
        if u % cfg.eval_every == 0 or u == cfg.updates:
            p = _to_target_day(_roll(net, X, W, cfg.block, lo), sc)
            if not np.isfinite(p[:, val[0]]).all():
                assert best_state is not None, "no forecast for some validation days: warmup reaches before the first input day, or the net returned NaN"
                break                                           # the net diverged after an earlier check: keep the best state found so far
            v = [[nse(q[i][val[j]], p[i, val[j], j]) for j in range(len(LEADS))] for i in range(len(ids))]
            score = float(np.mean(v))
            hist.append(dict(update=u, train_loss=float(np.mean(run_loss)), val_nse=score, val_nse_by_lead=[float(x) for x in np.mean(v, 0)],
                             seconds=round(time.perf_counter() - t0, 1), load=(round(os.getloadavg()[0], 1) if hasattr(os, "getloadavg") else float("nan"))))     # no load average on Windows (2026-10-28)
            run_loss = []
            if log:
                log(f"  update {u:5d}  train loss {hist[-1]['train_loss']:.4f}  val NSE {score:.4f}  ({hist[-1]['seconds']:.0f} s)")
            if score > best:
                best, best_state, best_u, bad = score, copy.deepcopy(net.state_dict()), u, 0
            else:
                bad += 1
                if bad >= cfg.patience:
                    break
    assert best_state is not None, "no validation check gave a finite score (updates=0, or a basin's validation flow is constant)"
    net.load_state_dict(best_state)
    net.eval()                                              # a trained model is handed back with dropout off
    return Run(cfg, sc, net, ids, hist, time.perf_counter() - t0, time.process_time() - c0, u, best_u, best,
               sum(p.numel() for p in net.parameters()), split)


# ---- one heavy job at a time -----------------------------------------------------------------------------------------
_HELD = False                                               # this process already holds the lock


@contextlib.contextmanager
def cpu_lock(name):
    """Wait until no other training script holds the CPU (this laptop has two cores), then hold it until the block ends.

    Smoke tests (FLOOD_QUICK=1) do not queue: they are small and run on one thread."""
    try:
        import fcntl
    except ImportError:                                     # Windows: no lock, just run
        fcntl = None
    global _HELD
    if QUICK or fcntl is None or _HELD:                     # already ours: a nested use must not wait for itself
        yield
        return
    f = open(ROOT / ".cpu_lock", "a+")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        f.seek(0)
        print(f"[{name}] waiting for the CPU, in use by: {f.read().strip() or 'another job'}", flush=True)
        fcntl.flock(f, fcntl.LOCK_EX)
    f.seek(0); f.truncate(); f.write(f"{name} (pid {os.getpid()}, since {time.strftime('%H:%M:%S')})"); f.flush()
    _HELD = True
    try:
        yield
    finally:
        _HELD = False
        f.seek(0); f.truncate(); f.flush()
        fcntl.flock(f, fcntl.LOCK_UN)
        f.close()
