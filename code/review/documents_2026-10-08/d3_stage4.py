"""d3: the real-time check of E1b (Stage IV rain in place of AORC on the issue day of every test-year onset), recomputed with the critic's own code.

What the builder's script (code/check_stage4_realtime.py -> results/stage4_realtime.json) states: with Stage IV rain on the issue day of each of the 611 onsets the three saved lstm_wqa networks warn of 294 onsets (against 313 with
AORC and 114 with the rain table alone), 275 with the last hour of the issue day taken out, 255 with the last two hours out, 319 with Stage IV times k = 1.0634, 296 / 274 with AORC itself minus its last hour / two hours,
289 with Stage IV on the day before as well; on the issue days of all 12 basins (the warnings given when the river was low) 291 hits and 97 false warnings with Stage IV against 313 and 132 with AORC; and the same for a second
set of three networks (lstm_wqa_rep: AORC in place of the Daymet rain), three controls on the rain of the issue day, the fit of the forecasts on the onset days and the agreement of the two rain products.

What this script does, from the raw material and its own code (not the builder's script, not its functions): it reads the hourly AORC basin means (data/aorc/aorc_<gauge>.npz) and sums them by the local wall-clock day (summer time included) to
its own 'rain to midnight' table; derives the 611 onsets from the flow table and the 95th percentile of the training years; loads the saved networks (results/models/lstm_wqa_s0..2.pt and lstm_wqa_rep_s0..2.pt) into its own
torch.nn.LSTM + Linear, builds its own input matrix from the scalers stored in the .pt files and rolls the networks over the record the way the engine defines a forecast (365 days of reading, blocks of 365 days); PROVES that this
reproduces the stored single-seed forecasts (results/preds/lstm_wqa_s<k>.npz, lstm_wqa_rep_s<k>.npz) and the 313 / 315 onsets; reads the Stage IV hours from data/aorc/stage4_issue_days.csv (hour-ending UTC labels), assigns each hour to the
local day in which it starts, replaces the value of the issue day, forecasts again, and counts. The controls are redrawn with the critic's own random numbers (so they are compared as ranges, not as equal numbers). The only engine
functions used are the loaders F.load_data() (the daily CAMELS tables: rain, sun, temperatures, humidity, day length, flow) and F.basins() (the ten static properties); F.thresholds only to assert that my levels equal the engine's.
Everything is forward passes with one thread; nothing is trained and nothing is written into the project.
Output: OUT/d3_stage4.json
"""
import time

from dcommon import *
import flood as F

t0 = time.time()
say = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)
D = F.load_data()
ids = list(D)
B = F.basins()
info = study_basins()
assert ids == list(info), "the engine and the audit file disagree on the order of the basins"
ZONE = {g: time_zone(info[g]) for g in ids}
DAYS = pd.date_range("1980-10-01", "2014-09-30")
THR = high_flow_levels(D)
assert all(abs(THR[g] - F.thresholds(D)[g]) < 1e-12 for g in ids), "my high-flow levels differ from the engine's"
ON = test_onsets(D, THR)
assert len(ON) == 611, len(ON)
ev = pd.read_csv(RES / "flood_events.csv", dtype={"gauge_id": str})
ev = ev[~ev.started_before_test.astype(bool)]
assert set(zip(ev.gauge_id, pd.to_datetime(ev.onset_date))) == set(zip(ON.gauge_id, ON.target)), "my 611 onsets are not the onsets of results/flood_events.csv"
say("onsets:", len(ON), "issue days:", ON.issue.nunique())


# ---- AORC: hourly basin means -> daily 'rain to midnight' by the local wall-clock day -----------------------------------------------------------------------
def local_day_of_hours(stamps, zone):
    """The local calendar day of each hour, for hour-ending UTC stamps: the hour (T-1h, T] belongs to the day in which it STARTS (so an hour that ends at 24:00 belongs to the day that ends then)."""
    start = pd.DatetimeIndex(stamps - pd.Timedelta(hours=1)).tz_localize("UTC").tz_convert(zone)
    return start.tz_localize(None).normalize()


AO_H, AO_DAILY = {}, {}
for g in ids:
    z = np.load(DATA / "aorc" / f"aorc_{g}.npz", allow_pickle=False)
    st = pd.DatetimeIndex(pd.to_datetime(z["stamp_utc_s"], unit="s"))
    AO_H[g] = pd.Series(z["rain_mm"].astype("float64"), index=st)
    day = pd.Series(AO_H[g].values, index=local_day_of_hours(st, ZONE[g])).groupby(level=0).sum()
    AO_DAILY[g] = day.reindex(DAYS)
    assert AO_DAILY[g].notna().all()
say("AORC daily tables built (local wall clock, summer time included)")

# ---- Stage IV hours -----------------------------------------------------------------------------------------------------------------------------------------
S4 = pd.read_csv(DATA / "aorc" / "stage4_issue_days.csv", index_col=0, parse_dates=True)
S4.index = pd.DatetimeIndex(S4.index)
LOCAL = {z: local_day_of_hours(S4.index, z) for z in set(ZONE.values())}
GROUPS = {z: {d: v.values for d, v in pd.Series(np.arange(len(S4))).groupby(LOCAL[z])} for z in LOCAL}      # local day -> positions of its hours in the Stage IV file


def day_hours(g, day):
    """(labels, AORC hours, Stage IV hours with the missing ones replaced by the AORC hour, number replaced) of the local day `day` of basin g; None if an hour of the day is not in the Stage IV file."""
    pos = GROUPS[ZONE[g]].get(day)
    if pos is None:
        return None
    lab = S4.index[pos]
    n_expected = int((pd.Timestamp(day + pd.Timedelta(days=1)).tz_localize(ZONE[g]).tz_convert("UTC") - pd.Timestamp(day).tz_localize(ZONE[g]).tz_convert("UTC")) / pd.Timedelta(hours=1))
    if len(lab) != n_expected:
        return None
    a = AO_H[g].reindex(lab).values
    s = S4.loc[lab, g].values.astype(float)
    miss = np.isnan(s)
    return lab, a, np.where(miss, a, s), int(miss.sum())


REC = {}
for g, t in zip(ON.gauge_id, ON.issue):
    r = day_hours(g, t)
    assert r is not None, f"Stage IV hours of {g} {t.date()} are not all in the file"
    assert abs(r[1].sum() - AO_DAILY[g].loc[t]) < 1e-9
    REC[(g, t)] = r
n_hours = sum(len(r[0]) for r in REC.values())
n_fallback = sum(r[3] for r in REC.values())
say(f"Stage IV hours of the 611 issue-days: {n_hours}, replaced by the AORC hour: {n_fallback}")
K = sum(r[1].sum() for r in REC.values()) / sum(r[2].sum() for r in REC.values())




# ---- the saved networks, my own forward pass ----------------------------------------------------------------------------------------------------------------
class Net:
    """LSTM(n_in -> 64) and a linear head of 3 numbers, loaded from a saved file (results/models/lstm_<name>_s<seed>.pt); dropout is off (evaluation)."""

    def __init__(self, name, seed):
        z = torch.load(RES / "models" / f"lstm_{name}_s{seed}.pt", map_location="cpu", weights_only=True)
        self.cfg, self.sc = z["cfg"], z["sc"]
        assert list(z["ids"]) == ids and "aorc_to_midnight" in self.cfg["dyn"] and self.cfg["use_flow"] and not self.cfg["future"] and len(self.cfg["static"]) == 10
        n_in = len(self.cfg["dyn"]) + 1 + len(self.cfg["static"])
        self.lstm, self.head = torch.nn.LSTM(n_in, self.cfg["hidden"], batch_first=True), torch.nn.Linear(self.cfg["hidden"], 3)
        sd = z["state_dict"]
        self.lstm.load_state_dict({k[len("lstm."):]: v for k, v in sd.items() if k.startswith("lstm.")})
        self.head.load_state_dict({k[len("head."):]: v for k, v in sd.items() if k.startswith("head.")})
        self.lstm.eval()
        self.head.eval()

    def inputs(self, rain):
        """X [basin, day, n_in] float32: the daily columns of the network (in its own order; the column named aorc_to_midnight is the rain table given) scaled with the stored mean and spread, the log flow, the ten properties (scaled with the stored mean and spread)."""
        sc, cfg = self.sc, self.cfg
        dyn_mu, dyn_sd = np.asarray(sc["dyn_mu"], float), np.asarray(sc["dyn_sd"], float)
        rows = []
        for g in ids:
            d = D[g]
            cols = np.column_stack([rain[g].values if c == "aorc_to_midnight" else d[c].values for c in cfg["dyn"]])
            parts = [(cols - dyn_mu) / dyn_sd, ((np.log1p(d.q_mm.values) - sc["lq_mu"]) / sc["lq_sd"])[:, None]]
            s = (B.loc[g, list(cfg["static"])].values.astype(float) - np.asarray(sc["st_mu"], float)) / np.asarray(sc["st_sd"], float)
            parts.append(np.repeat(s[None], len(d), 0))
            rows.append(np.hstack(parts))
        return torch.tensor(np.stack(rows), dtype=torch.float32)

    def forecast(self, X):
        """Forecasts by target day [basin, day, lead] in mm/day: blocks of 365 days read after 365 days of warm-up (the engine's definition of a forecast), the flow scaled back, never below zero, issue day t -> target day t+lead."""
        warm, block, T = self.cfg["warmup"], self.cfg["block"], X.shape[1]
        assert torch.isfinite(X).all()
        out = torch.full((X.shape[0], T, 3), float("nan"))
        with torch.no_grad():
            for c in range(warm, T, block):
                h = self.lstm(X[:, c - warm:c + block])[0]
                out[:, c:c + block] = self.head(h)[:, warm:]
        o = np.clip(out.numpy().astype("float64") * self.sc["q_sd"] + self.sc["q_mu"], 0, None)
        p = np.full_like(o, np.nan)
        for j, k in enumerate((1, 2, 3)):
            p[:, k:, j] = o[:, :-k, j]
        return p


MODELS = {m: [Net(m, s) for s in (0, 1, 2)] for m in ("wqa", "wqa_rep")}
for m, nets in MODELS.items():
    for n in nets[1:]:
        assert n.sc["dyn_mu"] == nets[0].sc["dyn_mu"] and n.sc["q_mu"] == nets[0].sc["q_mu"], f"the scalers differ between the seeds of {m}"
GI = {g: i for i, g in enumerate(ids)}
DI = {d: i for i, d in enumerate(DAYS)}
ON_I = [(GI[g], DI[t]) for g, t in zip(ON.gauge_id, ON.target)]                  # (basin, target day) of each onset
ON_THR = np.array([THR[g] for g in ON.gauge_id])
OBS_ON = np.array([D[g].q_mm.values[DI[t]] for g, t in zip(ON.gauge_id, ON.target)])      # the observed flow of each onset's target day
ISSUE = list(zip(ON.gauge_id, ON.issue))
from scipy.stats import spearmanr


def run(nets, rain, label):
    """The three networks of a model on a table of rain-to-midnight columns; returns (per-seed forecasts, their mean)."""
    t = time.time()
    per = [n.forecast(n.inputs(rain)) for n in nets]
    ens = sum(per) / 3
    say(f"  {label}: forward passes {time.time() - t:.0f} s")
    return per, ens


def warned(p):
    """Warned flag of each onset: the lead-1 forecast of its target day above the level of the basin (trigger 1.0)."""
    return np.array([p[b, d, 0] > thr for (b, d), thr in zip(ON_I, ON_THR)])


def table_with(changes):
    """The AORC daily tables with the given {(gauge, day): value} replaced."""
    tab = {g: AO_DAILY[g].copy() for g in ids}
    for (g, d), v in changes.items():
        tab[g].iloc[DI[d]] = v
    return tab


def fit(ens):
    """How well the 1-day forecasts of the onset days fit the observed flow of those days: mean absolute error (mm/day), Spearman correlation, sum of forecasts over sum of observations."""
    f = np.array([ens[b, d, 0] for b, d in ON_I])
    return dict(mae_mm_day=float(np.mean(np.abs(f - OBS_ON))), spearman=float(spearmanr(f, OBS_ON)[0]), forecast_over_observed=float(f.sum() / OBS_ON.sum()))


# ---- the substitutions (the same for every model) --------------------------------------------------------------------------------------------------------------------
VAR = {
    "stage4": {k: r[2].sum() for k, r in REC.items()},
    "stage4_k": {k: K * r[2].sum() for k, r in REC.items()},
    "stage4_no_last_hour": {k: r[2][:-1].sum() for k, r in REC.items()},
    "stage4_no_last_two_hours": {k: r[2][:-2].sum() for k, r in REC.items()},
    "aorc_no_last_hour": {k: r[1][:-1].sum() for k, r in REC.items()},
    "aorc_no_last_two_hours": {k: r[1][:-2].sum() for k, r in REC.items()},
}
BEFORE = {}
for (g, t), r in REC.items():
    rb = day_hours(g, t - pd.Timedelta(days=1))
    assert rb is not None
    BEFORE[(g, t - pd.Timedelta(days=1))] = rb[2].sum()
VAR["stage4_and_day_before"] = {**VAR["stage4"], **BEFORE}
issue_days = sorted(ON.issue.unique())
ALL12 = {}
for g in ids:
    for t in issue_days:
        r = day_hours(g, t)
        assert r is not None, (g, t)
        ALL12[(g, t)] = r
say("substitution tables built")
zq = np.load(RES / "preds" / "lstm_wq.npz", allow_pickle=False)
TABLE_ALONE = warned(zq["pred"])
assert int(TABLE_ALONE.sum()) == 114
rec_replaced = np.array([r[3] > 0 for r in REC.values()])              # the onsets whose issue day has an hour replaced by the AORC hour
S = jload("stage4_realtime")
cmp = []


def chk(what, mine, theirs, tol=0.0):
    cmp.append(dict(what=what, mine=mine, builder=theirs, equal=bool(abs(mine - theirs) <= tol)))


def tally(ens):
    """Hits, misses, false warnings and correct negatives of the warnings given on the issue days when the river was at or below the level (target day in the test years)."""
    h = m = f = c = 0
    for g in ids:
        b = GI[g]
        for t in issue_days:
            d1 = DI[t + pd.Timedelta(days=1)]
            if not (pd.Timestamp("2005-10-01") <= t + pd.Timedelta(days=1) <= pd.Timestamp("2014-09-30")):
                continue
            q_t, q_n = D[g].q_mm.values[DI[t]], D[g].q_mm.values[d1]
            if q_t > THR[g]:
                continue
            w, high = ens[b, d1, 0] > THR[g], q_n > THR[g]
            h += int(w and high); m += int((not w) and high); f += int(w and not high); c += int((not w) and (not high))
    return dict(hits=h, misses=m, false_warnings=f, correct_negatives=c, far=f / (h + f))


def analyse(mname, with_controls):
    nets, res, Sm = MODELS[mname], {}, S["models"][mname]
    Vb = Sm["variants"]
    # 1. reproduction
    base_per, base = run(nets, table_with({}), f"{mname}: AORC as trained")
    max_diff, seed_hits = 0.0, []
    for s, p in enumerate(base_per):
        z = np.load(RES / "preds" / f"lstm_{mname}_s{s}.npz", allow_pickle=False)
        assert list(z["basins"]) == ids and np.array_equal(np.isnan(z["pred"]), np.isnan(p))
        max_diff = max(max_diff, float(np.nanmax(np.abs(z["pred"] - p))))
        seed_hits.append(int(warned(p).sum()))
    ze = np.load(RES / "preds" / f"lstm_{mname}.npz", allow_pickle=False)
    res["reproduction"] = dict(max_abs_difference_to_the_stored_single_seed_forecasts_mm_per_day=max_diff, max_abs_difference_to_the_stored_seed_average=float(np.nanmax(np.abs(ze["pred"] - base))),
                               onsets_warned_aorc_mine=int(warned(base).sum()), onsets_warned_aorc_single_seeds_mine=seed_hits, onsets_warned_rain_table_alone_stored_lstm_wq=int(TABLE_ALONE.sum()), onsets=len(ON))
    assert max_diff < 1e-6 and int(warned(base).sum()) == Sm["aorc"]
    say(f"{mname} reproduction:", res["reproduction"])
    flags, ens_by, out = {"aorc": warned(base), "table": TABLE_ALONE}, {"aorc": base}, {}
    for name, vals in VAR.items():
        per, ens = run(nets, table_with(vals), f"{mname}: {name}")
        flags[name], ens_by[name] = warned(ens), ens
        out[name] = dict(onsets_warned=int(flags[name].sum()), share=float(flags[name].mean()), single_seeds=[int(warned(p).sum()) for p in per])
        say(f"{mname} {name}: {out[name]['onsets_warned']} of 611  single seeds {out[name]['single_seeds']}")
    res["variants"] = out
    n_a, n_s, n_t = flags["aorc"].sum(), flags["stage4"].sum(), flags["table"].sum()
    res["share_of_the_aorc_effect_surviving"] = dict(stage4=float((n_s - n_t) / (n_a - n_t)), stage4_no_last_hour=float((flags["stage4_no_last_hour"].sum() - n_t) / (n_a - n_t)),
                                                      stage4_no_last_two_hours=float((flags["stage4_no_last_two_hours"].sum() - n_t) / (n_a - n_t)))
    res["onsets_lost_and_gained_stage4_against_aorc"] = dict(lost=int((flags["aorc"] & ~flags["stage4"]).sum()), gained=int((~flags["aorc"] & flags["stage4"]).sum()))
    # the onsets whose issue day has no replaced hour
    k = ~rec_replaced
    a_, s_, t_ = flags["aorc"][k].sum(), flags["stage4"][k].sum(), flags["table"][k].sum()
    res["onsets_without_any_replaced_hour"] = dict(onsets=int(k.sum()), aorc=int(a_), rain_table_alone=int(t_), stage4=int(s_), share_surviving=float((s_ - t_) / (a_ - t_)))
    # 2. intervals (2,000 draws)
    rng = np.random.default_rng(20261008)
    n = len(ON)
    days = ON.groupby(["issue"]).indices
    day_keys = list(days)
    A_, S_, T_ = flags["aorc"].astype(float), flags["stage4"].astype(float), flags["table"].astype(float)
    sh_on, sh_day = [], []
    for _ in range(2000):
        i = rng.integers(0, n, n)
        sh_on.append((S_[i].sum() - T_[i].sum()) / (A_[i].sum() - T_[i].sum()))
        j = rng.integers(0, len(day_keys), len(day_keys))
        idx = np.concatenate([days[day_keys[q]] for q in j])
        sh_day.append((S_[idx].sum() - T_[idx].sum()) / (A_[idx].sum() - T_[idx].sum()))
    res["interval_of_the_share_surviving_95"] = dict(drawing_onsets=[float(np.percentile(sh_on, 2.5)), float(np.percentile(sh_on, 97.5))], drawing_issue_days=[float(np.percentile(sh_day, 2.5)), float(np.percentile(sh_day, 97.5))], draws=2000)
    # 3. the warnings on the issue days of all 12 basins
    r_aorc = tally(base)
    _, e1 = run(nets, table_with({k_: r[2].sum() for k_, r in ALL12.items()}), f"{mname}: all 12 basins, Stage IV")
    _, e2 = run(nets, table_with({k_: K * r[2].sum() for k_, r in ALL12.items()}), f"{mname}: all 12 basins, Stage IV x K")
    res["issue_days_of_all_12_basins"] = dict(aorc=r_aorc, stage4=tally(e1), stage4_times_k=tally(e2), issue_days=len(issue_days))
    # 4. the fit on the onset days
    res["fit_on_the_onset_days"] = dict(aorc=fit(base), stage4=fit(ens_by["stage4"]), rain_table_alone=fit(zq["pred"]))
    # 5. the controls on the rain of the issue day (own random numbers)
    if with_controls:
        by_basin = {g: [i for i, (gg, _) in enumerate(ISSUE) if gg == g] for g in ids}
        ordinary = {}
        for g in ids:
            near = {d + pd.Timedelta(days=o) for (gg, d) in ISSUE if gg == g for o in range(-2, 3)}
            ordinary[g] = [d for d in pd.date_range("2005-10-01", "2014-09-30") if d not in near]
        rng = np.random.default_rng(20261009)
        ctl = {"another_onset_stage4": dict(counts=[], fits=[]), "ordinary_day_aorc": dict(counts=[], fits=[]), "no_rain": dict(counts=[], fits=[])}
        for draw in range(10):
            vals = {}
            for g, idx in by_basin.items():
                perm = rng.permutation(len(idx))
                for q in range(len(idx)):
                    if perm[q] == q:                                      # another onset, not itself
                        o = (q + 1) % len(idx)
                        perm[q], perm[o] = perm[o], perm[q]
                for q, p_ in enumerate(perm):
                    vals[ISSUE[idx[q]]] = REC[ISSUE[idx[p_]]][2].sum()
            _, ens = run(nets, table_with(vals), f"{mname}: control, the Stage IV rain of another onset ({draw + 1}/10)")
            ctl["another_onset_stage4"]["counts"].append(int(warned(ens).sum())); ctl["another_onset_stage4"]["fits"].append(fit(ens))
            vals = {(g, t): float(AO_DAILY[g].loc[ordinary[g][int(rng.integers(0, len(ordinary[g])))]]) for (g, t) in ISSUE}
            _, ens = run(nets, table_with(vals), f"{mname}: control, the AORC rain of an ordinary day ({draw + 1}/10)")
            ctl["ordinary_day_aorc"]["counts"].append(int(warned(ens).sum())); ctl["ordinary_day_aorc"]["fits"].append(fit(ens))
        _, ens = run(nets, table_with({k_: 0.0 for k_ in ISSUE}), f"{mname}: control, no rain")
        ctl["no_rain"]["counts"].append(int(warned(ens).sum())); ctl["no_rain"]["fits"].append(fit(ens))
        res["controls"] = {k_: dict(onsets_warned=v["counts"], mean=float(np.mean(v["counts"])), lowest=min(v["counts"]), highest=max(v["counts"]),
                                    fit_on_the_onset_days={m_: float(np.mean([f_[m_] for f_ in v["fits"]])) for m_ in ("mae_mm_day", "spearman", "forecast_over_observed")}) for k_, v in ctl.items()}
    # 6. against the builder's file
    chk(f"{mname}: onsets warned with AORC", int(n_a), Sm["aorc"])
    chk(f"{mname}: onsets warned with the rain table alone", int(n_t), Sm["rain_table_alone"])
    for k_ in VAR:
        chk(f"{mname}: onsets warned: {k_}", out[k_]["onsets_warned"], Vb[k_]["onsets_warned"])
        for s in range(3):
            chk(f"{mname}:   single seed {s}: {k_}", out[k_]["single_seeds"][s], Vb[k_]["single_seed_onsets_warned"][s])
    chk(f"{mname}: share of the AORC effect surviving (Stage IV)", res["share_of_the_aorc_effect_surviving"]["stage4"], Vb["stage4"]["share_of_the_aorc_effect_surviving"], 1e-9)
    chk(f"{mname}: share surviving, last hour out", res["share_of_the_aorc_effect_surviving"]["stage4_no_last_hour"], Vb["stage4_no_last_hour"]["share_of_the_aorc_effect_surviving"], 1e-9)
    chk(f"{mname}: onsets lost against AORC", res["onsets_lost_and_gained_stage4_against_aorc"]["lost"] - res["onsets_lost_and_gained_stage4_against_aorc"]["gained"], -Vb["stage4"]["differs_from_aorc_by_onsets"])
    for key_, mine_ in (("aorc", r_aorc), ("stage4", res["issue_days_of_all_12_basins"]["stage4"]), ("stage4_times_the_level_of_aorc", res["issue_days_of_all_12_basins"]["stage4_times_k"])):
        for f_ in ("hits", "misses", "false_warnings", "correct_negatives"):
            chk(f"{mname}: all 12 basins, {key_}: {f_}", mine_[f_], Sm["issue_days_of_all_basins"][key_][f_])
    iv = Vb["stage4"].get("intervals")
    if iv:
        for kind, key_ in (("drawing_onsets", "over_onsets"), ("drawing_issue_days", "over_issue_days")):
            for q, j in ((0, "low"), (1, "high")):
                chk(f"{mname}: interval {kind} {j}", res["interval_of_the_share_surviving_95"][kind][q], iv[key_]["share_surviving"][q], 0.012)
    w = Sm["onsets_without_any_replaced_hour"]
    for f_ in ("onsets", "aorc", "rain_table_alone", "stage4"):
        chk(f"{mname}: without any replaced hour: {f_}", res["onsets_without_any_replaced_hour"][f_], w[f_])
    chk(f"{mname}: without any replaced hour: share surviving", res["onsets_without_any_replaced_hour"]["share_surviving"], w["share_surviving"], 1e-9)
    ref = Sm.get("reference_fits_on_the_onset_days")
    if ref:
        for kk in ("aorc", "rain_table_alone"):
            for m_ in ("mae_mm_day", "spearman", "forecast_over_observed"):
                chk(f"{mname}: fit on the onset days, {kk}: {m_}", res["fit_on_the_onset_days"][kk][m_], ref[kk][m_], 1e-9)
    fv = Vb["stage4"].get("fit_on_the_onset_days")
    if fv:
        for m_ in ("mae_mm_day", "spearman", "forecast_over_observed"):
            chk(f"{mname}: fit on the onset days, stage4: {m_}", res["fit_on_the_onset_days"]["stage4"][m_], fv[m_], 1e-9)
    if with_controls:
        c_b = Sm["controls_on_the_rain_of_the_issue_day"]
        res["controls_against_the_builders"] = {}
        for k_ in ("another_onset_stage4", "ordinary_day_aorc", "no_rain"):
            mine_, theirs = res["controls"][k_], c_b[k_]
            ok = (mine_["onsets_warned"] == theirs["onsets_warned"]) if k_ == "no_rain" else bool(abs(mine_["mean"] - theirs["mean"]) <= 5 and mine_["lowest"] <= theirs["highest"] and mine_["highest"] >= theirs["lowest"])
            res["controls_against_the_builders"][k_] = dict(mine_mean=mine_["mean"], mine_range=[mine_["lowest"], mine_["highest"]], builder_mean=theirs["mean"], builder_range=[theirs["lowest"], theirs["highest"]], consistent=ok,
                                                           mine_mae=mine_["fit_on_the_onset_days"]["mae_mm_day"], builder_mae=theirs["fit_on_the_onset_days"]["mae_mm_day"])
    return res


t_start = time.time()
res = analyse("wqa", True)
res["wqa_rep"] = analyse("wqa_rep", False)
res["K_aorc_over_stage4"] = float(K)
res["hours"] = dict(stage4_hours_of_the_issue_days_of_the_611_onsets=int(n_hours), replaced_by_the_aorc_hour=int(n_fallback), basin_days_with_a_replaced_hour=int(sum(r[3] > 0 for r in REC.values())))
res["mean_rain_mm_of_the_issue_days"] = dict(aorc=float(np.mean([r[1].sum() for r in REC.values()])), stage4=float(np.mean([r[2].sum() for r in REC.values()])))
# the agreement of the two products on the issue days
Ad, Sd = np.array([r[1].sum() for r in REC.values()]), np.array([r[2].sum() for r in REC.values()])
last_a, last_s = np.array([r[1][-1] for r in REC.values()]), np.array([r[2][-1] for r in REC.values()])
# note: the builder's median ratio is taken over the basin-days with AORC rain above 1 mm; my first guess (above 0 mm) gave 0.9389 against its 0.9425 and was corrected after reading its code (median_ratio_over_days_with_aorc_above_0 keeps it).
agr = dict(basin_days=len(Ad), correlation=float(np.corrcoef(Ad, Sd)[0, 1]), mean_ratio_stage4_over_aorc=float(Sd.sum() / Ad.sum()), median_ratio=float(np.median(Sd[Ad > 1] / Ad[Ad > 1])), median_ratio_over_days_with_aorc_above_0=float(np.median(Sd[Ad > 0] / Ad[Ad > 0])), mean_mm_aorc=float(Ad.mean()), mean_mm_stage4=float(Sd.mean()),
           share_of_the_day_in_the_last_hour_aorc=float(last_a.sum() / Ad.sum()), share_of_the_day_in_the_last_hour_stage4=float(last_s.sum() / Sd.sum()),
           share_of_the_day_in_the_last_hour_aorc_mean_of_ratios=float(np.mean(last_a[Ad > 0] / Ad[Ad > 0])), share_of_the_day_in_the_last_hour_stage4_mean_of_ratios=float(np.mean(last_s[Sd > 0] / Sd[Sd > 0])))
per_basin = {}
for g in ids:
    m_ = np.array([gg == g for gg, _ in REC])
    per_basin[g] = dict(correlation=float(np.corrcoef(Ad[m_], Sd[m_])[0, 1]), ratio=float(Sd[m_].sum() / Ad[m_].sum()))
agr["per_basin"] = per_basin
res["agreement_of_the_two_products"] = agr
Sa = S["agreement_of_the_two_products_on_the_issue_days"]
for k_ in ("basin_days", "correlation", "mean_ratio_stage4_over_aorc", "median_ratio", "mean_mm_aorc", "mean_mm_stage4"):
    chk(f"agreement of the products: {k_}", agr[k_], Sa[k_], 1e-9)
for k_ in ("share_of_the_day_in_the_last_hour_aorc", "share_of_the_day_in_the_last_hour_stage4"):
    ok_sum = abs(agr[k_] - Sa[k_]) <= 1e-9
    chk(f"agreement of the products: {k_} (ratio of sums)" if ok_sum else f"agreement of the products: {k_} (mean of ratios)", agr[k_] if ok_sum else agr[k_ + "_mean_of_ratios"], Sa[k_], 1e-9)
for g in ids:
    chk(f"agreement of the products: correlation in basin {g}", per_basin[g]["correlation"], Sa["per_basin"][g]["correlation"], 1e-9)
    chk(f"agreement of the products: ratio in basin {g}", per_basin[g]["ratio"], Sa["per_basin"][g]["ratio"], 1e-9)
chk("k", float(K), S["k_level_aorc_over_stage4"], 1e-9)
chk("hours replaced by AORC", n_fallback, S["fallback"]["hours_replaced_by_aorc"])
chk("hours in total", n_hours, S["fallback"]["hours_total"])
res["against_the_builders_file"] = cmp
res["n_compared"] = len(cmp)
res["not_equal"] = [c for c in cmp if not c["equal"]]
res["all_equal"] = not res["not_equal"]
# the controls are statistical: they are in 'controls_against_the_builders' (consistent = my mean within 5 onsets of the builder's and the ranges overlap; no rain: equal)
res["controls_consistent"] = all(v["consistent"] for v in res["controls_against_the_builders"].values())
res["seconds_total"] = round(time.time() - t0, 1)
jsave("d3_stage4.json", res)
say(f"compared {len(cmp)} numbers with results/stage4_realtime.json; all equal: {res['all_equal']}; not equal: {[(c['what'], c['mine'], c['builder']) for c in res['not_equal']]}; controls consistent: {res['controls_consistent']}")
say("controls:", json.dumps(res["controls_against_the_builders"]))
