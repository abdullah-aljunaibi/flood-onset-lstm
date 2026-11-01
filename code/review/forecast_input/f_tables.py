"""The reviewer's own rebuild of the daily tables the E2 networks read (columns dated by the ISSUE day t), from the project's Daymet table (flood.load_data), the AORC cache (my own local days, f_aorc.py), and the forecast columns,
either (variant 'csv') from the bias-corrected columns of data/gefs/basins/gefs_c00_<gauge>.csv, or (variant 'own') from the reviewer's own day totals (f4_series.npz) times the reviewer's own bias factors, rounded to the six
decimals the tables are written with. The design, read from code/18_forecast_input.py and rebuilt here (not imported):
 main arm (E2a, the usual split):  base = Daymet FORCING + aorc_to_midnight (AORC rain of the wall-clock day t) + p1..p3 (AORC rain of the local days t+1..t+3, dated by the issue day t);
     fs_k = p_k for the issue days up to 2000-09-30 and the bias-corrected forecast from 2000-10-01 on; fr_k = p_k up to 2000-09-30, then the RAW forecast; fk_k = p_k up to 2000-09-30, then the bias-corrected forecast SHUFFLED within
     (basin, calendar month) across years, the three leads of a day together, no day keeping its own calendar year, fixed seeds (the algorithm of the builder's shuffled_rows is COPIED here: the control's content is proved by the
     properties checked in f6_tables.py and by the bit-for-bit reproduction of the saved control networks);
 sat arm (E2b):  the same base from 2000-01-01 on, fc_k = the bias-corrected forecast in every year, fx_k = the shuffled forecast in every year (seeds (18, 2, basin number))."""
from f_common import *
from f_aorc import *
import flood as F

SUB_FROM = "2000-10-01"
ROUND = 6


def shuffled_rows(df, key):
    """COPY of code/18_forecast_input.py shuffled_rows: the rows of df permuted within each calendar month so that no row stays in its own calendar year; the three leads of one day move together; key seeds the generator."""
    out = df.copy()
    yrs, mon = df.index.year.values, df.index.month.values
    for m in range(1, 13):
        pos = np.flatnonzero(mon == m)
        rng = np.random.default_rng([*key, m])
        y = yrs[pos]
        perm = rng.permutation(len(pos))
        for _ in range(500):
            bad = np.flatnonzero(y[perm] == y)
            if not len(bad):
                break
            others = np.setdiff1d(np.arange(len(pos)), bad)
            pool = np.concatenate([bad, rng.choice(others, size=min(len(bad), len(others)), replace=False)])
            perm[pool] = perm[rng.permutation(pool)]
        assert np.array_equal(np.sort(perm), np.arange(len(pos))) and not (y[perm] == y).any()
        out.iloc[pos] = df.iloc[pos[perm]].values
    return out


def build_tables(variant="csv", ids=None):
    D = F.load_data(); ids = ids or list(D)
    zones, _ = gauge_zones(ids)
    DAYS = pd.date_range(*F.PERIOD)
    T_ISS = pd.date_range("2000-01-01", "2014-09-30")
    days_ext = np.arange(np.datetime64("1980-10-01"), np.datetime64("2014-10-04"))
    if variant == "own":
        Z = np.load(f"{OUT}/f4_series.npz"); sk = jread("f5_skill.json")["bias"]["factors_mine"]
        assert [str(g) for g in Z["ids"]] == ids
    TAB = {"main": {}, "sat": {}}
    for bi, g in enumerate(ids):
        ser, _ = aorc_local_days(g, zones[g], days_ext)
        A = pd.Series(ser, index=pd.DatetimeIndex(days_ext.astype("datetime64[ns]")))
        base = D[g].assign(aorc_to_midnight=A.reindex(DAYS).values)
        for k in KS:
            base[f"p{k}"] = A.shift(-k).reindex(DAYS).values                          # dated by the ISSUE day t: the AORC rain of local day t+k
        if variant == "csv":
            c = pd.read_csv(f"{PROJ}/data/gefs/basins/gefs_c00_{g}.csv", index_col="issue_day", parse_dates=True)
            fcb = c[[f"fc_bc_{k}" for k in KS]].copy(); fcr = c[[f"fc_raw_{k}" for k in KS]].copy()
        else:
            raw = pd.DataFrame(Z["raw"][ids.index(g)], index=T_ISS, columns=list(KS)).round(ROUND)
            fcr = raw.copy(); fcb = (raw * np.array([sk[g][f"l{k}"] for k in KS])[None, :]).round(ROUND)
            # the table's bias-corrected column is its own rounded raw column times the factor, rounded again (as the csv is written)
            fcr.columns = [f"fc_raw_{k}" for k in KS]; fcb.columns = [f"fc_bc_{k}" for k in KS]
        fcb.columns = fcr.columns = list(KS)
        sh_main = shuffled_rows(fcb.loc[SUB_FROM:], (18, 1, bi))
        m = base.copy()
        for k in KS:
            for pre, src in (("fs", fcb.loc[SUB_FROM:]), ("fr", fcr.loc[SUB_FROM:]), ("fk", sh_main)):
                v = m[f"p{k}"].copy(); v.loc[SUB_FROM:] = src[k].values; m[f"{pre}{k}"] = v
        TAB["main"][g] = m
        s = base.loc[str(T_ISS[0].date()):].copy()
        sh_sat = shuffled_rows(fcb, (18, 2, bi))
        for k in KS:
            s[f"fc{k}"], s[f"fx{k}"] = fcb[k].values, sh_sat[k].values
        TAB["sat"][g] = s
    return TAB, ids
