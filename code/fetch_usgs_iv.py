"""Fetch the 15-minute (instantaneous) USGS discharge of the study gauges from the USGS Water Data OGC API and reduce it to hourly series.

Written 2026-10-28 for experiment E4e of NEXT_EXPERIMENTS_2026-10-28.md (the state of the river at the issue time as an LSTM input). Checked that day:
the legacy service waterservices.usgs.gov/nwis/iv returns nothing for these years any more; the OGC API serves APPROVED 15-minute discharge of all 12
study gauges back to 1985-10-01 (02472000 from 1983-10, 02128000 and 02298608 from 1986-10, 02221525 and 02349900 from 1989-10, 03473000 from 1990-10,
02296500 from 1992-10). One month of one gauge is one request of about 0.5 MB and 3-4 s (2,976 rows); a few requests needed no key. Heavy use may be
throttled: the free key is at https://api.waterdata.usgs.gov/signup/ (pass it with --key or the environment variable USGS_API_KEY). Measured on the Windows PC on 2026-10-28: without a key the API
allows 1000 requests an hour (X-Ratelimit-Limit; a 429 answer carries Retry-After, about 32 min), so fetch_month now waits as told instead of giving up, and one thread is enough. Whole plan:
12 gauges x 348 months (1985-10..2014-09) = about 4,200 requests, about 2 GB, about 4 h in one thread, free. Check the plan's cost line before running it all.

Usage (from the project root; needs requests, pandas, numpy):
  python code/fetch_usgs_iv.py --gauges 03439000 --start 2003-05 --end 2003-06      a test: two months of one gauge
  python code/fetch_usgs_iv.py --start 1985-10 --end 2014-09                        all 12 study gauges; resumable: a month already on disk is never fetched again
  python code/fetch_usgs_iv.py --assemble                                            only rebuild the hourly files from the monthly files on disk
Writes
  data/hourly_flow/raw/<gauge>/<YYYY-MM>.csv.gz     the API's CSV as received (x, y, time_series_id, monitoring_location_id, parameter_code, statistic_id, time, value, ...)
  data/hourly_flow/usgs_iv_<gauge>.npz              hourly: stamp_utc_s = hour-ENDING UTC stamps (the convention of data/aorc: the value at T describes the hour (T-1h, T]),
                                                    q_mean_cfs = mean of the 15-minute values in the hour (NaN when the hour has none), q_last_cfs = the last value in the hour
                                                    (the river at the end of the hour), n_values, approved = share of the hour's values with approval_status 'Approved'
  data/hourly_flow/usgs_iv_meta.json                what was fetched when, rows, hours, gaps
Units: ft3/s as served. Convert as camels.py does for the daily flow: q_mm_per_day = q_cfs * 0.0283168 * 86400 / (area_km2 * 1e6) * 1000. The daily USGS mean of
day d (CAMELS q_mm) is the mean over the LOCAL day, so compare day totals only after shifting to the gauge's wall clock (16_aorc_rain_to_midnight.py, ZONE).
"""
import argparse
import gzip
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, RES

API = "https://api.waterdata.usgs.gov/ogcapi/v1/collections/continuous/items"
OUTDIR = DATA / "hourly_flow"
RAW = OUTDIR / "raw"
META = OUTDIR / "usgs_iv_meta.json"
PAGE = 10000                                                 # rows per request; a month of 15-minute values has at most 2,976 (5-minute: 8,928)


def study_gauges():
    return [r["gauge_id"] for r in json.loads((RES / "data_audit.json").read_text())["selected"]]


def fetch_month(g, m, session, key=None, tries=6):
    """One gauge-month of 15-minute discharge as CSV, cached; returns (file, rows or 'cached')."""
    f = RAW / g / f"{m}.csv.gz"
    if f.exists():
        return f, "cached"
    t0, t1 = m.to_timestamp(how="start"), m.to_timestamp(how="end").floor("s")
    params = dict(monitoring_location_id=f"USGS-{g}", parameter_code="00060", datetime=f"{t0:%Y-%m-%dT%H:%M:%S}Z/{t1:%Y-%m-%dT%H:%M:%S}Z", f="csv", limit=PAGE)
    headers = {"X-Api-Key": key} if key else {}
    err, k, waited = None, 0, 0
    while True:
        try:
            r = session.get(API, params=params, headers=headers, timeout=180)
            if r.status_code == 200:
                break
            err = f"HTTP {r.status_code}: {r.text[:200]}"
            if r.status_code == 429:                      # the API's hourly quota (X-Ratelimit-Limit 1000 without a key, 2026-10-28): wait as told, do not count it as a try
                wait = min(int(float(r.headers.get("Retry-After", "60"))) + 15, 3700)
                waited += wait
                print(f"  {g} {m}: rate limit reached, waiting {wait} s as the API asks (remaining {r.headers.get('X-Ratelimit-Remaining')}, limit {r.headers.get('X-Ratelimit-Limit')})", flush=True)
                time.sleep(wait)
                continue
            if r.status_code not in (500, 502, 503, 504):
                raise RuntimeError(f"{g} {m}: {err}")
        except requests.RequestException as e:
            err = repr(e)
        k += 1
        if k >= tries:
            raise RuntimeError(f"{g} {m}: gave up after {tries} tries ({err})")
        time.sleep(5 * 2 ** k)
    text = r.text
    rows = max(text.count("\n") - 1, 0)
    if rows >= PAGE:
        raise RuntimeError(f"{g} {m}: {rows} rows reached the page limit; split the month")
    f.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(f, "wt", encoding="utf-8") as fh:
        fh.write(text)
    print(f"  {g} {m}: {rows} rows", flush=True)
    return f, rows


def assemble(g):
    """Hourly series of one gauge from its monthly files: mean and last value of the 15-minute values in each hour (hour-ending UTC stamps)."""
    files = sorted((RAW / g).glob("*.csv.gz"))
    frames = []
    for f in files:
        try:
            d = pd.read_csv(f, usecols=["time", "value", "approval_status"], dtype={"value": str, "approval_status": str})
        except pd.errors.EmptyDataError:                     # a month the API answered with an empty body (no values): nothing to add (seen 2026-10-29)
            continue
        if len(d):
            frames.append(d)
    if not frames:
        return dict(values=0, note="no rows on disk")
    d = pd.concat(frames, ignore_index=True)
    t = pd.to_datetime(d["time"], utc=True).dt.tz_convert(None)
    v = pd.to_numeric(d["value"], errors="coerce")                       # 'Ice', 'Eqp' and the like become NaN and are dropped
    ok = v.notna().values
    s = pd.DataFrame(dict(v=v.values[ok], appr=(d["approval_status"].values[ok] == "Approved")), index=pd.DatetimeIndex(t.values[ok])).sort_index()
    s = s[~s.index.duplicated(keep="last")]
    hour_end = (s.index - pd.Timedelta(microseconds=1)).ceil("h")          # the value at 00:15 belongs to the hour ending 01:00; the one at exactly 01:00 too
    grp = s.groupby(hour_end)
    H = pd.DataFrame(dict(q_mean=grp["v"].mean(), q_last=grp["v"].last(), n=grp.size(), approved=grp["appr"].mean()))
    H = H.reindex(pd.date_range(H.index.min(), H.index.max(), freq="h"))
    OUTDIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUTDIR / f"usgs_iv_{g}.npz", stamp_utc_s=(H.index.values.astype("datetime64[s]").astype("int64")), q_mean_cfs=H["q_mean"].values.astype("float64"),
                        q_last_cfs=H["q_last"].values.astype("float64"), n_values=H["n"].fillna(0).values.astype("int16"), approved=H["approved"].values.astype("float32"))
    return dict(values=int(len(s)), first_hour_end=str(H.index[0]), last_hour_end=str(H.index[-1]), hours=int(len(H)), hours_without_values=int(H["q_mean"].isna().sum()),
                months_on_disk=len(files), share_approved=float(np.nanmean(H["approved"].values)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--gauges", default=None, help="comma-separated USGS gauge ids (default: the 12 study basins of results/data_audit.json)")
    ap.add_argument("--start", default="1985-10", help="first month YYYY-MM")
    ap.add_argument("--end", default="2014-09", help="last month YYYY-MM")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--key", default=os.environ.get("USGS_API_KEY"), help="optional API key (or USGS_API_KEY)")
    ap.add_argument("--assemble", action="store_true", help="do not fetch; rebuild the hourly files from the monthly files on disk")
    a = ap.parse_args()
    gauges = a.gauges.split(",") if a.gauges else study_gauges()
    meta = json.loads(META.read_text()) if META.exists() else {"api": API, "gauges": {}}
    if not a.assemble:
        months = list(pd.period_range(a.start, a.end, freq="M"))
        t0 = time.time()
        with requests.Session() as session, ThreadPoolExecutor(a.threads) as pool:
            for g in gauges:
                done = list(pool.map(lambda m: fetch_month(g, m, session, a.key), months))
                fetched = [r for _, r in done if r != "cached"]
                print(f"{g}: {len(months)} months, {len(fetched)} fetched now ({sum(fetched) if fetched else 0} rows), {len(done) - len(fetched)} were on disk  [{time.time() - t0:.0f} s]", flush=True)
                meta["gauges"].setdefault(g, {})["fetched"] = dict(start=a.start, end=a.end, when=time.strftime("%Y-%m-%d %H:%M"), months=len(months), months_fetched_now=len(fetched))
    for g in gauges:
        meta["gauges"].setdefault(g, {})["hourly"] = assemble(g)
        print(f"{g}: hourly file {meta['gauges'][g]['hourly']}", flush=True)
    meta["how_to_read"] = "stamp_utc_s are hour-ENDING UTC stamps; q_mean_cfs is the mean of the 15-minute values in (T-1h, T], q_last_cfs the last of them; convert to mm/day with the basin area as in camels.py"
    OUTDIR.mkdir(parents=True, exist_ok=True)
    META.write_text(json.dumps(meta, indent=1))
    print(f"wrote {META}")


if __name__ == "__main__":
    main()
