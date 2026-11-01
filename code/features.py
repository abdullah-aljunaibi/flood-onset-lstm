"""Hand-made daily features for the simple models (ridge regression, random forest): what is known when a forecast is issued?

One function, feature_table(d, flow, lead). Row t describes the END of issue day t: recent rain, how wet the basin already is,
temperature, sunlight, humidity and the season - optionally the river flow measured up to day t (needs a river gauge) and,
for the perfect-weather-forecast variant, the actual rain and temperature of the target days t+1..t+lead.
Run this file to prove that no feature looks further ahead than it is allowed to: python code/features.py
"""
import numpy as np
import pandas as pd
from scipy.signal import lfilter

RAIN_SUMS = (3, 7, 14, 30, 90)      # days; the longest one decides how much history a row needs
API_DECAY = (0.85, 0.95)            # antecedent-rain index: yesterday's index * decay + today's rain (memory of ~7 and ~20 days)


def feature_table(d, flow=False, lead=0):
    """Features known at the end of each issue day, same index as d (one basin's table from flood.load_data()).

    flow=True adds the observed river flow up to the issue day. lead=k>0 adds the actual rain and mean temperature of the
    target days t+1..t+k (a perfect weather forecast). Rows without the full 90-day history - or, with lead=k, without k
    further days of weather - are NaN in every column; callers drop them."""
    rain, temp = d.prcp, (d.tmax + d.tmin) / 2
    X = pd.DataFrame(index=d.index)
    for j in range(3):
        X[f"rain_d{j}"] = rain.shift(j)                                   # rain on day t, t-1, t-2 (mm)
    for n in RAIN_SUMS:
        X[f"rain_sum{n}"] = rain.rolling(n).sum()                         # rain of the last n days, day t included (mm)
    for a in API_DECAY:
        X[f"wetness_{int(a * 100):03d}"] = lfilter([1.0], [1.0, -a], rain.values)      # sum of a**i * rain(t-i), i = 0, 1, 2, ...
    X["temp_d0"], X["temp_mean7"], X["temp_mean30"] = temp, temp.rolling(7).mean(), temp.rolling(30).mean()      # deg C
    X["sun_mean7"], X["humid_d0"] = d.srad.rolling(7).mean(), d.vp        # sunlight (W/m2), vapour pressure (Pa)
    doy = 2 * np.pi * d.index.dayofyear.values / 365.25
    X["season_sin"], X["season_cos"] = np.sin(doy), np.cos(doy)
    if flow:
        q = d.q_mm
        for j in range(3):
            X[f"flow_d{j}"] = q.shift(j)                                  # flow on day t, t-1, t-2 (mm/day)
        X["flow_change"], X["flow_mean7"], X["flow_mean30"] = q - q.shift(1), q.rolling(7).mean(), q.rolling(30).mean()
    for j in range(1, lead + 1):
        X[f"rain_ahead{j}"], X[f"temp_ahead{j}"] = rain.shift(-j), temp.shift(-j)      # the target days' actual weather
    return X.where(X.notna().all(axis=1))


if __name__ == "__main__":
    # Self-check on made-up data: change EVERYTHING after a cut day and see which rows of the table move.
    # Rows up to the cut (minus the lead) must stay exactly the same; the very next row must change, or the check proves nothing.
    idx = pd.date_range("1980-10-01", "1983-09-30")
    rng = np.random.default_rng(0)
    d = pd.DataFrame({c: rng.gamma(2.0, 3.0, len(idx)) for c in ["prcp", "srad", "tmax", "tmin", "vp", "dayl", "q_mm"]}, index=idx)
    cut = 700                                                             # position of the last untouched day
    bad = d.copy()
    bad.iloc[cut + 1:] = bad.iloc[cut + 1:] * 7 + 100
    need = max(RAIN_SUMS) - 1
    for use_flow in (False, True):
        for k in (0, 1, 2, 3):
            A, B = feature_table(d, use_flow, k), feature_table(bad, use_flow, k)
            assert A.index.equals(d.index) and A.columns.equals(B.columns)
            last = cut - k                                                # last row that may not see the corrupted days
            assert A.iloc[:last + 1].equals(B.iloc[:last + 1]), f"flow={use_flow} lead={k}: a feature reads data after day t+{k}"
            assert not np.allclose(A.iloc[last + 1].values, B.iloc[last + 1].values), "the check has no power"
            ok = A.notna().all(axis=1).values
            assert not ok[:need].any() and ok[need:len(d) - k].all() and not ok[len(d) - k:].any(), "unexpected NaN rows"
            assert A.isna().all(axis=1).values[~ok].all(), "a row without full history must be NaN in every column"
            print(f"flow={use_flow!s:5}  lead={k}:  {A.shape[1]:2d} features, rows 0..{last} unchanged, row {last + 1} changes  - ok")
    past = [c for c in feature_table(d, True, 3).columns if "ahead" not in c]
    assert feature_table(d, True, 3)[past].iloc[need:-3].equals(feature_table(d, True, 0)[past].iloc[need:-3]), "lead must only ADD columns"
    print("no feature at row t uses data after day t (lead=0) or after day t+k (lead=k): all checks passed")
