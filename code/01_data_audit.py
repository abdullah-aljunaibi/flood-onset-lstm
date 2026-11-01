"""Stage 2 - data audit: which south-eastern US basins have a clean enough record to train on?

Candidates = CAMELS-US basins in HUC 03 (South Atlantic-Gulf) and 06 (Tennessee) that are humid, rain-fed and 100-2500 km2.
Every candidate's daily flow record is checked for gaps, estimated values and zero-flow days; the final study basins are
picked by rule (complete record, spread over three landscapes). Writes results/data_audit.json + results/basin_audit.csv.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from camels import attributes, load_basin
from common import *

use_style()
A = attributes()
cand = A[A.huc_02.isin([3, 6]) & (A.frac_snow < 0.1) & (A.aridity < 1) & A.area_gages2.between(100, 2500)]
PERIOD = ("1980-10-01", "2014-09-30")                      # 34 full water years
N_PICK = 4                                                 # per landscape

rows = []
for g in cand.index:
    d = load_basin(g).loc[PERIOD[0]:PERIOD[1]]
    miss = d.q_mm.isna()
    gap = (miss != miss.shift()).cumsum()[miss]            # lengths of consecutive missing runs
    yearly_missing = miss.groupby(d.index.year).mean()
    rows.append(dict(gauge_id=g, n_days=len(d), missing_share=float(miss.mean()), longest_gap_days=int(gap.value_counts().max()) if miss.any() else 0,
                     years_with_gaps=int((yearly_missing > 0).sum()), estimated_share=float(d.q_flag.astype(str).str.contains("e").mean()),
                     zero_flow_share=float((d.q_mm == 0).mean()), forcing_missing=int(d[["prcp", "tmax", "tmin", "srad", "vp"]].isna().sum().sum()),
                     q_mean_mm=float(d.q_mm.mean()), q_p95_mm=float(d.q_mm.quantile(.95)), q_max_mm=float(d.q_mm.max()),
                     p_mean_mm=float(d.prcp.mean()), runoff_ratio=float(d.q_mm.mean() / d.prcp.mean())))
T = pd.DataFrame(rows).set_index("gauge_id").join(cand[["gauge_name", "huc_02", "gauge_lat", "gauge_lon", "elev_mean", "area_gages2", "aridity", "frac_snow", "baseflow_index", "high_q_freq"]])
T["landscape"] = np.where(T.elev_mean > 600, "Mountains (above 600 m)", np.where(T.gauge_lat < 32.5, "Southern lowlands (south of 32.5N)", "Northern lowlands and Piedmont"))
T["complete"] = (T.missing_share == 0) & (T.forcing_missing == 0)

# pick by rule: complete record only, then within each landscape spread the picks across basin size
picked = []
for land, grp in T[T.complete].groupby("landscape"):
    grp = grp.sort_values("area_gages2")
    idx = np.unique(np.linspace(0, len(grp) - 1, min(N_PICK, len(grp))).round().astype(int))
    picked += grp.index[idx].tolist()
T["selected"] = T.index.isin(picked)
T.round(4).to_csv(RES / "basin_audit.csv")

S = T[T.selected]
R = dict(period=PERIOD, n_all_camels=len(A), n_candidates=len(T), n_complete=int(T.complete.sum()), n_with_gaps=int((~T.complete).sum()),
         worst_missing=T.sort_values("missing_share", ascending=False).head(5)[["gauge_name", "missing_share", "longest_gap_days"]].round(4).reset_index().to_dict("records"),
         max_estimated_share=float(T.estimated_share.max()), median_estimated_share=float(T.estimated_share.median()),
         candidates_by_landscape=T.landscape.value_counts().to_dict(), complete_by_landscape=T[T.complete].landscape.value_counts().to_dict(),
         selected=S.reset_index()[["gauge_id", "gauge_name", "landscape", "gauge_lat", "gauge_lon", "elev_mean", "area_gages2", "p_mean_mm", "q_mean_mm", "q_p95_mm", "q_max_mm",
                                   "runoff_ratio", "baseflow_index", "estimated_share", "zero_flow_share"]].round(3).to_dict("records"))
(RES / "data_audit.json").write_text(json.dumps(R, indent=2))

# figure: where the candidates and the selected basins are
fig, ax = plt.subplots(figsize=(7.6, 4.2))
col = {"Mountains (above 600 m)": BLUE, "Southern lowlands (south of 32.5N)": ORANGE, "Northern lowlands and Piedmont": AQUA}
for land, grp in T.groupby("landscape"):
    ax.scatter(grp.gauge_lon, grp.gauge_lat, s=18, color=col[land], alpha=.35, edgecolors="none")
    s = grp[grp.selected]
    ax.scatter(s.gauge_lon, s.gauge_lat, s=70, color=col[land], edgecolors=INK, linewidths=1.1, label=f"{land} ({len(s)} selected of {len(grp)})", zorder=3)
gaps = T[~T.complete]
ax.scatter(gaps.gauge_lon, gaps.gauge_lat, s=60, marker="x", color=INK2, linewidths=1.2, label=f"record has gaps ({len(gaps)})", zorder=2)
ax.set_xlabel("longitude", fontsize=8.5); ax.set_ylabel("latitude", fontsize=8.5); ax.tick_params(labelsize=8); ax.set_aspect(1.2)
ax.set_title("Candidate basins in the south-eastern US and the ones selected", fontsize=10.5); ax.legend(frameon=False, fontsize=8, loc="lower right")
fig.tight_layout(); fig.savefig(FIG / "basin_selection.png", dpi=220); plt.close(fig)

pd.set_option("display.width", 230)
print(json.dumps({k: v for k, v in R.items() if k != "selected"}, indent=1))
print(S[["gauge_name", "landscape", "area_gages2", "elev_mean", "p_mean_mm", "q_mean_mm", "q_p95_mm", "q_max_mm", "runoff_ratio", "estimated_share"]].round(2).sort_values("landscape").to_string())
