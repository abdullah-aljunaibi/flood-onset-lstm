"""Information (data quality of the 51 extra rivers): CAMELS ships a list of basins whose boundary area differs from the gauge's drainage area by more than 10%
(data/camels_us/basin_dataset_public_v1p2/basin_size_errors_10_percent.txt). Which of the 63 are on it, and how different are the three areas that could define the flow depth: the area written in the forcing
file (the one the project divides the flow by, camels.load_basin), the gauge's area (camels_topo area_gages2) and the outline's (HCDN shapefile AREA)?
Run: python m3b_area_errors.py"""
import re

import shapefile

from m_common import *

J = json.loads((PROJ / "results" / "more_basins.json").read_text())
ids63, ids12, ids50 = J["design"]["rivers"]["sixty_three"], J["design"]["rivers"]["twelve"], J["design"]["rivers"]["other_50"]
txt = (V12 / "basin_size_errors_10_percent.txt").read_text().splitlines()[1:]
listed = {ln.split()[1]: int(ln.split()[-1]) for ln in txt if ln.strip()}
A = attribute_tables()
r = shapefile.Reader(str(CAM / "basin_set_full_res" / "HCDN_nhru_final_671.shp"))
fields = [f[0] for f in r.fields[1:]]
outline = {str(int(rec[fields.index("hru_id")])).zfill(8): rec[fields.index("AREA")] / 1e6 for rec in r.iterRecords()}
rows = {}
for g in ids63:
    forc_area = raw_basin(g)[2] / 1e6
    rows[g] = dict(forcing_file_km2=forc_area, gauge_km2=float(A.loc[g, "area_gages2"]), outline_km2=outline[g], forcing_over_gauge=forc_area / float(A.loc[g, "area_gages2"]), outline_over_gauge=outline[g] / float(A.loc[g, "area_gages2"]))
on_list = sorted(set(ids63) & set(listed))
res = dict(file_lists=len(listed), of_the_63=on_list, of_the_12=sorted(set(ids12) & set(listed)), of_the_50=sorted(set(ids50) & set(listed)), size_diff_percent_in_the_file={g: listed[g] for g in on_list},
           areas_of_the_listed={g: rows[g] for g in on_list}, forcing_over_gauge_range_of_the_63=[min(v["forcing_over_gauge"] for v in rows.values()), max(v["forcing_over_gauge"] for v in rows.values())],
           flow_depth_error_if_forcing_area_is_wrong_percent={g: 100 * (rows[g]["gauge_km2"] / rows[g]["forcing_file_km2"] - 1) for g in on_list})
print(f"{len(listed)} basins on the CAMELS list; of the 63: {on_list} (of the 12: {res['of_the_12']}, of the 50: {res['of_the_50']}); forcing-file area over gauge area for the 63: {res['forcing_over_gauge_range_of_the_63'][0]:.3f} to {res['forcing_over_gauge_range_of_the_63'][1]:.3f}")
for g in on_list:
    v = rows[g]
    print(f"  {g}: listed {listed[g]:+d}%; forcing-file {v['forcing_file_km2']:.1f} km2, gauge {v['gauge_km2']:.1f}, outline {v['outline_km2']:.1f}  (forcing/gauge {v['forcing_over_gauge']:.3f})")
save("m3b_area_errors", res)
