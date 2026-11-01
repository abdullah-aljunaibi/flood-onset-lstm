"""Were the September 2004 floods on the French Broad River hurricane floods? A check against the best-track record.

Section 3 of the report says that the validation years of this river are dominated by a few flood days in September 2004.
This script looks up which Atlantic tropical cyclones passed within 500 km of the gauge in that month (HURDAT2, the file
cached by 06_flood_analysis.py under data/hurdat) and lists the largest flow days of the month next to them.
Writes results/floods_2004.json.
"""
import json
import math

import flood as F
from common import DATA

G = "03439000"
B = F.basins().loc[G]
d = F.load_data()[G].loc["2004-09-01":"2004-09-30"]


def km(lat1, lon1, lat2, lon2):
    p = math.pi / 180
    x = math.sin((lat2 - lat1) * p / 2) ** 2 + math.cos(lat1 * p) * math.cos(lat2 * p) * math.sin((lon2 - lon1) * p / 2) ** 2
    return 2 * 6371 * math.asin(math.sqrt(x))


f = sorted((DATA / "hurdat").glob("hurdat2-1851-*.txt"))[-1]
name, near = None, {}
for line in f.read_text().splitlines():
    c = [x.strip() for x in line.split(",")]
    if c[0].startswith("AL") and len(c[0]) == 8:
        name = c[1]
    elif c[0].startswith("200409"):
        lat, lon = float(c[4][:-1]) * (1 if c[4][-1] == "N" else -1), float(c[5][:-1]) * (-1 if c[5][-1] == "W" else 1)
        dist = km(B.gauge_lat, B.gauge_lon, lat, lon)
        if dist < 500 and dist < near.get(name, dict(closest_km=1e9))["closest_km"]:
            near[name] = dict(name=name.title(), closest_km=round(dist), date=f"{c[0][:4]}-{c[0][4:6]}-{c[0][6:]}", hour_utc=c[1], status_then=c[3])
top = d.q_mm.sort_values(ascending=False).head(3)
R = dict(basin=F.SHORT[G], gauge_id=G, month="2004-09", high_flow_level_mm_day=F.thresholds(F.load_data())[G],
         largest_flow_days=[dict(date=str(k.date()), flow_mm_day=round(float(v), 1), rain_stamped_that_day_mm=round(float(d.prcp[k]), 1)) for k, v in top.items()],
         cyclones_within_500_km=sorted(near.values(), key=lambda s: s["date"]), source=f"HURDAT2 best track, data/hurdat/{f.name} (NOAA National Hurricane Center)",
         notes=["status_then is the storm's status in the best-track record at its closest point (TD = tropical depression: the remnant of the hurricane).",
                "A cyclone nearby on the day of a flood is strong circumstantial evidence, not proof, that its rain caused the flood."])
(F.RES_OUT / "floods_2004.json").write_text(json.dumps(R, indent=1))
print(json.dumps(R, indent=1))
