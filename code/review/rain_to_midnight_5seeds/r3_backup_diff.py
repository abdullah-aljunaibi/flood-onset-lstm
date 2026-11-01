"""Check 3: what changed from three to five seeds. Every boolean and reading word (gain_rule, the readings, both_products, e1_success, yardstick readings, ladder
n_seeds) and the headline numbers of the backup file (Project-Flood_backup_2026-10-30/results/rain_to_midnight.json, the three-seed state) against the file now.
Also: are seeds 0-2 of the backup's by_seed lists the same values as seeds 0-2 now (the old trainings were kept, not retrained)? Writes OUT/backup_diff.json."""
from r_common import *

NEW = jload(PROJ / RESULTS)
OLD = jload(BACKUP / RESULTS)
WORD_KEYS = {"reading", "clear_gain", "clear_loss", "single_seeds_all_above", "single_seeds_all_below", "every_seed_above_every_seed", "every_seed_below_every_seed", "higher_in_most_basins",
             "p_below_cut", "met", "control_flat", "every_lstm_training_above", "every_lstm_training_below", "baseline_against_seed_spread", "reading_in_words", "n_seeds", "n_seeds_scored",
             "lstm_trainings_above_the_simple_model", "n_lstm_trainings"}
changed = []


def walk(path, a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in a:
            if k in b:
                walk(f"{path}.{k}", a[k], b[k])
    elif path.rsplit(".", 1)[-1] in WORD_KEYS or path.rsplit(".", 1)[-1].startswith(("every_seed", "control_")) and isinstance(a, bool):
        if a != b:
            changed.append(dict(where=path, three_seeds=a, five_seeds=b))


for blk in ("gain_rule", "value_of_hours_09_to_24", "value_of_evening_hours_beyond_utc_day", "value_of_hours_09_to_utc_day_end", "second_rain_product_alone",
            "nldas_rain_against_era5_over_the_utc_day", "both_products", "e1_success", "yardstick", "ladder", "models", "verdicts"):
    walk(blk, OLD.get(blk), NEW.get(blk))
# the seed rule results with the 'every_seed...' keys of e1_success are covered; drop duplicate model-level n_seeds rows of the stored sets (unchanged)
head = {}
for n in OWN:
    for k in (1, 2, 3):
        o, w = OLD["summary_test"]["sets"][n][f"l{k}"], NEW["summary_test"]["sets"][n][f"l{k}"]
        head[f"{n} l{k}"] = dict(nse_median=[o["nse_median"], w["nse_median"]], onset_hits=[o["onset"]["hits"], w["onset"]["hits"]], onset_pod=[o["onset"]["pod"], w["onset"]["pod"]],
                                 onset_far=[o["onset"]["far"], w["onset"]["far"]], all_hits=[o["all"]["hits"], w["all"]["hits"]])
first3 = {}
for n in OWN:
    for k in (1, 2, 3):
        ob, nb = OLD["seed_ranges"]["sets"][n][f"l{k}"]["by_seed"], NEW["seed_ranges"]["sets"][n][f"l{k}"]["by_seed"]
        first3[f"{n} l{k}"] = all(same(ob[c], nb[c][:3], 0.0) for c in ob)
unchanged_ladder = {n: NEW["summary_test"]["sets"][n] == OLD["summary_test"]["sets"][n] for n in LADDER}
texts = {k: [OLD.get(k), NEW.get(k)] for k in ("outcome", "outcome_caveat", "outcome_full")}
jsave("backup_diff.json", dict(changed=changed, n_changed=len(changed), headline=head, seeds_0_2_values_unchanged=first3, ladder_summary_unchanged=unchanged_ladder, texts=texts,
                               old_seeds=OLD.get("seeds"), new_seeds=NEW.get("seeds")))
print(len(changed))
for c in changed:
    print(c)
print("first3", all(first3.values()), "ladder unchanged", unchanged_ladder)
for k, (o, w) in texts.items():
    print("\n==", k, "\nOLD:", o, "\nNEW:", w)
