# Our own checks

**Project-Flood: four short checks, one for each of us.** The team lead, member 2, member 3, member 4. Each takes about 15 minutes. The hand-in goes by e-mail before 2 November 2026, and the team lead needs time to rebuild after you record yours, so do yours early.

## Why we do this

In the mid-term we promised "own checks: 4 of 4". All the code was written and run by Claude Code, under the team lead's direction. Independent AI agents checked it. What is still missing is us. Each of us now checks, with our own run and our own eyes, the numbers of the part we present.

## Why each check is personal

The report and the slides say today: **"The team's own checks are not yet done."** That sentence may change only when a member records in `team.json` that he or she ran the check. If somebody else records it for you, the report would claim a check that nobody made. So:

- Nobody fills in another person's entry. Not the team lead, not Claude, not any AI agent, not a script.
- Record a check only if you ran it yourself and every line said AGREE.
- If a line says DIFFER, do not record the check as passed. Tell the team lead.
- Write your sentence in your own words.

## What the script does, and what it does not

`team_check.py` reads the stored forecasts (`results/preds/*.npz`) and recomputes the headline numbers of your part with its own short code. It needs only Python 3.10 or newer, numpy and pandas. It prints each number next to the number in the registry (`results/final_report_numbers.json`), which is what the report and the slides print. A line looks like this:

```
    onsets warned            yours 19% (114 of 611)         registry 19% (114 of 611)         AGREE
```

*yours* is what you computed (114 hits out of 611 onsets). *registry* is what the documents print. The last word is the verdict. **AGREE** means the same to the printed rounding (19%, 0.59) and the same counts of hits, misses and false alarms. Otherwise it says **DIFFER**.

The script writes nothing and never touches `team.json`. It trains no network, and it does not prove that the forecasts are good. It shows that the numbers on our slides are what the stored forecasts give when you score them yourself.

## The six steps

(1) Read your page. (2) Run the command on it. (3) Do the "by hand" line with a calculator. (4) Open the figure on your page and read its one sentence. (5) Look at your slides (`Handin_2026-11-02\4_Final_Project_Presentation_Flood.pptx`; the speaker notes are in `4_Final_Project_Presentation_notes.md`): do they print the same numbers as the registry column? (6) Record your check in `team.json`, your own entry only, and tell the team lead, who rebuilds the report and the deck.

## If a line says DIFFER

Do not record. Tell the team lead which line. A five-seed run is changing some stored forecasts while we work. It rewrites the sets `lstm_wqa` and `lstm_wqa_s`, which the checks of the team lead and member 4 use. The output then shows a NOTE: a forecast file is newer than the registry. Run the check again when the team lead says the run has finished and the registry is rebuilt.

## This PC, or your own laptop

- **This PC:** open a terminal in `<project root>`. `team.json` is in the same folder, next to `README.md`. Edit your own entry there, yourself.
- **Your own laptop:** install Python 3.10 or newer, then `pip install numpy pandas` (if `python` is not found, try `py`). Put `team_checks/` and the files your check reads (`python team_checks/team_check.py --member <name> --files` lists them; your page names them) in one folder, in the same layout as the project. Edit your own entry in your own copy of `team.json` and send that file to the team lead from your own e-mail. He copies your three fields into the project's `team.json` word for word, and changes nothing else. Keep your e-mail.
- **After you save `team.json`:** it must be valid. On this PC run `python code\team.py` in the project folder: it lists who counts as done (a check without a date or a sentence counts as not done). On a laptop run `python -c "import json; json.load(open('team.json', encoding='utf-8')); print('valid')"`.

**The team lead, after the entries are in:** rebuild as in `Handin_2026-11-02\README_HANDIN.md` (the registry, then the report, then the deck). Read the rebuilt documents and check that they say how many checks are done.

---

## 1. The team lead: the headline ladder (slides 1, 5 and 6)

**What the check is.** The finding of our project is a ladder. The same network, given more information, warns of more flood onsets one day ahead: with the rain table, with rain up to midnight (AORC), with a perfect weather forecast. You recompute these three rungs from the stored forecasts, and the median NSE of the first rung.

**Why it matters for your slides.** Slides 5 and 6 quote the ladder: the rain table at one end, the rung with AORC rain at the other. You present them and you take the questions. If these numbers were wrong, the finding of the whole project would be wrong.

**Command, this PC** (terminal in the project folder):

```
python -X utf8 team_checks\team_check.py --member lead
```

**Command, your own laptop** (after `pip install numpy pandas`): `python team_checks/team_check.py --member lead`. It reads `results/final_report_numbers.json` and, in `results/preds/`, `lstm_wq`, `lstm_wqa`, `lstm_wqf` (`.npz`). Also copy the figure you open.

**What to look at.**

- Every comparison line says AGREE: the share of onsets warned for the three rungs, the NSE of the first rung, and the onsets in the test years that every percentage is out of.
- The AORC rung comes from `lstm_wqa`, which the five-seed run rewrites. If a slide shows another number than the registry column, the deck is older than the registry; you rebuild it.
- Read the words that stand beside the AORC rung and the perfect-forecast rung. Never quote either number without them.
- By hand: divide the hits by the onsets printed for `lstm_wq`. Do you get its percentage?

**Open** `figures\plain_ladder.png`. **One sentence to understand:** with the same network, the share of onsets warned one day ahead grows with each rung of information, so the limit is the information, not the network.

**Record it, yourself, by hand.** Open `team.json`, find your own entry, and change only its last three lines, as in the box. `true` has no quotes. The date is the day you ran the check. In `check_result` write one sentence in your own words: what you ran, that the numbers agreed, one thing you saw. Do not touch `role`, `name`, other entries, commas or braces. Save as UTF-8 and check the file (page 1).

```json
    {
      "role": "Member 1 (RS/GIS, team lead)",
      "name": "The team lead",
      "check_done": true,
      "check_date": "YYYY-MM-DD",
      "check_result": "One sentence in your own words."
    },
```

Then rebuild, last, after the other three have recorded (page 1).

---

## 2. Member 2: data, basins and GIS layers (slide 2)

**What the check is.** Three things. First, how many basins there are at each step of the choice, 671 to 78 to 63 to 12: you count the rows of the audit table yourself. Second, the GIS check table: our basin area and mean elevation against the CAMELS table. Third, one count from the river flow itself: the flood onsets of the test years.

**Why it matters for your slides.** Slide 2 gives the data and the split, and says how the 12 basins were chosen and that our GIS layers agree with the CAMELS table. If the counts or the layers were wrong, every later number would rest on the wrong basins. Your notes for slide 4 say "except one elevation". The table shows the elevation gaps of all the mountain basins: look at all of them before you say that sentence.

**Command, this PC** (terminal in the project folder):

```
python -X utf8 team_checks\team_check.py --member member2
```

**Command, your own laptop** (after `pip install numpy pandas`): `python team_checks/team_check.py --member member2`. It reads `results/final_report_numbers.json`, `data_audit.json`, `basin_audit.csv`, `gis_layers.json`, the folder `results/gis_layers/` and `results/preds/lstm_wq.npz`. Also copy the figure you open.

**What to look at.**

- Every comparison line says AGREE. No set that the five-seed run rewrites is used here.
- Step 2 prints the mountain basins. Those that differ from CAMELS by more than 25 m are marked.
- By hand: add the twelve onset counts of step 3. Then pick one basin in step 2 and subtract: ours minus CAMELS.
- Optional, if you have QGIS: open `results\gis_layers\basins_12.gpkg` and one `elevation_<gauge>.tif`, and read the area and the mean elevation of that basin yourself.

**Open** `figures\basin_gis_check.png`, the top-right panel (mean elevation). **One sentence to understand:** our area and elevation agree with the CAMELS table, except that our mean elevation is higher in some mountain basins; the cause was not found, and the grid size does not explain it.

**Record it, yourself, by hand.** Open `team.json`, find your own entry, and change only its last three lines, as in the box. `true` has no quotes. The date is the day you ran the check. In `check_result` write one sentence in your own words: what you ran, that the numbers agreed, one thing you saw. Do not touch `role`, `name`, other entries, commas or braces. Save as UTF-8 and check the file (page 1).

```json
    {
      "role": "RS/GIS member 2",
      "name": "Member 2",
      "check_done": true,
      "check_date": "YYYY-MM-DD",
      "check_result": "One sentence in your own words."
    },
```

Then tell the team lead, who rebuilds the report and the deck.

---

## 3. Member 3: model, fair test, number to beat (slide 3)

**What the check is.** First, the split: every day of the record belongs to one period, train, validation or test, by the day it is forecast for. The three periods add up to the whole record with no overlap. Second, the flood level is fitted on the training years only. Third, the number to beat: among six simple models you find the one with the highest NSE on the validation years, and read its NSE on the test years next to the LSTM's.

**Why it matters for your slides.** Slide 2's split bar: each period has one job. Slide 3: the number to beat, and the LSTM is not clearly better. If the split leaked, or the number to beat had been chosen on the test years, the comparison with the LSTM would not be fair.

**Command, this PC** (terminal in the project folder):

```
python -X utf8 team_checks\team_check.py --member member3
```

**Command, your own laptop** (after `pip install numpy pandas`): `python team_checks/team_check.py --member member3`. It reads `results/final_report_numbers.json` and, in `results/preds/`, `lstm_wq`, `base_persistence`, `base_climatology`, `base_linreg_w`, `base_linreg_wq`, `base_rf_w`, `base_rf_wq` (`.npz`). Also copy the figure you open.

**What to look at.**

- Every comparison line says AGREE. No set that the five-seed run rewrites is used here.
- Step 3: the star marks the simple model with the highest validation score at each lead. Step 4: the LSTM minus that model is small at every lead.
- Step 4 may print "for reading" lines: a simple model that scores higher on the test years than the one we chose. We do not switch to it. Only the validation years may choose.
- By hand: add the three day counts of step 1. Do you get all the days of the record?

**Open** `figures\baselines.png`, the top-left panel (accuracy without a weather forecast): the big filled markers are the medians of the 12 basins. **One sentence to understand:** we pick the simple model to beat on the validation years, never on the test years, so the test years stay a fair exam that the models have not seen.

**Record it, yourself, by hand.** Open `team.json`, find your own entry, and change only its last three lines, as in the box. `true` has no quotes. The date is the day you ran the check. In `check_result` write one sentence in your own words: what you ran, that the numbers agreed, one thing you saw. Do not touch `role`, `name`, other entries, commas or braces. Save as UTF-8 and check the file (page 1).

```json
    {
      "role": "GNSS member 1",
      "name": "Member 3",
      "check_done": true,
      "check_date": "YYYY-MM-DD",
      "check_result": "One sentence in your own words."
    },
```

Then tell the team lead, who rebuilds the report and the deck.

---

## 4. Member 4: results (slide 4)

**What the check is.** The ladder, one day ahead, for four sets: the rain table, plus rain to the evening (NLDAS), plus rain to midnight (AORC), plus a perfect weather forecast. For each, two numbers: the share of onsets warned, and the share of the warnings that were false. Then the newest results, E4e (the river's hourly record as an input), on the satellite-era split: the same two numbers for three sets.

**Why it matters for your slides.** Slide 4 shows the ladder. Slide 5 shows the targets: GO needs at least 70% warned and at most 30% false. Both halves are in your check, and the verdict NOT YET rests on them. The E4e table is on slide 12, which the team lead presents. You check its numbers because the results part is yours.

**Command, this PC** (terminal in the project folder):

```
python -X utf8 team_checks\team_check.py --member member4
```

**Command, your own laptop** (after `pip install numpy pandas`): `python team_checks/team_check.py --member member4`. It reads `results/final_report_numbers.json` and, in `results/preds/`, `lstm_wq`, `lstm_wqn`, `lstm_wqa`, `lstm_wqf`, `lstm_wqa_s`, `lstm_wqaH_s`, `lstm_wqaWHF_s` (`.npz`). Also copy the figure you open.

**What to look at.**

- Every comparison line says AGREE. Two sets are rewritten by the five-seed run: `lstm_wqa` and `lstm_wqa_s`. If their lines say DIFFER and a NOTE says the file is newer than the registry, that is the run: do not record, wait for the team lead.
- Part 2 is comparable only inside the satellite-era split, never with part 1.
- Read the words beside the AORC and the perfect-forecast rungs. Never quote those numbers without them.
- By hand: false share = false warnings divided by (hits plus false warnings). Do you get the percentage for `lstm_wq`?

**Open** `figures\flow_state.png`, the top panel: the first, second and seventh bars (wqa_s, wqaH_s, wqaWHF_s) carry your two numbers in their labels. **One sentence to understand:** with the river's hourly record the same network warns of more onsets and fewer of its warnings are false, on this split only.

**Record it, yourself, by hand.** Open `team.json`, find your own entry, and change only its last three lines, as in the box. `true` has no quotes. The date is the day you ran the check. In `check_result` write one sentence in your own words: what you ran, that the numbers agreed, one thing you saw. Do not touch `role`, `name`, other entries, commas or braces. Save as UTF-8 and check the file (page 1).

```json
    {
      "role": "GNSS member 2",
      "name": "Member 4",
      "check_done": true,
      "check_date": "YYYY-MM-DD",
      "check_result": "One sentence in your own words."
    }
```

Then tell the team lead, who rebuilds the report and the deck.
