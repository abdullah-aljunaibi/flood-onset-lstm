"""The calendar of the project after the proposal: ONE place for the deadline and the weeks, read by the proposal (build_proposal.py), the
six-slide deck (build_proposal_slides.py), the web presentation (build_proposal_web.py) and the talk plan (build_presentation_plan_proposal.py).

Set on 2026-10-14 from the team lead's message "we do not have 8 weeks to do the last project day is NOV 5th": the proposal talk is on
15 October 2026 and the last project day is 5 November 2026, three weeks later. PLAN.md's eight-week timeline (written before the
deadline was known, and run in one evening) is therefore replaced by the three weeks below for everything that is still to do.
"""
import datetime

PROPOSAL_DAY = datetime.date(2026, 10, 15)          # the proposal talk and the meeting with the professor
DEADLINE = datetime.date(2026, 11, 5)               # the last project day: final hand-in
WEEKS_N = 3

# (week number, dates, title, what is done, the deliverables it closes)
WEEKS = [
    dict(n=1, from_=datetime.date(2026, 10, 15), to=datetime.date(2026, 10, 21), title="Proposal, feedback and the team's own checks",
         what="Proposal talk and meeting with the professor (15 Oct); act on the feedback; each member re-runs and records their own check (team.json); "
              "names and IDs filled in; the 12-basin table and the data record frozen.",
         closes="D1"),
    dict(n=2, from_=datetime.date(2026, 10, 22), to=datetime.date(2026, 10, 28), title="Remaining experiments and the GIS layers",
         what="Five trainings per model where the NSE decides; the GIS layers of all 12 basins (outlines, elevation, land cover) exported and checked against the CAMELS tables; "
              "every experiment script with its review record. Stretch goal, only if the download finishes by 28 Oct: a network trained on the real-time radar product (Stage IV).",
         closes="D2, D3"),
    dict(n=3, from_=datetime.date(2026, 10, 29), to=datetime.date(2026, 11, 5), title="Final report, slides and hand-in",
         what="Final report and slides rebuilt from the results files; the question sheet; two rehearsals; final check of every number against results/*.json; hand-in on 5 Nov.",
         closes="D4, D5"),
]

MILESTONES = [
    (datetime.date(2026, 10, 15), "Proposal presented; feedback noted"),
    (datetime.date(2026, 10, 21), "Own checks recorded for 4 of 4 members; data frozen"),
    (datetime.date(2026, 10, 28), "Experiments frozen; results files final"),
    (datetime.date(2026, 11, 3), "Report and slides final; rehearsed"),
    (datetime.date(2026, 11, 5), "Final hand-in"),
]

STRETCH = ("A network trained on the real-time radar product (Stage IV) needs its hourly archive for 2002-2014 (about 200 GB through the proxy); "
           "it is attempted only if the download finishes by 28 October, otherwise the real-time check already made (Stage IV in place of AORC on the issue days) stands.")


def d(x, year=False):
    """15 Oct, or 15 October 2026."""
    return f"{x.day} {x.strftime('%B %Y')}" if year else f"{x.day} {x.strftime('%b')}"      # %-d is not valid on Windows (2026-10-28)


DURATION = f"three weeks, from the proposal on {d(PROPOSAL_DAY, True)} to the final hand-in on {d(DEADLINE, True)}"
HEADING = f"Three weeks: {d(PROPOSAL_DAY)} to {d(DEADLINE, True)}"
ROWS = [[f"{w['n']}  ({d(w['from_'])}–{d(w['to'])})", f"{w['title']}: {w['what']}"] for w in WEEKS]
ROWS_SHORT = [[f"{d(w['from_'])}–{d(w['to'])}", w["title"]] for w in WEEKS]
