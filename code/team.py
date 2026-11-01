"""Who is in the team, and which of the members' own checks are done: read from team.json in the project folder.

team.json is filled in BY THE TEAM, by hand. No script writes it and no script may set a check to done: a member who has
run the check that section 9 of the report gives him or her sets "check_done" to true in his or her own entry and adds the
date and one sentence on what the check showed. The three document builders (build_report.py, build_slides.py, build_qa.py)
read the file through this module, so that a name or a finished check appears in the report, the slides and the
question-and-answer sheet after one rebuild, in the same words everywhere.

While a name is empty the documents print the role the plan gives ("GNSS member 1"); while no check is done they say, as
before, that the team's own checks are still to be done. A check that is marked done but has no date or no result
sentence is treated as NOT done, and one line on stderr says why.
"""
import datetime
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
FILE = ROOT / "team.json"

# the four roles of PLAN.md ("## Roles"), in the order of that table; "key" is the short key the slide builder uses
DEFAULT = [dict(key="A", role="Member 1 (RS/GIS, team lead)"), dict(key="R2", role="RS/GIS member 2"),
           dict(key="G1", role="GNSS member 1"), dict(key="G2", role="GNSS member 2")]
MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December")


def _day(iso):
    """'2026-10-04' -> '4 October 2026'; anything else is returned as it is."""
    try:
        d = datetime.date.fromisoformat(str(iso)[:10])
        return f"{d.day} {MONTHS[d.month - 1]} {d.year}"
    except ValueError:
        return str(iso)


def load():
    """The content of team.json as {'course', 'university', 'members': [...]}, with the defaults for whatever is missing."""
    try:
        raw = json.loads(FILE.read_text(encoding="utf-8")) if FILE.exists() else {}
    except (OSError, ValueError) as e:
        print(f"team: team.json could not be read ({type(e).__name__}: {e}); the documents print the roles of the plan", file=sys.stderr)
        raw = {}
    given = {str(m.get("role", "")).strip(): m for m in raw.get("members", []) if isinstance(m, dict)}
    out = []
    for d in DEFAULT:
        m = given.get(d["role"], {})
        name, date, result = (str(m.get(k) or "").strip() for k in ("name", "check_date", "check_result"))
        done = m.get("check_done") is True
        if done and not (date and result):
            print(f"team: the check of {d['role']} is marked done in team.json but has no date or no result sentence; it is treated as not done",
                  file=sys.stderr)
            done = False
        out.append(dict(d, name=name, check_done=done, check_date=date, check_result=result.rstrip(".")))
    return dict(course=str(raw.get("course") or "").strip(), university=str(raw.get("university") or "").strip(), members=out)


def _field(role):
    return "RS/GIS" if "RS/GIS" in role else "GNSS" if "GNSS" in role else ""


def label(role_or_key):
    """The words to print for a member: 'Name (GNSS)' when team.json gives a name, otherwise the role of the plan ('GNSS member 1').
    Takes the role as PLAN.md writes it or the short key (A, R2, G1, G2); an unknown value is returned as it is."""
    for m in load()["members"]:
        if role_or_key in (m["role"], m["key"]):
            return f"{m['name']} ({_field(m['role'])})" if m["name"] else m["role"]
    return role_or_key


def names_line(sep=", "):
    """All four members in the order of the plan: 'Member 1 (RS/GIS, team lead), RS/GIS member 2, GNSS member 1, GNSS member 2' until names are given."""
    return sep.join(label(m["role"]) for m in load()["members"])


def course_line(sep=" · "):
    """'course · university' as far as team.json gives them; '' when it gives neither."""
    t = load()
    return sep.join(x for x in (t["course"], t["university"]) if x)


def checks_state():
    """How many of the members' own checks are done: {'n', 'n_done', 'done': [labels], 'open': [labels]}."""
    ms = load()["members"]
    done = [label(m["role"]) for m in ms if m["check_done"]]
    return dict(n=len(ms), n_done=len(done), done=done, open=[label(m["role"]) for m in ms if not m["check_done"]])


def check_status(role_or_key):
    """One member's check for the table of report section 9: 'still to be done', or 'done on 4 October 2026: <result> (entered by
    the member in team.json)'."""
    for m in load()["members"]:
        if role_or_key in (m["role"], m["key"]):
            return (f"done on {_day(m['check_date'])}: {m['check_result']} (entered by the member in team.json)" if m["check_done"]
                    else "still to be done")
    return "still to be done"


def checks_sentence(short=False):
    """The state of the team's own checks as one sentence without a full stop - the same words in every document.
    short=True is the form for a slide or a list ('The team's own checks are not yet done')."""
    s = checks_state()
    if s["n_done"] == 0:
        return "The team's own checks are not yet done" if short else "the team members' own checks that the plan assigns are still to be done"
    if s["n_done"] < s["n"]:
        who = ", ".join(s["done"])
        return (f"{s['n_done']} of the team's {s['n']} own checks are done ({who}); the others are not yet done" if short else
                f"of the team members' own checks that the plan assigns, {s['n_done']} of {s['n']} are done ({who}); the others are still to be done")
    return "The team's own checks are done (report section 9)" if short else "the team members' own checks that the plan assigns are done (section 9)"


if __name__ == "__main__":
    print(names_line()); print(course_line() or "(no course line)"); print(checks_state()); print(checks_sentence()); print(checks_sentence(short=True))
