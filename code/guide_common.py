"""Shared pieces of the step-by-step guide (build_guide.py -> Project_Guide.html / .pdf).

The guide explains the project to readers who are new to machine learning. Three things are fixed here so that the parts of
the guide, which live in code/guide_sections/, read as one text:

  STEPS   the steps of the guide, in order, with their titles (steps 17 to 23: the work after the mid-term talk, g6_final.py);
  WORDS   ONE plain definition for every concept the guide teaches - a part shows a concept with concept(key, here=...),
          and the list 'The concepts on one page' at the end prints the same sentences;
  the look: step(), concept(), remember(), try_it(), keep(), small_figure(), full_detail(), GUIDE_CSS.

Where the full detail stands: since 30 October 2026 every step points to the final report of the hand-in (FINAL_REPORT, item 3), by its section,
table and figure numbers as the .docx prints them; the long report of 8 October 2026 (Flood_Report.pdf) is named only as that earlier report.

As in the report, no result number is typed into a text: every number is read from results/*.json when the guide is built.
"""
import re
import struct

from common import FIG
from report_common import BLUE  # noqa: F401  (the colours of the report, for parts that draw with them)

# ---- the steps, in order: (number, title, the module of code/guide_sections that writes it) -------------------------------
STEPS = [
    (1, "The question: can a model warn of a flood?", "g1_start"),
    (2, "The data: twelve rivers, 34 years", "g1_start"),
    (3, "A fair test: learn on some years, score on others", "g1_start"),
    (4, "The score: how good is a forecast?", "g1_start"),
    (5, "Simple models first: the number to beat", "g1_start"),
    (6, "The LSTM: a network with a memory", "g2_model"),
    (7, "Training: how the network learns, and when to stop", "g2_model"),
    (8, "One network for all twelve rivers", "g2_model"),
    (9, "Did it beat the simple models?", "g2_model"),
    (10, "Could its warnings be trusted?", "g3_findings"),
    (11, "Why not? What the model is given matters most", "g3_findings"),
    (12, "Looking from above: maps and satellite data", "g3_findings"),
    (13, "A newer network: the Transformer", "g3_findings"),
    (14, "How sure are these numbers?", "g4_concepts"),
    (15, "What we found, and what it teaches", "g4_concepts"),
    (16, "What we did next, and what it showed", "g5_next"),
    (17, "A second model from the team, and what it pointed to", "g6_final"),
    (18, "A stricter warning level: fewer false warnings, fewer warned", "g6_final"),
    (19, "How the day&rsquo;s rain fell, hour by hour", "g6_final"),
    (20, "The river at the issue time", "g6_final"),
    (21, "Five trainings instead of three", "g6_final"),
    (22, "The basins as GIS layers", "g6_final"),
    (23, "Where the project ended", "g6_final"),
]
TITLE = {n: t for n, t, _ in STEPS}
OPTIONAL_STEPS = (16, 17, 18, 19, 20, 21, 22, 23)   # a step that is left out, with one line on stderr, while the results it stands on are missing (without 16 the guide is the guide of the first fifteen steps)

# ---- one plain definition per concept: key -> (the term as printed, the definition) ---------------------------------------
# Rules for these sentences: no number of this project (those belong in 'In this project'), no word that is itself unexplained,
# two or three short sentences at most.
WORDS = {
    "basin": ("Basin", "The land whose rain drains to one point of a river. Rain that falls anywhere in the basin ends up, sooner or later, at that point."),
    "model": ("Model", "A rule that turns inputs into an output. In machine learning the rule is not written by hand: it is found from examples."),
    "inputs_target": ("Inputs and target", "The inputs are what the model is given. The target is what it should produce. The examples it learns from are pairs of "
                      "inputs with the right target."),
    "lead_time": ("Lead time", "How far ahead a forecast looks. A forecast made today for tomorrow has a lead time of one day."),
    "split": ("Training, validation and test data", "Three separate parts of the data, each with one job. The model learns from the training part. Its settings are "
              "chosen on the validation part. The test part is used only to score. A score on data that the model has already seen says nothing."),
    "leak": ("Leak", "When information from the future, or from the test data, reaches the model by accident. A leak makes the scores look better than they really are."),
    "nse": ("NSE (the score for river flow)", "A number that says how good a series of forecasts is. 1 is a perfect forecast. 0 is no better than always forecasting "
            "the river&rsquo;s average flow. Below 0 is worse than that."),
    "high_flow": ("High-flow day and onset", "A high-flow day is a day when the river is above a level that it passes on only a few days in a hundred. An onset is the "
                  "first such day of an event: the river was still low when the forecast was made. Warning of an onset is what counts."),
    "hit_miss": ("Hit, miss and false warning", "A hit: a warning was given and the high flow came. A miss: the high flow came without a warning. A false warning: a "
                 "warning was given and the high flow did not come."),
    "baseline": ("Baseline", "A simple method that every new model has to beat. If a complicated model is no better than the baseline, its complexity is not paying."),
    "weights": ("Weights", "The adjustable numbers inside a model. Learning means finding good values for them. They are also called the parameters of the model."),
    "network": ("Neural network", "A model built from many simple units. Each unit adds up its inputs, each multiplied by a weight, and bends the result. Many units "
                "together can follow rules that are far from a straight line."),
    "normalising": ("Normalising the inputs", "Putting every input on the same scale before a model reads it. From each value the input&rsquo;s average is taken "
                    "away, and the rest is divided by the input&rsquo;s usual spread; both come from the training data only. Then no input counts more only "
                    "because its numbers are larger."),
    "lstm": ("LSTM", "A neural network for sequences such as daily weather. It reads the days one after another and carries a memory from each day to the next. It "
             "learns what to keep in that memory, what to add and what to pass on."),
    "training": ("Training", "How a model learns. It makes forecasts for examples whose answer is known, the error is measured, and the weights are changed a little "
                 "so that the error gets smaller. This is repeated hundreds or thousands of times."),
    "loss": ("Loss", "The one number that training makes small. It measures how wrong the forecasts are on the training examples."),
    "gradient": ("Gradient descent", "The computer works out, for every weight, the slope of the loss: whether the loss goes up or down when that weight is raised, "
                 "and how steeply. Then every weight is moved a little downhill, so that the loss falls; where the slope is steep, the move is larger. The method "
                 "that works out all these slopes at once is called backpropagation."),
    "learning_rate": ("Learning rate", "The number that sets how far the weights are moved for a given slope. Too large, and the training jumps about and never "
                      "settles. Too small, and it takes far too long."),
    "update": ("Update and batch", "The weights are not changed after every single example. The model looks at a small group of examples, a batch, and then changes "
               "the weights once: that is one update."),
    "overfitting": ("Overfitting", "When a model learns its training examples by heart, with all their accidents, and becomes worse on new data. The sign: the error on "
                    "the training data keeps falling while the score on the validation data gets worse."),
    "early_stopping": ("Early stopping", "Stopping the training at the point where the score on the validation data was best, and keeping the model as it was then."),
    "hyperparameter": ("Hyper-parameter", "A setting that is chosen before the training starts, such as the size of the network or the learning rate. It is chosen by "
                       "trying a few values and comparing them on the validation data."),
    "underfitting": ("Underfitting", "The opposite of overfitting. A model that is too simple, or trained too little, cannot follow even its training examples: its "
                     "error is large everywhere. A good model lies between the two: flexible enough to follow its training examples, but not their accidents."),
    "regularisation": ("Regularisation", "Anything that keeps a model from learning its training examples by heart. Early stopping is one way, and a small model is "
                       "another. A third is dropout. During training, a random part of the numbers passed on inside a network is set to zero, so the network "
                       "cannot lean on any one of them."),
    "seed": ("Seed", "The number that fixes everything random in a training: the starting weights, the examples drawn for each batch, and what dropout sets to "
             "zero. Two trainings that differ only in their seed give slightly different models. So a difference between two kinds of model counts only if it "
             "is larger than the difference between two seeds."),
    "generalisation": ("Generalisation", "How well a model works on cases that it was not trained on: other years, or a river it has never seen."),
    "one_change": ("One change at a time", "To find out what one thing is worth, change only that thing and keep everything else the same. Specialists call such a "
                   "test an ablation."),
    "upper_bound": ("Upper bound", "What a model reaches when it is given perfect information that nobody really has. It shows how much room better inputs leave for "
                    "that model; a real system would reach less."),
    "control": ("Control", "A second run that differs from the first in only the one thing being tested. If the control shows the same gain, the gain did not come "
                "from that thing."),
    "reanalysis": ("Reanalysis", "The weather of the past, rebuilt by running a weather model over the observations of that time. It gives values everywhere and for "
                   "every day, but they are model output, not measurements."),
    "transformer": ("Transformer and attention", "Another kind of neural network for sequences. It does not carry a memory from day to day. For each forecast it looks "
                    "back at all earlier days at once and weighs them anew, by what they hold; this is called attention."),
    "real_time": ("Real-time data and data made after the fact", "Real-time data is what exists when a forecast is made. Data made after the fact is "
                  "produced later, with corrections and with information that did not exist then. A model scored on it can look better than it could ever be in use, so ask "
                  "of every input: could a real service have it at the time?"),
    "operating_point": ("Operating point and warning level", "A warning is given when the forecast passes a level, the warning level. A higher level gives fewer warnings and "
                        "fewer false ones; a lower level gives more of both. The resulting pair, the share warned and the share of warnings that were false, is the operating "
                        "point: moving the level moves the point, not what the model knows."),
    "resampling": ("Resampling", "A way to see how much a result depends on the cases that happened to be measured. The cases are drawn again at random, many times, "
                   "and the result is worked out each time. The range of the answers shows how firm the result is."),
    # added 30 October 2026 for the steps after the mid-term talk (g6_final.py)
    "boosting": ("Gradient boosting", "A model made of many small decision trees, chains of yes/no questions that each end in a value. The trees are built one after "
                 "another: each new tree learns the errors that the trees before it still make, and the forecast adds up all the trees."),
    "accuracy": ("Accuracy, and why it misleads on rare events", "Accuracy is the share of all cases that a model gets right. When the event is rare, a model that never "
                 "warns is right almost every time, so a high accuracy can hide that the events themselves were missed. For rare events, count the events that "
                 "were warned of and the warnings that were false."),
    "latency": ("Delivery delay (latency)", "The time between a measurement and the moment it reaches the forecaster. A forecast can use only what has arrived when it is "
                "issued, so a test that hands the model its data with no delay can make it look better than it could be in use."),
    "after_looking": ("A choice made after seeing the test scores", "Any change made once the test scores are known: another training, another input, another warning "
                      "level. Even an honest change can lean towards a better-looking result, because the test has already been seen. Such a choice must be "
                      "written down and disclosed, and its result counts for less than one whose plan was fixed in advance."),
}
USED = []                                                   # keys in the order in which the guide shows them (for the list at the end)


# ---- the look -----------------------------------------------------------------------------------------------------------
def step(n, lead):
    """The heading of step n (its title comes from STEPS) with one sentence that says what the step is about."""
    return f'<h2 class="step" id="step{n}"><span class="n">{n}</span>{TITLE[n]}</h2>\n<p class="lead">{lead}</p>'


def concept(key, here=""):
    """A concept box: the fixed plain definition of WORDS[key], and what it is in this project (here: one or two sentences, numbers read
    from the results files). Show each concept once, where it is first needed."""
    term, plain = WORDS[key]
    if key not in USED:
        USED.append(key)
    more = f'<span class="here"><b>In this project:</b> {here}</span>' if here else ""
    return f'<div class="concept"><span class="lab">Concept</span><b>{term}.</b> {plain}{more}</div>'


def remember(text):
    """The one or two sentences a reader should take away from a step."""
    return f'<p class="remember"><b>What to remember.</b> {text}</p>'


FINAL_REPORT = "3_Final_Project_Report_Flood.pdf"                  # the final report of the hand-in (item 3), the source of every section number below
OLD_REPORT = "the earlier long report of 8 October 2026 (Flood_Report.pdf)"


# Steps 1 to 16 name the sections of the long report of 8 October 2026; the final report (hand-in item 3) holds the same material here
# (located in its text on 30 October 2026). Steps 2 and 3 both passed 'section 1', steps 6 and 7 'section 3': their entries name both places.
FINAL_FOR_OLD = {
    "page 1 and the introduction in front of section 1": "Sections 1.1 and 2.3",
    "section 1": "Sections 4.1, 5.3 and 6.3",
    "sections 2 and 5": "Sections 3.2 and 6.3",
    "section 2": "Section 5.4",
    "section 3": "Sections 5.2 and 5.3",
    "section 4.1": "Sections 5.2 and 7.1",
    "sections 4.2, 3 and 4.4": "Sections 7.1 and 7.3",
    "section 5": "Section 7.1 and Table 16 (Appendix C)",
    "sections 4.3, 4.5 and 5": "Sections 7.1, 7.3 and 8.1",
    "section 6": "Section 4.1, Appendix B and Table 14 (Appendix C)",
    "section 7": "Sections 7.1 and 10.3",
    "sections 4.6 to 4.8 and 8": "Sections 7.1, 7.3 and 8.2",
}


def final_detail_for(old_sections):
    """'Full detail' for a step written against the long report of 8 October 2026: the final report's parts that hold the same material."""
    where = FINAL_FOR_OLD.get(old_sections)
    if where is None:
        import sys
        print(f"guide_common: no final-report section for the old pointer {old_sections!r}; the line names the final report only", file=sys.stderr)
        return '<p class="who">Full detail: the final report.</p>'
    return full_detail(where)


def full_detail(where):
    """The line at the end of a step that names where the final report holds the full detail (where: its sections, tables, figures)."""
    return f'<p class="who">Full detail: the final report, {where}.</p>'


def try_it(text):
    """A box 'Try it yourself': a command to run, with one sentence on what it shows (text is HTML; put commands in <code>)."""
    return f'<div class="try"><b>Try it yourself.</b> {text}</div>'


def concepts_list():
    """'The concepts on one page': every concept of WORDS as a two-column table, those the guide showed first, in the order shown."""
    keys = USED + [k for k in WORDS if k not in USED]
    rows = "".join(f"<tr><th>{WORDS[k][0]}</th><td>{WORDS[k][1]}</td></tr>" for k in keys)
    return f'<table class="words">{rows}</table>'


def keep(*blocks):
    """Blocks that stay together on one page and start below the pictures before them: a heading with its first paragraph, a list
    with the line that introduces it, 'What to remember' with the line 'Full detail'."""
    return '<div class="keep">' + "\n".join(b for b in blocks if b) + "</div>"


# ---- pictures that were drawn narrower than the text ------------------------------------------------------------------------
# Every picture is shown at the full text width, with one exception: a picture that was drawn narrower. Blown up to the full width its
# lettering would be far larger than that of the other pictures, so it is shown at the width at which the lettering matches theirs.
FULL_PX = 1474                                              # a picture drawn for the full width: 6.7 inches at 220 dots per inch
FULL_UNITS = 6.7 * 72                                       # the same width in points, the unit of the drawings (SVG) of the report


def lettering_width(name):
    """The share of the text width (%) at which the lettering of figures/<name> is as large as in a picture drawn for the full width;
    100 for such a picture, and when the size of the file cannot be read."""
    try:
        with open(FIG / name, "rb") as f:
            head = f.read(24)
    except OSError:
        return 100
    if head[:8] != b"\x89PNG\r\n\x1a\n" or len(head) < 24:
        return 100
    return min(100, round(100 * struct.unpack(">I", head[16:20])[0] / FULL_PX))


def drawing_width(svg):
    """The same for a drawing (SVG text) whose units are points: the width of its viewBox against that of a full picture."""
    m = re.search(r'viewBox="0 0 ([\d.]+) ', svg)
    return min(100, round(100 * float(m.group(1)) / FULL_UNITS)) if m else 100


def small_figure(inner, caption, width, beside=False):
    """A picture (inner: an <img> or an <svg>) at `width` % of the text width: in the middle, with its caption below at the full text
    width; or, with beside=True, on the left with its caption beside it. At 100 % it is an ordinary figure."""
    if width >= 100:
        return f"<figure>{inner}<figcaption>{caption}</figcaption></figure>"
    side = ' class="beside"' if beside else ""
    return f'<figure{side}><div class="pic" style="width:{width}%">{inner}</div><figcaption>{caption}</figcaption></figure>'


FOOT = "font:7.5pt 'Helvetica Neue',sans-serif;color:#898781"
GUIDE_CSS = (
    "@page{@bottom-left{content:'Predicting river floods with AI \\00B7  the project step by step';" + FOOT + "}"
    "@bottom-right{content:counter(page) ' / ' counter(pages);" + FOOT + "}} "
    # a paragraph or a list item is never cut in two by a page break or by a picture that moves to the next page
    "body{font-size:10.3pt;line-height:1.46} p{margin:0 0 6pt;break-inside:avoid} li{margin-bottom:3pt;break-inside:avoid} "
    "h1{font-size:21pt;margin:0 0 2pt} .sub{font-size:10.4pt} "
    "h2.step{font-size:14.5pt;margin:17pt 0 3pt;display:flex;align-items:center;gap:8pt;break-after:avoid} "
    "h2.step .n{display:inline-flex;align-items:center;justify-content:center;width:22pt;height:22pt;border-radius:50%;background:var(--blue);"
    "color:#fff;font-size:11.5pt;flex:none} "
    "h2.plain{font-size:14.5pt;margin:17pt 0 4pt;border-left:4px solid var(--blue);padding-left:7pt} "
    "p.lead{font-size:11.2pt;color:var(--ink2);margin:0 0 7pt} h3{font-size:11pt;margin:10pt 0 3pt} "
    ".concept{border:1px solid #cfe0f7;background:#f3f7fd;border-radius:8px;padding:6pt 10pt 7pt;margin:8pt 0;break-inside:avoid;font-size:9.9pt} "
    ".concept .lab{display:inline-block;background:var(--blue);color:#fff;border-radius:20px;padding:0 7pt;font-size:7.6pt;font-weight:600;"
    "margin-right:6pt;vertical-align:1pt;text-transform:uppercase;letter-spacing:.05em} .concept .here{display:block;margin-top:3pt} "
    ".remember{border-left:3px solid var(--aqua);background:#eef9f4;padding:5pt 9pt;margin:8pt 0;break-inside:avoid} "
    ".try{border:1px dashed var(--axis);border-radius:8px;padding:5pt 9pt;margin:8pt 0;font-size:9.4pt;break-inside:avoid} .try code{font-size:8.5pt} "
    "figcaption,.cap{font-size:8.9pt} table{font-size:9.3pt} "
    ".toc{columns:2;column-gap:18pt;margin:4pt 0 6pt;padding:0;list-style:none} .toc li{margin:0 0 2.5pt;break-inside:avoid} "
    ".toc .n{display:inline-block;width:15pt;font-weight:600;color:var(--blue)} "
    ".words th{width:25%;color:var(--ink);font-weight:600} .words td,.words th{padding:3.5pt 6pt} "
    ".who{font-size:9.3pt;color:var(--ink2)} "
    # keep(): one block on one page, below the pictures before it; small_figure(): a narrower picture, in the middle or beside its caption.
    # The two cells of 'beside' are laid out as a table of fixed widths: with a CSS grid, Chrome took the width of the image file for the
    # width of the page in some page breaks and printed the whole guide smaller.
    ".keep{break-inside:avoid;clear:both} "
    "figure .pic{margin:0 auto} figure.beside{display:table;table-layout:fixed;width:100%} "
    "figure.beside .pic,figure.beside figcaption{display:table-cell;vertical-align:middle} figure.beside figcaption{padding-left:11pt} "
    # on a screen the page is as wide as the printed page, not as wide as the window
    "@media screen{body{max-width:184mm;margin:12mm auto;padding:0 13mm}}")
