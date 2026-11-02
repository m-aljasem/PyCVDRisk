"""Build reference/edacs/REPORT.md. Run from repo root: python reference/edacs/report.py"""

from __future__ import annotations

import csv
import datetime as dt
from pathlib import Path

import numpy as np

from cvd_risk import edacs

HERE = Path(__file__).parent
FX = HERE / "fixtures"
BASE = {"age": 30, "sex": "female", "known_cad": 0, "risk_factors": 0, "diaphoresis": 0,
        "radiates_arm_shoulder": 0, "pleuritic": 0, "palpation": 0}
BANDS = [((18, 45), 2), ((46, 50), 4), ((51, 55), 6), ((56, 60), 8), ((61, 65), 10), ((66, 70), 12),
         ((71, 75), 14), ((76, 80), 16), ((81, 85), 18), ((86, 110), 20)]  # Box 3, a)
RULES = [(f"age {a}", {"age": a}, pts) for (lo, hi), pts in BANDS for a in (lo, hi)] + [
    ("male", {"sex": "male"}, 2 + 6), ("age 50, known CAD", {"age": 50, "known_cad": 1}, 4 + 4),
    ("age 50, 3 risk factors", {"age": 50, "risk_factors": 3}, 4 + 4),
    ("age 50, 2 risk factors", {"age": 50, "risk_factors": 2}, 4),
    ("age 51, known CAD", {"age": 51, "known_cad": 1}, 6), ("diaphoresis", {"diaphoresis": 1}, 2 + 3),
    ("radiates to arm or shoulder", {"radiates_arm_shoulder": 1}, 2 + 5),
    ("pain with inspiration", {"pleuritic": 1}, 2 - 4), ("pain on palpation", {"palpation": 1}, 2 - 6),
]


def main():
    p = list(csv.DictReader((FX / "patients.csv").open()))
    ref = np.array([float(r["score"]) for r in csv.DictReader((FX / "reference.csv").open())])
    c = lambda k: np.array([x[k] for x in p])
    f = lambda k: c(k).astype(float)
    ours = edacs(f("age"), c("sex"), f("known_cad"), f("risk_factors"), f("diaphoresis"), f("radiates"),
                 f("pleuritic"), f("palpation"))
    same = int((ours == ref).sum())
    rules = [(n, float(edacs(**{**BASE, **kw})), w) for n, kw, w in RULES]
    ok = sum(g == w for _, g, w in rules)
    L = ["# EDACS: verification report", "", f"Generated {dt.date.today()} by `reference/edacs/report.py`.", "",
         "## 1. Source (evidence level C)", "",
         "Than et al., Emerg Med Australas 2014, Box 3 (`source/than2014.pdf`). Every age band is checked at",
         "both ends; other items are checked on a woman aged 30 (2 points).", "",
         "| Rule | Score | Expected |", "|---|---:|---:|"]
    L += [f"| {n} | {g:g} | {w} |" for n, g, w in rules]
    L += ["", f"{ok}/{len(rules)} rules give the published points.", "",
          "## 2. RiskScorescvd (evidence level B)", "",
          *[f"- {v}" for v in (FX / "VERSION").read_text().splitlines()], "",
          "RiskScorescvd has no known-CAD input: its +4 applies to >= 3 risk factors only. The fixtures",
          "therefore set known CAD to 0; the CAD rule is covered in section 1. 40% of ages are band edges.", "",
          f"{len(p)} random patients aged 18-99 (seed 2014): {same}/{len(p)} identical scores.", "",
          "## 3. Differences by design", "",
          "- Ages are taken in completed years (50.5 counts as 50); RiskScorescvd mixes `Age < 51` and `Age <= 50`.",
          "- Ages under 18 give NaN (the score is for adults); RiskScorescvd gives 0 age points.",
          "- The EDACS-ADP (score < 16, no new ischaemia, negative 0 and 2 h troponins) is left to the caller."]
    (HERE / "REPORT.md").write_text("\n".join(L) + "\n")
    print(f"rules {ok}/{len(rules)}; RiskScorescvd {same}/{len(p)}")
    assert ok == len(rules) and same == len(p)


if __name__ == "__main__":
    main()
