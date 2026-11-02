"""Build reference/heart/REPORT.md. Run from repo root: python reference/heart/report.py"""

from __future__ import annotations

import csv
import datetime as dt
from pathlib import Path

import numpy as np

from cvd_risk import heart

HERE = Path(__file__).parent
FX = HERE / "fixtures"


def load():
    p = list(csv.DictReader((FX / "patients.csv").open()))
    r = list(csv.DictReader((FX / "reference.csv").open()))
    col = lambda k: np.array([float(x[k]) for x in p])
    ours = heart(col("history"), col("ecg"), col("age"), col("risk_factors"), col("atherosclerosis"),
                 col("troponin_ratio"))
    ref = np.array([float(x["score"]) if x["score"] else np.nan for x in r])
    return p, ours, ref


# One row per rule in Melki & Jernberg 2013 (Table 1 and Methods): inputs -> expected points total
# with the other four items held at 0 (history 0, ECG 0, age 30, no risk factors, troponin ratio 0).
RULES = [
    ("history slightly suspicious", {"history": 0}, 0), ("history moderately suspicious", {"history": 1}, 1),
    ("history highly suspicious", {"history": 2}, 2), ("ECG normal", {"ecg": 0}, 0),
    ("ECG non-specific repolarisation", {"ecg": 1}, 1), ("ECG significant ST depression", {"ecg": 2}, 2),
    ("age 44 (<45)", {"age": 44}, 0), ("age 45", {"age": 45}, 1), ("age 64", {"age": 64}, 1),
    ("age 65", {"age": 65}, 2), ("1 risk factor", {"risk_factors": 1}, 1), ("2 risk factors", {"risk_factors": 2}, 1),
    ("3 risk factors", {"risk_factors": 3}, 2), ("atherosclerosis, no risk factors", {"atherosclerosis": 1}, 2),
    ("troponin below the limit", {"troponin_ratio": 0.99}, 0), ("troponin at the limit", {"troponin_ratio": 1}, 1),
    ("troponin 3x the limit", {"troponin_ratio": 3}, 1), ("troponin above 3x", {"troponin_ratio": 3.01}, 2),
]
BASE = {"history": 0, "ecg": 0, "age": 30, "risk_factors": 0, "atherosclerosis": 0, "troponin_ratio": 0}


def main():
    p, ours, ref = load()
    same = (ours == ref) | (np.isnan(ours) & np.isnan(ref))
    rules = [(name, float(heart(**{**BASE, **kw})), want) for name, kw, want in RULES]
    rules_ok = sum(got == want for _, got, want in rules)
    version = (FX / "VERSION").read_text().splitlines()
    L = [
        "# HEART score: verification report", "",
        f"Generated {dt.date.today()} by `reference/heart/report.py`.", "",
        "## 1. Source (evidence level C: the published rules)", "",
        "Six et al. (Neth Heart J 2008) define the score; their paper is not in the project library. The point",
        "table and its operational definitions are taken from Melki & Jernberg (Crit Pathw Cardiol 2013,",
        "Table 1 and Methods; `source/melki2013.pdf`), which cites www.heartscore.nl. Table 1 reads",
        "\"<=normal limit / 1-3x / >=3x\" for troponin and \"<=45 / 45-65 / >=65\" for age, which overlap at",
        "the cut points; the Methods resolve both (\"younger than 45\" 0, \"45-64\" 1, \"65 years or older\" 2;",
        "\"below the local decision limit\" 0, \"between 1 and 3 times\" 1, \"higher than 3 times\" 2).",
        "These are the rules implemented.", "",
        "| Rule | Points | Expected |", "|---|---:|---:|",
    ]
    L += [f"| {n} | {g:g} | {w} |" for n, g, w in rules]
    L += [
        "", f"{rules_ok}/{len(rules)} rules give the published points.", "",
        "## 2. RiskScorescvd (evidence level B)", "",
        *[f"- {v}" for v in version], "",
        "RiskScorescvd grades history from a count of typical symptoms (0-1, 2-4, 5-6) and troponin from",
        "hs-troponin I against fixed sex-specific limits (34 and 16 ng/L). `make_patients.py` therefore",
        "draws each patient in both forms: our history grade and a symptom count in the matching range,",
        "and our troponin ratio equal to its hs-troponin I divided by its limit. 30% of troponin values",
        "and 20% of ages sit on or next to a cut point.", "",
        f"{len(p)} random patients (seed 2008): {int(same.sum())}/{len(p)} identical scores.", "",
        "## 3. Differences by design", "",
        "- Ages are not restricted to whole years: 64.5 scores 1 here, while RiskScorescvd's `45 <= Age <= 64`",
        "  gives NA.",
        "- RiskScorescvd counts five risk factors; Melki also counts obesity and angina pectoris. Here the",
        "  caller passes the count.",
        "- RiskScorescvd scores an ECG given as all-zero flags as 1 point; here the ECG is the clinician's",
        "  0-2 grade.",
    ]
    (HERE / "REPORT.md").write_text("\n".join(L) + "\n")
    print(f"rules {rules_ok}/{len(rules)}; RiskScorescvd {int(same.sum())}/{len(p)}")
    assert rules_ok == len(rules) and same.all()


if __name__ == "__main__":
    main()
