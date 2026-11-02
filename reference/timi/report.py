"""Build reference/timi/REPORT.md. Run from repo root: python reference/timi/report.py"""

from __future__ import annotations

import csv
import datetime as dt
from pathlib import Path

import numpy as np

from cvd_risk import timi

HERE = Path(__file__).parent
FX = HERE / "fixtures"
BASE = {"age": 50, "risk_factors": 0, "coronary_stenosis": 0, "st_deviation": 0, "anginal_events_24h": 0,
        "aspirin_7d": 0, "elevated_markers": 0}
RULES = [  # Antman 2000, Table 1 (bold rows) and Results: one point per predictor present
    ("age 64", {"age": 64}, 0), ("age 65", {"age": 65}, 1), ("2 risk factors", {"risk_factors": 2}, 0),
    ("3 risk factors", {"risk_factors": 3}, 1), ("coronary stenosis >= 50%", {"coronary_stenosis": 1}, 1),
    ("ST deviation", {"st_deviation": 1}, 1), ("1 anginal event in 24 h", {"anginal_events_24h": 1}, 0),
    ("2 anginal events in 24 h", {"anginal_events_24h": 2}, 1), ("aspirin in last 7 days", {"aspirin_7d": 1}, 1),
    ("elevated cardiac markers", {"elevated_markers": 1}, 1),
    ("all seven", {"age": 70, "risk_factors": 5, "coronary_stenosis": 1, "st_deviation": 1,
                   "anginal_events_24h": 3, "aspirin_7d": 1, "elevated_markers": 1}, 7),
]


def load():
    p = list(csv.DictReader((FX / "patients.csv").open()))
    ref = np.array([float(r["score"]) for r in csv.DictReader((FX / "reference.csv").open())])
    col = lambda k: np.array([float(x[k]) for x in p])
    ours = timi(col("age"), col("risk_factors"), col("coronary_stenosis"), col("st_deviation"), col("episodes"),
                col("aspirin"), col("elevated_markers"))
    return p, ours, ref


def main():
    p, ours, ref = load()
    same = int((ours == ref).sum())
    rules = [(n, float(timi(**{**BASE, **kw})), w) for n, kw, w in RULES]
    rules_ok = sum(g == w for _, g, w in rules)
    L = ["# TIMI risk score for UA/NSTEMI: verification report", "",
         f"Generated {dt.date.today()} by `reference/timi/report.py`.", "",
         "## 1. Source (evidence level C)", "",
         "Antman et al., JAMA 2000 (`source/antman2000.pdf`): the 7 predictors that stayed significant in",
         "the multivariate model (Table 1, bold) score 1 point each, because their coefficients were of",
         "similar size. Risk factors for CAD (Table 1 footnote): family history of CAD, hypertension,",
         "hypercholesterolaemia, diabetes, current smoking.", "",
         "| Rule | Score | Expected |", "|---|---:|---:|"]
    L += [f"| {n} | {g:g} | {w} |" for n, g, w in rules]
    L += ["", f"{rules_ok}/{len(rules)} rules give the published points.", "",
          "## 2. RiskScorescvd (evidence level B)", "",
          *[f"- {v}" for v in (FX / "VERSION").read_text().splitlines()], "",
          "RiskScorescvd stands in previous PCI or CABG for known coronary stenosis, and hs-troponin I",
          "above fixed sex-specific limits (34 and 16 ng/L) for elevated markers. `make_patients.py` draws both",
          "forms consistently, with 30% of ages and troponins on or next to a cut point.", "",
          f"{len(p)} random patients (seed 2000): {same}/{len(p)} identical scores."]
    (HERE / "REPORT.md").write_text("\n".join(L) + "\n")
    print(f"rules {rules_ok}/{len(rules)}; RiskScorescvd {same}/{len(p)}")
    assert rules_ok == len(rules) and same == len(p)


if __name__ == "__main__":
    main()
