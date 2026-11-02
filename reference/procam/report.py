"""Build reference/procam/REPORT.md. Run from repo root: python reference/procam/report.py

Each band of Table 3 must occur in the PDF text and give its points at both ends (inputs given in
mg/dL and converted, so the unit round trip is tested at every edge). Table 4 is checked against the
PDF the same way.
"""

from __future__ import annotations

import datetime as dt
import logging
import re
from pathlib import Path

import pypdf

from cvd_risk import procam
from cvd_risk.prevent.prevent import MG_PER_MMOL
from cvd_risk.procam import RISK_PERCENT
from cvd_risk.procam.procam import TG_MG_PER_MMOL

HERE = Path(__file__).parent
logging.getLogger("pypdf").setLevel(logging.ERROR)
GLYPHS = {"/H11021": "<", "/H11350": "≥", "–": "-"}


def text():
    t = " ".join(p.extract_text() or "" for p in pypdf.PdfReader(HERE / "source" / "assmann2002.pdf").pages)
    for k, v in GLYPHS.items():
        t = t.replace(k, v)
    return re.sub(r"\s+", " ", t)


# all-zero man: age 35, LDL 90, HDL 60, TG 90 mg/dL, SBP 110, no smoking/diabetes/family history
BASE = {"age": 35, "sex": "male", "ldl_chol": 90, "hdl_chol": 60, "triglycerides": 90, "sbp": 110,
        "smoker": 0, "diabetes": 0, "family_history": 0}
TABLE3 = [
    ("age", "Age, y 35-39 0 40-44 6 45-49 11 50-54 16 55-59 21 60-65 26", [0, 6, 11, 16, 21, 26],
     [(35, 39.99), (40, 44.99), (45, 49.99), (50, 54.99), (55, 59.99), (60, 65.99)]),
    ("ldl_chol", "LDL cholesterol, mg/dL <100 0 100-129 5 130-159 10 160-189 14 ≥190 20", [0, 5, 10, 14, 20],
     [(50, 99.99), (100, 129.99), (130, 159.99), (160, 189.99), (190, 300)]),
    ("hdl_chol", "HDL cholesterol, mg/dL <35 11 35-44 8 45-54 5 ≥55 0", [11, 8, 5, 0],
     [(20, 34.99), (35, 44.99), (45, 54.99), (55, 100)]),
    ("triglycerides", "Triglycerides, mg/dL <100 0 100-149 2 150-199 3 ≥200 4", [0, 2, 3, 4],
     [(50, 99.99), (100, 149.99), (150, 199.99), (200, 500)]),
    ("smoker", "Smoker No 0 Yes 8", [0, 8], [(0, 0), (1, 1)]),
    ("diabetes", "Diabetes mellitus No 0 Yes 6", [0, 6], [(0, 0), (1, 1)]),
    ("family_history", "MI in family history No 0 Yes 4", [0, 4], [(0, 0), (1, 1)]),
    ("sbp", "Systolic blood pressure, mm Hg <120 0 120-129 2 130-139 3 140-159 5 ≥160 8", [0, 2, 3, 5, 8],
     [(100, 119.99), (120, 129.99), (130, 139.99), (140, 159.99), (160, 220)]),
]
FACTOR = {"ldl_chol": MG_PER_MMOL, "hdl_chol": MG_PER_MMOL, "triglycerides": TG_MG_PER_MMOL}


def score(**kw):
    return float(procam(**{k: v / FACTOR.get(k, 1) if k in FACTOR else v for k, v in kw.items()}))


def main():
    t = text()
    base = score(**BASE)
    rows, ok, n, found = [], 0, 0, 0
    for name, printed, pts, ranges in TABLE3:
        hit = printed in t
        found += hit
        for i, (p, (lo, hi)) in enumerate(zip(pts, ranges)):
            got = [score(**{**BASE, name: v}) - base for v in (lo, hi)]  # the base man scores 0 on every item
            n += 1
            ok += got == [p, p]
            note = f" \"{printed}\" {'found' if hit else 'NOT FOUND'}" if i == 0 else ""
            rows.append(f"| {name} | {lo}-{hi} | {got[0]:g} / {got[1]:g} | {p} |{note} |")
    t4 = " ".join(f"{k} {v:.1f}" for k, v in RISK_PERCENT.items())
    t4_found = t4 in t
    L = ["# PROCAM score: verification report", "", f"Generated {dt.date.today()} by `reference/procam/report.py`.", "",
         "Evidence level C: Assmann et al., Circulation 2002 (`source/assmann2002.pdf`), Table 3 (points)",
         "and Table 4 (points -> 10-year risk). The paper gives no worked example and no independent",
         "implementation was available. Inputs are entered in mg/dL and converted, so each band edge also",
         "tests the mmol/L round trip.", "",
         "| Input | Band (low-high, source units) | Points at low / high | Printed | |", "|---|---|---|---:|---|",
         *rows, "",
         f"{ok}/{n} bands give the printed points at both ends; {found}/{len(TABLE3)} printed rows found in the PDF.",
         f"Table 4 (points 21-59) found in the PDF as transcribed: {'yes' if t4_found else 'NO'}.", "",
         "Not implemented: the full Cox model (Table 2). The paper gives no example to check it against."]
    (HERE / "REPORT.md").write_text("\n".join(L) + "\n")
    print(f"bands {ok}/{n}; rows found {found}/{len(TABLE3)}; table 4 {t4_found}")
    assert ok == n and found == len(TABLE3) and t4_found


if __name__ == "__main__":
    main()
