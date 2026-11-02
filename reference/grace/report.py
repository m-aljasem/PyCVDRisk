"""Build reference/grace/REPORT.md. Run from repo root: python reference/grace/report.py

Each band of both nomograms is checked twice: its printed labels and points must occur in the PDF
text (so the transcription cannot drift), and the function must give those points at both ends of
the band. Then Granger's two worked examples.
"""

from __future__ import annotations

import datetime as dt
import logging
import re
from pathlib import Path

import pypdf

from cvd_risk import grace_6month, grace_hospital

HERE = Path(__file__).parent
logging.getLogger("pypdf").setLevel(logging.ERROR)


def text(name):
    return re.sub(r"\s+", " ", " ".join(p.extract_text() or "" for p in pypdf.PdfReader(HERE / "source" / name).pages))


H_BASE = {"killip": 1, "sbp": 250, "heart_rate": 40, "age": 20, "creatinine": 0.2, "cardiac_arrest": 0,
          "st_deviation": 0, "elevated_enzymes": 0}  # 0+0+0+0+1+0+0+0
M_BASE = {"age": 20, "chf_history": 0, "mi_history": 0, "heart_rate": 40, "sbp": 250, "st_depression": 0,
          "creatinine": 0.2, "elevated_enzymes": 0, "in_hospital_pci": 1}  # 1 point
# (input, printed text in PDF order, points, (low, high) inputs inside each band, points of the base value)
HOSPITAL = [
    ("killip", "I 0 II 20 III 39 IV 59", [0, 20, 39, 59], [(1, 1), (2, 2), (3, 3), (4, 4)], 0),
    ("sbp", "≤80 58 80-99 53 100-119 43 120-139 34 140-159 24 160-199 10 ≥200 0", [58, 53, 43, 34, 24, 10, 0],
     [(0, 80), (80.1, 99.9), (100, 119.9), (120, 139.9), (140, 159.9), (160, 199.9), (200, 300)], 0),
    ("heart_rate", "≤50 0 50-69 3 70-89 9 90-109 15 110-149 24 150-199 38 ≥200 46", [0, 3, 9, 15, 24, 38, 46],
     [(0, 50), (50.1, 69.9), (70, 89.9), (90, 109.9), (110, 149.9), (150, 199.9), (200, 300)], 0),
    ("age", "≤30 0 30-39 8 40-49 25 50-59 41 60-69 58 70-79 75 80-89 91 ≥90 100", [0, 8, 25, 41, 58, 75, 91, 100],
     [(18, 30), (30.1, 39.9), (40, 49.9), (50, 59.9), (60, 69.9), (70, 79.9), (80, 89.9), (90, 110)], 0),
    ("creatinine", "0-0.39 1 0.40-0.79 4 0.80-1.19 7 1.20-1.59 10 1.60-1.99 13 2.00-3.99 21 >4.0 28",
     [1, 4, 7, 10, 13, 21, 28], [(0, 0.39), (0.4, 0.79), (0.8, 1.19), (1.2, 1.59), (1.6, 1.99), (2, 3.99), (4.01, 9)], 1),
    ("cardiac_arrest", "Cardiac Arrest at Admission 39", [0, 39], [(0, 0), (1, 1)], 0),
    ("st_deviation", "ST-Segment Deviation 28", [0, 28], [(0, 0), (1, 1)], 0),
    ("elevated_enzymes", "Elevated Cardiac Enzyme Levels 14", [0, 14], [(0, 0), (1, 1)], 0),
]
SIX_MONTH = [
    ("age", "≤29 30–39 40–49 50–59 60–69 70–79 80–89 ≥90 0 0 18 36 55 73 91 100", [0, 0, 18, 36, 55, 73, 91, 100],
     [(18, 29.9), (30, 39.9), (40, 49.9), (50, 59.9), (60, 69.9), (70, 79.9), (80, 89.9), (90, 110)], 0),
    ("chf_history", "Heart Failure 24", [0, 24], [(0, 0), (1, 1)], 0),
    ("mi_history", "Myocardial Infarction 12", [0, 12], [(0, 0), (1, 1)], 0),
    ("heart_rate", "≤49.9 50–69.9 70–89.9 90–109.9 110–149.9 150–199.9 ≥200 0 3 9 14 23 35 43",
     [0, 3, 9, 14, 23, 35, 43], [(0, 49.9), (50, 69.9), (70, 89.9), (90, 109.9), (110, 149.9), (150, 199.9), (200, 300)], 0),
    ("sbp", "≤79.9 80–99.9 100–119.9 120–139.9 140–159.9 160–199.9 ≥200 24 22 18 14 10 4 0",
     [24, 22, 18, 14, 10, 4, 0], [(0, 79.9), (80, 99.9), (100, 119.9), (120, 139.9), (140, 159.9), (160, 199.9), (200, 300)], 0),
    ("st_depression", "11ST-Segment Depression", [0, 11], [(0, 0), (1, 1)], 0),
    ("creatinine", "0–0.39 0.4–0.79 0.8–1.19 1.2–1.59 1.6–1.99 2–3.99 ≥4 1 3 5 7 9 15 20", [1, 3, 5, 7, 9, 15, 20],
     [(0, 0.39), (0.4, 0.79), (0.8, 1.19), (1.2, 1.59), (1.6, 1.99), (2, 3.99), (4, 9)], 1),
    ("elevated_enzymes", "15Elevated Cardiac Enzymes", [0, 15], [(0, 0), (1, 1)], 0),
    ("in_hospital_pci", "Coronary Invervention 14", [14, 0], [(0, 0), (1, 1)], 0),
]


def check(fn, base, table, pdf):
    ref = float(fn(**base))
    rows, ok, n, found = [], 0, 0, 0
    for name, printed, pts, ranges, base_pts in table:
        hit = printed in pdf
        found += hit
        for p, (lo, hi) in zip(pts, ranges):
            got = [float(fn(**{**base, name: v})) - ref + base_pts for v in (lo, hi)]
            n += 1
            ok += got == [p, p]
            rows.append(f"| {name} | {lo}-{hi} | {got[0]:g} / {got[1]:g} | {p} |")
        rows[-len(pts)] += f" printed: \"{printed}\" {'found' if hit else 'NOT FOUND'} |"
    return rows, ok, n, found


def main():
    g, e = text("granger2003.pdf"), text("eagle2004.pdf")
    hrows, hok, hn, hf = check(grace_hospital, H_BASE, HOSPITAL, g)
    mrows, mok, mn, mf = check(grace_6month, M_BASE, SIX_MONTH, e)
    ex1 = float(grace_hospital(2, 100, 100, 65, 1, 0, 1, 1))
    ex2 = float(grace_hospital(1, 80, 60, 55, 0.4, 0, 0, 0))
    alt = (float(grace_hospital(2, 99.99, 100, 65, 1, 0, 1, 1)), float(grace_hospital(1, 80, 60, 55, 0.39, 0, 0, 0)))
    quotes = ["His score would be: 20 + 53 + 15 + 58 + 7 + 0 + 28 + 14 = 196", "0 + 58 + 3 + 41 + 1 = 103"]
    qf = sum(q in g for q in quotes)
    head = ["| Input | Band (low-high input) | Points at low / high | Printed |  |", "|---|---|---|---:|---|"]
    L = ["# GRACE (in-hospital and 6-month post-discharge): verification report", "",
         f"Generated {dt.date.today()} by `reference/grace/report.py`.", "",
         "Evidence level C: the published nomograms. No independent implementation exists for them:",
         "RiskScorescvd's `GRACE()` is a different, continuous (interpolated) version with other weights",
         "(Killip 0/15/29/44), so it is not a reference for either paper.", "",
         "## 1. In-hospital death (Granger 2003, Figure 4)", "", *head, *hrows, "",
         f"{hok}/{hn} bands give the printed points at both ends; {hf}/{len(HOSPITAL)} printed rows found in the PDF.",
         "", "### Worked examples", "",
         "| Example (Figure 4) | Printed | This implementation |", "|---|---:|---:|",
         f"| Killip II, SBP 100, HR 100, age 65, creatinine 1, ST deviation, enzymes | 196 | {ex1:g} |",
         f"| Killip I, SBP 80, HR 60, age 55, creatinine 0.4 | 103 | {ex2:g} |", "",
         f"The examples are quoted from the PDF ({qf}/2 found). They do not follow the printed table:",
         "example 1 gives SBP 100 53 points (table: 100-119 = 43) and its terms add to 195, not 196;",
         "example 2 gives creatinine 0.4 1 point (table: 0.40-0.79 = 4). Both examples put a value on a",
         f"band's printed lower edge into the band below. With that reading they give {alt[0]:g} and {alt[1]:g}.",
         "We follow the table, whose 0.40 is explicit; this is a note for the authors.", "",
         "## 2. 6-month post-discharge death (Eagle 2004, Figure 2)", "", *head, *mrows, "",
         f"{mok}/{mn} bands give the printed points at both ends; {mf}/{len(SIX_MONTH)} printed rows found in the PDF.",
         "Figure 2 also prints a stray \"1\" under SBP >=200; it belongs to no row and is ignored.",
         "The paper gives no worked example."]
    (HERE / "REPORT.md").write_text("\n".join(L) + "\n")
    print(f"hospital {hok}/{hn} found {hf}/{len(HOSPITAL)}; 6-month {mok}/{mn} found {mf}/{len(SIX_MONTH)}; "
          f"examples {ex1:g} {ex2:g} quotes {qf}/2")
    assert hok == hn and mok == mn and hf == len(HOSPITAL) and mf == len(SIX_MONTH) and qf == 2
    assert (ex1, ex2) == (185, 106) and alt == (195, 103)


if __name__ == "__main__":
    main()
