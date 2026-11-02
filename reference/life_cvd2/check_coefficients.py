"""Check src/cvd_risk/life_cvd2/coefficients.py against the supplement
(source/hageman2024_supplement.pdf): Table 3 coefficients (diabetes rows excluded, as the
supplement instructs), Table 4 baseline survival (ages 35-100) and Table 5 recalibration scales.
Run from repo root: python reference/life_cvd2/check_coefficients.py (exit code 1 on any mismatch).
"""

from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path

import pypdf

HERE = Path(__file__).parent
C = importlib.import_module("cvd_risk.life_cvd2.coefficients")
LABELS = {  # Table 3 row label -> term
    "Age (per 5 years)": "age", "Current smoking": "smoker", "SBP (per 20mmHg)": "sbp",
    "Total cholesterol (per 1 mmol/L)": "tc", "HDL cholesterol (per 0.5 mmol/L)": "hdl",
    "Current smoking* age (per 5 year)": "smoker_age", "SBP (per 20mmHg) * age (per 5 year)": "sbp_age",
    "Total cholesterol (per 1 mmol/L) * age (per 5 year)": "tc_age",
    "HDL cholesterol (per 0.5 mmol/L) * age (per 5 year)": "hdl_age",
}
NUM = r"-?\d+\.\d+"


def supplement():
    t = "\n".join(p.extract_text() for p in pypdf.PdfReader(HERE / "source" / "hageman2024_supplement.pdf").pages)
    t3 = t[t.index("Supplementary Table 3:"):t.index("Supplementary Table 4:")]
    t4 = t[t.index("Supplementary Table 4:"):t.index("Supplementary Table 5:")]
    t5 = t[t.index("Supplementary Table 5:"):t.index("Supplementary Table 6:")]
    beta = {}
    for ln in t3.splitlines():
        m = re.match(rf"(.*?) ({NUM}) ({NUM}) ({NUM}) ({NUM})\s*$", ln)
        if m and m[1] in LABELS:
            beta[LABELS[m[1]]] = [float(v) for v in m.groups()[1:]]
    assert set(beta) == set(LABELS.values()), beta
    rows = re.findall(r"^(\d{2,3}) (\d\.\d{5}) (\d\.\d{5}) (\d\.\d{5}) (\d\.\d{5})", t4, re.MULTILINE)
    baseline = {int(a): ((float(b), float(c)), (float(d), float(e))) for a, b, c, d, e in rows}
    scales = {}
    for outcome, block in (("cvd", t5[t5.index("CVD events"):t5.index("Non-CVD mortality")]),
                           ("non_cvd", t5[t5.index("Non-CVD mortality"):])):
        for region in ("very_high", "low", "moderate", "high"):
            m = re.search(rf"^{region.replace('_', ' ').capitalize()} risk region ({NUM}) ({NUM}) ({NUM}) ({NUM})",
                          block, re.MULTILINE)
            v = [float(x) for x in m.groups()]
            scales.setdefault(outcome, {})[region] = {"male": (v[0], v[1]), "female": (v[2], v[3])}
    return beta, baseline, scales


def main() -> int:
    beta, baseline, scales = supplement()
    bad = []
    for k, term in enumerate(C.TERMS):
        want = beta[term]  # men CVD, men non-CVD, women CVD, women non-CVD
        ours = [C.BETA["male"]["cvd"][k], C.BETA["male"]["non_cvd"][k],
                C.BETA["female"]["cvd"][k], C.BETA["female"]["non_cvd"][k]]
        if ours != want:
            bad.append(f"{term}: ours {ours} vs Table 3 {want}")
    if C.BASELINE != baseline or sorted(baseline) != list(range(35, 101)):
        bad.append("baseline survival differs from Table 4")
    if C.SCALES != scales:
        bad.append(f"scales differ from Table 5: {scales}")
    print(f"{4 * len(C.TERMS)} coefficients, {len(baseline)} baseline rows x 4, "
          f"{sum(len(r) * 2 for r in scales.values())} region scales x 2 checked; mismatches: {bad or 'none'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
