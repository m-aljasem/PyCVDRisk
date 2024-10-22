"""Check src/cvd_risk/dial2/coefficients.py against the supplement: Supplementary Table 3 (core
model), Table 4 (core baseline survival, ages 30-94) and Table 5 (recalibration scales), read from
the authors' Word file (source/ostergaard2023_supplement.docx, whose tables are cleanly structured).
Every number must also appear in the PDF version of the supplement.
Run from repo root: python reference/dial2/check_coefficients.py (exit code 1 on any mismatch).
"""

from __future__ import annotations

import importlib
import re
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import pypdf

HERE = Path(__file__).parent
C = importlib.import_module("cvd_risk.dial2.coefficients")
W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
LABELS = {  # Table 3 transformed variable -> term
    "-": "smoker", "(sbp-138)/17": "sbp", "(log(egfr)-4.4)/0.26": "egfr", "(chol-5.1)/1.1": "tc",
    "(hdl-1.3)/0.4": "hdl", "(hba1c-55)/16": "hba1c", "(age.diab-58)/12": "age_diab",
    "((age-63)/12)*smoking": "smoker_age", "((age-63)/12)*( (sbp-138)/17)": "sbp_age",
    "((age-63)/12)*( (log(egfr)-4.4)/0.26)": "egfr_age", "((age-63)/12)*( (chol-5.1)/1.1)": "tc_age",
    "((age-63)/12)*( (hdl-1.3)/0.4)": "hdl_age", "((age-63)/12)*( (hba1c-55)/16)": "hba1c_age",
    "((age-63)/12)*( (age.diab-58)/12)": "age_diab_age",
}


def docx_tables():
    body = ET.fromstring(zipfile.ZipFile(HERE / "source" / "ostergaard2023_supplement.docx").read("word/document.xml"))
    return [[["".join(t.text or "" for t in tc.iter(W + "t")).strip() for tc in tr.iter(W + "tc")]
             for tr in tbl.iter(W + "tr")] for tbl in body.iter(W + "tbl")]


def supplement():
    tables = docx_tables()
    beta = {LABELS[r[1]]: r[2:] for r in tables[2] if len(r) == 6 and r[1] in LABELS}  # Table 3, core model
    baseline = {int(r[0]): r[1:] for r in tables[4] if r[0].isdigit()}  # Table 4, core model
    scales, outcome = {}, None
    for r in tables[6]:  # Table 5
        if r[0] in ("CVD events", "Non-CVD mortality"):
            outcome = "cvd" if r[0] == "CVD events" else "non_cvd"
        elif r[0].endswith("risk region"):
            scales.setdefault(outcome, {})[r[0].split()[0].lower()] = r[1:]
    return beta, baseline, scales


def main() -> int:
    beta, baseline, scales = supplement()
    pdf = re.sub(r"\s+", "", "".join(p.extract_text() for p in
                                     pypdf.PdfReader(HERE / "source" / "ostergaard2023_supplement.pdf").pages))
    bad = []
    for k, term in enumerate(C.TERMS):
        ours = [C.BETA["male"]["cvd"][k], C.BETA["female"]["cvd"][k],
                C.BETA["male"]["non_cvd"][k], C.BETA["female"]["non_cvd"][k]]
        if ours != [float(v) for v in beta[term]]:
            bad.append(f"{term}: ours {ours} vs Table 3 {beta[term]}")
    want = {a: ((float(v[0]), float(v[1])), (float(v[2]), float(v[3]))) for a, v in baseline.items()}
    if C.BASELINE != want or sorted(want) != list(range(30, 95)):
        bad.append("baseline survival differs from Table 4")
    want = {o: {r: {"male": (float(v[0]), float(v[1])), "female": (float(v[2]), float(v[3]))}
                for r, v in regs.items()} for o, regs in scales.items()}
    if C.SCALES != want:
        bad.append(f"scales differ from Table 5: {want}")
    strings = [v for vs in beta.values() for v in vs] + [v for vs in baseline.values() for v in vs] \
        + [v for regs in scales.values() for vs in regs.values() for v in vs]
    missing = [v for v in strings if v.lstrip("-") not in pdf]
    if missing:
        bad.append(f"{len(missing)} docx values not in the PDF: {missing[:5]}")
    print(f"{4 * len(C.TERMS)} coefficients, {len(want and baseline)} baseline rows x 4, 8 region scales x 2 checked "
          f"(docx; {len(strings) - len(missing)}/{len(strings)} also found in the PDF); mismatches: {bad or 'none'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
