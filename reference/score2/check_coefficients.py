"""Check every SCORE2 / SCORE2-OP / SCORE2-Diabetes coefficient against the supplement text.
Run from repo root: python reference/score2/check_coefficients.py   (exit 1 on any mismatch)

Each value is located by its row label in the source table and compared as printed:
  SCORE2           updated supplement, Supplementary Methods Table 2 + rescaling table
  SCORE2-OP        supplement: worked example rows (Suppl. Methods Table 3, one per sex),
                   rescaling table, and the printed formulas 1 - S0^exp(LP - meanLP)
  SCORE2-Diabetes  supplement PDF, Log SHR table
"""

from __future__ import annotations

import re
import sys
import zipfile
from pathlib import Path

import pypdf

from cvd_risk.score2 import coefficients as C

SRC = Path(__file__).parent / "source"


def docx_rows(name):
    x = zipfile.ZipFile(SRC / name).read("word/document.xml").decode()
    t = re.sub(r"</w:tr>", "\n", re.sub(r"</w:tc>", " | ", x))
    t = re.sub(r"<[^>]+>", "", re.sub(r"</w:p>", " ", t))
    return t, [[c.strip() for c in r.split("|")] for r in t.splitlines()]


def num(s):
    return float(s.replace("–", "-").replace("−", "-"))


S2_LABELS = {"age": "Age (yrs)", "smoker": "Smoking (current", "sbp": "Systolic blood pressure",
             "diabetes": "Diabetes* (yes", "tc": "Total cholesterol (mmol", "hdl": "HDL cholesterol (mmol",
             "age_smoker": "Smoking x age", "age_sbp": "SBP x age", "age_tc": "Total cholesterol x age",
             "age_hdl": "HDL cholesterol x age", "age_diabetes": "Diabetes* x age", "s0": "Baseline survival"}
OP_LABELS = {"age": "Age (per year)", "diabetes": "Diabetes", "smoker": "Smoking",
             "sbp": "SBP (per mmHg)", "tc": "Total cholesterol (per", "hdl": "HDL cholesterol (per",
             "age_diabetes": "Diabetes interaction", "age_smoker": "Smoking interaction",
             "age_sbp": "SBP interaction", "age_tc": "Total cholesterol interaction",
             "age_hdl": "HDL cholesterol interaction"}
DM_LABELS = {"age": "Age (yrs)", "smoker": "Smoking (current", "sbp": "SBP (mm Hg)", "diabetes": "Diabetes (yes",
             "tc": "Total cholesterol (mmol", "hdl": "HDL cholesterol (mmol", "age_smoker": "Smoking interaction",
             "age_sbp": "SBP interaction", "age_diabetes": "Diabetes interaction", "age_tc": "TCHOL interaction",
             "age_hdl": "HDL interaction", "agediag": "if hxdiabbin=1", "hba1c": "HbA1c (mmol",
             "egfr": "Ln eGFR (ml", "egfr2": "Ln eGFR2", "age_hba1c": "HbA1c interaction",
             "age_egfr": "Ln eGFR interaction"}
REGION_ROWS = ("Low risk region", "Moderate risk region", "High risk region", "Very high risk region")


def scales(rows, first_label_row):
    """The four region rows following the rescaling table header, as ((m1, m2), (f1, f2))."""
    i = next(k for k, r in enumerate(rows) if r[0] == "Risk region" or (r[0] == "" and "Scale 1" in r))
    out = []
    for label in REGION_ROWS:
        r = next(r for r in rows[i:] if r[0] == label)
        out.append(tuple(num(v) for v in r[1:5]))
    return out


def main():
    bad, n = [], 0

    def check(what, ours, printed):
        nonlocal n
        n += 1
        if ours != printed:
            bad.append(f"{what}: ours {ours}, source {printed}")

    # SCORE2
    text, rows = docx_rows("score2_ehab309_updated_supplement.docx")
    t2 = rows[next(i for i, r in enumerate(rows) if "Table 2: Model coefficients" in r[0]):]
    for key, label in S2_LABELS.items():
        r = next(r for r in t2 if r[0].startswith(label))
        vals = [v for v in r[1:] if re.fullmatch(r"-?\d+\.\d+", v)]
        for j, sex in enumerate(("male", "female")):
            check(f"SCORE2 {sex} {key}", C.SCORE2[sex][key], num(vals[j]))
    for g, vals in enumerate(scales(rows, "Risk region")):
        for j, (sex, k) in enumerate((("male", 0), ("male", 1), ("female", 0), ("female", 1))):
            check(f"SCORE2 scale {sex} {REGION_ROWS[g]} {k + 1}", C.SCORE2_SCALES[sex][g][k], vals[j])

    # SCORE2-OP
    text, rows = docx_rows("score2_op_ehab312_supplement.docx")
    starts = {s: next(i for i, r in enumerate(rows) if f"Sex = {s}" in r[0]) for s in ("male", "female")}
    for sex, i in starts.items():
        for key, label in OP_LABELS.items():
            r = next(r for r in rows[i:i + 11] if r[1].startswith(label))  # first match: main effect
            check(f"SCORE2-OP {sex} {key}", C.SCORE2_OP[sex][key], num(r[2]))
    for sex, (s0, mlp) in {"male": ("0.7576", "0.0929"), "female": ("0.8082", "0.229")}.items():
        assert re.search(re.escape(s0) + r"\^?exp\S*\(βx-xcen-" + re.escape(mlp) + r"\)", text), sex
        check(f"SCORE2-OP {sex} s0", C.SCORE2_OP[sex]["s0"], num(s0))
        check(f"SCORE2-OP {sex} mean_lp", C.SCORE2_OP[sex]["mean_lp"], num(mlp))
    for g, vals in enumerate(scales(rows, "")):
        for j, (sex, k) in enumerate((("male", 0), ("male", 1), ("female", 0), ("female", 1))):
            check(f"SCORE2-OP scale {sex} {REGION_ROWS[g]} {k + 1}", C.SCORE2_OP_SCALES[sex][g][k], vals[j])

    # SCORE2-Diabetes
    lines = "\n".join(p.extract_text() for p in pypdf.PdfReader(SRC / "score2_diabetes_ehad260_supplement.pdf").pages)
    lines = lines.splitlines()
    i = next(k for k, ln in enumerate(lines) if ln.startswith("SCORE2 variables"))
    for key, label in DM_LABELS.items():
        ln = next(ln for ln in lines[i:i + 30] if label in ln)
        vals = re.findall(r"-?\d+\.\d+", ln)[-2:]
        for j, sex in enumerate(("male", "female")):
            check(f"SCORE2-Diabetes {sex} {key}", C.SCORE2_DIABETES[sex][key], num(vals[j]))

    print(f"{n - len(bad)}/{n} values match the source text")
    for b in bad:
        print("MISMATCH", b)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
