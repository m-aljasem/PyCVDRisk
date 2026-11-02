"""Build reference/qrisk2/REPORT.md. Run from repo root: python reference/qrisk2/report.py"""

from __future__ import annotations

import datetime as dt
import importlib
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

mod = importlib.import_module("cvd_risk.qrisk2.qrisk2")
HERE = Path(__file__).parent
FX = HERE / "fixtures"
FLAGS = ("af", "rheumatoid_arthritis", "ckd", "bp_treated", "family_history")
_spec = importlib.util.spec_from_file_location("q3r", HERE.parent / "qrisk3" / "report.py")
q3r = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(q3r)  # reuse leaves() and mutated()


def run(p):
    return mod.qrisk2(p.age, p.sex, p.ethnicity, p.smoking, p.bmi, p.chol_hdl_ratio, p.sbp,
                      townsend=p.townsend, diabetes=p.diabetes, **{f: p[f] for f in FLAGS})


def main():
    p = pd.read_csv(FX / "patients.csv")
    r = pd.read_csv(FX / "reference.csv").clinrisk_c
    na = r.isna()
    ok = lambda got: (np.isnan(got) == na).all() and np.allclose(got[~na], r[~na], rtol=1e-12, atol=1e-15)  # noqa: E731
    got = run(p)
    diff = float(np.max(np.abs(got[~na] / r[~na] - 1)))

    original = mod.MODELS
    caught = []
    for sex in ("female", "male"):
        for path, v in q3r.leaves(original[sex]):
            mod.MODELS = q3r.mutated(original, sex, path, v * (1 + 1e-6))
            try:
                caught.append((sex, path, not ok(run(p))))
            finally:
                mod.MODELS = original
    missed = [f"{s} {pth}" for s, pth, c in caught if not c]

    lines = [
        "# QRISK2-2014 — verification report", "",
        f"Generated {dt.date.today()} by `reference/qrisk2/report.py`.", "",
        "## 1. Source", "",
        "ClinRisk's open-source C implementation of QRISK2-2014 (LGPL-3.0-or-later; release",
        "packaged by Debian as `qrisk2`), saved unmodified in `source/` (female model",
        "`Q80_model_4_0.c`, male `Q80_model_4_1.c`, `utils.c`, headers, licence).",
        "`extract_coefficients.py` generates `src/cvd_risk/qrisk2/coefficients.py` from it with the",
        "QRISK3 parser, and a test re-parses the C to confirm the file is unchanged.", "",
        "## 2. Agreement with the reference (evidence level A)", "",
        "The same C code compiled unmodified, called through its public functions `cvd_female` /",
        "`cvd_male`, so ClinRisk's own input validation is part of the comparison.", "",
        "```", (FX / "VERSION").read_text().strip(), "```", "",
        f"{len(p):,} seeded synthetic patients: every whole age 23-86 for both sexes, every",
        "ethnicity, smoking and diabetes category, every flag, and continuous inputs running 5%",
        "past ClinRisk's validation ranges with 2% of rows at each exact bound.", "",
        f"- Rows ClinRisk rejects: {na.sum():,}; the port returns NaN for exactly these rows.",
        f"- Other rows: max relative difference {diff:.1e}.", "",
        "## 3. Mutation check", "",
        "Every non-zero number in the model (survivor, ethnicity and smoking terms, centring",
        f"means, coefficients) multiplied by 1 + 1e-6: {len(caught) - len(missed)}/{len(caught)} caught.",
        *([f"Missed: {', '.join(missed)}."] if missed else []), "",
        "## 4. Notes", "",
        "- QRISK2-2014's coefficients are not published in a paper (the 2008 BMJ paper describes",
        "  an earlier version), so the C release is the only primary source.",
        "- Age is truncated to whole years because ClinRisk's functions take an integer age.",
        "- Validation ranges are ClinRisk's (inclusive): age 25-84, BMI 20-40, cholesterol/HDL",
        "  ratio 1-12, SBP 70-210, Townsend -7 to 11.",
        "- Licence: like QRISK3, the module is LGPL-3.0-or-later and carries ClinRisk's disclaimer.",
    ]
    (HERE / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[12:]))


if __name__ == "__main__":
    main()
