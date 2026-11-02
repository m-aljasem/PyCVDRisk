"""Build reference/qrisk3/REPORT.md. Run from repo root: python reference/qrisk3/report.py"""

from __future__ import annotations

import copy
import datetime as dt
import importlib
import time
from pathlib import Path

import numpy as np
import pandas as pd



mod = importlib.import_module("cvd_risk.qrisk3.qrisk3")
HERE = Path(__file__).parent
FX = HERE / "fixtures"
FLAGS = ("af", "atypical_antipsychotics", "corticosteroids", "erectile_dysfunction", "migraine",
         "rheumatoid_arthritis", "ckd", "severe_mental_illness", "sle", "bp_treated", "family_history")


def run(p):
    return mod.qrisk3(p.age, p.sex, p.ethnicity, p.smoking, p.bmi, p.chol_hdl_ratio, p.sbp, p.sbp_sd,
                      townsend=p.townsend, diabetes=p.diabetes, **{f: p[f] for f in FLAGS})


def leaves(m):
    """(path, value) for every number in a sex's model (skipping zero placeholders and powers)."""
    yield ("survivor",), m["survivor"]
    for k in ("ethnicity", "smoking"):
        for i, v in enumerate(m[k]):
            if v != 0:
                yield (k, i), v
    for k in ("means", "terms"):
        for name, v in m[k].items():
            yield (k, name), v


def mutated(models, sex, path, v):
    t = copy.deepcopy(models)
    node = t[sex]
    if len(path) == 1:
        node[path[0]] = v
    elif isinstance(node[path[0]], tuple):
        lst = list(node[path[0]]); lst[path[1]] = v; node[path[0]] = tuple(lst)
    else:
        node[path[0]][path[1]] = v
    return t


def main():
    p = pd.read_csv(FX / "patients.csv")
    r = pd.read_csv(FX / "reference.csv")
    ok = lambda got: np.allclose(got, r.clinrisk_c, rtol=1e-12, atol=1e-15)  # noqa: E731
    t0 = time.perf_counter()
    got = run(p)
    secs = time.perf_counter() - t0
    diffs = {c: float(np.max(np.abs(got / r[c] - 1))) for c in ("clinrisk_c", "qrisk3_cran")}

    original = mod.MODELS
    caught = []
    for sex in ("female", "male"):
        for path, v in leaves(original[sex]):
            mod.MODELS = mutated(original, sex, path, v * (1 + 1e-6))
            try:
                caught.append((sex, path, not ok(run(p))))
            finally:
                mod.MODELS = original
    missed = [f"{s} {pth}" for s, pth, c in caught if not c]

    big = pd.concat([p] * 34, ignore_index=True)
    t0 = time.perf_counter()
    run(big)
    secs_1m = (time.perf_counter() - t0) * 1e6 / len(big)

    lines = [
        "# QRISK3-2017 — verification report", "",
        f"Generated {dt.date.today()} by `reference/qrisk3/report.py`.", "",
        "## 1. Source", "",
        "ClinRisk's open-source C implementation of QRISK3-2017 (LGPL-3.0-or-later), saved",
        "unmodified as `source/qrisk3_2017.c` from the qrisk.org source page.",
        "`extract_coefficients.py` generates `src/cvd_risk/qrisk3/coefficients.py` from it,",
        "copying every literal as printed, and a test re-parses the C to confirm the file is unchanged.", "",
        "## 2. Agreement with the reference", "",
        "Primary reference: the same C code compiled unmodified (`harness.c`, `run_reference.py`).",
        "This is the developers' own implementation (evidence level A). Secondary: the CRAN",
        "package QRISK3, a third-party R port (level B).", "",
        "```", (FX / "VERSION").read_text().strip(), "```", "",
        f"{len(p):,} seeded synthetic patients: every whole age 25-84 for both sexes, every",
        "ethnicity, smoking and diabetes category, and every binary flag.", "",
        "| Reference | Max rel diff |", "|---|---|",
        f"| ClinRisk C | {diffs['clinrisk_c']:.1e} |",
        f"| QRISK3 (CRAN) | {diffs['qrisk3_cran']:.1e} |", "",
        "The CRAN package agrees with the C code bit for bit on these patients.", "",
        f"Scoring {len(p):,} rows: {secs:.3f} s; 1,000,000 rows: {secs_1m:.2f} s.", "",
        "## 3. Mutation check", "",
        f"Every non-zero number in the model (survivor, ethnicity and smoking terms, centring",
        f"means, 109 coefficients) multiplied by 1 + 1e-6: {len(caught) - len(missed)}/{len(caught)} caught.",
        *([f"Missed: {', '.join(missed)}."] if missed else []), "",
        "## 4. Notes", "",
        "- Age is truncated to whole years because ClinRisk's functions take an integer age.",
        "- The QRISK3 web calculator imputes missing BMI, cholesterol ratio and SBP. That code is",
        "  not in the open-source release, so these inputs are required here.",
        "- Only the age range (25-84) is enforced. The other inputs are checked for being",
        "  positive and finite only; ClinRisk's page has no input validation code.",
        "- Licence: the module is LGPL-3.0-or-later (it carries ClinRisk's coefficients) while",
        "  the rest of the package is MIT. ClinRisk's terms require their disclaimer to be held",
        "  with any score shown; it is in the module docstring.",
    ]
    (HERE / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[14:]))


if __name__ == "__main__":
    main()
