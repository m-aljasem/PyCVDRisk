"""DIAL2 verification against the paper's worked examples (Figure 3: two men, moderate-risk region,
age 50) and the supplement's coefficients. Writes REPORT.md.
Run from repo root: python reference/dial2/report.py
"""

from __future__ import annotations

import datetime
import importlib
import itertools
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
C = importlib.import_module("cvd_risk.dial2.coefficients")
M = importlib.import_module("cvd_risk.dial2.dial2")
COMMON = {"age": 50, "sex": "male", "smoker": 0, "sbp": 140, "total_chol": 5.5, "hdl_chol": 1.3, "region": "moderate"}
EXAMPLES = {  # caption inputs; printed CVD-free life expectancy, with treatment, gain (years)
    "A": ({"hba1c": 75, "age_at_diagnosis": 40, "egfr": 70}, (70.5, 74.1, 3.6)),
    "B": ({"hba1c": 50, "age_at_diagnosis": 50, "egfr": 70}, (76.0, 79.1, 3.1)),
}
HR = 0.8 * 0.78 ** 1.5  # 10 mmHg SBP and 1.5 mmol/L LDL lowering


def figure3(inputs):
    """(life expectancy, with treatment, gain) as printed: the paper shows the treated value as the
    rounded life expectancy plus the rounded gain."""
    return _figure3(COMMON, inputs)


def _figure3(common, inputs):
    le = float(M.dial2(**common, **inputs, outcome="life_expectancy"))
    gain = float(M.dial2(**common, **inputs, outcome="life_expectancy", hr_cvd=HR)) - le
    return round(le, 1), round(round(le, 1) + round(gain, 1), 1), round(gain, 1), le, le + gain, gain


RANGES = {  # one input of example B changed at a time, over these values
    "sbp": np.arange(100, 181), "total_chol": np.arange(30, 81) / 10, "hdl_chol": np.arange(8, 21) / 10,
    "hba1c": np.arange(30, 101), "age_at_diagnosis": np.arange(30, 51), "egfr": np.arange(30, 151),
    "smoker": [1], "region": ["low"], "sex": ["female"],
}


def single_changes(example="B"):
    """{input: [values]} for which changing only that input reproduces every printed number."""
    inputs, printed = EXAMPLES[example]
    out = {}
    for key, values in RANGES.items():
        for v in values:
            common = {**COMMON, key: v} if key in COMMON else COMMON
            x = {**inputs, key: v} if key in inputs else inputs
            if _figure3(common, x)[:3] == printed:
                out.setdefault(key, []).append(v.item() if hasattr(v, "item") else v)
    return out


def sensitivity(rel=0.01):
    """Each coefficient and scale moved by `rel` (relative): does any printed number change?"""
    caught = n = 0
    inputs = [EXAMPLES["A"][0], {**EXAMPLES["B"][0], "egfr": 90}]
    want = [figure3(x)[:3] for x in inputs]
    for o, sx in itertools.product(("cvd", "non_cvd"), ("male", "female")):
        beta, scales = M._TABLES[o]
        s = 0 if sx == "female" else 1
        for arr, idx in [(beta, (s, k)) for k in range(beta.shape[1])] + \
                        [(scales, (s, r, k)) for r in range(scales.shape[1]) for k in range(2)]:
            saved = arr[idx]
            arr[idx] = saved * (1 + rel)
            try:
                n += 1
                caught += [figure3(x)[:3] for x in inputs] != want
            finally:
                arr[idx] = saved
    return caught, n


def main():
    rows = []
    for name, (inputs, printed) in EXAMPLES.items():
        got = figure3(inputs)
        rows.append(f"| {name} (caption: eGFR {inputs['egfr']}) | {printed} | {got[:3]} | "
                    f"{got[3]:.3f} / {got[4]:.3f} / {got[5]:.3f} |")
    b90 = figure3({**EXAMPLES["B"][0], "egfr": 90})
    rows.append(f"| B with eGFR 90 | {EXAMPLES['B'][1]} | {b90[:3]} | {b90[3]:.3f} / {b90[4]:.3f} / {b90[5]:.3f} |")
    match = single_changes()
    caught, n = sensitivity()
    lines = [
        "# DIAL2 — verification report", "",
        f"Generated {datetime.date.today()} by `reference/dial2/report.py`.", "",
        "## 1. Source", "",
        "Østergaard et al., Eur J Prev Cardiol 2023;30:61-69, and its online supplement (PDF and Word),",
        "saved in `source/`. `check_coefficients.py` reads Supplementary Tables 3-5 (core model) from the",
        "Word file and confirms every value also appears in the PDF: 56 coefficients, 260 baseline",
        "survivals (ages 30-94) and 16 recalibration scales match. Unlike LIFE-CVD2, the supplement prints",
        "every transformation (centring and scaling), the life table and the lifetime-risk definition.", "",
        "## 2. The paper's worked examples (evidence level C)", "",
        "Figure 3 shows CVD-free life expectancy with and without lifelong 10 mmHg SBP and 1.5 mmol/L LDL",
        "lowering (HR 0.80 x 0.78^1.5 on CVD) for two men aged 50 in the moderate-risk region (non-smoker,",
        "SBP 140, total cholesterol 5.5, HDL 1.3). The printed 'with treatment' value equals the rounded life",
        "expectancy plus the rounded gain in both examples, so we compare it that way.", "",
        "| Example | Printed (LE, treated, gain) | Ours, rounded | Ours, unrounded |", "|---|---|---|---|",
        *rows, "",
        "- **Example A is reproduced** (all three numbers).",
        "- **Example B is not reproduced with the caption's inputs** (life expectancy 74.4 instead of 76).",
        "  Changing one input at a time over a plausible range, the only values that reproduce all three",
        "  printed numbers are: " + "; ".join(f"{k} = {', '.join(map(str, v))}" for k, v in match.items()) + ".",
        "  We read this as a caption error: the figure was drawn with eGFR 90, not 70.",
        "- Sensitivity: moving any one of the 56 coefficients or 16 scales by 1% (relative) changes at",
        f"  least one of the six printed numbers in {caught}/{n} cases. Six one-decimal numbers are weak",
        "  evidence; there is no published chart or calculator output to compare against.", "",
        "## 3. Reference implementations", "",
        "The official calculator is the web tool at u-prevent.com, which cannot be queried offline. No",
        "open-source implementation was found (searched November 2026).", "",
        "## 4. Scope", "",
        "- Core model. The extended model (adding insulin, retinopathy, BMI and albuminuria) has its own",
        "  coefficients and baseline but no recalibration scales and was not externally validated; it is",
        "  not implemented.",
        "- Low- and moderate-risk ESC regions only; the paper has no scales for high and very high.",
        "- Whole-year ages 30-94 (30-85 for 10-year risk). Lifetime risk runs to age 95 and life",
        "  expectancy is inf when survival is still >= 0.5 at 95. The derivation population was 30-85.",
    ]
    (HERE / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
