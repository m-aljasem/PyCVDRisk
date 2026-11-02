"""Build reference/smart_reach/REPORT.md. Run from repo root: python reference/smart_reach/report.py"""

from __future__ import annotations

import copy
import datetime as dt
import importlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
FX = HERE / "fixtures"
mod = importlib.import_module("cvd_risk.smart_reach.smart_reach")


def inputs(p):
    return dict(age=p.age, sex=p.sex, smoker=p.smoker, diabetes=p.diabetes, sbp=p.sbp, total_chol=p.total_chol,
                creatinine=p.creatinine, cad=p.cad, cerebrovascular=p.cerebrovascular, pad=p["pad"],  # p.pad is a method
                af=p.af, heart_failure=p.heart_failure, region=p.region)


def calculator_life_expectancy(p):
    """The calculator's own median rule (cells O28:O73, G76-G78) applied to our life table."""
    ok, _, _, _, g = mod._life_table(**inputs(p), max_age=89)
    m = 44 + np.sum(~((g > 0) & (g < 0.5)), axis=-1)  # 45 + COUNTIF(O28:O73, 0) - 1
    k = np.clip(m - 45, 0, g.shape[-1] - 2)[:, None]
    gm, gm1 = np.take_along_axis(g, k, -1)[:, 0], np.take_along_axis(g, k + 1, -1)[:, 0]
    return np.where(ok, np.where(m < 90, m + (gm - 0.5) / (gm - gm1), np.inf), np.nan), (g < 0).any(axis=-1)


def reference():
    r = pd.read_csv(FX / "reference.csv")
    return (pd.to_numeric(r.risk10, errors="coerce").to_numpy(),
            pd.to_numeric(r.life_expectancy.replace(">90", "inf"), errors="coerce").to_numpy())


def rel(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.max(np.abs(a[m] / b[m] - 1)))


def main():
    p = pd.read_csv(FX / "patients.csv")
    ref10, ref_le = reference()
    risk = mod.smart_reach(**inputs(p))
    le = mod.smart_reach(**inputs(p), outcome="life_expectancy")
    calc_le, neg = calculator_life_expectancy(p)
    same_nan10 = bool((np.isnan(risk) == np.isnan(ref10)).all())
    ok_rows = ~neg
    ok_le = np.isclose(le, ref_le, rtol=1e-12, atol=0) | (np.isinf(le) & np.isinf(ref_le))
    below_age = int(np.sum(neg & (ref_le < p.age)))

    ok, start, h, i, g = mod._life_table(**inputs(p), max_age=80)
    win = (mod._AGES >= start) & (mod._AGES < start + 11)
    neg10 = ok & ((g < 0) & win).any(axis=-1)

    chk = subprocess.run([sys.executable, str(HERE / "check_coefficients.py")], capture_output=True, text=True).stdout.strip()

    original = (mod.CVD, mod.NON_CVD)
    caught, total = 0, 0
    for which in (0, 1):
        for k, v in original[which].items():
            if v == 0:
                continue
            mutated = [copy.deepcopy(original[0]), copy.deepcopy(original[1])]
            mutated[which][k] = v * (1 + 1e-6)
            mod.CVD, mod.NON_CVD = mutated
            try:
                total += 1
                caught += not np.allclose(mod.smart_reach(**inputs(p)), ref10, rtol=1e-12, atol=0, equal_nan=True)
            finally:
                mod.CVD, mod.NON_CVD = original

    lines = [
        "# SMART-REACH — verification report", "",
        f"Generated {dt.date.today()} by `reference/smart_reach/report.py`.", "",
        "## 1. Sources", "",
        "Kaasenbrood et al., J Am Heart Assoc 2018;7:e009217, saved in `source/`: the paper, its",
        "supplement (Data S1; Tables S2-S3 print the model) and the authors' calculator spreadsheet",
        "(Data S2), which holds the coefficients to 7 significant digits. `extract_calculator.py` dumps",
        "the calculator's model formulas to `fixtures/calculator_formulas.txt`.", "",
        f"`check_coefficients.py`: {chk}", "",
        "- The coefficients are the calculator's. Each supplement value is the calculator's rounded to",
        "  4 decimals, except the CVD three-locations term: printed 0.2896, calculator 0.2895387.", "",
        "## 2. Agreement with the authors' calculator (evidence level A)", "",
        "The calculator was recalculated by LibreOffice Calc with its formulas unmodified",
        "(`run_reference.py`): each patient's codes go into the engine sheet's input cells, the cells",
        "the front sheet's drop-down controls normally fill. LibreOffice writes 15 significant digits.", "",
        "```", (FX / "VERSION").read_text().strip(), "```", "",
        f"{len(p)} seeded synthetic patients: every whole age 45-89, both sexes, all combinations of",
        "risk factors, disease locations and the three regions.", "",
        "| Output | Rows | Agreement |", "|---|---|---|",
        f"| 10-year risk (ages 45-80) | {np.isfinite(ref10).sum()} | max rel diff {rel(risk, ref10):.1e}; NaN on the same rows: {same_nan10} |",
        f"| Life expectancy, normal life tables | {ok_rows.sum()} | max rel diff {rel(le[ok_rows], ref_le[ok_rows]):.1e}; '>90' on the same rows: {bool((np.isinf(le) == np.isinf(ref_le))[ok_rows].all())} |",
        f"| Life expectancy, all rows, calculator's own median rule | {len(p)} | max rel diff {rel(calc_le, ref_le):.1e} |", "",
        "## 3. A defect in the calculator's life expectancy", "",
        "The two models were fitted separately, so for high-risk people the yearly CVD risk H and",
        "non-CVD death risk I can sum to more than 1 at older ages. Event-free survival",
        "G(t+1) = G(t)(1 - H - I) then turns negative. The calculator finds the median age by",
        "counting the table rows whose G is not in (0, 0.5) (`COUNTIF(O28:O73, 0)`), so negative",
        "rows late in the table are counted as if survival were still above 0.5 and the median is",
        "misplaced even when it was reached decades earlier.", "",
        f"- {neg.sum()}/{len(p)} synthetic patients ({neg.mean():.1%}) have such a table; for every one of",
        f"  them, and no other, our life expectancy differs from the calculator's. In {below_age} of them the",
        "  calculator's life expectancy is below the person's current age.",
        f"- `smart_reach` returns the first crossing of 0.5 instead. Applying the calculator's rule to",
        f"  our tables reproduces its output on all {len(p)} rows (third row of the table above), which",
        f"  confirms the diagnosis. {ok_le.sum()}/{len(p)} rows agree directly.",
        f"- The 10-year risk follows the calculator's arithmetic; {neg10.sum()} of {ok.sum()} patients have a",
        f"  negative G within their 10-year window (ages {sorted(set(p.age[neg10]))}).", "",
        "## 4. Mutation check", "",
        f"Every non-zero coefficient multiplied by 1 + 1e-6: {caught}/{total} change the 10-year risks",
        "beyond the 1e-12 tolerance.", "",
        "## 5. Scope", "",
        "- Outputs: 10-year risk of MI, stroke or CVD death (ages 45-80) and life expectancy free of",
        "  these events (ages 45-89). The calculator's treatment-effect module (statins, PCSK9",
        "  inhibitors, blood pressure lowering, antithrombotics, canakinumab) is not implemented.",
        "- Age in whole years, as in the calculator's table. Regions: the calculator's three",
        "  (Western Europe = reference, North America, the Netherlands).",
        "- A row with no disease location (CAD, cerebrovascular disease, PAD) is NaN: the model is",
        "  for people with established atherosclerotic disease (the calculator does compute it).",
    ]
    (HERE / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[12:]))


if __name__ == "__main__":
    main()
