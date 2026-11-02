"""Build reference/score/REPORT.md. Run from repo root: python reference/score/report.py"""

from __future__ import annotations

import copy
import datetime as dt
import importlib
import sys
from decimal import Decimal
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import chart_check  # noqa: E402

C = importlib.import_module("cvd_risk.score.coefficients")


def last_digit(v):
    return float(Decimal(10) ** Decimal(repr(v)).as_tuple().exponent)


def main():
    reg, sex, sm, age, sbp, chol, v = chart_check.cells()
    ok = chart_check.consistent()
    lo, hi = chart_check.bounds()
    point = 100 * chart_check.mod.score(age, sex, chol, sbp, sm, reg)
    known = np.array([(r, s, m, a, b, c) in chart_check.KNOWN_ANOMALIES
                      for r, s, m, a, b, c in zip(reg, sex, sm, age, sbp, chol)])

    caught, total = 0, 0
    for k, base in C.BASELINE.items():
        for cause, (a, p) in base.items():
            for j, val in enumerate((a, p)):
                for sign in (1, -1):
                    b = copy.deepcopy(C.BASELINE)
                    pair = list(b[k][cause]); pair[j] = val + sign * last_digit(val); b[k][cause] = tuple(pair)
                    total += 1
                    caught += bool((~chart_check.consistent(baseline=b) & ~known).any())
    for cause, betas in C.BETA.items():
        for key, val in betas.items():
            for sign in (1, -1):
                be = copy.deepcopy(C.BETA)
                be[cause][key] = val + sign * last_digit(val)
                total += 1
                caught += bool((~chart_check.consistent(beta=be) & ~known).any())

    lines = [
        "# SCORE (2003) — verification report", "",
        f"Generated {dt.date.today()} by `reference/score/report.py`.", "",
        "## 1. Source", "",
        "Conroy et al., Eur Heart J 2003, Appendix A (Tables A and B), saved in `source/`.",
        "`check_coefficients.py` parses both tables from the PDF: all 22 values match.", "",
        "## 2. The official SCORE charts (evidence level A)", "",
        "The two total-cholesterol charts (Fig 1 high-risk, Fig 2 low-risk regions; 800 cells)",
        "were transcribed by eye from the PDF at 400-800 dpi (`fixtures/charts_transcribed.txt`).",
        "The paper prints alpha to one decimal and the betas to two or three, so the published",
        "function cannot reproduce the charts exactly: a change of 0.05 in alpha alone moves risk by",
        "about 5%. Each cell is therefore checked against the interval of risks the printed",
        "coefficients allow (each moved by up to half a unit of its last digit).", "",
        f"- Printed coefficients, rounded to whole %: {np.mean(np.round(point) == v):.1%} of cells exact,",
        f"  {np.mean(np.abs(np.round(point) - v) <= 1):.1%} within 1 point. The misses are all high-risk cells (men,",
        "  smokers, cholesterol 7-8) where the chart is lower, as expected from coefficient rounding.",
        f"- **{ok.sum()}/{len(v)} cells lie within the printed-precision interval** (median width",
        f"  {np.median(hi - lo):.2f} points). The two outside it are chart anomalies, not transcription errors:",
        "  - High risk, man, non-smoker, 65, SBP 160, cholesterol 8: printed **16**, shaded as >=15%,",
        "    in a row reading 9 11 13 15 16; the function gives 17.4-21.0. Almost certainly a typo for 18/19.",
        "  - High risk, woman, non-smoker, 50, SBP 120, cholesterol 6: printed 1; the function gives",
        "    0.40-0.46 for every admissible coefficient value.",
        f"- Sensitivity: moving any one of the 22 printed coefficients by one unit of its last digit",
        f"  pushes at least one other chart cell out of its interval in {caught}/{total} cases.", "",
        "## 3. Reference implementations", "",
        "No working third-party implementation was found. The only one on CRAN, `calculateScoreEur`",
        "in PBImisc 1.0, returns NA for every input (it uses `coefs[\"w\"]`, which does not exist,",
        "instead of the weighted sum `w`), and always adds the smoking coefficient regardless of",
        "`currentSmoker`. v1 of this package used coefficients that appear nowhere in the paper.", "",
        "## 4. Scope", "",
        "- Total-cholesterol model only. The cholesterol/HDL-ratio charts (Figs 3-4) use a function",
        "  whose coefficients are not published.",
        "- Ages 40-65, the range of the charts; the paper gives no other validity range.",
        "- The formula follows Appendix A literally: S0(t) = exp(-exp(alpha)(t - 20)^p), risk for each",
        "  cause = 1 - S(age + 10)/S(age), CVD risk = CHD risk + non-CHD risk (a sum, as printed).",
    ]
    (HERE / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[8:]))


if __name__ == "__main__":
    main()
