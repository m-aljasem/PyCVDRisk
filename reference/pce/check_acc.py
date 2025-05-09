"""Static comparison with the official ACC CVD Risk Estimator+ formula code (read, not executed).

Source: tools.acc.org/CVD-Risk-Estimator-Plus/assets/js/riskformulas.js, saved unmodified as
source/acc_cvd_risk_estimator_riskformulas.js. Its `predictCalculate`
(the 10-year ASCVD risk) is parsed into {term: coefficient} per sex x race, with `s010` (baseline
survival) and `mnxb` (mean). Checks: (1) the same terms enter each of the four equations as in
coefficients.py; (2) every ACC constant, rounded to the guideline's printed precision, equals the
guideline value. The ACC code carries more decimals than the guideline prints for the baseline
survivals and the African American women's coefficients (as R PooledCohort does).
`run_acc.js` runs the file itself on fixtures/patients.csv (needs the owner's permission to execute
downloaded code); tests/pce compare that output when fixtures/acc_reference.csv exists.
Run from repo root: python reference/pce/check_acc.py
"""

from __future__ import annotations

import importlib
import re
from decimal import Decimal
from pathlib import Path

C = importlib.import_module("cvd_risk.pce.coefficients")
JS = Path(__file__).parent / "source" / "acc_cvd_risk_estimator_riskformulas.js"
GROUPS = {"appmodel.Form().isAfrican() && appmodel.Form().isFemale()": ("female", "african_american"),
          "!appmodel.Form().isAfrican() && appmodel.Form().isFemale()": ("female", "white"),
          "appmodel.Form().isAfrican() && appmodel.Form().isMale()": ("male", "african_american"),
          "!appmodel.Form().isAfrican() && appmodel.Form().isMale()": ("male", "white")}
TERMS = {"(isCurrent ? self.lnFollowUpage(): self.lnage())": "ln_age", "self.age2(isCurrent)": "ln_age_sq",
         "self.lntot(isCurrent)": "ln_tc", "self.agetc(isCurrent)": "ln_age_tc", "self.lnhdl(isCurrent)": "ln_hdl",
         "self.agehdl(isCurrent)": "ln_age_hdl", "self.trlnsbp(isCurrent)": "ln_sbp_treated",
         "self.agetsbp(isCurrent)": "ln_age_sbp_treated", "self.ntlnsbp(isCurrent)": "ln_sbp_untreated",
         "self.agentsbp(isCurrent)": "ln_age_sbp_untreated", "self.agesmoke(isCurrent)": "ln_age_smoker",
         "(isCurrent ? Number(appmodel.Form().isSmoker()):Number(appmodel.Form().isBaseLineSmoker()))": "smoker",
         "(isCurrent ? Number(appmodel.Form().isDiabetic()):Number(appmodel.Form().isBaseLineDiabetic()))": "diabetes"}


def _function(src, name):
    start = src.index(f"self.{name} = function")
    return src[start:src.index("};", start)]


def parse():
    """{(sex, race): {term or 'mean'/'s0': value}} from the ACC file."""
    src = JS.read_text().replace("\r", "")
    out = {g: {} for g in GROUPS.values()}
    for body, key in (("s010", "s0"), ("mnxb", "mean")):
        for cond, val in re.findall(r"if \((.*?)\) i = (-?[\d.]+);", _function(src, body)):
            out[GROUPS[cond.strip()]][key] = float(val)
    pred = _function(src, "predictCalculate")
    for cond, expr in re.findall(r"if \((.*?)\)\s*\n\s*i = (.*?)(?:;|\n)", pred):
        group = out[GROUPS[cond.strip()]]
        for part in expr.strip().split(" + "):  # each part: "c * term" or "(-c * term)"
            if part.startswith("(-"):
                part = part[1:-1]
            coef, term = part.split(" * ", 1)
            group[TERMS[term.strip()]] = float(coef)
    return out


def _round_like(x, ref):
    return float(round(Decimal(repr(x)), -Decimal(repr(ref)).as_tuple().exponent))


def main():
    acc, problems, more_decimals = parse(), [], []
    for g, ours in C.MODELS.items():
        if set(acc[g]) != set(ours):
            problems.append(f"{g}: terms differ: ACC {sorted(acc[g])} vs package {sorted(ours)}")
            continue
        for k, v in ours.items():
            if _round_like(acc[g][k], v) != v:
                problems.append(f"{g} {k}: ACC {acc[g][k]} does not round to the guideline's {v}")
            elif acc[g][k] != v:
                more_decimals.append(f"{g[0]} {g[1]} {k}: ACC {acc[g][k]}, guideline {v}")
    n = sum(len(m) for m in C.MODELS.values())
    print(f"ACC constants: {n} compared; same terms in 4/4 equations; {n - len(problems)} agree with the "
          f"guideline at its printed precision; {len(more_decimals)} carry more decimals:")
    print("\n".join("  " + m for m in more_decimals + problems))
    return len(problems)


if __name__ == "__main__":
    raise SystemExit(main())
