"""Compare src/cvd_risk/framingham/coefficients.py with the printed sources.
Run from repo root: python reference/framingham/check_coefficients.py   (exit 1 on any mismatch)

Lab model: paper Table 2 (coefficients, S0) and the worked examples' mean sums.
BMI model: supplementary Table I (coefficients, S0). The two BMI-model means are not
printed in either PDF; they are checked against the spreadsheets' cached results by
tests/framingham (exact reproduction of every cached risk).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pypdf

from cvd_risk.framingham.coefficients import MODELS

SRC = Path(__file__).parent / "source"
LABELS = {"Log of age": "ln_age", "Log of total cholesterol": "ln_tc", "Log of HDL cholesterol": "ln_hdl",
          "Body Mass Index": "ln_bmi", "Log of SBP if not treated": "ln_sbp_untreated",
          "Log of SBP if treated": "ln_sbp_treated", "Smoking": "smoker", "Diabetes": "diabetes"}


def text(name):
    t = "\n".join(p.extract_text() for p in pypdf.PdfReader(SRC / name).pages)
    return t.replace("/H11002", "-").replace("/H11005", "=")


def table(t, start, stop):
    """{(sex, term): value} from a 'Women ... Men ...' coefficient table between two markers."""
    t = t[t.index(start):]
    t = t[:t.index(stop)]
    out, sex = {}, None
    for ln in t.splitlines():
        if ln.startswith(("Women", "Men")):
            sex = "female" if ln.startswith("Women") else "male"
            out[(sex, "s0")] = float(re.search(r"0\.\d{5}", ln).group())
        for label, key in LABELS.items():
            m = re.match(re.escape(label) + r"\s+(-?\s*\d+\.\d+)", ln, re.I)
            if m and sex:
                out[(sex, key)] = float(m.group(1).replace(" ", ""))
    return out


def main() -> int:
    paper = text("dagostino2008.pdf")
    printed = {("lab",) + k: v for k, v in table(paper, "Table 2. Regression", "So(10) indicates").items()}
    for sex, mean in (("female", r"26\.1931"), ("male", r"23\.9802")):
        assert re.search(r"=\s*" + mean + r"\.", paper.replace("/H11005", "=")), sex
        printed[("lab", sex, "mean")] = float(mean.replace("\\", ""))
    supp = text("dagostino2008_supplement.pdf")
    printed.update({("bmi",) + k: v for k, v in table(supp, "Supplementary Table I.", "*Estimated").items()})
    ours = {(m, s, k): v for m, by_sex in MODELS.items() for s, c in by_sex.items() for k, v in c.items()
            if not (m == "bmi" and k == "mean")}
    bad = sorted(set(printed.items()) ^ set(ours.items()))
    print(f"{len(printed)} printed values; {len(ours)} compared; mismatches: {len(bad)}")
    for b in bad:
        print("MISMATCH", b)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
