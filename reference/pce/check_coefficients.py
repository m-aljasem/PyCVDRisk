"""Compare src/cvd_risk/pce/coefficients.py with Table A of the source (PMC4700825 XML).
Run from repo root: python reference/pce/check_coefficients.py   (exit 1 on any mismatch)
"""

from __future__ import annotations

import html
import re
import sys
from pathlib import Path

from cvd_risk.pce.coefficients import MODELS

SRC = Path(__file__).parent / "source" / "goff2013_PMC4700825.xml"
LABELS = {  # row label in Table A (whitespace removed, lower case) -> our term
    "lnage(y)": "ln_age", "logage(y)": "ln_age", "lnage,squared": "ln_age_sq",
    "lntotalcholesterol(mg/dl)": "ln_tc", "logtotalcholesterol(mg/dl)": "ln_tc",
    "lnage×lntotalcholesterol": "ln_age_tc", "logage×logtotalcholesterol": "ln_age_tc",
    "lnhdl–c(mg/dl)": "ln_hdl", "loghdl–c(mg/dl)": "ln_hdl", "lnage×lnhdl–c": "ln_age_hdl",
    "logage×loghdl–c": "ln_age_hdl", "logtreatedsystolicbp(mmhg)": "ln_sbp_treated",
    "logage×logtreatedsystolicbp": "ln_age_sbp_treated", "loguntreatedsystolicbp(mmhg)": "ln_sbp_untreated",
    "logage×loguntreatedsystolicbp": "ln_age_sbp_untreated", "currentsmoker(1=yes,0=no)": "smoker",
    "logage×currentsmoker": "ln_age_smoker", "diabetes(1=yes,0=no)": "diabetes",
    "mean(coefficient×value)": "mean", "baselinesurvival": "s0",
}


def table_a() -> dict:
    x = SRC.read_text()
    tw = next(m.group(0) for m in re.finditer(r"<table-wrap\b.*?</table-wrap>", x, re.S) if "29.799" in m.group(0))
    rows = [[html.unescape(re.sub(r"<[^>]+>", "", c)) for c in re.findall(r"<t[dh]\b.*?</t[dh]>", r, re.S)]
            for r in re.findall(r"<tr\b.*?</tr>", tw, re.S)]
    out, sex = {}, None
    for r in rows:
        label = re.sub(r"\s+", "", r[0]).lower()
        if label.startswith(("women", "men")):
            sex = "female" if label.startswith("women") else "male"
        elif label in LABELS:
            v = [c.strip().replace("–", "-") for c in r[1:]]
            coef = (v[0], v[3]) if LABELS[label] not in ("mean", "s0") else (v[2], v[5])
            for race, c in zip(("white", "african_american"), coef):
                if c != "N/A":
                    out[(sex, race, LABELS[label])] = float(c)
    return out


def main() -> int:
    printed = table_a()
    ours = {(s, r, k): v for (s, r), m in MODELS.items() for k, v in m.items()}
    bad = sorted(set(printed.items()) ^ set(ours.items()))
    print(f"{len(printed)} values in Table A; {len(ours)} in coefficients.py; mismatches: {len(bad)}")
    for b in bad:
        print("MISMATCH", b)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
