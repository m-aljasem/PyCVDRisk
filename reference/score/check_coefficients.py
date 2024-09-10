"""Compare src/cvd_risk/score/coefficients.py with Appendix A, Tables A and B of the PDF.
Run from repo root: python reference/score/check_coefficients.py   (exit 1 on any mismatch)
"""

import re
import sys
from pathlib import Path

import pypdf

from cvd_risk.score.coefficients import BASELINE, BETA


def main() -> int:
    t = "\n".join(p.extract_text() for p in pypdf.PdfReader(Path(__file__).parent / "source" / "conroy2003.pdf").pages)
    t = t.replace("−", "-")
    t = t[t.index("Table A Coefficients for Eq. (1)"):]
    t = t[:t.index("Appendix B")]
    printed, region = {}, None
    for ln in t.splitlines():
        m = re.match(r"(Low risk|High risk)?\s*(Men|Women)\s+(-?\d+\.\d+)\s+(\d+\.\d+)\s+(-?\d+\.\d+)\s+(\d+\.\d+)", ln)
        if m:
            region = (m.group(1) or region or "").split()[0].lower() or region
            sex = "male" if m.group(2) == "Men" else "female"
            a1, p1, a2, p2 = map(float, m.groups()[2:])
            printed[(region, sex)] = {"chd": (a1, p1), "non_chd": (a2, p2)}
    rows = {"Current smoker": "smoker", "Cholesterol (mmol/L)": "chol", "Systolic BP (mmHg)": "sbp"}
    beta = {"chd": {}, "non_chd": {}}
    for ln in t.splitlines():
        for label, key in rows.items():
            if ln.startswith(label):
                a, b = re.findall(r"\d+\.\d+", ln[len(label):])
                beta["chd"][key], beta["non_chd"][key] = float(a), float(b)
    bad = [k for k in BASELINE if printed.get(k) != BASELINE[k]] + ([] if beta == BETA else ["beta"])
    print(f"{len(printed)} baseline rows, {sum(map(len, beta.values()))} betas parsed; mismatches: {bad or 'none'}")
    return 1 if bad or len(printed) != 4 else 0


if __name__ == "__main__":
    sys.exit(main())
