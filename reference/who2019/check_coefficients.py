"""Check every coefficient in src/cvd_risk/who2019/coefficients.py against the PDFs.

Log HRs: appendix 1 Table 1.6 (printed to 7 decimals) must contain each value in
order. Baseline survival: main paper Table 2 prints 3 decimals; ours must round to it.
Run from repo root: python reference/who2019/check_coefficients.py <main paper pdf>
"""

import logging
import re
import sys
from pathlib import Path

import pypdf

from cvd_risk.who2019 import coefficients as C

HERE = Path(__file__).parent
ORDER = {"lab": ["age", "chol", "sbp", "diabetes", "smoker", "chol_age", "sbp_age", "diabetes_age", "smoker_age"],
         "nonlab": ["age", "bmi", "sbp", "smoker", "bmi_age", "sbp_age", "smoker_age"]}


def text(pdf):
    logging.disable(logging.CRITICAL)
    t = " ".join((p.extract_text() or "") for p in pypdf.PdfReader(pdf).pages)
    return re.sub(r"\s+", " ", t).replace("·", ".").replace("‐", "-")


def main(paper: Path):
    t = text(HERE / "source" / "kaptoge2019_supplementary_appendix.pdf")
    t = t[t.index("Table 1.6: Log hazard ratios"):]
    t = t[t.index("Laboratory-based model:"):t.index("* By region implies")]
    # Each row: label, men logHR, se, I2 (lo, hi), p, p, women logHR, se, I2 (lo, hi), p, p
    row = re.compile(r"(-?\d+\.\d+) (\d+\.\d+) \d+% \(\d+%, \d+%\) [\d.]+ [\d.]+ (-?\d+\.\d+) (\d+\.\d+) \d+% \(\d+%, \d+%\) [\d.]+ [\d.]+")
    found = [(float(m[1]), float(m[3])) for m in row.finditer(t)]
    ours = [(table[o]["male"][k], table[o]["female"][k])
            for name, table in (("lab", C.LAB), ("nonlab", C.NONLAB)) for o in ("mi", "stroke") for k in ORDER[name]]
    assert len(found) == len(ours) == 32, (len(found), len(ours))
    assert found == ours, [(a, b) for a, b in zip(found, ours) if a != b]
    p = text(paper)
    printed = [float(x) for x in re.findall(r"Baseline survival estimate at 10 years† (0\.\d{3})", p)] + \
              [float(x) for x in re.findall(r"Baseline survival estimate at 10 years† 0\.\d{3} \.\. (0\.\d{3})", p)]
    s0 = [(t_[o]["male"]["s0"], t_[o]["female"]["s0"]) for t_ in (C.LAB, C.NONLAB) for o in ("mi", "stroke")]
    men, women = printed[:4], printed[4:]
    assert len(men) == len(women) == 4, printed
    assert [round(m, 3) for m, _ in s0] == men and [round(w, 3) for _, w in s0] == women, (s0, printed)
    print("64 log HRs match appendix Table 1.6 exactly; 8 baseline survivals round to paper Table 2")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
