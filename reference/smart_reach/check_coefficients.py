"""Check src/cvd_risk/smart_reach/coefficients.py against both primary sources:
  1. the authors' calculator (fixtures/calculator_formulas.txt, from extract_calculator.py):
     every coefficient, region shift and baseline survival must be identical;
  2. the supplement (source/kaasenbrood2018_supplement.pdf, Tables S2 and S3): every printed
     value must equal the calculator's value rounded to the printed 4 decimals.
Run from repo root: python reference/smart_reach/check_coefficients.py (exit code 1 on any mismatch).
"""

from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path

import pypdf

HERE = Path(__file__).parent
C = importlib.import_module("cvd_risk.smart_reach.coefficients")
ORDER = {  # term order in the calculator cells D28 / E28 (region shifts last)
    "cvd": ("male", "smoker", "diabetes", "sbp", "sbp_sq", "tc", "tc_sq", "creatinine",
            "two_locations", "three_locations", "af", "chf"),
    "non_cvd": ("male", "smoker", "smoker_age", "diabetes", "sbp", "tc", "creatinine", "creatinine_sq",
                "two_locations", "three_locations", "af", "chf"),
}
NUM = r"([+\-]?)\s*(\d+\.\d+(?:e-?\d+)?)"
# The supplement prints 0.2896 for the three-locations CVD term; the calculator's 0.2895387 rounds to 0.2895.
KNOWN = {("cvd", "three_locations")}


def calculator():
    text = (HERE / "fixtures" / "calculator_formulas.txt").read_text()
    cells = dict(re.findall(r"^([A-Z]+\d+) = (.*)$", text, re.M))
    out = {}
    for name, cell in (("cvd", "D28"), ("non_cvd", "E28")):
        body = cells[cell].removeprefix("IF(Condition1+Condition2=2.0,").removesuffix(',"NA")')
        body = re.sub(r"IF\(\w+=\d\.0,1\.0,0\.0\)", "X", body).replace("A28", "AGE").replace("^2.0", "SQ")
        body = re.sub(r"([+\-])X\*AGE\*", r"X*AGE*\1", body)  # "-X*AGE*0.0486" carries its sign in front
        nums = [float(s + v) for s, v in re.findall(NUM, body)]
        keys = ORDER[name] + ("netherlands", "north_america")
        assert len(nums) == len(keys), (cell, nums)
        out[name] = dict(zip(keys, nums))
    rows = re.findall(r"^(\d+),([\d.]+),([\d.]+)$", text, re.M)
    out["baseline"] = {int(a): (float(b), float(c)) for a, b, c in rows}
    return out


def supplement():
    pages = pypdf.PdfReader(HERE / "source" / "kaasenbrood2018_supplement.pdf").pages
    t = re.sub(r"(\d\.) (\d)", r"\1\2", "\n".join(p.extract_text() for p in pages)).replace("–", "-")
    s2 = t[t.index("Table S2."):t.index("Table S3.")]
    s3 = t[t.index("Table S3."):t.index("Figure S1.")]
    out = {"baseline": {int(a): (float(b), float(c)) for a, b, c in re.findall(r"^(\d\d) (\d\.\d{4}) (\d\.\d{4})", s2, re.M)}}
    a_txt = s3[s3.index("A = "):s3.index("Non-cardiovascular")].removeprefix("A = ")
    b_txt = s3[s3.index("B = "):s3.index("¥Age")].removeprefix("B = ")
    for name, txt in (("cvd", a_txt), ("non_cvd", b_txt)):
        nums = [float(s + v) for s, v in re.findall(NUM, txt)]
        assert len(nums) == len(ORDER[name]), (name, nums)
        out[name] = dict(zip(ORDER[name], nums))
    m = re.search(r"Dutch.*?add\s*(-?[\d.]+) to A and ([\d.]+) to B.*?North American.*?add\s*([\d.]+) to A and ([\d.]+) to B", s3, re.S)
    out["regions"] = {"netherlands": (float(m[1]), float(m[2])), "north_america": (float(m[3]), float(m[4]))}
    return out


def main() -> int:
    calc, supp, bad = calculator(), supplement(), []
    ours = {"cvd": C.CVD, "non_cvd": C.NON_CVD}
    for name in ("cvd", "non_cvd"):
        for k in ORDER[name]:
            if ours[name][k] != calc[name][k]:
                bad.append(f"{name}.{k}: ours {ours[name][k]} vs calculator {calc[name][k]}")
            if round(calc[name][k], 4) != supp[name][k] and (name, k) not in KNOWN:
                bad.append(f"{name}.{k}: calculator {calc[name][k]} vs supplement {supp[name][k]}")
        zero = set(ours[name]) - set(ORDER[name])
        bad += [f"{name}.{k} should be 0" for k in zero if ours[name][k] != 0]
    for reg in ("netherlands", "north_america"):
        want = (calc["cvd"][reg], calc["non_cvd"][reg])
        if C.REGIONS[reg] != want or supp["regions"][reg] != want:
            bad.append(f"region {reg}: ours {C.REGIONS[reg]}, calculator {want}, supplement {supp['regions'][reg]}")
    if not (C.BASELINE == calc["baseline"] == supp["baseline"]) or len(C.BASELINE) != 45:
        bad.append("baseline survival table differs between ours, calculator and Table S2")
    n = sum(len(v) for v in ORDER.values())
    print(f"{n} coefficients, 2 region shifts, {len(C.BASELINE)} baseline rows checked; mismatches: {bad or 'none'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
