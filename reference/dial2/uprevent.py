"""DIAL2 against the U-Prevent calculator (u-prevent.com), filled in by hand or by a browser assistant.

    python reference/dial2/uprevent.py patients   # writes fixtures/uprevent_patients.json and
                                                  # fixtures/uprevent_prompt.md (instructions to paste)
    python reference/dial2/uprevent.py compare    # reads fixtures/uprevent_results.jsonl and prints
                                                  # the package's values next to U-Prevent's

U-Prevent shows risks with one decimal and life expectancy in whole years, apparently rounded to two
decimals first (shown()); agreement is judged at that display precision.
"""

from __future__ import annotations

import json
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np

from cvd_risk import dial2

FX = Path(__file__).parent / "fixtures"
PATIENTS, PROMPT, RESULTS = FX / "uprevent_patients.json", FX / "uprevent_prompt.md", FX / "uprevent_results.jsonl"
FIELDS = ("age", "sex", "smoker", "sbp", "total_chol", "hdl_chol", "hba1c", "egfr", "age_at_diagnosis", "region")


def patients(n=20, seed=2023):
    g = np.random.default_rng(seed)
    out = []
    for i in range(n):
        age = int(g.integers(40, 80))
        out.append(dict(id=i, age=age, sex=("male", "female")[i % 2], smoker=bool(g.random() < 0.3),
                        sbp=int(g.integers(110, 181)), total_chol=round(float(g.uniform(3.5, 7.5)), 1),
                        hdl_chol=round(float(g.uniform(0.8, 2.0)), 1), hba1c=int(g.integers(42, 86)),
                        egfr=int(g.integers(45, 106)), age_at_diagnosis=age - int(g.integers(0, 16)),
                        region=("low", "moderate")[(i // 2) % 2]))
    return out


def prompt(ps):
    rows = "\n".join(json.dumps(p) for p in ps)
    return f"""# Task: read DIAL2 results from U-Prevent for {len(ps)} patients

On the U-Prevent DIAL2 calculator page, for each patient below:

1. Enter exactly these values. Units: SBP mmHg; total and HDL cholesterol mmol/L; HbA1c mmol/mol;
   eGFR mL/min/1.73 m2; age_at_diagnosis = age when type 2 diabetes was diagnosed; region = the
   ESC risk region (pick a country of that region, e.g. low: Netherlands, moderate: Germany).
   Leave every other field (BMI, albuminuria, insulin, retinopathy, treatment goals) empty or at
   its default; write down any field that is required and not listed.
2. Copy the results exactly as displayed, with the shown decimals: 10-year CVD risk, lifetime CVD
   risk, and CVD-free life expectancy.
3. Output ONE line of JSON per patient, nothing else:
   {{"id": 0, "risk10": "12%", "lifetime_risk": "45%", "life_expectancy": "81", "country": "Netherlands", "notes": ""}}
   Keep values as the displayed strings. Put anything unusual (a field you had to fill, a warning,
   a value out of range) in "notes".

Patients (one JSON object per line):

{rows}

When done, paste all output lines into reference/dial2/fixtures/uprevent_results.jsonl.
"""


def shown(v, q):
    """U-Prevent's display: rounded half-up to 2 decimals, then to the shown precision (9.448 -> 9.45 -> 9.5)."""
    return Decimal(str(v)).quantize(Decimal("0.01"), ROUND_HALF_UP).quantize(Decimal(q), ROUND_HALF_UP)


def compare():
    """(id, outcome, U-Prevent's string, package value, agrees) for every value in RESULTS."""
    ps = {p["id"]: p for p in json.loads(PATIENTS.read_text())}
    out = []
    for ln in RESULTS.read_text().splitlines():
        if ln.strip():
            r = json.loads(ln)
            for o, q, scale in (("risk10", "0.1", 100), ("lifetime_risk", "0.1", 100), ("life_expectancy", "1", 1)):
                v = scale * float(dial2(**{k: ps[r["id"]][k] for k in FIELDS}, outcome=o))
                out.append((r["id"], o, r[o], v, shown(v, q) == Decimal(r[o].rstrip("%"))))
    return out


if __name__ == "__main__":
    if sys.argv[1:] == ["patients"]:
        FX.mkdir(exist_ok=True)
        ps = patients()
        PATIENTS.write_text(json.dumps(ps, indent=1) + "\n")
        PROMPT.write_text(prompt(ps))
        print(f"wrote {PATIENTS} and {PROMPT}")
    elif sys.argv[1:] == ["compare"]:
        for row in compare():
            print(*row[:3], f"{row[3]:.3f}", "" if row[4] else "DIFFERENT")
    else:
        sys.exit(__doc__)
