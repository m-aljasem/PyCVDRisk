"""DIAL2 against the U-Prevent calculator (u-prevent.com), filled in by hand or by a browser assistant.

    python reference/dial2/uprevent.py patients   # writes fixtures/uprevent_patients.json and
                                                  # fixtures/uprevent_prompt.md (instructions to paste)
    python reference/dial2/uprevent.py compare    # reads fixtures/uprevent_results.jsonl and prints
                                                  # the package's values next to U-Prevent's

U-Prevent shows risks rounded to whole percent and life expectancy to whole years (at best one
decimal), so agreement is judged to that display precision.
"""

from __future__ import annotations

import json
import sys
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


def compare():
    ps = {p["id"]: p for p in json.loads(PATIENTS.read_text())}
    print(f"{'id':>3} {'outcome':16} {'U-Prevent':>10} {'package':>10}")
    for ln in RESULTS.read_text().splitlines():
        if not ln.strip():
            continue
        r = json.loads(ln)
        args = {k: ps[r["id"]][k] for k in FIELDS}
        for o in ("risk10", "lifetime_risk", "life_expectancy"):
            v = float(dial2(**args, outcome=o))
            v = v if o == "life_expectancy" else 100 * v
            print(f"{r['id']:>3} {o:16} {r[o]:>10} {v:10.2f}  {r.get('notes', '')}")


if __name__ == "__main__":
    if sys.argv[1:] == ["patients"]:
        FX.mkdir(exist_ok=True)
        ps = patients()
        PATIENTS.write_text(json.dumps(ps, indent=1) + "\n")
        PROMPT.write_text(prompt(ps))
        print(f"wrote {PATIENTS} and {PROMPT}")
    elif sys.argv[1:] == ["compare"]:
        compare()
    else:
        sys.exit(__doc__)
