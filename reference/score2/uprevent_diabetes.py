"""SCORE2-Diabetes against the U-Prevent calculator (u-prevent.com), entered through its web form by a browser
assistant. `python reference/score2/uprevent_diabetes.py` writes fixtures/uprevent_diabetes_patients.json and
uprevent_diabetes_prompt.md; results go in fixtures/uprevent_diabetes_results.jsonl, compared by compare().

U-Prevent shows one decimal after rounding to two (15.447 -> 15.45 -> 15.5), as found for DIAL2 (shown()).
Its SCORE2-Diabetes form accepts ages 55-90 (version 1.238.308.20261120.3), not the model's 40-69, so the
patients under 55 were rejected and are kept with risk10 null.

1 base patient, one-change variants of it (every predictor, every region), and random patients.
"""

import json
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np

FX = Path(__file__).parent / "fixtures"
REGIONS = ("low", "moderate", "high", "very_high")
BASE = dict(age=55, sex="male", smoker=0, sbp=140, total_chol=5.5, hdl_chol=1.2, age_at_diabetes_diagnosis=50,
            hba1c=55, egfr=85, region="moderate")
CHANGES = [dict(age=40), dict(age=69), dict(sex="female"), dict(smoker=1), dict(sbp=110), dict(sbp=180),
           dict(total_chol=7.5), dict(hdl_chol=1.8), dict(age_at_diabetes_diagnosis=35), dict(age_at_diabetes_diagnosis=55),
           dict(hba1c=40), dict(hba1c=90), dict(egfr=45), dict(egfr=110),
           dict(region="low"), dict(region="high"), dict(region="very_high")]


def patients(seed=2023):
    g = np.random.default_rng(seed)
    ps = [BASE] + [{**BASE, **c} for c in CHANGES]
    for i in range(14):
        age = int(g.integers(40, 70))
        hdl = round(float(g.uniform(0.8, 2.0)), 1)
        ps.append(dict(age=age, sex=("male", "female")[i % 2], smoker=int(g.random() < 0.3), sbp=int(g.integers(100, 191)),
                       total_chol=round(hdl + float(g.uniform(2.0, 5.0)), 1), hdl_chol=hdl,
                       age_at_diabetes_diagnosis=age - int(g.integers(0, 16)), hba1c=int(g.integers(40, 96)),
                       egfr=int(g.integers(40, 116)), region=REGIONS[i % 4]))
    return [dict(id=i, ldl_chol=round(p["total_chol"] - p["hdl_chol"] - 0.5, 1), **p) for i, p in enumerate(ps)]


def prompt(ps):
    rows = "\n".join(json.dumps(p) for p in ps)
    return f"""# Task: read SCORE2-Diabetes results from U-Prevent for {len(ps)} patients

Open the SCORE2-Diabetes calculator on u-prevent.com (10-year cardiovascular risk for people with type 2
diabetes). For each patient below:

1. Before the first patient, output ONE JSON line listing every input field the calculator shows, with its
   unit, allowed range and whether it is required: {{"fields": [...]}}.
2. Enter exactly the patient's values. Every patient has type 2 diabetes. Units: age in years; SBP mmHg;
   total_chol, hdl_chol and ldl_chol in mmol/L (ldl_chol is only there in case the calculator requires it);
   HbA1c in mmol/mol; eGFR in mL/min/1.73 m2; age_at_diabetes_diagnosis in years. smoker 1 = current smoker.
   region: low / moderate / high / very_high = the calculator's Low / Moderate / High / Very high risk region.
   Leave everything else off or at its default (no medication, no treatment goals).
3. Do NOT change a value to get it accepted. If the calculator rejects one, record "risk10": null and quote
   the error in "notes". If a required field is not given here, stop and ask me; never fill a placeholder.
4. Copy every result exactly as displayed, with its shown decimals (10-year risk and anything else shown).
5. Output ONE line of JSON per patient, nothing else:
   {{"id": 0, "risk10": "12.3%", "other": {{"name": "value"}}, "notes": ""}}

Patients (one JSON object per line):

{rows}

When done, save all output lines to reference/score2/fixtures/uprevent_diabetes_results.jsonl.
"""


def shown(v):
    return Decimal(str(v)).quantize(Decimal("0.01"), ROUND_HALF_UP).quantize(Decimal("0.1"), ROUND_HALF_UP)


def compare():
    """(id, U-Prevent's string, package risk in %, agrees at display) for every accepted patient."""
    from cvd_risk import score2_diabetes
    ps = {p["id"]: p for p in json.loads((FX / "uprevent_diabetes_patients.json").read_text())}
    out = []
    for ln in (FX / "uprevent_diabetes_results.jsonl").read_text().splitlines()[1:]:
        r = json.loads(ln)
        if r["risk10"] is not None:
            p = ps[r["id"]]
            v = 100 * float(score2_diabetes(p["age"], p["sex"], p["smoker"], p["sbp"], p["total_chol"], p["hdl_chol"], 1,
                                            p["age_at_diabetes_diagnosis"], p["hba1c"], p["egfr"], p["region"]))
            out.append((r["id"], r["risk10"], v, shown(v) == Decimal(r["risk10"].rstrip("%"))))
    return out


if __name__ == "__main__":
    ps = patients()
    (FX / "uprevent_diabetes_patients.json").write_text(json.dumps(ps, indent=1) + "\n")
    (FX / "uprevent_diabetes_prompt.md").write_text(prompt(ps))
