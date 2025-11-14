"""PREVENT against the AHA online calculator (professional.heart.org/.../prevent-calculator), entered through
its web form by a browser assistant (the page refuses automated requests). `python
reference/prevent/aha_calculator.py` writes fixtures/aha_patients.json and aha_prompt.md; results go in
fixtures/aha_results.jsonl, compared by compare().

The calculator shows one decimal (trailing zero dropped: "4%"), 30-year risks only for ages 30-59, and names the
equations it used; with no ZIP code it uses the base, UACR, HbA1c or full (both) equations.

Cholesterol is given in mg/dL, the calculator's unit; the package takes mmol/L (mg/dL x 0.02586, the source's
factor). SDI needs a ZIP code and is left out. Optional UACR and HbA1c are given for some patients.
"""

import json
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np

FX = Path(__file__).parent / "fixtures"
BASE = dict(age=55, sex="female", sbp=135, bp_treated=0, total_chol=200, hdl_chol=50, statin=0, diabetes=0,
            smoker=0, egfr=85, bmi=28.0, uacr=None, hba1c=None)
CHANGES = [dict(age=30), dict(age=79), dict(sex="male"), dict(sbp=100), dict(sbp=180), dict(bp_treated=1),
           dict(total_chol=280), dict(hdl_chol=80), dict(statin=1), dict(diabetes=1), dict(smoker=1),
           dict(egfr=45), dict(egfr=120), dict(bmi=22.0), dict(bmi=38.0),
           dict(uacr=30.0), dict(uacr=300.0), dict(hba1c=5.5), dict(hba1c=8.0), dict(diabetes=1, hba1c=8.0),
           dict(uacr=30.0, hba1c=6.5)]


def patients(seed=2024):
    g = np.random.default_rng(seed)
    ps = [BASE] + [{**BASE, **c} for c in CHANGES]
    for i in range(12):
        dm = int(g.random() < 0.3)
        ps.append(dict(age=int(g.integers(30, 80)), sex=("male", "female")[i % 2], sbp=int(g.integers(95, 196)),
                       bp_treated=int(g.random() < 0.4), total_chol=int(g.integers(140, 311)),
                       hdl_chol=int(g.integers(25, 96)), statin=int(g.random() < 0.3), diabetes=dm,
                       smoker=int(g.random() < 0.25), egfr=int(g.integers(30, 141)), bmi=round(float(g.uniform(19, 39.5)), 1),
                       uacr=round(float(np.exp(g.uniform(0, 6))), 1) if i % 3 else None,
                       hba1c=round(float(g.uniform(5.0, 10.0)), 1) if i % 2 else None))
    return [dict(id=i, **p) for i, p in enumerate(ps)]


def prompt(ps):
    rows = "\n".join(json.dumps(p) for p in ps)
    return f"""# Task: read PREVENT results from the AHA online calculator for {len(ps)} patients

Open the American Heart Association PREVENT online calculator:
https://professional.heart.org/en/guidelines-and-statements/prevent-calculator
For each patient below:

1. Before the first patient, output ONE JSON line listing every input field the calculator shows, with its
   unit, allowed range and whether it is required, and every result it displays: {{"fields": [...], "results": [...]}}.
2. Enter exactly the patient's values. Units: age in years; sex; SBP mmHg; bp_treated 1 = on blood-pressure
   medication; total_chol and hdl_chol in mg/dL; statin 1 = on a statin; diabetes 1 = has diabetes;
   smoker 1 = current smoker; eGFR in mL/min/1.73 m2 (if the calculator asks for creatinine instead, stop
   and tell me); bmi in kg/m2 (if it asks for height and weight instead, stop and tell me).
   uacr (urine albumin-to-creatinine ratio, mg/g) and hba1c (%) are optional: enter them only when they are
   not null, and leave them empty when null. Leave the ZIP code (social deprivation) empty for every patient.
3. Do NOT change a value to get it accepted. If the calculator rejects one, record the patient with empty
   results and quote the error in "notes". If a required field is not given here, stop and ask me; never fill
   a placeholder.
4. Copy every result exactly as displayed, with its shown decimals: 10-year and 30-year risk of total CVD,
   ASCVD and heart failure, and anything else shown (e.g. PREVENT-Age). If 30-year risk is not shown for a
   patient (e.g. above an age limit), use null and say so in "notes".
5. Output ONE line of JSON per patient, nothing else:
   {{"id": 0, "cvd10": "5.1%", "ascvd10": "3.2%", "hf10": "2.0%", "cvd30": "20.4%", "ascvd30": "12.1%", "hf30": "11.0%", "other": {{}}, "notes": ""}}

Patients (one JSON object per line):

{rows}

When done, save all output lines to reference/prevent/fixtures/aha_results.jsonl.
"""


EQUATIONS = {"base equations": "base", "equations that included uacr": "uacr",
             "equations that included hba1c": "hba1c", "full equations": "full"}
SHOWN = (("cvd", "total_cvd"), ("ascvd", "ascvd"), ("hf", "heart_failure"))


def compare():
    """(id, model, value name, calculator's string, package %, agrees at 1 decimal) for every displayed risk."""
    from cvd_risk import prevent
    ps = {p["id"]: p for p in json.loads((FX / "aha_patients.json").read_text())}
    out = []
    for ln in (FX / "aha_results.jsonl").read_text().splitlines()[1:]:
        r = json.loads(ln)
        p, model = ps[r["id"]], EQUATIONS[r["other"]["equations"].lower()]
        for key, outcome in SHOWN:
            for years in (10, 30):
                s = r[f"{key}{years}"]
                if s is None:
                    continue
                v = 100 * float(prevent(p["age"], p["sex"], p["sbp"], p["bp_treated"], p["total_chol"] * 0.02586,
                                        p["hdl_chol"] * 0.02586, p["statin"], p["diabetes"], p["smoker"], p["egfr"],
                                        p["bmi"], uacr=p["uacr"], hba1c=p["hba1c"], model=model, years=years,
                                        outcome=outcome))
                ok = Decimal(str(v)).quantize(Decimal("0.1"), ROUND_HALF_UP) == Decimal(s.rstrip("%"))
                out.append((r["id"], model, f"{key}{years}", s, v, ok))
    return out


if __name__ == "__main__":
    ps = patients()
    (FX / "aha_patients.json").write_text(json.dumps(ps, indent=1) + "\n")
    (FX / "aha_prompt.md").write_text(prompt(ps))
