"""SMART2 against the U-Prevent calculator (u-prevent.com), filled in by hand or by a browser assistant.

    python reference/smart2/uprevent.py patients   # writes fixtures/uprevent_patients.json and uprevent_prompt.md

The set is 1 base patient, 18 one-change variants of it (each predictor and region moved alone, so
the effect of every coefficient and any unknown constant can be read off), and 11 random patients.
Results are in fixtures/uprevent_results.jsonl (first line: the calculator's input fields); the
comparison is tests/smart2/test_smart2.py::test_matches_uprevent_calculator.
"""

import json
import sys
from pathlib import Path

import numpy as np

FX = Path(__file__).parent / "fixtures"
BASE = dict(age=60, sex="male", smoker=0, diabetes=0, sbp=130, total_chol=5.0, hdl_chol=1.2, egfr=80, hscrp=2.0,
            years_since_diagnosis=3, cad=1, cerebrovascular=0, pad=0, aaa=0, antithrombotic=0, region="europe_moderate")
CHANGES = [dict(age=70), dict(age=45), dict(sex="female"), dict(smoker=1), dict(diabetes=1), dict(sbp=160),
           dict(total_chol=6.5), dict(egfr=50), dict(egfr=100), dict(hscrp=8.0), dict(hscrp=0.5),
           dict(years_since_diagnosis=10), dict(years_since_diagnosis=0), dict(cad=0, cerebrovascular=1),
           dict(pad=1), dict(aaa=1), dict(antithrombotic=1), dict(region="europe_very_high")]


def patients(seed=2022):
    g = np.random.default_rng(seed)
    ps = [BASE] + [{**BASE, **c} for c in CHANGES]
    for i in range(11):
        loc = g.random(4) < 0.35
        loc[g.integers(4)] = True
        hdl = round(float(g.uniform(0.8, 2.0)), 1)
        age = int(g.integers(42, 80))
        ps.append(dict(age=age, sex=("male", "female")[i % 2], smoker=int(g.random() < 0.3), diabetes=int(g.random() < 0.4),
                       sbp=int(g.integers(110, 181)), total_chol=round(hdl + float(g.uniform(1.5, 5.0)), 1), hdl_chol=hdl,
                       egfr=int(g.integers(40, 106)), hscrp=round(float(g.uniform(0.3, 12.0)), 1),
                       years_since_diagnosis=int(g.integers(0, min(20, age - 30))), cad=int(loc[0]),
                       cerebrovascular=int(loc[1]), pad=int(loc[2]), aaa=int(loc[3]), antithrombotic=int(g.random() < 0.7),
                       region=("europe_low", "europe_moderate", "europe_high", "europe_very_high")[i % 4]))
    return [dict(id=i, non_hdl_chol=round(p["total_chol"] - p["hdl_chol"], 1), **p) for i, p in enumerate(ps)]


def prompt(ps):
    rows = "\n".join(json.dumps(p) for p in ps)
    return f"""# Task: read SMART2 results from U-Prevent for {len(ps)} patients

On the U-Prevent SMART2 calculator (secondary prevention, patients with established atherosclerotic
disease; u-prevent.com), for each patient below:

1. Before the first patient, list every input field the calculator shows, with its unit and whether
   it is required, as ONE JSON line: {{"fields": [...]}}. Note which risk regions or countries it offers.
2. Enter exactly the patient's values. Units: SBP mmHg; cholesterol mmol/L (enter total and HDL if
   the calculator asks for those; non_hdl_chol = total_chol - hdl_chol is given for reference only);
   eGFR mL/min/1.73 m2 (if it asks for creatinine instead, tell me and stop); hsCRP mg/L (if it asks
   for mg/dL, convert and say so); years_since_diagnosis = years since the first manifestation of
   cardiovascular disease. cad / cerebrovascular / pad / aaa = 1 means that disease is present
   (coronary artery disease / cerebrovascular disease / peripheral artery disease / abdominal aortic
   aneurysm). antithrombotic = 1 means aspirin or an equivalent antithrombotic drug is taken.
   Region: europe_low / europe_moderate / europe_high / europe_very_high = the ESC risk region
   (pick a country of that region, e.g. low: Netherlands, moderate: Germany, high: Poland,
   very_high: Ukraine; use the region itself if the calculator offers regions).
   Leave every field not listed here empty or at its default and write down any that is required.
3. Copy every result exactly as displayed, with its shown decimals (10-year risk, and any other
   number the calculator shows, such as life expectancy, lifetime risk or treatment benefit).
4. Output ONE line of JSON per patient, nothing else:
   {{"id": 0, "risk10": "20%", "other": {{"name": "value"}}, "country": "Germany", "notes": ""}}
   Keep values as the displayed strings. Put anything unusual (a field you had to fill, a warning,
   a value out of range, a value you had to convert) in "notes".

Patients (one JSON object per line):

{rows}

When done, paste all output lines into reference/smart2/fixtures/uprevent_results.jsonl.
"""


# Round 2 (python reference/smart2/uprevent.py patients2): what round 1 left open. HbA1c, required
# for diabetics, was a placeholder 53 in round 1; inputs at the calculator's limits (squared terms);
# the antithrombotic options other than aspirin. ids 100+; patient 100 repeats round-1 patient 0.
DIABETIC = {**BASE, "diabetes": 1}
OLD_DIABETIC = {**BASE, "age": 72, "sex": "female", "diabetes": 1, "pad": 1, "egfr": 55, "region": "europe_high"}
ROUND2 = ([BASE] + [{**DIABETIC, "hba1c": h} for h in (33, 42, 53, 64, 75, 90, 108)]
          + [{**OLD_DIABETIC, "hba1c": h} for h in (42, 86)]
          + [{**BASE, **c} for c in (dict(age=40), dict(age=80), dict(egfr=30), dict(egfr=120), dict(hscrp=0.1),
                                     dict(hscrp=15.0), dict(years_since_diagnosis=30), dict(sbp=95), dict(sbp=200),
                                     dict(total_chol=8.0, hdl_chol=0.6))]
          + [{**BASE, "antithrombotic": a} for a in ("dual_antiplatelet", "aspirin_plus_low_dose_doac")])


def patients2():
    ps = []
    for i, p in enumerate(ROUND2):
        nh = round(p["total_chol"] - p["hdl_chol"], 1)
        ps.append(dict(id=100 + i, non_hdl_chol=nh, ldl_chol=round(nh - 0.5, 1), hba1c=p.get("hba1c"),
                       **{k: v for k, v in p.items() if k != "hba1c"}))
    return ps


def prompt2(ps):
    rows = "\n".join(json.dumps(p) for p in ps)
    return f"""# Task: read SMART2 results from U-Prevent for {len(ps)} more patients (round 2)

Same U-Prevent SMART2 calculator as round 1. For each patient below:

1. Enter exactly the patient's values, with the same units and meanings as round 1: SBP mmHg;
   total_chol, hdl_chol and ldl_chol in mmol/L (enter all three as given; non_hdl_chol is for
   reference only); eGFR mL/min/1.73 m2; hsCRP mg/L; years_since_diagnosis = years since the
   first cardiovascular event; cad / cerebrovascular / pad / aaa = 1 means that disease is present.
   hba1c is in mmol/mol: enter it when diabetes = 1 (it is null when diabetes = 0).
   Region: europe_low / europe_moderate / europe_high / europe_very_high = the calculator's
   Low / Moderate / High / Very high risk region.
   antithrombotic: 0 = "No antithrombotic", 1 = "Aspirin or equivalent", "dual_antiplatelet" =
   "Dual antiplatelet therapy", "aspirin_plus_low_dose_doac" = "Aspirin + low dose DOAC".
   Leave everything else off or at its default (smoking only as given; Mediterranean diet off; step
   count empty; no statin, ezetimibe, bempedoic acid, PCSK9-inhibitor, GLP1-RA, SGLT2-inhibitor,
   colchicine or icosapent-ethyl).
2. Several patients sit exactly at an input limit on purpose (age 40 and 80, eGFR 30, hsCRP 0.1 and
   15, 30 years since the event, SBP 200, total cholesterol 8.0, HDL 0.6, HbA1c 33 and 108). Do NOT
   change a value to get it accepted. If the calculator rejects a value, record that patient with
   "risk10": null and quote the error message in "notes", then move on.
3. Do not fill in any placeholder. If a field is required that is not given here, stop and ask me.
4. Copy every result exactly as displayed, with its shown decimals.
5. Output ONE line of JSON per patient, nothing else:
   {{"id": 100, "risk10": "26.1%", "other": {{"reduction_due_to_treatment": "0.0%"}}, "notes": ""}}
   Keep values as the displayed strings; put anything unusual in "notes".

Patients (one JSON object per line):

{rows}

When done, save all output lines to reference/smart2/fixtures/uprevent_results2.jsonl.
"""


if __name__ == "__main__" and sys.argv[1:] == ["patients"]:
    FX.mkdir(exist_ok=True)
    ps = patients()
    (FX / "uprevent_patients.json").write_text(json.dumps(ps, indent=1) + "\n")
    (FX / "uprevent_prompt.md").write_text(prompt(ps))
elif __name__ == "__main__" and sys.argv[1:] == ["patients2"]:
    ps = patients2()
    (FX / "uprevent_patients2.json").write_text(json.dumps(ps, indent=1) + "\n")
    (FX / "uprevent_prompt2.md").write_text(prompt2(ps))
