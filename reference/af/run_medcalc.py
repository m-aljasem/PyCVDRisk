"""MedCalc-Bench (ncbi-nlp, commit 20b10f9) CHA2DS2-VASc and HAS-BLED on fixtures/*.csv.

The calculators are vendored unmodified in source/medcalc_bench (read in full before running).
Run with isolation from the repo root: python3 -I reference/af/run_medcalc.py
"""
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "source" / "medcalc_bench"))
from cha2ds2_vasc_score import generate_cha2ds2_vasc_explanation  # noqa: E402
from has_bled_score import compute_has_bled_score_explanation  # noqa: E402


def rows(name):
    with open(HERE / "fixtures" / f"{name}.csv") as f:
        return list(csv.DictReader(f))


def write(name, out):
    with open(HERE / "fixtures" / f"medcalc_{name}.csv", "w", newline="") as f:
        csv.writer(f).writerows([("id", "score")] + out)


b = lambda r, k: r[k] == "1"  # noqa: E731
write("cha2ds2_vasc", [(r["id"], generate_cha2ds2_vasc_explanation({
    "age": [float(r["age"]), "years"], "sex": r["sex"].capitalize(), "chf": b(r, "heart_failure"),
    "hypertension": b(r, "hypertension"), "stroke": b(r, "stroke_tia_te"), "vascular_disease": b(r, "vascular_disease"),
    "diabetes": b(r, "diabetes")})["Answer"]) for r in rows("cha2ds2_vasc")])
# alcohol is a drinks/week count in MedCalc (>= 8 scores); our flag 1 -> 8 drinks, 0 -> 7
write("has_bled", [(r["id"], compute_has_bled_score_explanation({
    "age": [float(r["age"]), "years"], "alcoholic_drinks": 8 if b(r, "alcohol") else 7,
    "hypertension": b(r, "hypertension"), "renal_disease_has_bled": b(r, "abnormal_renal"),
    "liver_disease_has_bled": b(r, "abnormal_liver"), "stroke": b(r, "stroke"), "prior_bleeding": b(r, "bleeding"),
    "labile_inr": b(r, "labile_inr"), "medications_for_bleeding": b(r, "drugs")})["Answer"]) for r in rows("has_bled")])
