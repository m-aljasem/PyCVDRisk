"""Build reference/af/REPORT.md. Run from repo root: python reference/af/report.py

The three scores are sums of published points; the evidence is each paper's definition (level C),
checked rule by rule, plus Gage's worked example. The rule text quoted below is checked to occur in
the PDF text, so a quote cannot drift from the source.
"""

from __future__ import annotations

import datetime as dt
import logging
import re
from pathlib import Path

import pypdf

from cvd_risk import cha2ds2_vasc, chads2, has_bled

HERE = Path(__file__).parent
logging.getLogger("pypdf").setLevel(logging.ERROR)  # font warnings, irrelevant to text


def text(name):
    t = " ".join(p.extract_text() or "" for p in pypdf.PdfReader(HERE / "source" / name).pages)
    return re.sub(r"\s+", " ", t.replace("/H11350", ">="))


C0 = {"age": 50, "heart_failure": 0, "hypertension": 0, "diabetes": 0, "stroke_tia": 0}
V0 = {"age": 50, "sex": "male", "heart_failure": 0, "hypertension": 0, "diabetes": 0, "stroke_tia_te": 0, "vascular_disease": 0}
H0 = {"age": 50, "hypertension": 0, "abnormal_renal": 0, "abnormal_liver": 0, "stroke": 0, "bleeding": 0, "labile_inr": 0,
          "drugs": 0, "alcohol": 0}

# (paper, quote that must occur in its text, function, changes from the all-zero patient, expected score)
RULES = [
    ("gage2001.pdf", "adding 1 point each for any of the following—recent CHF, hyper- tension, age 75 years or older, and DM—and 2 points for a history of stroke or TIA",
     chads2, [({"heart_failure": 1}, 1), ({"hypertension": 1}, 1), ({"age": 74.9}, 0), ({"age": 75}, 1),
              ({"diabetes": 1}, 1), ({"stroke_tia": 1}, 2),
              ({"heart_failure": 1, "hypertension": 1, "age": 80, "diabetes": 1, "stroke_tia": 1}, 6)]),
    ("gage2001.pdf", "an 82-year-old (+1) patient who had hypertension (+1) and a prior stroke (+2) would have a CHADS2 score of 4",
     chads2, [({"age": 82, "hypertension": 1, "stroke_tia": 1}, 4)]),
    ("lip2010.pdf", "C ongestive heart failure/LV dysfunction 1 H ypertension 1 A ge >= 75 y 2 D iabetes mellitus 1 S troke/TIA/TE 2",
     cha2ds2_vasc, [({"heart_failure": 1}, 1), ({"hypertension": 1}, 1), ({"age": 75}, 2), ({"diabetes": 1}, 1),
                    ({"stroke_tia_te": 1}, 2), ({"vascular_disease": 1}, 1), ({"age": 64.9}, 0), ({"age": 65}, 1),
                    ({"age": 74.9}, 1), ({"sex": "female"}, 1),
                    ({"age": 80, "sex": "female", "heart_failure": 1, "hypertension": 1, "diabetes": 1,
                      "stroke_tia_te": 1, "vascular_disease": 1}, 9)]),
    ("lip2010.pdf", "A ge 65-74 y 1 S ex c ategory (ie female gender) 1", cha2ds2_vasc, []),
    ("pisters2010.pdf", "A Abnormal renal and liver function (1 point each) 1 or 2 S Stroke 1 B Bleeding 1 L Labile INRs 1 E Elderly 1 D Drugs or alcohol (1 point each) 1 or 2",
     has_bled, [({"hypertension": 1}, 1), ({"abnormal_renal": 1}, 1), ({"abnormal_liver": 1}, 1),
                ({"abnormal_renal": 1, "abnormal_liver": 1}, 2), ({"stroke": 1}, 1), ({"bleeding": 1}, 1),
                ({"labile_inr": 1}, 1), ({"age": 65}, 0), ({"age": 65.1}, 1), ({"drugs": 1}, 1), ({"alcohol": 1}, 1),
                ({"drugs": 1, "alcohol": 1}, 2),
                ({k: 1 for k in H0 if k != "age"} | {"age": 70}, 9)]),
    ("pisters2010.pdf", "elderly ( . 65 years)", has_bled, []),
]
BASE = {chads2: C0, cha2ds2_vasc: V0, has_bled: H0}


def main():
    texts = {f: text(f) for f in {r[0] for r in RULES}}
    L = ["# CHADS2, CHA2DS2-VASc, HAS-BLED: verification report", "",
         f"Generated {dt.date.today()} by `reference/af/report.py`.", "",
         "Evidence level C: each paper's own definition, checked rule by rule. No independent",
         "implementation is used: the scores are sums of 0/1 items, so the risk is a wrong rule, not",
         "arithmetic. The quoted definitions are matched against the PDF text by the script.", ""]
    n_ok = n = q_ok = 0
    for paper, quote, fn, cases in RULES:
        found = quote in texts[paper]
        q_ok += found
        L += [f"**{fn.__name__}** ({paper}): \"{quote}\" {'(found in PDF)' if found else '(NOT FOUND)'}", ""]
        if cases:
            L += ["| Inputs (others 0, age 50, male) | Score | Expected |", "|---|---:|---:|"]
        for change, want in cases:
            got = float(fn(**{**BASE[fn], **change}))
            n += 1
            n_ok += got == want
            L.append(f"| {change} | {got:g} | {want} |")
        L.append("")
    L += [f"{n_ok}/{n} cases give the published score; {q_ok}/{len(RULES)} quotes found in the PDFs.", "",
          "Notes: HAS-BLED's \"elderly\" is printed as \">65 years\" (age 65 scores 0). CHADS2's \"recent\" CHF",
          "and the HAS-BLED item definitions (renal: dialysis, transplant or creatinine >= 200 umol/L; alcohol",
          ">= 8 units/week) are the caller's judgement; the functions take flags."]
    (HERE / "REPORT.md").write_text("\n".join(L) + "\n")
    print(f"cases {n_ok}/{n}; quotes {q_ok}/{len(RULES)}")
    assert n_ok == n and q_ok == len(RULES)


if __name__ == "__main__":
    main()
