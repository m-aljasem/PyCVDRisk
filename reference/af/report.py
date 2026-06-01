"""Build reference/af/REPORT.md. Run from repo root: python reference/af/report.py

The three scores are sums of published points; the evidence is each paper's definition (level C),
checked rule by rule, plus Gage's worked example. The rule text quoted below is checked to occur in
the PDF text, so a quote cannot drift from the source.
"""

from __future__ import annotations

import datetime as dt
import logging
import csv
import itertools
import re
import subprocess
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

# Independent re-encoding of each paper's point table, written from the quotes above (not from af.py):
# (item, points if present). Age enters as (lower bound inclusive?, bound, points).
PAPER = {
    "chads2": ({"heart_failure": 1, "hypertension": 1, "diabetes": 1, "stroke_tia": 2},
               lambda a: 1 if a >= 75 else 0),                          # Gage: "age 75 years or older"
    "cha2ds2_vasc": ({"heart_failure": 1, "hypertension": 1, "diabetes": 1, "stroke_tia_te": 2,
                      "vascular_disease": 1, "female": 1},
                     lambda a: 2 if a >= 75 else 1 if a >= 65 else 0),  # Lip: ">=75 y" 2, "65-74 y" 1
    "has_bled": ({k: 1 for k in ("hypertension", "abnormal_renal", "abnormal_liver", "stroke", "bleeding",
                                 "labile_inr", "drugs", "alcohol")},
                 lambda a: 1 if a > 65 else 0),                         # Pisters: "elderly (>65 years)"
}
AGES = [40, 64, 64.9, 65, 65.1, 74, 74.9, 75, 75.1, 90]  # every cut point and both sides of it
FNS = {"chads2": chads2, "cha2ds2_vasc": cha2ds2_vasc, "has_bled": has_bled}


def enumerate_all():
    """Every combination of every item, at each age in AGES; returns {score: (rows, ours, paper)}."""
    out = {}
    for name, (items, age_pts) in PAPER.items():
        rows = []
        for age in AGES:
            for flags in itertools.product((0, 1), repeat=len(items)):
                r = dict(zip(items, flags)) | {"age": age}
                rows.append(r)
        for i, r in enumerate(rows):
            r["id"] = i
            r["paper"] = sum(items[k] * r[k] for k in items) + age_pts(r["age"])
        args = [{k: v for k, v in r.items() if k not in ("id", "paper")} for r in rows]
        for a in args:
            if "female" in a:
                a["sex"] = "female" if a.pop("female") else "male"
        ours = [float(FNS[name](**a)) for a in args]
        with open(HERE / "fixtures" / f"{name}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, ["id"] + list(args[0]))
            w.writeheader()
            w.writerows({"id": r["id"]} | a for r, a in zip(rows, args))
        out[name] = (rows, ours)
    return out


def read_ref(fname):
    p = HERE / "fixtures" / fname
    if not p.exists():
        return None
    with open(p) as f:
        return {int(r["id"]): float(r["score"]) for r in csv.DictReader(f)}


def main():
    texts = {f: text(f) for f in {r[0] for r in RULES}}
    L = ["# CHADS2, CHA2DS2-VASc, HAS-BLED: verification report", "",
         f"Generated by `reference/af/report.py`.", "",
         "Evidence level C (complete) + B: each paper's definition, checked rule by rule (section 1) and over",
         "every input combination (section 2), plus two independent implementations (section 3). The quoted",
         "definitions are matched against the PDF text by the script.", "", "## 1. Published rules", ""]
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
          ">= 8 units/week) are the caller's judgement; the functions take flags.", ""]
    L += section_exhaustive()
    (HERE / "REPORT.md").write_text("\n".join(L) + "\n")
    print(f"cases {n_ok}/{n}; quotes {q_ok}/{len(RULES)}")
    assert n_ok == n and q_ok == len(RULES)


def section_exhaustive():
    ex = enumerate_all()
    root = HERE.parent.parent
    subprocess.run(["Rscript", "reference/af/run_reference.R"], cwd=root, check=True)
    subprocess.run(["python3", "-I", "reference/af/run_medcalc.py"], cwd=root, check=True)
    L = ["## 2. Exhaustive check against the papers' point tables (level C, complete)", "",
         "Every combination of every item, at ages " + ", ".join(f"{a:g}" for a in AGES) + " (each cut point",
         "and both sides of it), scored by the package and by an independent re-encoding of each paper's point",
         "table (`PAPER` in this script, written from the quoted definitions, not from `af.py`).", "",
         "| Score | Combinations | Package = paper | Score range | cliot 1.0.0 | MedCalc-Bench |", "|---|---:|---:|---|---:|---:|"]
    ok = True
    for name, (rows, ours) in ex.items():
        same = sum(o == r["paper"] for o, r in zip(ours, rows))
        cells = []
        for pkg in ("cliot", "medcalc"):
            ref = read_ref(f"{pkg}_{name}.csv")
            if ref is None:
                cells.append("not available")
                continue
            m = sum(ref[r["id"]] == o for o, r in zip(ours, rows))
            ok &= m == len(rows)
            cells.append(f"{m}/{len(rows)}")
        ok &= same == len(rows)
        L.append(f"| {name} | {len(rows)} | {same}/{len(rows)} | {min(ours):g}-{max(ours):g} | " + " | ".join(cells) + " |")
    L += ["", "## 3. Independent implementations (level B)", "",
          "- **cliot 1.0.0** (CRAN, N. Agarwal, 2025-11-30, GPL-3): `chads2_score`, `cha2ds2_vasc_score`,",
          "  `has_bled_score`, cut unmodified into `source/cliot/af_functions.R` (from",
          "  https://cran.r-project.org/src/contrib/cliot_1.0.0.tar.gz, sha256 83794e0c...19aa1bd)",
          "  and run by `run_reference.R`. Its HAS-BLED takes \"elderly\" as a flag, so its age rule is not",
          "  tested; we pass age > 65.",
          "- **MedCalc-Bench** (ncbi-nlp/MedCalc-Bench, commit 20b10f9, 2025-12-18; the NeurIPS 2024",
          "  benchmark's reference calculators): `cha2ds2_vasc_score.py`, `has_bled_score.py` and their",
          "  `age_conversion.py` helper vendored unmodified in `source/medcalc_bench/` (sha256 9c60a3dd...,",
          "  0f0e2062..., 8a9a02d4...), run by `run_medcalc.py` under `python3 -I`. It has no CHADS2. Its",
          "  HAS-BLED takes age (> 65) and alcoholic drinks per week (>= 8): our flag maps to 8 or 7 drinks.",
          "- RiskScorescvd 0.3.1 (vendored) has none of the three scores.", "",
          "Developer artefacts: the three scores were published only as point tables; no developer code and no",
          "official ESC/EHRA calculator was found, so level A does not apply.",
          "The package returns scores, not event rates, so the papers' per-score rate tables have nothing to",
          "be compared with. (cliot's printed annual rates are not Lip 2010's; we do not use them.)"]
    assert ok, "exhaustive or B check failed"
    return L


if __name__ == "__main__":
    main()
