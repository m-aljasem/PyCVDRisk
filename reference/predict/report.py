"""Build reference/predict/REPORT.md. Run from repo root: python reference/predict/report.py

1. coefficients.py vs the authors' R source (parsed as text, not run) and vs the papers' printed tables
   (Pylypchuk 2018 Table 3; Selak 2019 Table 4).
2. The papers' worked examples.
3. If reference/predict/fixtures/<model>_reference.csv exist (written by run_reference.R, the authors'
   code run unmodified), every patient is compared.
"""

from __future__ import annotations

import csv
import datetime as dt
import logging
import math
import re
from pathlib import Path

import numpy as np

from cvd_risk.predict import coefficients as C
from cvd_risk.predict import predict as P

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
RSRC = ROOT / "reference" / "vendor" / "PredictRiskScores" / "R"
logging.getLogger("pypdf").setLevel(logging.ERROR)
MODELS = {"cvd": ("NoPriorCVDRisk", C.CVD, P.predict_cvd), "cvd_bmi": ("NoPriorCVDRisk_BMI", C.CVD_BMI, P.predict_cvd_bmi),
          "cvd_policy": ("NoPriorCVDRisk_Policy", C.CVD_POLICY, P.predict_cvd_policy),
          "t2d": ("PriorT2DRisk", C.T2D, P.predict_t2d), "bleed": ("NoPriorCVDBleedRisk", C.BLEED, P.predict_bleed),
          "post_cvd": ("PostCVDRisk", C.POST_CVD, P.predict_post_cvd),
          "post_acs": ("PostACSRisk", C.POST_ACS, P.predict_post_acs)}


def pdf_text(name):
    import pypdf  # only for regenerating the report from the PDFs; tests import this module without it

    t = " ".join(p.extract_text() or "" for p in pypdf.PdfReader(HERE / "source" / name).pages)
    return re.sub(r"\s+", " ", t.replace("·", ".").replace("–", "-").replace("−", "-"))


def r_numbers(fn):
    """Every number in the R function's coefficient lists, centring, survival and mean-LP expressions."""
    s = "\n".join(line for line in (RSRC / f"{fn}.R").read_text().splitlines() if not line.startswith("#'"))
    nums = [float(v) for block in re.findall(r"coeff?s? <- list\((.*?)\)\n", s, re.DOTALL)
            for v in re.findall(r"=\s*(-?[\d.]+)", block)]
    nums += [float(v) for v in re.findall(r"values\$\w+\[\w\.ind\] - ([\d.]+)", s)]
    nums += [float(v) for v in re.findall(r"\) \+ ([\d.]+)\n", s)]
    for s0, mlp in re.findall(r"1 - ([\d.]+) \^ exp\(sum\.score(?:\[\w\.ind\])?(?: - ([\d.]+))?\)", s):
        nums += [float(s0)] + ([float(mlp)] if mlp else [])
    return sorted(nums)


def ours(m):
    nums = [v for d in m["beta"].values() for v in d.values()]
    nums += [v for d in m.get("centre", {}).values() for v in d.values()]
    nums += list(m.get("log_acr_shift", {}).values()) + list(m["s0"].values())
    nums += [v for v in m["mean_lp"].values() if v]
    return sorted(nums)


def table_pairs(t, start, end, rows):
    """For each (label, key): the two numbers (women, men) printed after the label in t[start:end]."""
    seg = t[t.index(start):]
    seg = seg[:seg.index(end)]
    out = []
    for label, key in rows:
        m = re.search(re.escape(label) + r"\s+(-?\d*\.\d+)\s+(-?\d*\.\d+)", seg)
        out.append((label, key, (float(m.group(1)), float(m.group(2))) if m else None))
    return out


T3_2018 = [("Age", "age"), ("Māori", "maori"), ("Pacific", "pacific"), ("Indian", "indian"),
           ("Chinese or other Asian", "asian"), ("NZDep quintile", "nzdep"), ("Ex-smoker", "ex_smoker"),
           ("Smoker", "smoker"), ("cardiovascular disease", "family_history"), ("Atrial fibrillation", "af"),
           ("Diabetes", "diabetes"), ("SBP per 1 mm Hg", "sbp"), ("TC/HDL", "tc_hdl"), ("OBPLM*", "bp_meds"),
           ("OLLM", "lipid_meds"), ("OATM", "antithrombotic"), ("Age × diabetes", "age_x_diabetes"),
           ("Age × SBP", "age_x_sbp"), ("OBPLM × SBP", "bp_meds_x_sbp")]
T3_MEANS = [("Age", "age"), ("NZDep quintle", "nzdep"), ("SBP", "sbp"), ("TC/HDL", "tc_hdl")]
T4_2019 = [("Age, per year", "age"), ("ori", "maori"), ("Paciﬁc", "pacific"), ("Indian", "indian"),
           ("Chinese or other Asian", "asian"), ("NZDep quintile, per 1 quintile", "nzdep"),
           ("Former smoker", "ex_smoker"), ("Current smoker", "smoker"), ("Family history of premature CVD", "family_history"),
           ("Diabetes", "diabetes"), ("Cancer", "cancer"), ("Prior bleeding event", "gi_bleed"),
           ("Peptic ulcer disease (nonbleeding)", "peptic_ulcer"), ("Alcohol-related condition", "alcohol"),
           ("Chronic liver disease or pancreatitis", "liver"), ("SBP, per mm Hg", "sbp"),
           ("Ratio of total-HDL cholesterol, per 1 unit", "tc_hdl"), ("Blood pressure-lowering", "bp_meds"),
           ("Lipid-lowering", "lipid_meds"), ("Peptic ulcer disease", "peptic_ulcer_meds"),
           ("Nonsteroidal anti-inﬂammatory", "nsaid"), ("Corticosteroid", "steroids"),
           ("Selective serotonin reuptake inhibitor", "ssri"), ("Baseline survival at 5 y", "s0"),
           ("Mean prognostic index‡", "mean_lp")]


def paper_checks():
    rows, ok, n = [], 0, 0
    t = pdf_text("pylypchuk2018.pdf")
    checks = [(lbl, key, got, (C.CVD["beta"]["female"][key], C.CVD["beta"]["male"][key]))
              for lbl, key, got in table_pairs(t, "Women Men Age 0.", "Means for centering", T3_2018)]
    checks += [(f"mean {lbl}", key, got, (C.CVD["centre"]["female"][key], C.CVD["centre"]["male"][key]))
               for lbl, key, got in table_pairs(t, "Means for centering", "NZDep=New Zealand", T3_MEANS)]
    m = re.search(r"\(at 5 years\) (0\.\d+) (0\.\d+)", t)
    checks.append(("baseline survival", "s0", m and (float(m.group(1)), float(m.group(2))),
                   (C.CVD["s0"]["female"], C.CVD["s0"]["male"])))
    for lbl, key, got, want in checks:
        n += 1
        good = got is not None and all(math.isclose(g, w, abs_tol=0) for g, w in zip(got, want))
        ok += good
        rows.append(f"| Pylypchuk 2018 T3 | {lbl} | {got} | {want} | {'ok' if good else 'MISMATCH'} |")
    t = pdf_text("selak2019.pdf")
    for lbl, key, got in table_pairs(t, "Characteristic Coe", "CVD = cardiovascular disease;", T4_2019):
        want = ((C.BLEED[key]["female"], C.BLEED[key]["male"]) if key in ("s0", "mean_lp")
                else (C.BLEED["beta"]["female"][key], C.BLEED["beta"]["male"][key]))
        n += 1
        good = got is not None and all(math.isclose(g, w, abs_tol=0) for g, w in zip(got, want))
        ok += good
        rows.append(f"| Selak 2019 T4 | {lbl} | {got} | {want} | {'ok' if good else 'MISMATCH'} |")
    return rows, ok, n


def examples():
    ex = [("Pylypchuk 2018 panel: European woman 55, diabetes, ex-smoker, NZDep 3, SBP 135, TC/HDL 5, on BP drugs",
           P.predict_cvd(55, "f", "european", 3, "ex", 1, 0, 0, 135, 5, 1, 0, 0), 0.0511, 4),
          ("Selak 2019 Table 4: European man 65, diabetes, former smoker, NZDep 3, SBP 130, TC/HDL 5, BP and lipid drugs",
           P.predict_bleed(65, "m", "european", 3, "ex", 1, 0, 1, 1, *[0] * 9, 130, 5), 0.019038994, 9)]
    return [(d, float(g), p, round(float(g), k) == p) for d, g, p, k in ex]


def load(name):
    return list(csv.DictReader((HERE / "fixtures" / f"{name}_patients.csv").open()))


def col(rows, k):
    return np.array([np.nan if r[k] == "NA" else (float(r[k]) if re.fullmatch(r"-?[\d.]+", r[k]) else r[k])
                     for r in rows], dtype=object)


def python_risk(name, rows):
    c = lambda k: col(rows, k)
    num = lambda k: c(k).astype(float)
    base = {"age": num("age"), "sex": c("sex"), "ethnicity": c("ethnicity"), "nzdep": num("nzdep")}
    f = {k: num(k) for k in rows[0] if k not in ("id", "sex", "ethnicity", "smoking", "acs_type")}
    f.update(base)
    for k in ("smoking", "acs_type"):
        if k in rows[0]:
            f[k] = c(k)
    return MODELS[name][2](**{k: v for k, v in f.items() if k != "id"})


def r_comparison():
    rows, all_ok = [], True
    for name in MODELS:
        ref = HERE / "fixtures" / f"{name}_reference.csv"
        if not ref.exists():
            rows.append(f"| {name} | not run | | |")
            all_ok = None if all_ok is not False else False
            continue
        r = np.array([float(x["risk"]) if x["risk"] != "NA" else np.nan for x in csv.DictReader(ref.open())])
        p = python_risk(name, load(name))
        d = np.nanmax(np.abs(p - r))
        same_nan = bool((np.isnan(p) == np.isnan(r)).all())
        good = same_nan and d < 1e-10
        all_ok = all_ok if good else False
        rows.append(f"| {name} | {len(r)} | {d:.1e} | {'ok' if good else 'DIFF'} |")
    return rows, all_ok


DEFECTS = [
    ("All primary-prevention models", ("`values$age[which(values$age > 79)] <- 80` clamps ages above 79 to 80, not 79; "
     "ages below 30 become 30. The input check only accepts whole-number ages 18-110."),
     "NaN outside 30-79 (real-valued ages accepted)."),
    ("NoPriorCVDRisk_BMI", ("`bmi %in% 25:29.9` (and 30:34.9, 35:39.9) is membership in the integers 25..29 etc., so a "
     "BMI of 27.3 falls in no band and is scored as the 18.5-25 reference."), "Bands [25, 30), [30, 35), [35, 40)."),
    ("PostCVDRisk, PostACSRisk", ("BMI `%in% 20:24`, `30:34`, `35:39`; HbA1c `>= 40 & <= 64`; creatinine `>= 100 & <= 149`; "
     "SBP `>= 120 & <= 139`, `>= 140 & <= 159`; days `>= 182 & <= 365`: non-integer values between bands "
     "(BMI 22.3, HbA1c 64.5, creatinine 149.5, SBP 139.5) get the reference category."),
     "Gap-free bands [20, 25), [40, 65), [100, 150), [120, 140), [140, 160), [182, 366) days."),
    ("PostCVDRisk, PostACSRisk", ("`male = +(input$sex %in% ok.male)` has no `tolower`, so sex \"M\" or \"Male\" is "
     "scored as female (the f/m indices elsewhere use `tolower`)."), "Case-insensitive sex."),
    ("PriorT2DRisk", ("`log((acr + 0.01) / 1000)` uses the function argument `acr`, not the column: called with `dat` it "
     "evaluates a variable named `acr` in the caller's environment, and the whole vector is assigned to each sex's "
     "subset, so mixed-sex batches get other rows' ACR."), "ACR taken per row."),
    ("All", "Missing binary inputs (NA) are scored as 0 by default (`allow.na = TRUE`).", "NaN."),
    ("Package metadata", "DESCRIPTION says GPL-2; the repository's LICENSE file is GPL-3.", "-"),
]


def main():
    static = [(name, fn, r_numbers(fn) == ours(m), len(ours(m))) for name, (fn, m, _) in MODELS.items()]
    prow, pok, pn = paper_checks()
    ex = examples()
    rrows, r_ok = r_comparison()
    L = ["# PREDICT (New Zealand): verification report", "", f"Generated {dt.date.today()} by `reference/predict/report.py`.", "",
         "Standard implementation: the VIEW team's R package PredictRiskScores 0.9 (University of Auckland; vendored",
         "unmodified in `reference/vendor/PredictRiskScores`, commit in `COMMIT`). Papers: Pylypchuk 2018 (Lancet,",
         "`source/pylypchuk2018.pdf`) and Selak 2019 (Ann Intern Med, `source/selak2019.pdf`).", "",
         "## 1. Coefficients", "",
         "Every number in each R function (coefficients, centring values, ACR shift, baseline survival, mean linear",
         "predictor), parsed from the source text, against `coefficients.py`:", "",
         "| Model | R function | Numbers | Identical |", "|---|---|---:|---|",
         *[f"| {n} | `{fn}` | {k} | {'yes' if s else 'NO'} |" for n, fn, s, k in static], "",
         f"Printed tables against `coefficients.py`: {pok}/{pn} rows identical to the last printed digit.", "",
         "| Source | Row | Printed (women, men) | Package | |", "|---|---|---|---|---|", *prow, "",
         "## 2. Worked examples", "",
         "| Example | Package | Printed | |", "|---|---:|---:|---|",
         *[f"| {d} | {g:.9f} | {p} | {'ok' if good else 'MISMATCH'} |" for d, g, p, good in ex], "",
         "## 3. Comparison with the authors' R code", "",
         "`run_reference.R` runs the seven R functions unmodified on `fixtures/<model>_patients.csv` (3,000 patients",
         "each, drawn where the R code behaves as documented: whole-number ages, BMI, creatinine, HbA1c and days;",
         "lower-case sex) with `dp = 12`.", "",
         "| Model | Patients | Max abs difference | |", "|---|---:|---:|---|", *rrows, ""]
    if r_ok is None:
        L += ["**Not run yet.** Executing the downloaded R code needs the owner's permission (see",
              "`reference/README.md`). Until then the evidence for the T2D, BMI, policy and post-event models is the",
              "parsed coefficients only; the 2018 and bleeding models also rest on their papers.", ""]
    L += ["## 4. Defects in the R code (found by reading it)", "",
          "| Where | R behaviour | This package |", "|---|---|---|", *[f"| {a} | {b} | {c} |" for a, b, c in DEFECTS], "",
          "## 5. Not checked", "",
          "- Pylypchuk 2021 (diabetes model): the Zotero attachment is a different article, so the paper's tables were",
          "  not compared. The VARIANZ policy model (Mehta 2018) and the post-CVD / post-ACS models (Poppe) have no",
          "  paper in the library; for these the R package is the only source.", ""]
    (HERE / "REPORT.md").write_text("\n".join(L) + "\n")
    print(f"static {sum(s for *_, s, _ in static)}/{len(static)}; paper rows {pok}/{pn}; "
          f"examples {sum(e[3] for e in ex)}/{len(ex)}; R {r_ok}")
    assert all(s for _, _, s, _ in static) and pok == pn and all(e[3] for e in ex) and r_ok is not False


if __name__ == "__main__":
    main()
