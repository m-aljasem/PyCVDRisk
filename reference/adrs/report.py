"""Build reference/adrs/REPORT.md. Run from repo root: python reference/adrs/report.py

1. Every coefficient and centring value against the supplement's SPSS syntax (Data S1, PDF text) and
   every coefficient against Table 2 (article XML from Europe PMC).
2. Checks of the values the paper does not state (GFR mean, mean blood pressure, stroke mean score).
"""

from __future__ import annotations

import datetime as dt
import logging
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pypdf

from cvd_risk.adrs.adrs import ASCVD, CAD, GFR_MEAN, STROKE, STROKE_MEAN

HERE = Path(__file__).parent
logging.getLogger("pypdf").setLevel(logging.ERROR)
SPSS = {"vitamin_d": "VITD_VALUE", "dev_gfr": "deviationGFR", "male": "SEX", "smoker": "CurrentSmoker", "age": "Age",
        "bp_treated": "BPtreated", "tc_hdl": "Chole_HDL_ratio", "diabetes": "DMDx", "dev_gfr_x_male": "deviationGFRSEX",
        "map": "mean_BP", "age2": "AGE_SQR", "hypertension": "Hypertension", "diabetes_x_male": "DMSEX",
        "diabetes_x_dev_gfr": "DMdevGFR", "bp_normal": "BP1", "bp_elevated": "BP2", "bp_stage1": "BP3",
        "bp_stage2": "BP4", "bp_crisis": "BP5"}
T2 = {"ascvd": {"vitamin_d": "Vitamin D (ng/mL)", "male": "Sex (0 female and 1 male)", "smoker": "Current smoker",
                "diabetes": "Diabetes mellitus", "bp_treated": "On hypertension treatment", "map": "Mean blood pressure",
                "tc_hdl": "Cholesterol/HDL ratio (mmol/L)", "age2": "Age square (year)",
                "dev_gfr": "Deviation from cohort GFR mean (mL/min/1.73 m2)", "age": "Age (year)",
                "dev_gfr_x_male": "Interaction of deviation from cohort GFR mean (mL/min/1.73 m2) and Sex (0 female and 1 male)"},
      "cad": {"vitamin_d": "Vitamin D (ng/mL)", "dev_gfr": "Deviation from cohort GFR mean (mL/min/1.73 m2)",
              "male": "Sex (0 female and 1 male)", "smoker": "Current smoker (yes = 1)", "age": "Age (year)",
              "hypertension": "Hypertension", "tc_hdl": "Cholesterol/HDL ratio (mmol/L)", "diabetes": "Diabetes mellitus",
              "bp_normal": "BP normal category: below 120/80", "bp_elevated": "BP elevated category: 120–129/less than 80",
              "bp_stage1": "BP stage 1 high blood pressure category: 130–139/80–89",
              "bp_stage2": "BP stage 2 high blood pressure category: 140 and above/90 and above",
              "bp_crisis": "BP hypertension crisis category",
              "dev_gfr_x_male": "Deviation from cohort GFR mean (mL/min/1.73 m2)* sex (0 female and 1 male)",
              "diabetes_x_male": "Interaction of diabetes mellitus and sex (0 female and 1 male)",
              "diabetes_x_dev_gfr": "Interaction of diabetes mellitus and deviation from cohort GFR mean (mL/min/1.73 m2)"},
      "stroke": {"age": "Age (year)", "smoker": "Current smoker", "sbp": "SBP",
                 "hypertension_x_male": "Interaction of hypertension and sex (0 female and 1 male)",
                 "age_x_male": "Interaction of age (year) and sex (0 female and 1 male)"}}
T2_TITLES = {"cad": "Hard Coronary", "stroke": "new stroke", "ascvd": "Atherosclerotic", "death": "new death"}


def spss_text():
    t = " ".join(p.extract_text() or "" for p in pypdf.PdfReader(HERE / "source" / "adrs_data_s1.pdf").pages)
    return re.sub(r"\s+", "", t.replace("−", "-"))


def table2():
    """{model: {row label: B}} from the article XML."""
    root = ET.parse(HERE / "source" / "PMC11681588.xml").getroot()
    out, cur = {}, None
    for tw in root.iter("table-wrap"):
        if "".join(tw.find("label").itertext()) != "Table 2":
            continue
        for tr in tw.iter("tr"):
            cells = [" ".join("".join(c.itertext()).split()) for c in tr]
            title = next((m for m, k in T2_TITLES.items() if k in cells[0]), None) if len(cells) == 1 else None
            if title:
                cur = out.setdefault(title, {})
            elif cur is not None and len(cells) >= 2 and re.fullmatch(r"[−-]?\d*\.\d+", cells[1]):
                cur[cells[0]] = float(cells[1].replace("−", "-"))
    return out


def term(b, m, name):
    """The SPSS term '((NAME - mean) * b)' in the forms the syntax uses."""
    ms = f"({m:g})" if m < 0 else f"{m:g}"
    bs = [f"{b:g}", f"({b:g})"] if b >= 0 else [f"{b:g}", f"-{abs(b):g}", f"({b:g})"]
    return [f"(({name}-{mv})*{bv})" for mv in {ms, f"{m:.2f}" if m >= 0 else ms, f"{m:.1f}" if m >= 0 else ms} for bv in bs]


def main():
    sp = spss_text()
    t2 = table2()
    rows, ok, n = [], 0, 0
    for model, coefs in (("ascvd", ASCVD), ("cad", CAD)):
        for k, (b, m) in coefs.items():
            in_s1 = any(f.lower() in sp.lower() for f in term(b, m, SPSS[k]))
            in_t2 = t2[model].get(T2[model][k]) == b
            n += 1
            ok += in_s1 and in_t2
            rows.append(f"| {model} | {k} | {b:g} | {m:g} | {'yes' if in_s1 else 'NO'} | {'yes' if in_t2 else 'NO'} |")
    for k, b in STROKE.items():
        in_t2 = t2["stroke"].get(T2["stroke"][k]) == b
        n += 1
        ok += in_t2
        rows.append(f"| stroke | {k} | {b:g} | - | (see 3) | {'yes' if in_t2 else 'NO'} |")
    s0_ok = "1-(0.994**" in sp and "1-(0.998**" in sp
    stroke_mean_ok = f"-{STROKE_MEAN}" in sp
    # 2. inferred values, from Table 1 (ASCVD-free cohort: women n 3965 = 452/0.114, men 4153 = 598/0.144)
    t1 = {"female": {"gfr": 114.0, "dev": 2.6, "sbp": 116.3, "dbp": 70.6, "map": 85.8, "age": 37.6, "sd_age": 14.1, "smk": 0.005, "htn": 0.150},
          "male": {"gfr": 108.9, "dev": -2.5, "sbp": 124.7, "dbp": 76.6, "map": 92.7, "age": 39.6, "sd_age": 15.4, "smk": 0.166, "htn": 0.209}}
    pm = ASCVD["male"][1]
    gfr_means = {s: v["gfr"] - v["dev"] for s, v in t1.items()}
    maps = {s: v["dbp"] + (v["sbp"] - v["dbp"]) / 3 for s, v in t1.items()}
    dev_x_male = pm * t1["male"]["dev"]
    mean_map = (1 - pm) * t1["female"]["map"] + pm * t1["male"]["map"]

    def stroke_mean(with_htn):
        f, m = t1["female"], t1["male"]
        avg = lambda k: (1 - pm) * f[k] + pm * m[k]
        return (STROKE["smoker"] * avg("smk") + STROKE["sbp"] * avg("sbp") + STROKE["age"] * avg("age")
                + STROKE["age_x_male"] * pm * m["age"] + with_htn * STROKE["hypertension_x_male"] * pm * m["htn"])

    bp_sum = sum(CAD[k][1] for k in ("bp_normal", "bp_elevated", "bp_stage1", "bp_stage2", "bp_crisis"))
    L = ["# Abu Dhabi Risk Study (ADRS): verification report", "", f"Generated {dt.date.today()} by `reference/adrs/report.py`.", "",
         "Evidence level C, and weak: the paper (J Am Heart Assoc 2024;13:e035930, open access, PMC11681588) gives the",
         "coefficients (Table 2) and the supplement (Data S1, `source/adrs_data_s1.pdf`) gives SPSS syntax with centring",
         "values and baseline survival, but there is no worked example, no calculator and no independent implementation.", "",
         "## 1. Transcription", "",
         "| Model | Term | Coefficient | Centring | In Data S1 syntax | Equals Table 2 B |", "|---|---|---:|---:|---|---|",
         *rows, "", f"{ok}/{n} terms confirmed in both sources (stroke: Table 2 only).",
         f"Baseline survival 0.994 (ASCVD, CAD) and 0.998 (stroke) found in Data S1: {'yes' if s0_ok else 'NO'};",
         f"stroke mean score {STROKE_MEAN} found: {'yes' if stroke_mean_ok else 'NO'}.", "",
         "## 2. Values the paper does not state", "",
         "- **Cohort GFR mean** (the deviation is entered, the mean is not printed): Table 1 gives GFR and deviation by sex;",
         f"  GFR - deviation = {gfr_means['female']:.1f} (women) and {gfr_means['male']:.1f} (men). The package uses {GFR_MEAN}.",
         f"  Consistency: the Data S1 centring of deviation x sex, -1.28, against {pm} x (-2.5) = {dev_x_male:.3f}.",
         f"- **Mean blood pressure** = DBP + (SBP - DBP)/3: Table 1 means give {maps['female']:.1f} (women; printed 85.8)",
         f"  and {maps['male']:.1f} (men; printed 92.7). Their sex-weighted mean, {mean_map:.2f}, against the Data S1 centring",
         "  89.32 (the cohort used for the centring is not stated exactly).",
         "- **CAD blood-pressure stages**: Table 2 names the ACC/AHA 2017 stages; crisis taken as SBP > 180 and/or DBP > 120.",
         f"  All five stages have a coefficient and their centring values sum to {bp_sum:.2f}, so with no sixth (reference)",
         "  category the five indicators are collinear in the fitted model. The printed formula is still well defined and",
         "  is used as printed.", "",
         "## 3. Stroke: two versions in Data S1", "",
         "Data S1 computes `sumrfFSTRK` twice; the second assignment (which SPSS would use) drops the hypertension x sex",
         "term that Table 2 lists. The mean score subtracted, 6.37, should equal the score at the cohort means. From Table 1",
         f"(sex-weighted, male share {pm}): {stroke_mean(True):.2f} with the term, {stroke_mean(False):.2f} without it. The printed",
         "6.37 fits the version with the term, which the package therefore uses.", "",
         "A smaller slip: the ASCVD block stores exp(score) in `ASCVD_SUMRF_` but raises 0.994 to `ASCVD_SUMRF` (no",
         "trailing underscore); the intent, 1 - S0 ** exp(score), is clear and is what the package computes.", "",
         "## 4. Owner action", "",
         "Ask the authors to confirm the cohort GFR mean (111.4?), the stroke formula, and to provide a few calculated",
         "risks; with those the evidence would rise above weak C.", ""]
    (HERE / "REPORT.md").write_text("\n".join(L) + "\n")
    print(f"terms {ok}/{n}; s0 {s0_ok}; stroke mean {stroke_mean_ok}; gfr {gfr_means}; stroke {stroke_mean(True):.3f}/{stroke_mean(False):.3f}")
    assert ok == n and s0_ok and stroke_mean_ok and abs(stroke_mean(True) - STROKE_MEAN) < abs(stroke_mean(False) - STROKE_MEAN)


if __name__ == "__main__":
    main()
