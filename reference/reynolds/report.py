"""Build reference/reynolds/REPORT.md. Run from repo root: python reference/reynolds/report.py

1. Every coefficient, as a term of the printed formula, must occur in the PDF text.
2. The printed worked examples (women: Table 6, 16 values; men: Clinical Examples, 1 value).
3. The developers' SAS code (run_cook_sas.py); the Wayback captures and third-party packages are described.
   Where plain rounding disagrees, the example is tested against the range of risks the formula can
   give with every printed number moved by up to half its last printed digit.
"""

from __future__ import annotations

import datetime as dt
import itertools
import logging
import re
from decimal import Decimal
from pathlib import Path

import numpy as np
import pypdf

from cvd_risk import reynolds
from cvd_risk.prevent.prevent import MG_PER_MMOL
from cvd_risk.reynolds.coefficients import MEN, WOMEN

import run_cook_sas  # same directory

HERE = Path(__file__).parent
logging.getLogger("pypdf").setLevel(logging.ERROR)
GLYPHS = {"/H11001": "+", "/H11003": "×", "−": "-", "–": "-", "x ": "× "}


def text(name):
    t = " ".join(p.extract_text() or "" for p in pypdf.PdfReader(HERE / "source" / name).pages)
    for k, v in GLYPHS.items():
        t = t.replace(k, v)
    return re.sub(r"\s+", "", t).replace("lipopro-tein", "lipoprotein")  # whitespace-free


QUOTES = {
    "ridker2007.pdf": {"s0": "[1-0.98634(exp[B-22.325])]", "mean": "[1-0.98634(exp[B-22.325])]",
                       "age": "B=0.0799×age", "ln_sbp": "3.137×naturallogarithm(systolicbloodpressure)",
                       "ln_hscrp": "0.180×naturallogarithm(high-sensitivityC-reactiveprotein)",
                       "ln_tc": "1.382×naturallogarithm(totalcholesterol)",
                       "ln_hdl": "-1.172×naturallogarithm(high-densitylipoproteincholesterol)",
                       "hba1c_if_diabetic": "0.134×hemoglobinA1c(%)(ifdiabetic)+0.818(ifcurrentsmoker)",
                       "smoker": "0.818(ifcurrentsmoker)", "parental_history": "0.438(iffamilyhistory"},
    "ridker2008.pdf": {"s0": "[1-0.8990(exp[B-33.097])]", "mean": "[1-0.8990(exp[B-33.097])]",
                       "ln_age": "B=4.385×naturallogarithm(age)", "ln_sbp": "2.607×naturallogarithm(systolicbloodpressure)",
                       "ln_tc": "0.963×naturallogarithm(totalcholesterol)",
                       "ln_hdl": "-0.772×naturallogarithm(high-densitylipoproteincholesterol)",
                       "smoker": "0.405(ifcurrentsmoker)", "ln_hscrp": "0.102×naturallogarithm(high-sensitivityC-reactiveprotein)",
                       "parental_history": "0.541(ifparentalhistory"},
}
TABLE6 = [(0.1, 4.9, 7.5), (0.5, 6.5, 9.9), (1.0, 7.4, 11.2), (3.0, 8.9, 13.4), (5.0, 9.7, 14.6),
          (8.0, 10.5, 15.8), (10.0, 10.9, 16.4), (20.0, 12.3, 18.4)]  # hsCRP, model B without / with parental history


def half_digit(v):
    return float(Decimal(5) * Decimal(10) ** (Decimal(repr(v)).as_tuple().exponent - 1))


def bounds(coefs, terms):
    """Min and max risk (%) over coefficients moved by up to half their last printed digit."""
    keys = [k for k in coefs if k not in ("s0", "mean")]
    out = []
    # risk rises with each term's contribution and the mean of B, falls with s0: the extremes are corners
    for signs in itertools.product((-1, 1), repeat=2):
        lo_b = sum((coefs[k] - np.sign(terms[k]) * half_digit(coefs[k])) * terms[k] for k in keys)
        hi_b = sum((coefs[k] + np.sign(terms[k]) * half_digit(coefs[k])) * terms[k] for k in keys)
        s0 = coefs["s0"] + signs[0] * half_digit(coefs["s0"])
        mean = coefs["mean"] + signs[1] * half_digit(coefs["mean"])
        out += [100 * (1 - s0 ** np.exp(x - mean)) for x in (lo_b, hi_b)]
    return min(out), max(out)


def main():
    found = {f: {k: q in text(f) for k, q in qs.items()} for f, qs in QUOTES.items()}
    n_found = sum(sum(v.values()) for v in found.values())
    n_q = sum(len(v) for v in found.values())
    assert set(QUOTES["ridker2007.pdf"]) == set(WOMEN) and set(QUOTES["ridker2008.pdf"]) == set(MEN)

    rows, n_round, n_bound = [], 0, 0
    examples = [("woman", 50, "female", 155, 240, 35, crp, 1, fh, printed)
                for crp, p0, p1 in TABLE6 for fh, printed in ((0, p0), (1, p1))]
    examples.append(("man", 65, "male", 130, 205, 45, 4.0, 0, 1, 20.4))
    for who, age, sex, sbp, tc, hdl, crp, sm, fh, printed in examples:
        got = 100 * float(reynolds(age, sex, sbp, tc / MG_PER_MMOL, hdl / MG_PER_MMOL, crp, sm, fh))
        if sex == "female":
            terms = {"age": age, "ln_sbp": np.log(sbp), "ln_hscrp": np.log(crp), "ln_tc": np.log(tc),
                     "ln_hdl": np.log(hdl), "hba1c_if_diabetic": 0, "smoker": sm, "parental_history": fh}
            lo, hi = bounds(WOMEN, terms)
        else:
            terms = {"ln_age": np.log(age), "ln_sbp": np.log(sbp), "ln_tc": np.log(tc), "ln_hdl": np.log(hdl),
                     "smoker": sm, "ln_hscrp": np.log(crp), "parental_history": fh}
            lo, hi = bounds(MEN, terms)
        rounds = round(got, 1) == printed
        within = lo <= printed + 0.05 and hi >= printed - 0.05
        n_round += rounds
        n_bound += within
        rows.append(f"| {who} | {crp} | {'yes' if fh else 'no'} | {printed} | {got:.3f} | "
                    f"{'yes' if rounds else 'no'} | {lo:.3f}-{hi:.3f} | {'yes' if within else 'NO'} |")

    cook_lines = run_cook_sas.main()
    L = ["# Reynolds Risk Score: verification report", "",
         f"Generated by `reference/reynolds/report.py`.", "",
         "## 1. Source", "",
         "Model B, \"the Reynolds Risk Score\": women from Ridker et al., JAMA 2007 (reprinted with",
         "corrections; Box), men from Ridker et al., Circulation 2008 (Appendix C). Both PDFs are in",
         f"`source/`. Each coefficient is quoted as a term of the printed formula: {n_found}/{n_q} found in the PDF text.", "",
         "Evidence: level A for men (the developers' SAS code, section 3, agrees exactly); level A with one",
         "documented discrepancy for women (the SAS code's baseline survival differs from the paper's;",
         "the paper's own Table 6 supports the printed value); plus level C, the papers' worked examples.", "",
         "## 2. Worked examples", "",
         "Women: Table 6 (age 50, smoker, no diabetes, BP 155/85, total cholesterol 240, HDL 35 mg/dL).",
         "Men: Clinical Examples (age 65, non-smoker, SBP 130, total cholesterol 205, HDL 45 mg/dL, hsCRP 4.0,",
         "parental history), model B. Printed risks have one decimal.", "",
         "| Sex | hsCRP | Parental history | Printed % | This package % | Rounds to printed | Range at printed precision | Printed in range |",
         "|---|---:|---|---:|---:|---|---|---|", *rows, "",
         f"{n_round}/{len(examples)} round to the printed value; {n_bound}/{len(examples)} lie within the range",
         "the formula gives when every printed number (coefficients, S0 and mean) moves by up to half",
         "its last printed digit. The papers computed with unrounded coefficients, so differences of",
         "one in the last printed decimal are expected.", "", *cook_lines,
         "## 4. The official calculator (reynoldsriskscore.org)", "",
         "Wayback CDX for reynoldsriskscore.org (2007-2026) lists default.aspx, home.aspx, Details.aspx, faq.aspx,",
         "commonScripts.js, styles.css and images; from 2024 the domain is parked. Captures saved in",
         "`source/wayback/` (file name = page_capture date). The calculator was an ASP.NET form that posted to",
         "the server: the risk came back in hidden fields (`resultRisk`, `resultOptimal`, ...), and",
         "`commonScripts.js` only draws the bars and converts mmol/L to mg/dL (x 39 for both total and HDL",
         "cholesterol). No result page, client-side formula, spreadsheet or handheld version was captured, so",
         "the site cannot be compared. Input rules shown on the form: age at most 80, hsCRP accepted up to",
         "50 mg/L, diabetes excluded (FAQ: people with diabetes should not use the score), parental history =",
         "heart attack or stroke before 60. This package enforces none of these (documented in its docstring).", "",
         "## 5. Third-party implementations", "",
         "- **cliot 1.0.0** (CRAN, N. Agarwal, 2025; `source/third_party/cliot_1.0.0.tar.gz`, sha256",
         "  83794e0c8a3f15434a61678558bcc19e30bc42139122f797183727ac019aa1bd): `reynolds_risk_score_women()` only.",
         "  It uses **3.193** for ln(SBP) where the paper prints 3.137, and its help page suggests entering",
         "  HbA1c 5.0 for non-diabetic women, whereas the paper enters HbA1c only if diabetic. `run_cliot.R`",
         "  (function cut out and run unchanged): with HbA1c 0 for non-diabetics, 5/978 seeded women equal this",
         "  package at its one-decimal output, cliot higher by 0.4-54.6%, and Table 6 is not reproduced (e.g.",
         "  hsCRP 1, no parental history: 8.6 vs printed 7.4). With 3.193 replaced by 3.137, 978/978 equal",
         "  (max 0.05 points, its rounding). Classification: cliot's error (coefficient typo; HbA1c advice).",
         "- **CVrisk 1.2.1** (CRAN): no Reynolds function in the source (`source/third_party/CVrisk_1.2.1.tar.gz`,",
         "  sha256 cddba08b5978a281db07939c8bc9dbad2b51162e98c855e8fed6e13b3571e630).",
         "- **MDCalc** (calc 3932, women only): the page prints the paper's formula (3.137, S0 0.98634, mean",
         "  22.325); its computation code is not in the page, so it was not run."]
    (HERE / "REPORT.md").write_text("\n".join(L) + "\n")
    print(f"quotes {n_found}/{n_q}; examples round {n_round}/{len(examples)}, in range {n_bound}/{len(examples)}")
    assert n_found == n_q and n_bound == len(examples)


if __name__ == "__main__":
    main()
