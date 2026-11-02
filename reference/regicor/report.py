"""Build reference/regicor/REPORT.md. Run from repo root: python reference/regicor/report.py

1. Every coefficient (Table 1), mean and event rate (Table 2) is rebuilt as the printed row and found
   in the PDF text; every term of the Appendix example is matched to the tables.
2. Every band gives its coefficient at both ends (inputs in mg/dL, converted).
3. The Appendix worked example (man, 50) is recomputed.
"""

from __future__ import annotations

import datetime as dt
import logging
import math
import re
from pathlib import Path

import pypdf

from cvd_risk import regicor
from cvd_risk.prevent.prevent import MG_PER_MMOL as K
from cvd_risk.regicor import mean_lp
from cvd_risk.regicor.coefficients import BETA, MEANS

HERE = Path(__file__).parent
logging.getLogger("pypdf").setLevel(logging.ERROR)
SEXES = ("male", "female")
LABELS = {
    "tc": ("<160", "160-199", "200-239", "240-279", ">280"),
    "hdl": ("<35", "35-44", "45-49", "50-59", ">59"),
    "bp": ("Optimal (S<120)/(D<80)", "Normal (S 120-129)/(D 80-84)", "High normal (S 130-139)/(D 85-89)",
           "Stage I (S 140-159)/(D90-99)", "Stage II-IV (S>160)/(D>100)"),
}
EDGES = {"tc": [(100, 159.99), (160, 199.99), (200, 239.99), (240, 279.99), (280, 400)],
         "hdl": [(20, 34.99), (35, 44.99), (45, 49.99), (50, 59.99), (60, 100)],
         "sbp": [(100, 119.99), (120, 129.99), (130, 139.99), (140, 159.99), (160, 220)],
         "dbp": [(60, 79.99), (80, 84.99), (85, 89.99), (90, 99.99), (100, 120)]}
BASE = {"age": 50, "total_chol": 180 / K, "hdl_chol": 55 / K, "sbp": 125, "dbp": 82, "diabetes": 0, "smoker": 0}


def text():
    t = " ".join(p.extract_text() or "" for p in pypdf.PdfReader(HERE / "source" / "marrugat2003.pdf").pages)
    return re.sub(r"\s+", " ", t.replace("−", "-").replace("–", "-"))


def table_rows():
    """(description, regex) for every printed row of Tables 1 and 2."""
    ci = r"(?: \([^)]*\))?"
    f2 = lambda v: re.escape(f"{v:.2f}")
    rows = [("T1 age", rf"Age \(y\) Mean {f2(BETA['male']['age'])}{ci} {f2(BETA['female']['age'])}"),
            ("T1 age squared", rf"Mean squared - {BETA['female']['age2']:.3f}"),
            ("T1 diabetes", rf"Diabetes {f2(BETA['male']['diabetes'])}{ci} {f2(BETA['female']['diabetes'])}"),
            ("T1 smoking", rf"consumption {f2(BETA['male']['smoker'])}{ci} {f2(BETA['female']['smoker'])}")]
    for k, labels in LABELS.items():
        for i, lab in enumerate(labels):
            rows.append((f"T1 {k} {lab}",
                         rf"{re.escape(lab)} {f2(BETA['male'][k][i])}{ci} {f2(BETA['female'][k][i])}"))
    cols = [MEANS[(s, c)] for s in SEXES for c in ("framingham", "gerona")]
    vals = lambda key, i=None: " ".join(f"{(m[key] if i is None else m[key][i]):.1f}" for m in cols
                                        if (m[key] if i is None else m[key][i]))
    rows += [("T2 age", f"Age \\(y\\) Mean {vals('age')}"), ("T2 age squared", f"Mean squared {vals('age2')}"),
             ("T2 diabetes", f"Diabetes {vals('diabetes')}"), ("T2 smoking", f"consumption {vals('smoker')}"),
             ("T2 event rate", "infarction " + " ".join(f"{m['event10']:.3f}" for m in cols))]
    t2_labels = {**LABELS, "bp": LABELS["bp"][:4] + ("Stage III-V (S>160)/(D>100)",)}
    for k, labels in t2_labels.items():
        for i, lab in enumerate(labels):
            v = vals(k, i)
            rows.append((f"T2 {k} {lab}", re.escape(f"{lab} {v}" if v else lab)))
    return rows


def lp_diff(sex, **kw):
    """log-hazard difference from the base person, recovered from two risks."""
    r0, r = (float(regicor(sex=sex, **{**BASE, **x})) for x in ({}, kw))
    return math.log(math.log(1 - r) / math.log(1 - r0))


def bands():
    rows, ok, n = [], 0, 0
    ref = {"tc": 1, "hdl": 3, "bp": 1}  # BASE categories
    for sex in SEXES:
        for name, key, other in (("tc", "total_chol", None), ("hdl", "hdl_chol", None),
                                 ("sbp", "sbp", {"dbp": 70}), ("dbp", "dbp", {"sbp": 110})):
            k = "bp" if name in ("sbp", "dbp") else name
            for i, (lo, hi) in enumerate(EDGES[name]):
                conv = (lambda v: v / K) if name in ("tc", "hdl") else (lambda v: v)
                want = BETA[sex][k][i] - BETA[sex][k][ref[k]]
                got = [lp_diff(sex, **(other or {}), **{key: conv(v)}) for v in (lo, hi)]
                n += 1
                good = all(abs(g - want) < 1e-9 for g in got)
                ok += good
                rows.append(f"| {sex} | {name} | {lo}-{hi} | {got[0]:+.3f} / {got[1]:+.3f} | {want:+.2f} |"
                            f" {'ok' if good else 'MISMATCH'} |")
    return rows, ok, n


def example(t):
    """The Appendix man: 50, diabetic, non-smoker, TC 210, HDL 52 mg/dL, high-normal BP."""
    man = {"age": 50, "sex": "male", "total_chol": 210 / K, "hdl_chol": 52 / K, "sbp": 135, "dbp": 80,
           "diabetes": 1, "smoker": 0}
    b = BETA["male"]
    sum_bx = b["age"] * 50 + b["tc"][2] + b["hdl"][3] + b["bp"][2] + b["diabetes"]
    out = {"sum_bx": sum_bx}
    for cal, printed_mean, printed_risk in (("framingham", 3.84, 0.068), ("gerona", 4.32, 0.047)):
        m = MEANS[("male", cal)]
        s = 1 - m["event10"]
        # terms written out in the Appendix, in order; matched to Tables 1-2
        start = t.find(f"0.05({m['age']})")
        seg = t[start:t.find(f"={printed_mean:.2f}", start)]
        terms = [(float(c.replace(" ", "")), float(x)) for c, x in re.findall(r"([+-]?\s?\d\.\d+)\((\d+\.?\d*)\)", seg)]
        table = [(b["age"], m["age"])] + [(b[k][i], m[k][i] / 100) for k in ("tc", "hdl", "bp")
                                          for i in range(5) if b[k][i] and m[k][i]] \
            + [(b["diabetes"], m["diabetes"] / 100), (b["smoker"], m["smoker"] / 100)]
        out[cal] = {
            "terms_match": [(round(c, 2), round(x, 3)) for c, x in terms] == [(round(c, 2), round(x, 3)) for c, x in table],
            "n_terms": len(terms),
            "recomputed_mean": mean_lp("male", cal), "printed_mean": printed_mean,
            "printed_risk": printed_risk,
            "risk_from_printed": 1 - s ** math.exp(round(sum_bx, 2) - printed_mean),
            "risk_sign_flipped": 1 - s ** math.exp(printed_mean - round(sum_bx, 2)),
            "package": float(regicor(calibration=cal, **man)),
            "age_coef_implied": (printed_mean - (mean_lp("male", cal) - b["age"] * m["age"])) / m["age"],
        }
    return out


def main():
    t = text()
    trows = table_rows()
    found = [(d, bool(re.search(rx, t))) for d, rx in trows]
    brows, bok, bn = bands()
    ex = example(t)
    mF, mG = ex["framingham"], ex["gerona"]
    ex_ok = (round(ex["sum_bx"], 2) == 4.02 and mF["terms_match"] and mG["terms_match"]
             and round(mF["risk_from_printed"], 3) == 0.068 and round(mG["risk_from_printed"], 3) == 0.026
             and round(mG["risk_sign_flipped"], 3) == 0.047 and "exp(4.02-4.32)=0.741" in t)
    w = MEANS[("female", "gerona")]
    L = ["# REGICOR: verification report", "", f"Generated {dt.date.today()} by `reference/regicor/report.py`.", "",
         "Evidence level C: Marrugat et al., J Epidemiol Community Health 2003 (`source/marrugat2003.pdf`),",
         "Table 1 (coefficients), Table 2 (Framingham and Gerona means, 10-year event rates) and the",
         "Appendix (formula, one worked example for a man). No independent implementation was available.", "",
         "## 1. Transcription", "",
         f"{sum(f for _, f in found)}/{len(found)} rows of Tables 1-2, rebuilt from `coefficients.py`, are found in the PDF text.",
         *[f"- NOT FOUND: {d}" for d, f in found if not f], "",
         "## 2. Bands", "",
         "Each band at both ends, against a base person aged 50 (TC 180, HDL 55 mg/dL, BP 125/82). The",
         "log-hazard change is recovered from the two risks. SBP bands are tested with DBP 70, DBP bands",
         "with SBP 110 (so the other reading is in the lowest category).", "",
         "| Sex | Input | Band | Log-hazard change at low / high | Table 1 difference | |", "|---|---|---|---|---:|---|",
         *brows, "", f"{bok}/{bn} bands give the Table 1 coefficient at both ends.", "",
         "## 3. Appendix worked example", "",
         "Man aged 50, diabetic, non-smoker, total cholesterol 210, HDL 52 mg/dL, high-normal blood pressure.", "",
         "| Step | Framingham | Gerona |", "|---|---:|---:|",
         f"| sum(b*X), printed 4.02 | {ex['sum_bx']:.4f} | {ex['sum_bx']:.4f} |",
         f"| Terms of sum(b*Xmean) written out in the Appendix match Tables 1-2 | {'yes' if mF['terms_match'] else 'NO'} ({mF['n_terms']}) | {'yes' if mG['terms_match'] else 'NO'} ({mG['n_terms']}) |",
         f"| sum(b*Xmean), printed | {mF['printed_mean']:.2f} | {mG['printed_mean']:.2f} |",
         f"| sum(b*Xmean), those terms added up | {mF['recomputed_mean']:.4f} | {mG['recomputed_mean']:.4f} |",
         f"| Risk from the printed sums (printed {mF['printed_risk']} / {mG['printed_risk']}) | {mF['risk_from_printed']:.4f} | {mG['risk_from_printed']:.4f} |",
         f"| Same with the sign of the exponent flipped | {mF['risk_sign_flipped']:.4f} | {mG['risk_sign_flipped']:.4f} |",
         f"| Risk from this package | {mF['package']:.4f} | {mG['package']:.4f} |",
         f"| Age coefficient that would give the printed sum | {mF['age_coef_implied']:.4f} | {mG['age_coef_implied']:.4f} |", "",
         ("The Appendix's written-out terms are exactly Table 1 x Table 2, but they add up to "
         f"{mF['recomputed_mean']:.2f} and {mG['recomputed_mean']:.2f}, not the printed 3.84 and 4.32. Both printed sums are"),
         f"reproduced if the age term uses a coefficient of about {mF['age_coef_implied']:.4f} instead of the printed 0.05 (the",
         "same value fits both sums), while sum(b*X) = 4.02 uses 0.05. So the example probably mixes an",
         "unrounded age coefficient (means) with the rounded one (patient).",
         "",
         "A second slip in the Gerona step: the paper prints exp(4.02-4.32) = 0.741 (correct), then",
         f"1 - 0.965^0.741 = 0.047. In fact 1 - 0.965^0.741 = {mG['risk_from_printed']:.3f}; 0.047 is 1 - 0.965^exp(+0.30), the",
         "exponent with its sign flipped. The paper's remark that recalibration gives \"approximately two",
         "thirds\" of the Framingham risk (0.047/0.068) rests on that slip. The Framingham step (0.068) is",
         "consistent with its printed sums.",
         "",
         ("This package uses Table 1 and Table 2 throughout and gives "
         f"{mF['package']:.3f} and {mG['package']:.3f}."),
         "Owner action: ask the authors for the unrounded coefficients (or check Wilson et al.,",
         "Circulation 1998;97:1837-47, which the function comes from).", "",
         "## 4. Other notes", "",
         f"- Women, Gerona: Table 2 gives a mean age of {w['age']} and a mean squared age of {w['age2']}, below",
         f"  {w['age']}^2 = {w['age'] ** 2:.1f}; a mean of squares cannot be below the square of the mean. The",
         f"  effect on the linear predictor is {abs(BETA['female']['age2']) * (w['age'] ** 2 - w['age2']):.4f}. The package uses the printed value.",
         "- Women's age coefficients are printed as 0.17 and -0.001 only, and the paper has no example for women.",
         "- Table 1 writes the top bands as \">280\" and \"S>160\"/\"D>100\"; Table 2 writes \">=280\" and \">=160\"/\">=100\".",
         "  The package uses >=, so no value falls between bands.",
         "- v1 of this package used continuous ln-coefficients and a constant that do not appear in this paper.", ""]
    (HERE / "REPORT.md").write_text("\n".join(L) + "\n")
    nf = sum(f for _, f in found)
    print(f"rows {nf}/{len(found)}; bands {bok}/{bn}; example ok {ex_ok}; "
          f"recomputed {mF['recomputed_mean']:.4f} {mG['recomputed_mean']:.4f}")
    assert nf == len(found) and bok == bn and ex_ok


if __name__ == "__main__":
    main()
