"""Jcvrisk 0.1.3's JALS functions against cvd_risk.jals(variant="acc_aha") (the online supplement) and the main text.
Reads reference/jcvrisk/fixtures (3,000 seeded patients run through Jcvrisk unmodified). Writes nothing; prints counts.
Run from the repo root: PYTHONPATH=src python3 reference/jals/compare_jcvrisk.py
"""
import csv

import numpy as np

from cvd_risk.jals import jals, jals_risk

FX = "reference/jcvrisk/fixtures"
ROWS = ((130, 139, 80, 89), (140, 159, 90, 99), (160, 179, 100, 109))
REP_BP = (110, 125, 135, 150, 170, 185)  # an SBP in each supplement BP row, with DBP 70


def jcvrisk_bp_row(s, d):
    """Jcvrisk's nested ifelse: the first row (from the lowest) whose SBP or DBP range matches."""
    if s < 120 and d < 80:
        return 0
    if 120 <= s <= 129 and d < 80:
        return 1
    return next((i + 2 for i, (a, b, c, e) in enumerate(ROWS) if a <= s <= b or c <= d <= e), 5)


def higher_bp_row(s, d):
    sc = int(np.digitize(s, (120, 130, 140, 160, 180)))
    return max(sc, int(np.digitize(d, (80, 80, 90, 100, 110))))


def main():
    pts = [r for r in csv.DictReader(open(f"{FX}/patients.csv")) if 40 <= float(r["age"]) < 90]
    out = {r["id"]: r for r in csv.DictReader(open(f"{FX}/jcvrisk_output.csv"))}
    col = lambda k: np.array([float(r[k]) for r in pts])  # noqa: E731
    sbp, dbp, age = col("sbp"), col("dbp"), col("age")
    jrow = np.array([jcvrisk_bp_row(s, d) for s, d in zip(sbp, dbp)])
    same = jrow == np.array([higher_bp_row(s, d) for s, d in zip(sbp, dbp)])
    n = len(pts)
    print(f"{n} patients aged 40-89; Jcvrisk's BP row = the higher category for {same.sum()}")
    for o in ("stroke", "ami", "composite", "cvd_death"):
        for m, tag in ((1, "af"), (0, "noaf")):
            jc = np.array([float(out[r["id"]][f"jals_{o}_{tag}_score"]) for r in pts])
            kw = dict(outcome=o, hdl=col("hdl") if o != "stroke" else None,
                      non_hdl=col("tc") - col("hdl") if o == "ami" else None,
                      bmi=col("bmi") if o == "cvd_death" else None,
                      egfr=col("egfr") if o in ("cvd_death", "ami") else None, **(dict(af=col("af")) if m else {}))
            args = (age, np.where(col("male") == 1, "male", "female"))
            rest = (col("ht_medication"), col("smoking"), col("t2dm"))
            sc = lambda s, d, v: np.clip(jals(*args, s, d, *rest, variant=v, **kw), 0, 60)  # noqa: E731
            supp, main_text = sc(sbp, dbp, "acc_aha"), sc(sbp, dbp, "jsh2014")
            supp_jrow = sc(np.array(REP_BP)[jrow], np.full(n, 70.0), "acc_aha")  # Jcvrisk's BP row
            line = (f"{o:9s} {tag:4s} score: supplement {np.sum(supp == jc)}/{n} (same BP row {np.sum((supp == jc)[same])}"
                    f"/{same.sum()}), with Jcvrisk's BP row {np.sum(supp_jrow == jc)}/{n}; main text {np.sum(main_text == jc)}/{n}")
            if m:  # Table 5 covers the AF models only; Jcvrisk prints risk in % to 2 decimals
                r = np.array([float(out[x["id"]][f"jals_{o}_af"]) for x in pts])
                ours = np.round(100 * jals_risk(jc, age, o, variant="acc_aha"), 2)
                d = np.abs(ours - r)
                line += f"; 5-year risk from its score {np.sum(d < 1e-9)}/{n} (max diff {d.max():.2f})"
            print(line)


if __name__ == "__main__":
    main()
