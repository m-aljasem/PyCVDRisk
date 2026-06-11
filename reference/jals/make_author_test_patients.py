"""Author test set for the JALS score (run from the repo root: PYTHONPATH=src python3 reference/jals/make_author_test_patients.py).

fixtures/author_test_patients.csv: 20 patients to send to the authors (inputs only): JA01 is the worked example of
Table 6 / Supplementary Table 5; JA02-JA05 have SBP and DBP in different blood-pressure categories (where the main
text, the supplement and Jcvrisk part); JA06-JA20 are seeded (seed 2019), aged 40-89, integer BP and lipids in mg/dL.
fixtures/author_test_expected.csv: our scores (Table 6 score = total minus age points) for the 8 models and the
5- and 10-year risks (%) of the AF models, for the main text (jsh2014) and the supplement (acc_aha), kept back.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from cvd_risk.jals import jals, jals_risk

FIX = Path(__file__).parent / "fixtures"
N, SEED = 15, 2019


def main():
    rng = np.random.default_rng(SEED)
    p = pd.DataFrame(dict(
        age=rng.integers(40, 90, N), sex=rng.choice(["male", "female"], N), sbp=rng.integers(105, 196, N),
        dbp=rng.integers(60, 116, N), bp_treated=(rng.random(N) < 0.35).astype(int), smoker=rng.integers(0, 2, N),
        diabetes=(rng.random(N) < 0.25).astype(int), af=(rng.random(N) < 0.15).astype(int),
        total_chol=rng.integers(150, 291, N), hdl_chol=rng.integers(32, 91, N),
        bmi=np.round(rng.uniform(17, 32, N), 1), egfr=rng.integers(35, 106, N)))
    ex = pd.DataFrame([dict(age=a, sex=s, sbp=b, dbp=d, bp_treated=t, smoker=0, diabetes=0, af=f, total_chol=200,
                            hdl_chol=55, bmi=23.0, egfr=75) for a, s, b, d, t, f in (
        (75, "male", 165, 85, 0, 0),  # worked example: stroke score 25, 5-year 3.08%
        (55, "female", 125, 85, 0, 0), (65, "male", 170, 85, 0, 0), (70, "female", 118, 82, 1, 0),
        (60, "male", 135, 95, 1, 1))])
    p = pd.concat([ex, p], ignore_index=True)
    p.insert(10, "non_hdl_chol", p.total_chol - p.hdl_chol)
    p.insert(0, "id", [f"JA{i:02d}" for i in range(1, len(p) + 1)])
    p.to_csv(FIX / "author_test_patients.csv", index=False)
    e = p[["id"]].copy()
    for v in ("jsh2014", "acc_aha"):
        for o in ("stroke", "ami", "composite", "cvd_death"):
            for m in ("af", "noaf"):
                s = jals(p.age, p.sex, p.sbp, p.dbp, p.bp_treated, p.smoker, p.diabetes, outcome=o, variant=v,
                         hdl=p.hdl_chol, non_hdl=p.non_hdl_chol, bmi=p.bmi, egfr=p.egfr,
                         **(dict(af=p.af) if m == "af" else {}))
                e[f"{v}_{o}_{m}_score"] = s
                if m == "af":
                    for y in (5, 10):
                        e[f"{v}_{o}_af_{y}y_pct"] = np.round(100 * jals_risk(s, p.age, o, y, v), 4)
    assert e.notna().all().all() and e.loc[0, "jsh2014_stroke_af_score"] == 25
    assert round(e.loc[0, "jsh2014_stroke_af_5y_pct"], 2) == 3.08 == round(e.loc[0, "acc_aha_stroke_af_5y_pct"], 2)
    e.to_csv(FIX / "author_test_expected.csv", index=False)


if __name__ == "__main__":
    main()
