"""Author test set for the Suita CVD risk model 2020 (run from the repo root:
PYTHONPATH=src python3 reference/suita_cvd/make_author_test_patients.py).

fixtures/author_test_patients.csv: 20 patients to send to the authors (inputs only): 8 hand-picked cases on the
category boundaries (SBP 139 / DBP 89, where Jcvrisk differs; discordant SBP/DBP and LDL-C/non-HDL-C; treated BP;
HDL-C 40 and 60; LDL-C 140 and non-HDL-C 170; age-band edges; AF and LVH), then 12 seeded patients aged 30-79
(seed 2020). Lipids in mg/dL; non-HDL-C = TC - HDL-C (erratum, JAT 2021;28:304); LDL-C by Friedewald.
fixtures/author_test_expected.csv: our Table 3 points and Table 4 band for both models, kept back for the comparison.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from cvd_risk.suita import suita_cvd, suita_cvd_risk

FIX = Path(__file__).parent / "fixtures"
N, SEED = 12, 2020
COLS = ["age", "sex", "sbp", "dbp", "bp_treated", "total_chol", "hdl_chol", "triglycerides", "smoker", "diabetes",
        "proteinuria", "af", "lvh"]
EDGE = [  # age sex sbp dbp treated tc hdl tg smoker dm protein af lvh
    (55, "male", 139, 70, 0, 200, 50, 100, 0, 0, 0, 0, 0),     # SBP 139: 120-139 band (Jcvrisk: lowest)
    (62, "female", 118, 89, 0, 210, 60, 120, 0, 0, 0, 0, 0),   # DBP 89 with SBP < 120; HDL-C 60
    (68, "male", 150, 85, 1, 190, 40, 150, 1, 0, 0, 0, 0),     # treated: top BP category; HDL-C 40
    (45, "female", 125, 95, 0, 222, 52, 185, 0, 1, 0, 0, 0),   # DBP 95 > SBP band; non-HDL 170, LDL 133
    (60, "male", 110, 70, 0, 230, 75, 75, 0, 0, 1, 0, 0),      # LDL-C 140, non-HDL 155; age 60 edge
    (65, "female", 160, 100, 0, 180, 39, 200, 1, 1, 1, 0, 0),  # SBP 160 and DBP 100 edges; HDL-C 39; age 65
    (75, "male", 145, 88, 0, 240, 45, 250, 0, 1, 0, 1, 1),     # AF and LVH; age 75 edge
    (79, "female", 135, 75, 1, 170, 70, 90, 0, 0, 0, 1, 0),    # oldest age; AF only
]


def main():
    rng = np.random.default_rng(SEED)
    tc, hdl = rng.integers(140, 301, N), rng.integers(30, 96, N)
    p = pd.DataFrame(dict(
        age=rng.integers(30, 80, N), sex=rng.choice(["male", "female"], N), sbp=rng.integers(100, 191, N),
        dbp=rng.integers(60, 111, N), bp_treated=(rng.random(N) < 0.3).astype(int), total_chol=tc, hdl_chol=hdl,
        triglycerides=rng.integers(50, 301, N), smoker=rng.integers(0, 2, N),
        diabetes=(rng.random(N) < 0.2).astype(int), proteinuria=(rng.random(N) < 0.15).astype(int),
        af=(rng.random(N) < 0.1).astype(int), lvh=(rng.random(N) < 0.1).astype(int)))
    p = pd.concat([pd.DataFrame(EDGE, columns=COLS), p], ignore_index=True)
    p.insert(0, "id", [f"SC{i:02d}" for i in range(1, len(p) + 1)])
    p.insert(p.columns.get_loc("triglycerides") + 1, "non_hdl_chol", p.total_chol - p.hdl_chol)
    p["triglycerides"] = p.triglycerides.clip(upper=5 * (p.non_hdl_chol - 60))  # keeps Friedewald LDL-C >= 60
    p.insert(p.columns.get_loc("non_hdl_chol") + 1, "ldl_chol", (p.non_hdl_chol - p.triglycerides / 5).round(1))
    p.to_csv(FIX / "author_test_patients.csv", index=False)
    a = (p.age, p.sex, p.sbp, p.dbp, p.bp_treated, p.hdl_chol, p.smoker, p.diabetes, p.proteinuria)
    e = p[["id"]].copy()
    e["points_ecg"] = suita_cvd(*a, ldl=p.ldl_chol, non_hdl=p.non_hdl_chol, af=p.af, lvh=p.lvh)
    e["points_ecg"] = e.points_ecg.astype(int)
    e["risk_ecg_pct"] = suita_cvd_risk(e.points_ecg, ecg=True)
    e["points_no_ecg"] = suita_cvd(*a, ldl=p.ldl_chol, non_hdl=p.non_hdl_chol)
    e["points_no_ecg"] = e.points_no_ecg.astype(int)
    e["risk_no_ecg_pct"] = suita_cvd_risk(e.points_no_ecg, ecg=False)
    assert e.notna().all().all() and (e.risk_ecg_pct != "").all()
    assert e.loc[0, "points_no_ecg"] == 14 + 4 + 0 - 2  # SC01: age 50-59, man, SBP 139 in 120-139 (0), HDL-C 50
    e.to_csv(FIX / "author_test_expected.csv", index=False)


if __name__ == "__main__":
    main()
