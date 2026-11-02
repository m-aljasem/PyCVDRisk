"""Write reference/predict/fixtures/<model>_patients.csv: random inputs for the seven PREDICT models.
Run from repo root.

Values are drawn where the authors' R code behaves as documented: whole-number ages (its input check
rejects others), whole-number BMI, creatinine, HbA1c and days (its bands test integer sequences),
lower-case sex codes. The R defects are demonstrated separately in report.py.
"""

from pathlib import Path

import numpy as np

FX = Path(__file__).parent / "fixtures"
ETH = ["european", "maori", "pacific", "indian", "asian"]


def flag(rng, n, p=0.3):
    return (rng.random(n) < p).astype(int)


def with_missing(rng, x, p=0.15):
    return np.where(rng.random(len(x)) < p, np.nan, x)


def common(rng, n, age_lo, age_hi, eth=ETH):
    return {"sex": rng.choice(["m", "f"], n), "age": rng.integers(age_lo, age_hi + 1, n), "ethnicity": rng.choice(eth, n),
            "nzdep": rng.integers(1, 6, n), "sbp": rng.integers(85, 211, n),
            "tc_hdl": np.round(rng.uniform(1.5, 9, n), 2)}


def main(n=3000, seed=2018):
    rng = np.random.default_rng(seed)
    out = {}
    p = common(rng, n, 30, 79)
    p.update(smoking=rng.choice(["never", "ex", "current"], n), **{k: flag(rng, n) for k in (
        "diabetes", "af", "family_history", "bp_meds", "lipid_meds", "antithrombotic")})
    out["cvd"] = p
    out["cvd_bmi"] = {**p, "bmi": with_missing(rng, rng.integers(15, 50, n))}
    q = common(rng, n, 30, 79, ETH + ["other"])
    del q["sbp"], q["tc_hdl"]
    q.update({k: flag(rng, n) for k in ("diabetes", "af", "bp_meds", "lipid_meds", "antithrombotic")})
    out["cvd_policy"] = q
    t = common(rng, n, 30, 79)
    t.update({k: flag(rng, n) for k in ("smoker", "af", "family_history", "lipid_meds", "antithrombotic", "bp_meds",
                                        "oral_hypoglycaemic", "insulin")},
             bmi=np.round(rng.uniform(18, 50, n), 1), egfr=rng.integers(10, 130, n),
             acr=np.round(rng.lognormal(0.5, 1.5, n), 2) * (rng.random(n) > 0.05), hba1c=rng.integers(35, 120, n),
             years_since_diagnosis=rng.integers(0, 30, n))
    out["t2d"] = t
    b = common(rng, n, 30, 79)
    b.update(smoking=rng.choice(["never", "ex", "current"], n), **{k: flag(rng, n, 0.2) for k in (
        "diabetes", "family_history", "lipid_meds", "bp_meds", "cancer", "gi_bleed", "peptic_ulcer", "alcohol",
        "liver", "peptic_ulcer_meds", "nsaid", "steroids", "ssri")})
    out["bleed"] = b
    for name, days in (("post_cvd", "days_since_event"), ("post_acs", "days_since_acs")):
        s = common(rng, n, 18, 100)
        s.update({k: flag(rng, n) for k in ("smoker", "diabetes", "af", "heart_failure", "bp_meds", "lipid_meds",
                                            "antithrombotic")},
                 bmi=with_missing(rng, rng.integers(15, 50, n)), creatinine=with_missing(rng, rng.integers(40, 300, n)),
                 hba1c=with_missing(rng, rng.integers(25, 110, n)),
                 **{days: with_missing(rng, rng.choice([*range(400), *range(0, 4000, 7), 181, 182, 365, 366, 1825,
                                                        1826], n))})
        out[name] = s
    out["post_cvd"]["other_vd"] = flag(rng, n, 0.2)
    out["post_acs"]["acs_type"] = rng.choice(["unstable_angina", "nstemi", "stemi"], n)
    FX.mkdir(exist_ok=True)
    for name, cols in out.items():
        keys = ["id", *cols]
        rows = zip(range(n), *cols.values())
        lines = [",".join(keys)] + [",".join("NA" if isinstance(v, float) and np.isnan(v) else str(v) for v in r)
                                    for r in rows]
        (FX / f"{name}_patients.csv").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
