"""Compare REGICOR with the official Framingham-REGICOR charts (fixtures/official_charts_transcribed.txt).

The charts (regicor.cat, "calibrades per al seu ús a la població catalana", non-diabetic) print 400
cells: sex x smoking x age band 35-44..65-74 x 5 blood-pressure x 5 total-cholesterol categories.
Two functions are compared:
1. this package (Marrugat 2003 JECH, Table 1 coefficients printed to 2 decimals, Gerona calibration);
2. Wilson 1998 full-precision categorical coefficients with the Spanish calibration of Marrugat 2003
   (Rev Esp Cardiol 56:253; G = 3.4881 / 10.2973, S10 = 0.9510 / 0.9780), as coded in the third-party
   R package cvrcalc 3.0 (r-forge/cvrcalc, pkg/R/cvrcalc.R, function regicor; GPL >= 2). Neither the
   Rev Esp Cardiol paper nor Wilson 1998 was available to check these constants.
Chart inputs are not printed; for each function the age within band, HDL category and rounding rule
that give the most exact cells are searched (`fit`).
Run from repo root: python reference/regicor/chart_check.py
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np

from cvd_risk.prevent.prevent import MG_PER_MMOL
from cvd_risk.regicor import regicor

FX = Path(__file__).parent / "fixtures" / "official_charts_transcribed.txt"
BANDS = ("35-44", "45-54", "55-64", "65-74")
BP_ROWS = {"0": 0, "120": 1, "130": 2, "140": 3, "160": 4}  # optimal .. stage II
BP_VALUES = ((110, 70), (125, 82), (135, 87), (150, 95), (170, 105))  # one (SBP, DBP) inside each category
TC_VALUES = (150, 180, 220, 260, 300)  # mg/dL, one inside each category
HDL_VALUES = (30, 40, 47, 55, 65)  # mg/dL, one inside each category
CVRCALC = {  # cvrcalc regicor(): categorical coefficients (category order as BP_VALUES etc.), G, S10
    "male": dict(age=0.04826, age2=0.0, tc=(-0.65945, 0, 0.17692, 0.50539, 0.65713),
                 hdl=(0.49744, 0.24310, 0, -0.05107, -0.48660), bp=(0.00226, 0, 0.28320, 0.52168, 0.61859),
                 smoker=0.52337, g=3.4881, s10=0.9510),
    "female": dict(age=0.33766, age2=-0.00268, tc=(-0.26138, 0, 0.20771, 0.24385, 0.53513),
                   hdl=(0.84312, 0.37796, 0.19785, 0, -0.42951), bp=(-0.53363, 0, -0.06773, 0.26288, 0.46573),
                   smoker=0.29246, g=10.2973, s10=0.9780),
}
ROUND = {"round": lambda p: np.floor(p + 0.5), "ceil": np.ceil, "floor": np.floor}


def cells():
    rows = [ln.split() for ln in FX.read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    return [(sex, int(sm), band, BP_ROWS[bp], k, int(v)) for sex, sm, band, bp, *vals in rows
            for k, v in enumerate(vals)]


def package(sex, age, bp, tc, hdl, smoker):
    s, d = BP_VALUES[bp]
    return 100 * float(regicor(age, sex, TC_VALUES[tc] / MG_PER_MMOL, HDL_VALUES[hdl] / MG_PER_MMOL, s, d, 0, smoker))


def cvrcalc(sex, age, bp, tc, hdl, smoker):
    c = CVRCALC[sex]
    lp = c["age"] * age + c["age2"] * age**2 + c["tc"][tc] + c["hdl"][hdl] + c["bp"][bp] + c["smoker"] * smoker
    return 100 * (1 - c["s10"] ** np.exp(lp - c["g"]))


def fit(fn, sex):
    """Best (exact cells, age offset, HDL category, rounding) for one sex; same offset in every band."""
    cs = [c for c in cells() if c[0] == sex]
    best = (0,)
    for off, hdl, rnd in itertools.product(np.arange(0, 10, 0.5), range(5), ROUND):
        ok = sum(ROUND[rnd](fn(sex, int(b[:2]) + off, bp, tc, hdl, sm)) == v for _, sm, b, bp, tc, v in cs)
        best = max(best, (int(ok), float(off), hdl, rnd))
    return best


def main():
    out = {}
    for name, fn in (("package (JECH 2003 Table 1)", package), ("cvrcalc (Wilson 1998 + Rev Esp Cardiol 2003)", cvrcalc)):
        for sex in ("male", "female"):
            ok, off, hdl, rnd = out[name, sex] = fit(fn, sex)
            print(f"{name}, {sex}: {ok}/200 exact (age = band start + {off}, HDL category {hdl}, {rnd})")
    return out


if __name__ == "__main__":
    main()
