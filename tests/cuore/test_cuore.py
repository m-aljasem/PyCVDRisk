import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cvd_risk.cuore import cuore
from cvd_risk.cuore.cuore import COEF

FIXTURE = Path(__file__).parents[2] / "reference" / "cuore" / "fixtures" / "calculator_reference.csv"

# Giampaoli 2007 Table 1 (mean, SD) and Table 2 (HR for 1 SD higher / yes vs no), order of COEF
TABLE1 = {"male": ((50.6, 9.2), (138.7, 20.5), (225.4, 44.4), (50.4, 14.0), 0.401, 0.059, 0.094),
          "female": ((50.3, 8.5), (135.9, 22.0), (229.0, 44.6), (60.0, 15.4), 0.259, 0.038, 0.145)}
TABLE2_HR = {"male": (2.01, 1.31, 1.31, 1.60, 1.66, 1.59, 1.63),
             "female": (1.95, 1.41, 1.14, 1.26, 2.17, 1.40, 1.80)}


def test_matches_official_calculator():
    """ISS online calculator, one decimal; it takes two SBP readings and uses their mean rounded half-to-even."""
    g = pd.read_csv(FIXTURE, dtype={"shown": str})
    sbp = np.round((g.Sbp1 + g.Sbp2) / 2)
    r = cuore(g.Age, g.Sex.map({1: "male", 2: "female"}), sbp, g.Cholesterol, g.HDL, g.Smoke, g.Diabetes, g.HT)
    ours = [f"{x:.1f}" for x in np.floor(1000 * r + 0.5) / 10]
    assert len(g) == 300 and list(ours) == list(g.shown)


@pytest.mark.parametrize("sex", ["male", "female"])
def test_hazard_ratios_of_table2(sex):
    """HR = exp(beta x SD) (continuous) or exp(beta), within rounding of beta (3 decimals) and HR (2 decimals);
    the men's HDL entry (1.60) is not (REPORT.md, finding 2)."""
    b = COEF[sex][0]
    for i, (beta, t1, hr) in enumerate(zip(b, TABLE1[sex], TABLE2_HR[sex])):
        sd = t1[1] if isinstance(t1, tuple) else 1
        tol = 0.0005 * sd + 0.005 / hr
        ok = abs(abs(beta) * sd - math.log(hr)) <= tol
        assert ok == (sex != "male" or i != 3), (sex, i, math.exp(abs(beta) * sd), hr)


@pytest.mark.parametrize("sex", ["male", "female"])
def test_g_is_the_linear_predictor_at_the_means(sex):
    """G = sum(beta x mean), within the rounding of the 3-decimal betas."""
    b, g, _ = COEF[sex]
    means = [t[0] if isinstance(t, tuple) else t for t in TABLE1[sex]]
    assert abs(sum(bi * m for bi, m in zip(b, means)) - g) <= sum(0.0005 * m for m in means)


def test_ranges_and_monotonicity():
    edge = cuore([34, 35, 69, 70], "male", 140, 220, 50, 0, 0, 0)
    assert np.isnan(edge[[0, 3]]).all() and np.isfinite(edge[1:3]).all()
    for kw in ({"sbp": 89}, {"sbp": 201}, {"tc": 129}, {"tc": 321}, {"hdl": 19}, {"hdl": 101}):
        args = {"age": 50, "sex": "female", "sbp": 140, "tc": 220, "hdl": 50, "smoker": 0, "diabetes": 0,
                "bp_treatment": 0} | kw
        assert np.isnan(cuore(**args)), kw
    r = cuore(55, "male", [120, 140, 160], 220, 50, 0, 0, 0)
    assert (np.diff(r) > 0).all()
    assert cuore(55, "female", 140, 220, 50, 1, 0, 0) > cuore(55, "female", 140, 220, 50, 0, 0, 0)
