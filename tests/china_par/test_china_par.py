import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cvd_risk.china_par import china_par
from cvd_risk.china_par.china_par import COEF

mod = importlib.import_module("cvd_risk.china_par.china_par")

FIXTURE = Path(__file__).parents[2] / "reference" / "china_par" / "fixtures" / "calculator_reference.csv"
# Yang 2016 Supplemental Table 1 (2 decimals)
PRINTED = {"male": dict(ln_age=31.97, ln_sbp_treated=27.39, ln_sbp_untreated=26.15, ln_tc=0.62, ln_hdl=-0.69,
                        ln_waist=-0.71, smoker=3.96, diabetes=0.36, north=0.48, urban=-0.16, family_history=6.22,
                        ln_age_ln_sbp_treated=-6.02, ln_age_ln_sbp_untreated=-5.73, ln_age_smoker=-0.94,
                        ln_age_family_history=-1.53),
           "female": dict(ln_age=24.87, ln_sbp_treated=20.71, ln_sbp_untreated=19.98, ln_tc=0.06, ln_hdl=-0.22,
                          ln_waist=1.48, smoker=0.49, diabetes=0.57, north=0.54, ln_age_ln_sbp_treated=-4.53,
                          ln_age_ln_sbp_untreated=-4.36)}


def _ours(g):
    return china_par(g.age, g.sex.map({1: "male", 2: "female"}), g.sbp, g.bp_treated, g.tc * 0.0259, g.hdl * 0.0259, g.waist, g.smoker,
                     g.diabetes, g.north, g.urban, g.family_history)


@pytest.mark.parametrize("seed", [0, 1])
def test_matches_official_calculator(seed):
    g = pd.read_csv(FIXTURE)
    g = g[g.seed == seed]
    shown = [f"{x:.1f}" for x in g.risk_shown]
    ours = [f"{x:.1f}" for x in np.floor(1000 * _ours(g) + 0.5) / 10]
    miss = [(i, a, b) for i, a, b in zip(g.id, ours, shown) if a != b]
    assert len(g) >= 150 and not miss, miss


@pytest.mark.parametrize("sex", ["male", "female"])
def test_coefficients_round_to_the_paper(sex):
    c = COEF[sex]
    assert set(c) - {"intercept"} == set(PRINTED[sex])
    for k, v in PRINTED[sex].items():
        assert abs(c[k] - v) <= 0.005 + 1e-9, (sex, k, c[k], v)


def test_out_of_range_is_nan():
    r = china_par([19, 60], "male", 130, 0, 5.4, 1.4, 80, 0, 1, 1, 1, 0)
    assert np.isnan(r[0]) and 0 < r[1] < 1


def test_table2_example():
    """Table 2 profile (untreated SBP 130, TC 210, HDL 55, waist 80, non-smoker, diabetes, urban north): the calculator
    equals ours 8/8; the paper's Table 2 differs by up to 0.3 points in 5 of 8 (REPORT.md, findings)."""
    t = pd.read_csv(FIXTURE.parent / "table2.csv")
    r = 100 * china_par(t.age, t.sex.map({1: "male", 2: "female"}), 130, 0, 210 * 0.0259, 55 * 0.0259, 80, 0, 1, 1, 1, 0)
    assert (np.floor(10 * r + 0.5) / 10 == t.calculator_shown).all()
    assert (np.abs(t.paper_table2 - t.calculator_shown) > 0.05).sum() == 5


def test_held_out_check(monkeypatch):
    """Coefficients recovered from seed 0 alone, applied to the 150 held-out patients of seed 1: 144 equal, the other 6
    one decimal off, each within 0.015 point of a rounding edge."""
    seed0 = json.loads((FIXTURE.parent / "recovered_coefficients_seed0.json").read_text())
    monkeypatch.setattr(mod, "COEF", {"male": seed0["men"], "female": seed0["women"]})
    g = pd.read_csv(FIXTURE)
    g = g[g.seed == 1]
    r = 100 * _ours(g)
    off = np.floor(10 * r + 0.5) / 10 != g.risk_shown
    assert len(g) == 150 and off.sum() == 6
    assert (np.abs(r[off] - g.risk_shown[off]) < 0.065).all()  # shown +- 0.05 is the interval; 0.015 past its edge
