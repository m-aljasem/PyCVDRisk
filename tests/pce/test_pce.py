"""Verification of the 2013 Pooled Cohort Equations.

1. Every coefficient equals Table A of the guideline (check_coefficients.py).
2. The guideline's worked example (Table A/B, four race-sex groups).
3. Agreement with R CVrisk (rounding removed) on 20 000 synthetic patients; with R
   PooledCohort once its more precise constants are substituted.
4. Input handling.
"""

import copy
import importlib
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cvd_risk import pce
from cvd_risk.prevent.prevent import MG_PER_MMOL

REF = Path(__file__).parents[2] / "reference" / "pce"
mod = importlib.import_module("cvd_risk.pce.pce")
GROUPS = [("female", "white"), ("female", "african_american"), ("male", "white"), ("male", "african_american")]
# PooledCohort 0.0.2 (Goff_2013): constants with more decimals than Table A prints.
POOLEDCOHORT = {
    ("female", "african_american"): dict(
        ln_age=17.1141, ln_tc=0.9396, ln_hdl=-18.9196, ln_age_hdl=4.4748, ln_sbp_treated=29.2907,
        ln_age_sbp_treated=-6.4321, ln_sbp_untreated=27.8197, ln_age_sbp_untreated=-6.0873, smoker=0.6908,
        diabetes=0.8738, mean=86.6081, s0=0.95334),
    ("female", "white"): dict(mean=-29.1817, s0=0.96652),
    ("male", "african_american"): dict(mean=19.5425, s0=0.89536),
    ("male", "white"): dict(mean=61.1816, s0=0.91436),
}


def _load(name):
    spec = importlib.util.spec_from_file_location(name, REF / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_coefficients_match_table_a():
    assert _load("check_coefficients").main() == 0


def test_guideline_worked_example():
    # 55 y, TC 213, HDL 50 mg/dL, untreated SBP 120, non-smoker, no diabetes.
    sexes, races = zip(*GROUPS)
    r = pce(55, sexes, races, 213 / MG_PER_MMOL, 50 / MG_PER_MMOL, 120, 0, 0, 0)
    # Printed 2.1, 3.0, 5.3, 6.1%. White men: the exact sum is 60.6995 (printed 60.69,
    # truncated), giving 5.38%; the printed 5.3% follows from the printed sum.
    np.testing.assert_array_equal(np.round(100 * r, 1), [2.1, 3.0, 5.4, 6.1])
    assert round(100 * (1 - 0.9144 ** np.exp(60.69 - 61.18)), 1) == 5.3


@pytest.fixture(scope="module")
def ref():
    p = pd.read_csv(REF / "fixtures" / "patients.csv")
    r = pd.read_csv(REF / "fixtures" / "reference.csv")
    got = lambda: mod.pce(p.age, p.sex, p.race, p.total_chol_mgdl / MG_PER_MMOL,  # noqa: E731
                          p.hdl_chol_mgdl / MG_PER_MMOL, p.sbp, p.bp_treated, p.smoker, p.diabetes)
    return p, r, got


def test_matches_cvrisk(ref):
    p, r, got = ref
    g = got()
    na = r.cvrisk.isna()
    assert ((p.age[na] > 79) & (p.age[na] < 80)).all()  # CVrisk rejects 79.01-79.99
    np.testing.assert_allclose(g[~na], r.cvrisk[~na], rtol=1e-12, atol=1e-15)


def test_matches_pooledcohort_with_its_constants(ref, monkeypatch):
    p, r, got = ref
    m = copy.deepcopy(mod.MODELS)
    for g, upd in POOLEDCOHORT.items():
        m[g].update(upd)
    monkeypatch.setattr(mod, "MODELS", m)
    np.testing.assert_allclose(got(), r.pooledcohort, rtol=1e-12, atol=1e-15)


BASE = dict(age=55, sex="male", race="white", total_chol=5.5, hdl_chol=1.3, sbp=130, bp_treated=0,
            smoker=0, diabetes=0)


def test_other_race_uses_white_equations():
    assert pce(**{**BASE, "race": "other"}) == pce(**BASE)


@pytest.mark.parametrize("kw", [{"age": 39.9}, {"age": 80}, {"sex": "x"}, {"race": "martian"},
                                {"total_chol": np.nan}, {"hdl_chol": 0}, {"sbp": -1}, {"smoker": 2}])
def test_invalid_rows_are_nan(kw):
    assert np.isnan(pce(**{**BASE, **kw}))
