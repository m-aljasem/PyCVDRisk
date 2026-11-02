"""Verification of QRISK2-2014.

1. The coefficient file is exactly what ClinRisk's C source prints.
2. Agreement with ClinRisk's C code, compiled unmodified and called with its own input
   validation, on 30 000 synthetic patients (NaN on exactly the rows ClinRisk rejects).
3. Input handling.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cvd_risk import qrisk2

REF = Path(__file__).parents[2] / "reference" / "qrisk2"
FLAGS = ("af", "rheumatoid_arthritis", "ckd", "bp_treated", "family_history")



# ClinRisk's C source is not redistributed here; see reference/README.md for where to download it.
@pytest.mark.skipif(not (REF / "source" / "c").exists(), reason="ClinRisk C source not downloaded (see reference/README.md)")
def test_coefficients_file_is_generated_from_clinrisk_source():
    spec = importlib.util.spec_from_file_location("q2x", REF / "extract_coefficients.py")
    ex = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ex)
    ms = ex.models()
    assert [len(m["terms"]) for m in ms.values()] == [42, 42]
    assert [m["age_powers"] for m in ms.values()] == [(0.5, 1), (-1, 2)]
    assert ex.OUT.read_text() == ex.render(ms)


def test_matches_clinrisk_c():
    p = pd.read_csv(REF / "fixtures" / "patients.csv")
    r = pd.read_csv(REF / "fixtures" / "reference.csv").clinrisk_c
    got = qrisk2(p.age, p.sex, p.ethnicity, p.smoking, p.bmi, p.chol_hdl_ratio, p.sbp,
                 townsend=p.townsend, diabetes=p.diabetes, **{f: p[f] for f in FLAGS})
    np.testing.assert_array_equal(np.isnan(got), r.isna())
    np.testing.assert_allclose(got, r, rtol=1e-12, atol=1e-15)


BASE = dict(age=50, sex="male", ethnicity="white_or_not_stated", smoking="non", bmi=25, chol_hdl_ratio=4, sbp=130)


def test_clinrisk_example():
    # The C code run directly on these inputs prints 4.140326751117418.
    np.testing.assert_allclose(qrisk2(**BASE), 0.04140326751117418, rtol=1e-14)


@pytest.mark.parametrize("kw", [
    {"age": 24.99}, {"age": 85}, {"bmi": 19.9}, {"bmi": 40.1}, {"chol_hdl_ratio": 12.1}, {"sbp": 69},
    {"townsend": 11.5}, {"townsend": np.nan}, {"sex": "x"}, {"ethnicity": "martian"}, {"diabetes": "type3"},
    {"af": 2},
])
def test_invalid_rows_are_nan(kw):
    assert np.isnan(qrisk2(**{**BASE, **kw}))


def test_bounds_are_inclusive():
    assert np.isfinite(qrisk2(**{**BASE, "age": 84.9, "bmi": 40, "chol_hdl_ratio": 1, "sbp": 210, "townsend": -7}))
