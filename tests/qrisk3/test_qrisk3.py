"""Verification of QRISK3-2017.

1. The coefficient file is exactly what ClinRisk's C source prints.
2. Agreement with ClinRisk's C code, compiled unmodified, on 30 000 synthetic patients
   (and with the CRAN package QRISK3, which matches the C code bit for bit).
3. Input handling.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cvd_risk import qrisk3

REF = Path(__file__).parents[2] / "reference" / "qrisk3"
FLAGS = ("af", "atypical_antipsychotics", "corticosteroids", "erectile_dysfunction", "migraine",
         "rheumatoid_arthritis", "ckd", "severe_mental_illness", "sle", "bp_treated", "family_history")


def test_coefficients_file_is_generated_from_clinrisk_source():
    spec = importlib.util.spec_from_file_location("extract", REF / "extract_coefficients.py")
    ex = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ex)
    ms = ex.models()
    assert [len(m["terms"]) for m in ms.values()] == [54, 55]
    assert ex.OUT.read_text() == ex.render(ms)


@pytest.mark.parametrize("column", ["clinrisk_c", "qrisk3_cran"])
def test_matches_reference(column):
    p = pd.read_csv(REF / "fixtures" / "patients.csv")
    r = pd.read_csv(REF / "fixtures" / "reference.csv")
    got = qrisk3(p.age, p.sex, p.ethnicity, p.smoking, p.bmi, p.chol_hdl_ratio, p.sbp, p.sbp_sd,
                 townsend=p.townsend, diabetes=p.diabetes, **{f: p[f] for f in FLAGS})
    assert not np.isnan(got).any()
    np.testing.assert_allclose(got, r[column], rtol=1e-12, atol=1e-15)


BASE = dict(age=64, sex="female", ethnicity="white_or_not_stated", smoking="non", bmi=25,
            chol_hdl_ratio=4, sbp=125, sbp_sd=0)


def test_clinrisk_example():
    # The C code run directly on these inputs prints 7.1534634471147047.
    np.testing.assert_allclose(qrisk3(**BASE), 0.071534634471147047, rtol=1e-14)


def test_age_is_truncated_to_whole_years():
    assert qrisk3(**{**BASE, "age": 64.99}) == qrisk3(**BASE)


def test_erectile_dysfunction_ignored_for_women():
    assert qrisk3(**BASE, erectile_dysfunction=1) == qrisk3(**BASE)
    m = {**BASE, "sex": "male"}
    assert qrisk3(**m, erectile_dysfunction=1) > qrisk3(**m)


@pytest.mark.parametrize("kw", [
    {"age": 24.99}, {"age": 85}, {"sex": "x"}, {"ethnicity": "martian"}, {"smoking": "vape"},
    {"diabetes": "type3"}, {"bmi": 0}, {"chol_hdl_ratio": np.nan}, {"sbp": -1}, {"sbp_sd": -1},
    {"townsend": np.nan}, {"af": 2},
])
def test_invalid_rows_are_nan(kw):
    assert np.isnan(qrisk3(**{**BASE, **kw}))


def test_broadcasts():
    r = qrisk3(**{**BASE, "age": [30, 50, 70], "sex": ["male", "female", "male"]})
    assert r.shape == (3,) and np.all(np.diff(r[[0, 2]]) > 0)
