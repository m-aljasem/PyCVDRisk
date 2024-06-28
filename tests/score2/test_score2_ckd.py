"""Verification of SCORE2-CKD (SCORE2 / SCORE2-OP with CKD add-ons).

1. The supplement's worked example (patient A), step by step.
2. Agreement with RiskScorescvd (unrounded) on 20 000 synthetic patients, all add-ons.
3. Input handling.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cvd_risk import score2_ckd
from cvd_risk.score2 import coefficients as C
from cvd_risk.score2.score2_ckd import _lin, ckd_addon

FX = Path(__file__).parents[2] / "reference" / "score2_ckd" / "fixtures"
COLS = ["age", "sex", "smoker", "sbp", "total_chol", "hdl_chol", "diabetes", "egfr", "region"]
A = (42, "male", 1, 128, 3.8, 1.4, 0, 45)  # patient A: age, sex, smoker, SBP, TC, HDL, DM, eGFR


def test_worked_example_patient_a():
    # Supplementary material 1.1.2. The supplement sums products rounded to 4 decimals,
    # so each step is checked to the precision it is printed with.
    t = ((42 - 60) / 5, 3.8 - 6, (1.4 - 1.3) / 0.5, (128 - 120) / 20, 0, 1)
    assert abs(_lin(C.CKD_EXPECTED_EGFR, 0, *t) - 103.1941) < 13 * 5e-5
    log8_ex = 1 + _lin(C.CKD_EXPECTED_LOG8_ACR, 0, *t) + C.CKD_EXPECTED_LOG8_ACR["egfr_lt60"] * 1
    assert abs(log8_ex - np.log(15.3906) / np.log(8)) < 17 * 5e-5
    # From the printed base risk 1.96%: 4.68% (eGFR add-on) and 6.07% (+ ACR 150 mg/g).
    assert abs(ckd_addon(0.0196, *A) - 0.0468) < 2e-4
    assert abs(ckd_addon(0.0196, *A, acr=150, addon="acr") - 0.0607) < 2e-4


def test_finding_patient_a_base_risk_not_reproducible():
    # The supplement gives SCORE2 = 1.96% (low-risk region) for patient A; the stated
    # inputs give 2.48% (RiskScorescvd agrees), and no region gives 1.96%.
    from cvd_risk import score2
    risks = [float(score2(*A[:7], rg)) for rg in ("low", "moderate", "high", "very_high")]
    assert round(risks[0], 4) == 0.0248 and all(abs(r - 0.0196) > 0.003 for r in risks)


@pytest.fixture(scope="module")
def ref():
    return pd.read_csv(FX / "patients.csv"), pd.read_csv(FX / "reference.csv")


@pytest.mark.parametrize("addon", ["egfr", "acr", "dipstick"])
def test_matches_r_reference(ref, addon):
    p, r = ref
    got = score2_ckd(*[p[c] for c in COLS], acr=p.acr, dipstick=p.dipstick, addon=addon)
    expected = r[addon].to_numpy()
    same = np.ones(len(p), bool)
    if addon == "dipstick":  # see test_finding_reference_uses_score2_dipstick_coefficients_for_op
        same = (p.age < 70) | (p.dipstick == "negative")
    assert not np.isnan(got).any()
    np.testing.assert_allclose(got[same], expected[same], rtol=1e-12, atol=1e-15)


def test_finding_reference_uses_score2_dipstick_coefficients_for_op(ref, monkeypatch):
    # RiskScorescvd applies the SCORE2 dipstick log HRs at age >= 70; the supplement
    # (Table C) gives separate SCORE2-OP values. Substituting them reproduces the
    # reference exactly, so this is the only difference.
    p, r = ref
    monkeypatch.setitem(C.CKD_DIPSTICK_ADDON, "score2_op", C.CKD_DIPSTICK_ADDON["score2"])
    got = score2_ckd(*[p[c] for c in COLS], dipstick=p.dipstick, addon="dipstick")
    np.testing.assert_allclose(got, r["dipstick"], rtol=1e-12, atol=1e-15)


ARGS = (55, "female", 0, 140, 5.5, 1.3, 0, 50, "low")


def test_addon_order_and_base():
    from cvd_risk import score2
    base = float(score2(*ARGS[:7], "low"))
    egfr = float(score2_ckd(*ARGS))
    assert egfr > base  # eGFR 50 is far below expected
    assert score2_ckd(*ARGS, dipstick="negative", addon="dipstick") == egfr


@pytest.mark.parametrize("kw", [
    {"acr": 0, "addon": "acr"}, {"acr": np.nan, "addon": "acr"}, {"addon": "acr"},
    {"dipstick": "4+", "addon": "dipstick"}, {"addon": "dipstick"},
])
def test_missing_or_invalid_addon_input_is_nan(kw):
    assert np.isnan(score2_ckd(*ARGS, **kw))


@pytest.mark.parametrize("i, v", [(0, 39), (0, 90), (7, 0), (7, np.nan), (8, "nowhere")])
def test_invalid_core_input_is_nan(i, v):
    args = list(ARGS)
    args[i] = v
    assert np.isnan(score2_ckd(*args))


def test_bad_addon_raises():
    with pytest.raises(ValueError):
        score2_ckd(*ARGS, addon="urine")
