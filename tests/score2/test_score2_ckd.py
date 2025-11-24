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


def _ckdpc():
    import importlib.util
    spec = importlib.util.spec_from_file_location("ckdpc", FX.parent / "ckdpc_calculator.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_matches_ckdpc_online_calculator():
    # The developers' calculator (CKD-PC, ckdpcrisk.org), 71 patients x 4 regions, eGFR and eGFR + ACR
    # add-ons, shown to 0.1%. Misses are within 0.02 points of a rounding edge (server-side precision).
    m = _ckdpc()
    ps = {p["id"]: p for p in m.patients()}
    rows = m.compare()
    op_ge90 = lambda r: ps[r[0]]["age"] >= 70 and ps[r[0]]["egfr"] >= 90  # noqa: E731
    base = [r for r in rows if r[2] == "base"]
    ckd = [r for r in rows if r[2] == "ckd" and not op_ge90(r)]
    assert len(base) == 284 and len(ckd) == 504
    assert all(abs(r[4] - r[5]) < 0.053 for r in base) and sum(m.agrees(r[4], r[5]) for r in base) == 280
    assert all(abs(r[4] - r[5]) < 0.07 for r in ckd) and sum(m.agrees(r[4], r[5]) for r in ckd) == 483
    # Patient A: the calculator gives 2.5% (low region) like the package, not the supplement's 1.96%.
    assert [r[4] for r in base if r[0] == 0][0] == 2.5


def test_finding_ckdpc_calculator_drops_score2_op_egfr_ge90_segment():
    # The supplement's SCORE2-OP equation has an eGFR >= 90 term (-0.4616, age x -0.0075); the
    # calculator leaves it out (as the SCORE2 add-on does by design), lowering risk at age >= 70.
    m = _ckdpc()
    ps = {p["id"]: p for p in m.patients()}
    rows = [r for r in m.compare() if r[2] == "ckd" and ps[r[0]]["age"] >= 70 and ps[r[0]]["egfr"] >= 90]
    assert len(rows) == 64
    assert not any(m.agrees(r[4], r[5]) for r in rows) and max(r[5] - r[4] for r in rows) > 28
    assert all(abs(r[4] - r[6]) < 0.06 for r in rows) and sum(m.agrees(r[4], r[6]) for r in rows) == 61
