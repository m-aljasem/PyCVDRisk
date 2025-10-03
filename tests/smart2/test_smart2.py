"""SMART2 (Hageman et al., Eur Heart J 2022): hand-computed value from Supplementary Table 2,
monotonicity of the horizon, and input handling."""

import json
from pathlib import Path

import numpy as np
import pytest

from cvd_risk import smart2

PERSON = dict(age=65, sex="male", smoker=0, diabetes=1, sbp=135, non_hdl_chol=3.5, egfr=75, hscrp=2.0,
              years_since_diagnosis=5, cad=1, cerebrovascular=0, pad=0, aaa=0, region="europe_moderate")


def test_hand_computed_value():
    lp = (-0.03496022 * 65 + 0.000551072 * 65**2 + 0.287658743 + 0.318170659 + 0.018913154 * 13.5
          + 0.540364249 * np.log(3.5) - 0.03967521 * 75 + 0.000218613 * 75**2 + 0.151760173 * np.log(2.0)
          + 0.047699585 * 5 - 0.00164973 * 25 + 0.294701954)
    expected = 1 - (1 - 0.165822797) ** np.exp(lp + 0.0463729 + 0.21072103 - np.log(0.6973285))
    assert smart2(**PERSON) == pytest.approx(expected, rel=1e-12)


def test_horizon_increases_and_antithrombotic_lowers_risk():
    r = [smart2(**PERSON, horizon=h) for h in range(1, 11)]
    assert all(a < b for a, b in zip(r, r[1:]))
    assert smart2(**PERSON, antithrombotic=1) < smart2(**PERSON)


def test_vectorised_and_invalid_rows_are_nan():
    out = smart2(**{**PERSON, "age": [65, 30, 65], "region": ["europe_low", "europe_low", "mars"]})
    assert np.isfinite(out[0]) and np.isnan(out[1:]).all()
    assert np.isnan(smart2(**{**PERSON, "cad": 0}))  # no disease location
    with pytest.raises(ValueError):
        smart2(**PERSON, horizon=11)


FX = Path(__file__).parents[2] / "reference" / "smart2" / "fixtures"


def test_matches_uprevent_calculator():
    """30 patients read from u-prevent.com, 10-year risk at its 0.1% display precision. Patient 21's
    total cholesterol was clamped to the calculator's minimum of 2.5 (non-HDL 1.6)."""
    ps = {p["id"]: p for p in json.loads((FX / "uprevent_patients.json").read_text())}
    ps[21]["non_hdl_chol"] = 1.6
    rows = [json.loads(ln) for ln in (FX / "uprevent_results.jsonl").read_text().splitlines()[1:]]
    assert len(rows) == 30
    keys = ["age", "sex", "smoker", "diabetes", "sbp", "non_hdl_chol", "egfr", "hscrp", "years_since_diagnosis",
            "cad", "cerebrovascular", "pad", "aaa", "region", "antithrombotic"]
    for r in rows:
        got = 100 * smart2(**{k: ps[r["id"]][k] for k in keys})
        assert got == pytest.approx(float(r["risk10"].rstrip("%")), abs=0.051), r["id"]


def test_matches_uprevent_calculator_round2():
    """22 more patients (ids 100-121): HbA1c 33-108 in diabetics (no effect on the risk), and every
    input at the calculator's limits. 120-121 use "dual antiplatelet" and "aspirin + low-dose DOAC",
    which U-Prevent treats as stronger than aspirin; SMART2 itself has only aspirin or equivalent,
    so they are excluded here."""
    ps = {p["id"]: p for p in json.loads((FX / "uprevent_patients2.json").read_text())}
    rows = [json.loads(ln) for ln in (FX / "uprevent_results2.jsonl").read_text().splitlines() if ln.strip()]
    keys = ["age", "sex", "smoker", "diabetes", "sbp", "non_hdl_chol", "egfr", "hscrp", "years_since_diagnosis",
            "cad", "cerebrovascular", "pad", "aaa", "region", "antithrombotic"]
    rows = [r for r in rows if ps[r["id"]]["antithrombotic"] in (0, 1)]
    assert len(rows) == 20
    for r in rows:
        got = 100 * smart2(**{k: ps[r["id"]][k] for k in keys})
        assert got == pytest.approx(float(r["risk10"].rstrip("%")), abs=0.051), r["id"]
