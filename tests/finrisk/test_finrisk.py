import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

from cvd_risk import finrisk

REF = Path(__file__).parents[2] / "reference" / "finrisk"


def _report():
    spec = importlib.util.spec_from_file_location("finrisk_report", REF / "report.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_table1_equations():
    rep = _report()
    for key, printed in rep.TABLE1.items():
        assert rep.table1(*key) == printed, key


def test_matches_official_calculator():
    # FINRISKI 2.0 server (fixtures/calculator_reference.jsonl); it floors its display at the optimal profile
    s = _report().summary()
    assert s["n"] >= 200 and s["rel"] < 1e-12 and s["floored_below"] and s["rel_optimal"] < 1e-12


def test_formula_and_combined_risk():
    lp = -9.081 + 0.075 * 60 + 0.579 + 0.320 * 6 + 0.011 * 140 - 1.082 * 1.2 + 0.338
    assert math.isclose(finrisk(60, "m", 1, 6, 1.2, 140, 0, 1, 0, outcome="chd"), 1 / (1 + math.exp(-lp)))
    args = (55, ["m", "f"], 1, 6, 1.2, 140, 1, 1, 1)
    c, s = finrisk(*args, outcome="chd"), finrisk(*args, outcome="stroke")
    np.testing.assert_allclose(finrisk(*args), 1 - (1 - c) * (1 - s))


def test_invalid():
    assert np.isnan(finrisk([29, 75, 50, 50], ["m", "m", "x", "m"], [0, 0, 0, 2], 5, 1.3, 130, 0, 0, 0)).all()
    assert not np.isnan(finrisk(74, "m", 0, 5, 1.3, 130, 0, 0, 0))
    assert np.isnan(finrisk(50, "f", 0, -1, 1.3, 130, 0, 0, 0, outcome="stroke"))
    with pytest.raises(ValueError):
        finrisk(50, "m", 0, 5, 1.3, 130, 0, 0, 0, outcome="mi")
