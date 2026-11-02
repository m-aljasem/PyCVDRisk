import math

import numpy as np
import pytest

from cvd_risk import adrs
from cvd_risk.adrs import bp_stage
from cvd_risk.adrs.adrs import ASCVD, CAD

P = {"age": 55, "sex": "male", "smoker": 1, "sbp": 135, "dbp": 85, "tc_hdl": 5.0, "diabetes": 1, "hypertension": 1, "bp_treated": 1,
         "vitamin_d": 20, "egfr": 90}


def test_formula_by_hand():
    # Data S1 syntax written out for one man; MAP = 85 + 50/3; stage 1 BP
    dev, mp = 90 - 111.4, 85 + 50 / 3
    x = {"vitamin_d": 20, "dev_gfr": dev, "male": 1, "smoker": 1, "age": 55, "bp_treated": 1, "tc_hdl": 5.0, "diabetes": 1,
             "dev_gfr_x_male": dev, "map": mp, "age2": 55**2}
    lp = sum(b * (x[k] - m) for k, (b, m) in ASCVD.items())
    assert adrs(**P) == pytest.approx(1 - 0.994 ** math.exp(lp))
    x.update(hypertension=1, diabetes_x_male=1, diabetes_x_dev_gfr=dev, bp_normal=0, bp_elevated=0, bp_stage1=1,
             bp_stage2=0, bp_crisis=0)
    lp = sum(b * (x[k] - m) for k, (b, m) in CAD.items())
    assert adrs(**P, outcome="cad") == pytest.approx(1 - 0.994 ** math.exp(lp))
    lp = 1.053 + 0.024 * 135 + 0.084 * 55 - 0.646 + 0.011 * 55 - 6.37
    assert adrs(**P, outcome="stroke") == pytest.approx(1 - 0.998 ** math.exp(lp))


def test_bp_stage_edges():
    s = [119, 120, 129, 130, 139, 140, 180, 181, 110, 110, 110, 110, 110]
    d = [79, 79, 79, 79, 79, 79, 79, 79, 80, 89, 90, 120, 121]
    assert bp_stage(s, d).tolist() == [0, 1, 1, 2, 2, 3, 3, 4, 2, 2, 3, 3, 4]
    assert np.isnan(bp_stage(0, 80))


def test_invalid_and_outcome():
    r = adrs(**{**P, "age": [55, 17, 55, 55, 55], "sex": ["m", "f", "x", "m", "m"], "egfr": [90, 90, 90, -1, 90],
                "smoker": [1, 1, 1, 1, 2]})
    assert np.isfinite(r[0]) and np.isnan(r[1:]).all()
    with pytest.raises(ValueError):
        adrs(**P, outcome="death")
