from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cvd_risk.epoch_japan import epoch_japan
from cvd_risk.epoch_japan.epoch_japan import COEF, MEAN, _lp

REF = Path(__file__).parents[2] / "reference"


def test_supplementary_table2_examples():
    """Example 1 (CHD death): woman 70, diabetes, SBP 135, TC 185, HDL-C 58: sums 265.45 and 267.24, risk 1.13%.
    Example 2 (stroke death): man 70, smoker, SBP 140: sums 163.44 and 165.43, risk 2.82%."""
    m = MEAN
    for o, x, xbar in [("chd", (np.log(70), np.log(135), np.log(185 / 58), 0, 0, 1, 0), 265.45),
                       ("stroke", (np.log(70), np.log(140), 0, 1, 1, 0, 0), 163.44)]:
        mean = _lp(COEF[o], np.log(m["age"]), np.log(m["sbp"]), np.log(m["tc_hdl"]), m["male"], m["smoker"],
                   m["diabetes"], m["proteinuria"])
        assert round(mean, 2) == xbar
    assert round(_lp(COEF["chd"], np.log(70), np.log(135), np.log(185 / 58), 0, 0, 1, 0), 2) == 267.24
    assert round(_lp(COEF["stroke"], np.log(70), np.log(140), 0, 0, 1, 0, 0), 2) == 165.43
    assert round(100 * float(epoch_japan(70, "female", 135, 0, 1, 0, total_chol=185, hdl_chol=58, outcome="chd")), 2) == 1.13
    assert round(100 * float(epoch_japan(70, "male", 140, 1, 0, 0, outcome="stroke")), 2) == 2.82


def test_jcvrisk():
    """R package Jcvrisk 0.1.3 (vendored) on 2,416 patients aged 40-79: stroke and CVD equal to 1e-12. Its CHD
    function enters ln(TC) where Table 2 and Supplementary Table 2 have ln(TC/HDL-C); with that substitution it
    equals ours too."""
    d = pd.read_csv(REF / "jcvrisk/fixtures/patients.csv")
    r = pd.read_csv(REF / "jcvrisk/fixtures/jcvrisk_output.csv")
    ok = d.age < 80
    sex = np.where(d.male == 1, "male", "female")
    f = lambda o, **kw: epoch_japan(d.age, sex, d.sbp, d.smoking, d.t2dm, d.urineprotein, outcome=o, **kw)[ok]  # noqa: E731
    assert ok.sum() == 2416
    for o in ("stroke", "cvd"):
        assert np.abs(f(o) - r["epoch_" + o][ok]).max() < 1e-12
    assert (np.abs(f("chd", total_chol=d.tc, hdl_chol=d.hdl) - r.epoch_chd[ok]) > 1e-6).mean() > 0.99
    assert np.abs(f("chd", total_chol=d.tc, hdl_chol=1) - r.epoch_chd[ok]).max() < 1e-12


def test_inputs():
    assert np.isnan(epoch_japan([39, 80], "male", 140, 0, 0, 0)).all()
    assert epoch_japan(60, "female", 140, 0, 0, 0, outcome="stroke") == epoch_japan(60, "male", 140, 0, 0, 0, outcome="stroke")
    with pytest.raises(ValueError):
        epoch_japan(60, "male", 140, 0, 0, 0, outcome="chd")
    with pytest.raises(ValueError):
        epoch_japan(60, "male", 140, 0, 0, 0, outcome="mi")
