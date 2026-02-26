import importlib
import math

import numpy as np
import pytest

from cvd_risk.suita import suita_cvd, suita_cvd_risk

mod = importlib.import_module("cvd_risk.suita.suita_cvd")

# Nakai 2020 Table 3 coefficients: (with ECG, without ECG)
COEF = {"men": (0.38, 0.38), "age": ((0, 0.82, 1.39, 1.76, 2.21, 2.53, 2.82), (0, 0.82, 1.40, 1.78, 2.23, 2.57, 2.85)),
        "bp": ((-0.38, 0, 0.29, 0.59), (-0.37, 0, 0.29, 0.61)), "lipid_high": (0.17, 0.17),
        "hdl": ((0, -0.23, -0.42), (0, -0.23, -0.42)), "smoker": (0.36, 0.35), "diabetes": (0.64, 0.67),
        "proteinuria": (0.21, 0.23), "af": (0.83, None), "lvh": (0.49, None)}
R = lambda x: math.floor(10 * x + 0.5) if x >= 0 else -math.floor(-10 * x + 0.5)


@pytest.mark.parametrize("m", [0, 1])
def test_points_are_ten_times_the_coefficients(m):
    """Methods: 'coefficients were multiplied by a factor of 10 to allow for integer scores'; all 2 x 21 hold."""
    assert [R(b) for b in COEF["age"][m]] == list(mod.AGE[1][m])
    for k, v in COEF.items():
        if k == "age" or v[m] is None:
            continue
        ours = mod.POINTS[k][m]
        assert (list(ours) if isinstance(ours, tuple) else ours) == ([R(b) for b in v[m]] if isinstance(v[m], tuple)
                                                                     else R(v[m])), k


def test_worked_cases():
    """Hand-summed from Table 3."""
    p = suita_cvd([45, 72], ["male", "female"], [135, 150], [85, 95], [0, 1], [45, 65], [1, 0], [0, 1], [0, 1],
                  ldl=[150, 120], non_hdl=[160, 180])
    assert list(p) == [8 + 4 + 0 + 2 - 2 + 4, 26 + 0 + 6 + 2 - 4 + 7 + 2]
    assert suita_cvd(45, "male", 135, 85, 0, 45, 1, 0, 0, ldl=150, af=1, lvh=0) == 8 + 4 + 0 + 2 - 2 + 4 + 8
    # higher of SBP/DBP; treatment puts BP in the top category
    assert suita_cvd(50, "female", 118, 92, 0, 50, 0, 0, 0, ldl=100) == 14 + 3 - 2
    assert suita_cvd(50, "female", 110, 70, 1, 50, 0, 0, 0, ldl=100) == 14 + 6 - 2


def test_risk_labels_table4():
    pts = [-3, 0, 1, 20, 21, 25, 26, 30, 31, 35, 36, 60, np.nan]
    assert list(suita_cvd_risk(pts, True)) == ["<1", "<1", "2", "2", "6", "6", "9", "9", "15", "15", "26", "26", ""]
    assert list(suita_cvd_risk(pts, False))[8:11] == ["14", "14", "25"]


def test_missing_and_out_of_range():
    assert np.isnan(suita_cvd([29, 80], "male", 120, 80, 0, 50, 0, 0, 0, ldl=120)).all()
    assert suita_cvd(50, "male", 120, 80, 0, 50, 0, 0, 0, ldl=np.nan, non_hdl=175) == 14 + 4 + 0 + 2 - 2
    assert np.isnan(suita_cvd(50, "male", 120, 80, 0, 50, 0, 0, 0, ldl=np.nan, non_hdl=np.nan))
    assert np.isnan(suita_cvd(50, "male", 120, 80, np.nan, 50, 0, 0, 0, ldl=120))
    with pytest.raises(ValueError):
        suita_cvd(50, "male", 120, 80, 0, 50, 0, 0, 0, ldl=120, af=1)
    with pytest.raises(ValueError):
        suita_cvd(50, "male", 120, 80, 0, 50, 0, 0, 0)


def test_jcvrisk():
    """R package Jcvrisk 0.1.3 suita_wECG / suita_woECG (vendored) on 2,416 patients aged 30-79: the same Table 4 band
    (its 1 for '<1') for all but 7 and 5, every one at SBP 139 or DBP 89, which its cut points (< 139, < 89) leave
    in the lowest category."""
    from pathlib import Path

    import pandas as pd
    ref = Path(__file__).parents[2] / "reference/jcvrisk/fixtures"
    d, r = pd.read_csv(ref / "patients.csv"), pd.read_csv(ref / "jcvrisk_output.csv")
    ok = ((d.age >= 30) & (d.age < 80)).values
    sex = np.where(d.male == 1, "male", "female")
    for ecg, col, n_bad in [(True, "suita_wecg", 7), (False, "suita_woecg", 5)]:
        kw = dict(af=d.af, lvh=d.lvh) if ecg else {}
        p = suita_cvd(d.age, sex, d.sbp, d.dbp, d.ht_medication, d.hdl, d.smoking, d.t2dm, d.urineprotein,
                      ldl=d.ldl, non_hdl=d.tc - d.hdl, **kw)
        lab = suita_cvd_risk(p, ecg)[ok]
        bad = np.where(lab == "<1", "1", lab).astype(float) != r[col].values[ok]
        assert bad.sum() == n_bad
        assert ((d.sbp.values[ok][bad] == 139) | (d.dbp.values[ok][bad] == 89)).all()
