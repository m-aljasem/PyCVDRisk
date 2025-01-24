import numpy as np

from cvd_risk import procam
from cvd_risk.prevent.prevent import MG_PER_MMOL as K
from cvd_risk.procam import RISK_PERCENT


def test_extremes_and_edges():
    assert procam(35, "m", 90 / K, 60 / K, 90 / 88.57, 110, 0, 0, 0) == 0
    assert procam(65, "m", 200 / K, 30 / K, 250 / 88.57, 170, 1, 1, 1) == 87
    np.testing.assert_array_equal(procam([39, 40, 65.9], "m", [99.9 / K, 100 / K, 190 / K], 60 / K, 1, 110, 0, 0, 0),
                                  [0, 6 + 5, 26 + 20])


def test_table4_and_validity():
    assert RISK_PERCENT[53] == 19.6 and RISK_PERCENT[54] == 21.7 and len(RISK_PERCENT) == 39
    assert np.isnan(procam([34, 66, 50, 50], ["m", "m", "f", "m"], 3, 1.2, 1.5, [120, 120, 120, -1], 0, 0, 0)).all()
