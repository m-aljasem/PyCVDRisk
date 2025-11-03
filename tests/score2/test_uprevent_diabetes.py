"""SCORE2-Diabetes against the official U-Prevent calculator (22 accepted patients, 1-decimal display)."""

import importlib.util
from pathlib import Path

import numpy as np

REF = Path(__file__).parents[2] / "reference" / "score2"
spec = importlib.util.spec_from_file_location("uprevent_diabetes", REF / "uprevent_diabetes.py")
U = importlib.util.module_from_spec(spec)
spec.loader.exec_module(U)


def test_matches_uprevent_display():
    rows = U.compare()
    assert len(rows) == 22
    misses = [(i, v) for i, _, v, ok in rows if not ok]
    # The one miss sits 0.0003 points below the edge where U-Prevent's two-step rounding turns up.
    assert [i for i, _ in misses] == [17]
    v = misses[0][1]
    assert abs(v - (np.floor(v * 10) / 10 + 0.045)) < 0.001
    assert max(abs(v - float(s.rstrip("%"))) for _, s, v, _ in rows) < 0.06
