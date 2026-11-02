import numpy as np
import pandas as pd

from cvd_risk._inputs import as_bool, as_category, as_float, as_sex


def test_as_float_scalars_lists_series():
    assert as_float(5).shape == ()
    np.testing.assert_array_equal(as_float([1, 2]), [1.0, 2.0])
    np.testing.assert_array_equal(as_float(pd.Series([3, 4], index=[10, 20])), [3.0, 4.0])


def test_as_float_bad_entries_become_nan():
    out = as_float([1, "x", None])
    assert out[0] == 1 and np.isnan(out[1]) and np.isnan(out[2])


def test_as_sex_normalises_and_flags_unknown():
    np.testing.assert_array_equal(as_sex(["Male", " female ", "F", "other", None]), [1, 0, 0, -1, -1])


def test_as_category_normalises_separators():
    allowed = ("low", "moderate", "high", "very_high")
    got = as_category(["Very High", "very-high", "LOW", "extreme"], allowed)
    np.testing.assert_array_equal(got, [3, 3, 0, -1])


def test_as_bool_rejects_non_binary():
    out = as_bool([0, 1, True, 2, np.nan])
    np.testing.assert_array_equal(out[:3], [0, 1, 1])
    assert np.isnan(out[3]) and np.isnan(out[4])
