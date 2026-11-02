"""Verification of the AHA PREVENT equations.

1. The authors' own worked examples (Supplemental Tables S12.A-J), all 100 values.
2. Agreement with the R reference (preventr, unrounded) on 5 000 synthetic patients,
   5 models x 2 horizons x 5 outcomes.
3. Properties and input handling.
"""

import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from cvd_risk import prevent
from cvd_risk.prevent import MODELS, OUTCOMES

mod = importlib.import_module("cvd_risk.prevent.prevent")  # the module, not the same-named function
FIXTURES = Path(__file__).parents[2] / "reference" / "prevent" / "fixtures"
MG = 0.02586  # source's mg/dL -> mmol/L factor
REQUIRED = ["age", "sex", "sbp", "bp_treated", "total_chol", "hdl_chol", "statin",
            "diabetes", "smoker", "egfr", "bmi"]


def run(q, model, years, outcome):
    return prevent(**{c: q[c] for c in REQUIRED}, uacr=q.uacr, hba1c=q.hba1c, sdi=q.sdi,
                   model=model, years=years, outcome=outcome)


# ---- 1. authors' worked examples -------------------------------------------

def pre_erratum_table():
    """S12 as printed: undo the 2026-10-14 HbA1c erratum (the examples predate it)."""
    swap = {"hba1c_dm": "hba1c_no_dm", "hba1c_no_dm": "hba1c_dm"}
    table = {k: ([swap.get(t, t) if k[0] in ("hba1c", "full") else t for t in terms], c)
             for k, (terms, c) in mod._table().items()}
    return lambda: table


@pytest.mark.parametrize("i", range(10))
def test_authors_worked_example(i, monkeypatch):
    e = pd.read_csv(FIXTURES / "worked_examples.csv").iloc[i]
    monkeypatch.setattr(mod, "_table", pre_erratum_table())
    for outcome in OUTCOMES:
        for sex in ("female", "male"):
            got = prevent(e.age, sex, e.sbp, e.bp_treated, e.total_chol_mgdl * MG, e.hdl_chol_mgdl * MG,
                          e.statin, e.diabetes, e.smoker, e.egfr, e.bmi, uacr=e.uacr, hba1c=e.hba1c,
                          sdi=e.sdi, model=e.model, years=int(e.years), outcome=outcome)
            np.testing.assert_allclose(got, e[f"{outcome}_{sex}"], rtol=1e-12, err_msg=f"{outcome} {sex}")


def test_erratum_changes_only_hba1c_models():
    e = pd.read_csv(FIXTURES / "worked_examples.csv")
    for _, row in e.iterrows():  # every example has diabetes and HbA1c 7.5 where used
        args = (row.age, "female", row.sbp, row.bp_treated, row.total_chol_mgdl * MG,
                row.hdl_chol_mgdl * MG, row.statin, row.diabetes, row.smoker, row.egfr, row.bmi)
        got = prevent(*args, uacr=row.uacr, hba1c=row.hba1c, sdi=row.sdi, model=row.model, years=int(row.years))
        same = np.isclose(got, row.total_cvd_female, rtol=1e-12)
        assert same == (row.model not in ("hba1c", "full")), row.model


# ---- 2. agreement with preventr --------------------------------------------

@pytest.fixture(scope="module")
def ref():
    p = pd.read_csv(FIXTURES / "patients.csv").set_index("id")
    r = pd.read_csv(FIXTURES / "reference.csv")
    return p, r


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("years", [10, 30])
def test_matches_preventr(ref, model, years):
    p, r = ref
    g = r[(r.model == model) & (r.years == years)]
    q = p.loc[g.id]
    for outcome in OUTCOMES:
        got, expected = run(q, model, years, outcome), g[outcome].to_numpy()
        assert not np.isnan(got).any()
        # preventr returns NA for ages 79.01-79.99; the source's range is "30-79 years".
        assert (q.age[np.isnan(expected)] > 79).all() and (q.age[q.age > 79].index.isin(g.id[np.isnan(expected)])).all()
        ok = ~np.isnan(expected)
        np.testing.assert_allclose(got[ok], expected[ok], rtol=1e-12, atol=1e-15, err_msg=outcome)


def test_batch_equals_row_by_row(ref):
    p, _ = ref
    q = p.head(100)
    batch = run(q, "full", 30, "heart_failure")
    rows = [float(run(q.iloc[[i]], "full", 30, "heart_failure")[0]) for i in range(len(q))]
    np.testing.assert_allclose(batch, rows, rtol=1e-15)


# ---- 3. properties ---------------------------------------------------------

@settings(max_examples=300)
@given(st.floats(30, 79.99), st.sampled_from(["male", "female"]), st.floats(90, 200),
       st.integers(0, 1), st.floats(3.37, 8.27), st.floats(0.52, 2.58), st.integers(0, 1),
       st.integers(0, 1), st.integers(0, 1), st.floats(15, 150), st.floats(18.5, 39.99),
       st.sampled_from(MODELS), st.sampled_from([10, 30]), st.sampled_from(OUTCOMES))
def test_risk_in_unit_interval(age, sex, sbp, tx, tc, hdl, st_, dm, sm, egfr, bmi, model, years, outcome):
    r = prevent(age, sex, sbp, tx, tc, hdl, st_, dm, sm, egfr, bmi, uacr=30, hba1c=6, sdi=5,
                model=model, years=years, outcome=outcome)
    assert 0 < r < 1


# ---- input handling and range edges ----------------------------------------

BASE = dict(age=50, sex="female", sbp=160, bp_treated=1, total_chol=5.172, hdl_chol=1.1637,
            statin=0, diabetes=1, smoker=0, egfr=90, bmi=35)


@pytest.mark.parametrize("field, value", [
    ("age", 29.99), ("age", 80), ("sbp", 89), ("sbp", 201), ("bmi", 40), ("bmi", 18.4),
    ("egfr", 14), ("egfr", 151), ("total_chol", 3.3), ("hdl_chol", 2.7), ("sex", "x"),
    ("smoker", 2), ("statin", np.nan),
])
def test_out_of_range_is_nan(field, value):
    assert np.isnan(prevent(**{**BASE, field: value}))


@pytest.mark.parametrize("field, value", [("age", 30), ("age", 79.99), ("bmi", 18.5), ("egfr", 150), ("sbp", 200)])
def test_range_edges_are_valid(field, value):
    assert not np.isnan(prevent(**{**BASE, field: value}))


def test_missing_optional_uses_indicator_not_nan():
    for model in ("uacr", "hba1c", "sdi", "full"):
        a = prevent(**BASE, model=model)
        b = prevent(**BASE, uacr=np.nan, hba1c=np.nan, sdi=np.nan, model=model)
        assert a == b and not np.isnan(a)


def test_invalid_optional_value_is_nan_only_where_used():
    assert np.isnan(prevent(**BASE, sdi=4.5, model="sdi"))
    assert np.isnan(prevent(**BASE, uacr=30000, model="full"))
    assert np.isnan(prevent(**BASE, hba1c=2, model="hba1c"))
    assert not np.isnan(prevent(**BASE, sdi=4.5, hba1c=2, model="uacr"))


def test_base_model_ignores_optional_inputs():
    assert prevent(**BASE) == prevent(**BASE, uacr=100, hba1c=9, sdi=10)


def test_bad_arguments_raise():
    for kw in ({"model": "nope"}, {"years": 20}, {"outcome": "death"}):
        with pytest.raises(ValueError):
            prevent(**BASE, **kw)
