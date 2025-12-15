import importlib
import re
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cvd_risk import grace2
mod = importlib.import_module("cvd_risk.grace.grace2")

REF = Path(__file__).parents[2] / "reference" / "grace2"
MODEL = {"OneYear": 1, "ThreeYear": 3}
VAR = {"Age": ("age", 10), "Pulse": ("pulse", 30), "BloodPressure": ("sbp", -20), "Creatine": ("creatinine", 1)}


def _lookups():
    s = (REF / "source" / "grace_lookups.js").read_text(encoding="utf-8")
    return {m.group(1): dict(re.findall(r'"([\w.]+)"\s*:\s*"([-\d.]+)"', m.group(2)))
            for m in re.finditer(r"var ((?:One|Three)Year\w*)\s*=\s*\{(.*?)\}", s, re.S)}


def test_coefficients_reproduce_every_calculator_lookup_value():
    """grace_lookups.js holds each term at the calculator's dropdown entries; every one is reproduced to
    the digits it prints."""
    n = 0
    for name, table in _lookups().items():
        parts = name.split("_")
        h, sub, out = MODEL[parts[0]], "Substituted" in parts, "death_mi" if "MI" in parts else "death"
        s0, age, pulse, sbp, renal, killip, arrest, enz, st = mod._M[h, out, sub]
        for key, value in table.items():
            v, term = parts[-1], None
            if v in VAR:
                arg, scale = VAR[v]
                b = {"age": age, "pulse": pulse, "sbp": sbp, "creatinine": renal}[arg]
                if arg == "sbp" and len(b) == 9:
                    term = b[int(np.digitize(float(key), mod._SBP_BANDS))]
                else:
                    term = mod._spline(np.float64(key) / scale, mod._KNOTS[arg], b)
            elif v == "Killip":
                term = ((0,) + killip)[int(key) - 1]
            else:
                term = {"Renal": renal, "Diuretic": killip, "Cardiac": arrest, "ElevatedEnzymes": enz, "STEMI": st}[v]
                term = term if key == "yes" else 0
            half_unit = 0.5 * 10.0 ** -len(value.partition(".")[2]) + 1e-12  # the value as printed
            assert abs(term - float(value)) <= max(half_unit, 1e-12), (name, key)
            n += 1
    assert n == 836  # 8 models: 72 ages, 10 pulse, 9 SBP, 7 creatinine (full), Killip or renal + diuretic, 3 flags


def _shown(v):
    """The calculator's Number.prototype.roundup(1): one decimal below 10, a whole number from 10."""
    return np.where(v < 10, np.floor(v * 10 + 0.5) / 10, np.floor(v + 0.5))


def test_matches_developers_calculator():
    """reference/grace2/run_app.js: the calculator's own functions on 6,143 dropdown combinations."""
    a = pd.read_csv(REF / "fixtures" / "app_reference.csv")
    yn = lambda s: (s == "yes").astype(int)
    for model, g in a.groupby("model"):
        kw = (dict(creatinine=g.creatinine, killip=g.killip) if model == "full"
              else dict(renal_insufficiency=yn(g.renal), diuretic=yn(g.diuretic)))
        for col, horizon, outcome in (("d1", 1, "death"), ("dmi1", 1, "death_mi"), ("d3", 3, "death"), ("dmi3", 3, "death_mi")):
            got = 100 * grace2(g.age, g.pulse, g.sbp, yn(g.cardiac), yn(g.stemi), yn(g.enzymes), horizon=horizon,
                               outcome=outcome, **kw)
            np.testing.assert_allclose(got, g[col], rtol=1e-6)  # the tables print 7-8 significant digits
            np.testing.assert_array_equal(_shown(got), g[col + "_shown"])


def test_matches_developers_grace_js():
    """grace.js, the developers' continuous 1-year death equations, on 500 random patients."""
    j = pd.read_csv(REF / "fixtures" / "gracejs_reference.csv")
    args = (j.age, j.pulse, j.sbp, j.cardiac, j.stemi, j.enzymes)
    np.testing.assert_allclose(grace2(*args, creatinine=j.creatinine, killip=j.killip), j.d1, rtol=1e-12)
    np.testing.assert_allclose(grace2(*args, renal_insufficiency=j.renal, diuretic=j.diuretic), j.d1_substituted,
                               rtol=1e-12)


def test_developers_worked_examples():
    """source/tests.js, 'One year long form' and 'One year substituted' (results to 2 decimals)."""
    full = grace2([90, 40], [92, 82], [139, 107], 0, 1, [1, 0], creatinine=[1.06, 1.00], killip=1)
    sub = grace2([90, 40], [92, 82], [139, 107], 0, 1, [1, 0], renal_insufficiency=0, diuretic=0)
    np.testing.assert_array_equal(np.round(full, 2), [0.28, 0.02])
    np.testing.assert_array_equal(np.round(sub, 2), [0.30, 0.02])


def _hr(b):
    """Table 2 values agree with exp(b) rounded to 2 and then to 1 decimal (22 of 24 when rounded once)."""
    two = Decimal(repr(float(np.exp(b)))).quantize(Decimal("0.01"), ROUND_HALF_UP)
    return float(two.quantize(Decimal("0.1"), ROUND_HALF_UP))


def test_hazard_ratios_match_fox2014_table2():
    m = mod._M
    d1, d1s, mi1, mi1s, d3, d3s = (m[1, "death", False], m[1, "death", True], m[1, "death_mi", False],
                                   m[1, "death_mi", True], m[3, "death", False], m[3, "death", True])
    table2 = {  # Killip II, III, IV; cardiac arrest; positive enzymes; ST deviation; renal; diuretics
        "1y death": ([*d1[5], d1[6], d1[7], d1[8], d1s[4], d1s[5]], [1.9, 2.4, 3.7, 2.4, 1.5, 1.6, 1.6, 2.0]),
        "1y death/MI": ([*mi1[5], mi1[6], mi1[7], mi1[8], mi1s[4], mi1s[5]], [1.7, 2.0, 3.2, 2.0, 1.3, 1.4, 1.6, 1.7]),
        "3y death": ([d3[5][0], d3[5][1], d3[6], d3[8], d3s[4], d3s[5], d3[1][0], d3[4][0]],
                     [1.1, 2.3, 2.9, 1.5, 2.0, 2.0, 1.8, 1.5]),  # + age per 10 years, creatinine per mg/dL
    }
    for name, (b, printed) in table2.items():
        assert [_hr(x) for x in b] == printed, name
    assert d3[5][1] == d3[5][2] and d3[7] == 0  # Killip III-IV pooled; enzymes not in the 3-year model


def test_three_year_death_sbp_slopes_follow_table2_above_130():
    """Table 2: per -20 mmHg, HR 0.9 at >= 160 and 1.6 at 130-159 (the band values; < 130 does not give 1.3)."""
    v = mod._M[3, "death", False][3]
    slope = lambda i, j, s1, s2: (v[j] - v[i]) / ((s2 - s1) / -20)
    assert _hr(slope(7, 8, 169.5, 180)) == 0.9
    assert _hr(slope(5, 6, 134.5, 149.5)) == 1.6


def test_arguments_and_invalid_rows():
    with pytest.raises(ValueError):
        grace2(60, 80, 120, 0, 0, 0, creatinine=1.0)
    with pytest.raises(ValueError):
        grace2(60, 80, 120, 0, 0, 0, creatinine=1.0, killip=1, renal_insufficiency=0, diuretic=0)
    with pytest.raises(ValueError):
        grace2(60, 80, 120, 0, 0, 0, creatinine=1.0, killip=1, horizon=2)
    r = grace2([60, 60, -1, 60, 60], 80, 120, [0, 0, 0, 2, 0], 0, 0, creatinine=[1.0, 1.0, 1.0, 1.0, -1],
               killip=[1, 5, 1, 1, 1])
    assert np.isfinite(r[0]) and np.isnan(r[1:]).all()


def test_risk_rises_with_each_adverse_finding():
    base = dict(age=65, heart_rate=80, sbp=130, cardiac_arrest=0, st_deviation=0, elevated_enzymes=0)
    for h, o in ((1, "death"), (1, "death_mi"), (3, "death"), (3, "death_mi")):
        r0 = grace2(**base, creatinine=1.0, killip=1, horizon=h, outcome=o)
        for k, v in (("age", 80), ("cardiac_arrest", 1), ("st_deviation", 1)):
            assert grace2(**{**base, k: v}, creatinine=1.0, killip=1, horizon=h, outcome=o) > r0, (h, o, k)
        assert grace2(**base, creatinine=1.0, killip=3, horizon=h, outcome=o) > r0
