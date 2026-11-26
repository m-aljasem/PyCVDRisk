"""ASSIGN: every constant is in the source PDFs, plus a hand calculation and input handling."""

import json
import logging
import math
import re
from pathlib import Path

import numpy as np
import pytest

from cvd_risk import assign
from cvd_risk.assign.assign import BETA, MEAN, SURVIVAL

pypdf = pytest.importorskip("pypdf")

SRC = Path(__file__).parents[2] / "reference" / "assign" / "source"
logging.getLogger("pypdf").setLevel(logging.ERROR)


def _text(name):
    return re.sub(r"\s+", "", " ".join(p.extract_text() or "" for p in pypdf.PdfReader(SRC / name).pages))


def _printed(x):  # "6.22520" is printed with trailing zeros, "0.0481" without; match the shortest form
    return f"{abs(x):.10g}"


@pytest.mark.skipif(not (Path(__file__).parents[2] / "reference/assign/source/woodward2007_appendix.pdf").exists(), reason="publisher document not in the repository (see reference/*/source/SOURCES.md)")
def test_version1_constants_in_2007_appendix():
    t = _text("woodward2007_appendix.pdf")
    for s in ("male", "female"):
        for v in (*BETA[1][s], *MEAN[1][s][:7], SURVIVAL[1][s]):
            assert _printed(v) in t, (s, v)
        assert _printed(MEAN[1][s][7]) in t


@pytest.mark.skipif(not (Path(__file__).parents[2] / "reference/assign/source/welsh2025_heart.pdf").exists(), reason="publisher document not in the repository (see reference/*/source/SOURCES.md)")
def test_version2_constants_in_2025_supplement():
    t = _text("welsh2025_heart.pdf")
    for s in ("male", "female"):
        for v in (*MEAN[2][s], SURVIVAL[2][s]):
            assert _printed(v) in t, (s, v)


def test_hand_calculation():
    # man 60, TC 6, HDL 1.2, SBP 150, diabetic, family history, 10 cigarettes, SIMD 40 (version 1)
    L = 0.05698 * 60 + 0.22286 * 6 - 0.53684 * 1.2 + 0.01183 * 150 + 0.81558 + 0.275 + 0.02005 * 10 + 0.06296 * 4
    Lbar = (0.05698 * 48.8706 + 0.22286 * 6.2252 - 0.53684 * 1.35042 + 0.01183 * 133.810 + 0.81558 * 0.0152905
            + 0.275 * 0.263762 + 0.02005 * 7.95841 + 0.06296 * 2.74038)
    want = 100 * (1 - 0.8831 ** math.exp(L - Lbar))
    assert assign(60, "male", 6, 1.2, 150, 1, 1, 10, 40, version=1) == pytest.approx(want, rel=1e-12)


def test_version2_lower_than_version1_and_mean_person():
    a = dict(age=55, sex="female", total_chol=5.8, hdl_chol=1.6, sbp=135, diabetes=0, family_history=0,
             cigarettes_per_day=0, simd_score=15)
    assert assign(**a, version=2) < assign(**a, version=1)
    # at the version-2 continuous means with both binary factors 0, A = -(b_dm m_dm + b_fh m_fh)
    z = -(0.97727 * 0.0268 + 0.492 * 0.6129)
    r = assign(55.1321, "f", 5.8199, 1.6086, 135.2402, 0, 0, 1.6107, 14.74)
    assert r == pytest.approx(100 * (1 - 0.9666 ** math.exp(z)), rel=1e-12)


def test_input_handling():
    ok = (50, "m", 5, 1.3, 130, 0, 0, 10, 30)
    out = assign([50, 29.9, 75, 50, 50, 50], ["m", "m", "m", "x", "m", "m"], 5, 1.3, 130, 0, 0, 10, [30, 30, 30, 30, np.nan, 30])
    assert np.isfinite(out[0]) and np.isnan(out[1:5]).all() and np.isfinite(out[5])
    assert np.isnan(assign(50, "m", 5, 1.3, 130, 2, 0, 10, 30))
    with pytest.raises(ValueError):
        assign(*ok, version=3)


def _site_constants():
    js = (SRC / "rightdecisions_executeAlgorithm.js").read_text()
    men, women = re.findall(r"var (A=.*?);2==Number|&&\((A=.*?)\);var W", js)
    out = {}
    for sex, block in (("male", men[0]), ("female", women[1])):
        d = {k: float(v) for k, v in re.findall(r"(\w)=(-?[\d.]+)", block)}
        out[sex] = ((d["A"], d["G"], -d["H"], d["B"], d["D"], d["E"], d["P"], d["K"]), d["L"],
                    (d["Y"], d["J"], d["F"], d["N"], d["T"], d["Q"], d["S"], d["R"] / 10))
    return out


def test_official_calculator_constants():
    # executeAlgorithm subtracts the HDL term, so its H is the positive magnitude
    for sex, (beta, s0, mean) in _site_constants().items():
        assert beta == BETA[2][sex] and s0 == SURVIVAL[2][sex], sex
        np.testing.assert_allclose(mean, MEAN[2][sex], rtol=0, atol=1e-12)


def _shown(x):  # the page: Math.round(10 M) / 10 below 10, Math.round(M) from 10
    return math.floor(10 * x + 0.5) / 10 if x < 10 else math.floor(x + 0.5)


def test_official_calculator_run():
    fx = Path(__file__).parents[2] / "reference" / "assign" / "fixtures" / "calculator_reference.jsonl"
    rows = [json.loads(ln) for ln in fx.read_text().splitlines()]
    assert len(rows) == 200
    keys = ("age", "sex", "total_chol", "hdl_chol", "sbp", "diabetes", "family_history", "cigarettes_per_day", "simd_score")
    ours = assign(*(np.array([r[k] for r in rows], dtype=object if k == "sex" else float) for k in keys))
    bad = [(r["id"], r["shown"], o) for r, o in zip(rows, ours) if _shown(o) != r["shown"]]
    assert bad == []


OFFLINE = SRC / "wayback" / "offline_1_5_1" / "includes" / "js"


def test_offline_v151_constants():
    # developers' offline calculator 1.5.1 (2014): betas.js holds the version-1 betas and centring means
    js = (OFFLINE / "betas.js").read_text()
    order = ("age", "tc", "hdl", "sbp", "diabetes", "chd", "cpd", "simd")
    blocks = dict(zip(("beta", "mean"), js.split("var beta_coefficients")))
    for kind, table in (("beta", BETA[1]), ("mean", MEAN[1])):
        for sex, vals in table.items():
            block = re.search(r'"%s"\s*:\s*\{(.*?)\}' % sex, blocks[kind], re.S).group(1)
            d = {k: float(v) for k, v in re.findall(r'"(\w+)":\s*(-?[\d.]+)', block)}
            assert tuple(d[k] for k in order) == vals, (kind, sex)
    src = (OFFLINE / "assign-offline.js").read_text()
    assert '{ "male" : 0.8831, "female" : 0.9365 }' in src and "cpd = parseInt(cpd, 10) + 10" in src


def test_offline_v151_run():
    # 1,000 seeded patients through the cut-out calculate() (reference/assign/run_offline_v151.js);
    # rheumatoid arthritis is entered in the calculator as 10 extra cigarettes a day
    fx = Path(__file__).parents[2] / "reference" / "assign" / "fixtures" / "offline_v151_reference.jsonl"
    rows = [json.loads(ln) for ln in fx.read_text().splitlines()]
    assert len(rows) == 1000 and sum(r["rheumatoid_arthritis"] for r in rows) == 200
    col = {k: np.array([r[k] for r in rows], dtype=object if k == "sex" else float) for k in rows[0]}
    cpd = col["cigarettes_per_day"] + 10 * col["rheumatoid_arthritis"]
    ours = assign(col["age"], col["sex"], col["total_chol"], col["hdl_chol"], col["sbp"], col["diabetes"],
                  col["family_history"], cpd, col["simd_score"], version=1)
    np.testing.assert_allclose(ours, col["raw"], rtol=0, atol=1e-12)
    assert (np.floor(ours + 0.5) == col["shown"]).all()
