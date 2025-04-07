"""Compare SCORE2 and SCORE2-OP with the official ESC 2021 risk charts (fixtures/esc2021_charts.csv).

Chart inputs are not printed. They are inferred from the charts and pinned by them: band
midpoints for age (e.g. 42.5), SBP (e.g. 110) and non-HDL cholesterol (e.g. 3.5 mmol/L), and
HDL 1.4 mmol/L for women and 1.2 for men (total cholesterol = non-HDL + HDL); `fit()` shows these
maximise exact agreement. The coefficients and region scales are printed rounded (SCORE2-OP
scales to 2 decimals), while the charts come from unrounded values. So each cell is also
checked against the interval of risks the printed values allow: every coefficient, baseline
survival, mean linear predictor and scale moved by up to half a unit of its last printed digit,
in the direction that raises (or lowers) that cell's risk.
Run from repo root: python reference/score2/chart_check.py
"""

from __future__ import annotations

import importlib
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

C = importlib.import_module("cvd_risk.score2.coefficients")
mod = importlib.import_module("cvd_risk.score2.score2")
FX = Path(__file__).parent / "fixtures" / "esc2021_charts.csv"
HDL = {"female": 1.4, "male": 1.2}
# Cells outside the printed-precision interval: (region, sex, smoker, age band, SBP band, non-HDL band).
# All four lie in the non-HDL 4.0-4.9 column; the charts' neighbouring cells agree.
KNOWN_ANOMALIES = {
    ("low", "female", 1, "55-59", "120-139", "4.0-4.9"),
    ("moderate", "female", 1, "60-64", "120-139", "4.0-4.9"),
    ("high", "male", 0, "60-64", "160-179", "4.0-4.9"),
    ("high", "female", 0, "40-44", "160-179", "4.0-4.9"),
}


def _mid(band):
    lo, hi = (float(x) for x in band.split("-"))
    return lo + (hi - lo + (1 if hi == int(hi) and lo == int(lo) else 0.1)) / 2


def cells(hdl=HDL):
    """Chart cells with model inputs; `hdl` maps sex -> HDL (mmol/L) used for the chart."""
    d = pd.read_csv(FX)
    d["age"] = d.age_band.map(_mid)  # 40-44 -> 42.5
    d["sbp"] = d.sbp_band.map(_mid)  # 100-119 -> 110
    d["nonhdl"] = d.nonhdl_band.map(_mid)  # 3.0-3.9 -> 3.5
    d["hdl"] = d.sex.map(hdl)
    d["tc"] = d.nonhdl + d.hdl
    return d


def predict(d):
    args = dict(age=d.age.to_numpy(), sex=d.sex.to_numpy(), smoker=d.smoker.to_numpy(), sbp=d.sbp.to_numpy(),
                total_chol=d.tc.to_numpy(), hdl_chol=d.hdl.to_numpy(), diabetes=0, region=d.region.to_numpy())
    return 100 * np.where(d.variant == "score2_op", mod.score2_op(**args), mod.score2(**args))


def _half(x):
    return float(Decimal(10) ** Decimal(repr(x)).as_tuple().exponent) / 2


def _terms(op, age, sm, sbp, tc, hdl):
    if op:
        a, s, t, h = age - 73, sbp - 150, tc - 6, hdl - 1.4
    else:
        a, s, t, h = (age - 60) / 5, (sbp - 120) / 20, tc - 6, (hdl - 1.3) / 0.5
    return {"age": a, "smoker": sm, "sbp": s, "tc": t, "hdl": h,
            "age_smoker": a * sm, "age_sbp": a * s, "age_tc": a * t, "age_hdl": a * h}


def bounds(d):
    """(lower, upper) risk in % per cell under the printed-precision interval."""
    out = np.empty((len(d), 2))
    for i, r in enumerate(d.itertuples()):
        op = r.variant == "score2_op"
        c = (C.SCORE2_OP if op else C.SCORE2)[r.sex]
        s1, s2 = (C.SCORE2_OP_SCALES if op else C.SCORE2_SCALES)[r.sex][C.REGIONS.index(r.region)]
        x = _terms(op, r.age, r.smoker, r.sbp, r.tc, r.hdl)
        for j, sg in enumerate((-1, 1)):
            lp = sum(c[k] * v + sg * _half(c[k]) * abs(v) for k, v in x.items())
            if op:
                lp -= c["mean_lp"] - sg * _half(c["mean_lp"])
            u = 1 - (c["s0"] - sg * _half(c["s0"])) ** np.exp(lp)
            cl = np.log(-np.log(1 - u))
            z = s1 + sg * _half(s1) + (s2 + sg * _half(s2) * np.sign(cl)) * cl
            out[i, j] = 100 * -np.expm1(-np.exp(z))
    return out[:, 0], out[:, 1]


def consistent(d=None):
    d = cells() if d is None else d
    lo, hi = bounds(d)
    v = d.risk_pct.to_numpy()
    return (v >= np.floor(lo + 0.5)) & (v <= np.floor(hi + 0.5))


def fit(grid=np.round(np.arange(1.0, 1.61, 0.1), 1)):
    """Exact-agreement share of SCORE2 cells for each chart HDL value, by sex."""
    d0 = cells()
    res = {}
    for sex in ("female", "male"):
        for h in grid:
            d = cells({**HDL, sex: h})
            m = (d.sex == sex) & (d.variant == "score2")
            res[sex, h] = float(np.mean(np.floor(predict(d[m]) + 0.5) == d0.risk_pct[m]))
    return res


def summary():
    d = cells()
    p = predict(d)
    lo, hi = bounds(d)
    d["exact"] = np.floor(p + 0.5) == d.risk_pct
    d["within"] = consistent(d)
    d["width"] = hi - lo
    d["p"], d["lo"], d["hi"] = p, lo, hi
    return d


def main():
    d = summary()
    for v, g in d.groupby("variant", sort=False):
        print(f"{v}: {len(g)} cells, exact {g.exact.sum()}, within printed precision {g.within.sum()}, "
              f"max |diff| {np.max(np.abs(np.floor(g.p + 0.5) - g.risk_pct)):.0f}, median interval width "
              f"{g.width.median():.2f} points")
    for r in d[~d.within].itertuples():
        print(f"outside: {r.region} {r.sex} smoker={r.smoker} age {r.age_band} SBP {r.sbp_band} "
              f"non-HDL {r.nonhdl_band}: chart {r.risk_pct}, model {r.p:.2f} (interval {r.lo:.2f}-{r.hi:.2f})")
    a = ap_summary()
    print(f"score2_asia_pacific: {len(a)} cells, exact {a.exact.sum()}")
    for r in a[~a.exact].itertuples():
        print(f"not exact: {r.region} {r.sex} smoker={r.smoker} age {r.age_band} SBP {r.sbp_band} "
              f"non-HDL {r.nonhdl_band}: chart {r.risk_pct}, model {r.p:.2f}")
    return d


# --- SCORE2 Asia-Pacific (Hageman 2025, Figure 3; raster charts transcribed by eye) ---
AP_FX = FX.parent / "asia_pacific_charts_transcribed.txt"
AP_KNOWN_ANOMALIES = {("high", "female", 0, "40-44", "100-119", "3.0-3.9")}  # printed 1; neighbours 3, 4, 4


def ap_cells():
    rows = [ln.split() for ln in AP_FX.read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    d = pd.DataFrame([dict(region=r[0], age_band=r[1], sbp_band=r[2], sex="female" if c < 8 else "male",
                           smoker=(c // 4) % 2, nonhdl_band=("3.0-3.9", "4.0-4.9", "5.0-5.9", "6.0-6.9")[c % 4],
                           risk_pct=int(v)) for r in rows for c, v in enumerate(r[3:])])
    d["age"], d["sbp"], d["nonhdl"] = d.age_band.map(_mid), d.sbp_band.map(_mid), d.nonhdl_band.map(_mid)
    d["hdl"] = d.sex.map(HDL)
    d["tc"] = d.nonhdl + d.hdl
    return d


def ap_summary():
    """Asia-Pacific chart cells with the model's risk; its scales are printed to 9 decimals, so exact match only."""
    d = ap_cells()
    d["p"] = 100 * mod.score2_asia_pacific(d.age.to_numpy(), d.sex.to_numpy(), d.smoker.to_numpy(), d.sbp.to_numpy(),
                                           d.tc.to_numpy(), d.hdl.to_numpy(), 0, d.region.to_numpy())
    d["exact"] = np.floor(d.p + 0.5) == d.risk_pct
    return d


if __name__ == "__main__":
    main()
