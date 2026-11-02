"""uniscore: the guideline-recommended CVD risk score for each person's country.

Routing (routing.csv, built and cited by reference/uniscore/build_routing.py), first match wins:
  USA                                  -> PREVENT, base model, 10-year total CVD (AHA 2023)
  GBR                                  -> QRISK3 (NICE NG238)
  55 countries of SCORE2 Figure 5      -> SCORE2 (40-69) / SCORE2-OP (70-89), ESC risk region
  48 countries of SCORE2 Asia-Pacific  -> SCORE2 Asia-Pacific (40-69), its risk region
  any other WHO 2019 country           -> WHO 2019 lab model, country recalibration

Inputs use the parameter names of the underlying functions and are broadcast together.
A row is NaN when its country is unknown or its score's own checks fail (e.g. age outside
that score's range). Two inputs are bridged for QRISK3: chol_hdl_ratio defaults to
total_chol / hdl_chol, and its diabetes category comes from `diabetes` (0/1) together
with `diabetes_type` ("type1"/"type2") for people with diabetes.
"""

from __future__ import annotations

import csv
import inspect
from functools import lru_cache
from pathlib import Path

import numpy as np

from cvd_risk._inputs import as_bool
from cvd_risk.prevent import prevent
from cvd_risk.qrisk3 import qrisk3
from cvd_risk.score2 import score2, score2_asia_pacific, score2_op
from cvd_risk.who2019 import who2019

SCORES = ("prevent", "qrisk3", "score2", "score2_asia_pacific", "who2019")
_FUNCS = {"prevent": prevent, "qrisk3": qrisk3, "score2": score2,
          "score2_asia_pacific": score2_asia_pacific, "who2019": who2019}
_SET_BY_ROUTER = {"region", "country", "model", "years", "outcome", "version", "recalibration", "year"}


@lru_cache(maxsize=None)
def _routing():
    with (Path(__file__).parent / "routing.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    return {r["iso3"]: (r["score"], r["region"]) for r in rows}


def _params(score):
    sig = inspect.signature(_FUNCS[score]).parameters
    names = {k for k in sig if k not in _SET_BY_ROUTER}
    required = {k for k, p in sig.items() if p.default is inspect.Parameter.empty}
    if score == "qrisk3":
        names |= {"total_chol", "hdl_chol", "diabetes_type"}
        required = (required - {"chol_hdl_ratio"}) | {"diabetes"}
    return names, required


def guideline_score(country) -> np.ndarray:
    """The score uniscore uses for each country (ISO3); "" when the country is unknown."""
    r = _routing()
    c = np.asarray(country, dtype=object)
    return np.vectorize(lambda v: r.get(str(v).strip().upper(), ("", ""))[0], otypes=[object])(c)


def _qrisk3_inputs(kw):
    kw = dict(kw)
    tc, hdl = kw.pop("total_chol", None), kw.pop("hdl_chol", None)
    if "chol_hdl_ratio" not in kw:
        if tc is None or hdl is None:
            raise ValueError("qrisk3 needs chol_hdl_ratio, or total_chol and hdl_chol")
        kw["chol_hdl_ratio"] = np.asarray(tc, float) / np.asarray(hdl, float)
    dm = as_bool(kw.pop("diabetes"))
    if "diabetes_type" not in kw and (dm == 1).any():
        raise ValueError('qrisk3 needs diabetes_type ("type1"/"type2") for people with diabetes')
    dtype = np.asarray(kw.pop("diabetes_type", "invalid"), dtype=object)
    kw["diabetes"] = np.where(dm == 0, "none", np.where(dm == 1, dtype, "invalid"))
    return kw


def uniscore(country, **inputs) -> np.ndarray:
    """10-year CVD risk (fraction) from the guideline score of each row's country (ISO3)."""
    known = set().union(*(_params(s)[0] for s in SCORES))
    unknown = set(inputs) - known
    if unknown:
        raise TypeError(f"unknown inputs: {sorted(unknown)}")

    shape = np.broadcast_shapes(np.shape(country), *(np.shape(v) for v in inputs.values()))
    flat = lambda v: np.broadcast_to(np.asarray(v, dtype=object if isinstance(v, str) else None), shape).ravel()  # noqa: E731
    iso = np.char.upper(np.char.strip(flat(country).astype(str)))
    data = {k: flat(v) for k, v in inputs.items()}
    route = _routing()
    score = np.array([route.get(c, ("", ""))[0] for c in iso], dtype=object)
    region = np.array([route.get(c, ("", ""))[1] for c in iso], dtype=object)

    out = np.full(iso.shape, np.nan)
    for s in SCORES:
        rows = score == s
        if not rows.any():
            continue
        names, required = _params(s)
        missing = required - set(data) - {"region", "country"}
        if missing:
            raise ValueError(f"{s} (used for {sorted(set(iso[rows]))}) needs: {sorted(missing)}")
        kw = {k: v[rows] for k, v in data.items() if k in names}
        if s == "prevent":
            out[rows] = prevent(**kw)
        elif s == "qrisk3":
            out[rows] = qrisk3(**_qrisk3_inputs(kw))
        elif s == "score2":
            old = np.asarray(kw["age"], float) >= 70
            out[rows] = np.where(old, score2_op(**kw, region=region[rows]), score2(**kw, region=region[rows]))
        elif s == "score2_asia_pacific":
            out[rows] = score2_asia_pacific(**kw, region=region[rows])
        else:
            out[rows] = who2019(**kw, country=iso[rows], recalibration="country")
    return out.reshape(shape)
