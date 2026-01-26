"""Does the official calculator use the coefficients printed in Yang 2016 Supplemental Table 1 (2 decimals)?

    python reference/china_par/fit_calculator.py        # all seeds -> fixtures/recovered_coefficients.json (package)
    python reference/china_par/fit_calculator.py 0      # seed 0 only -> recovered_coefficients_seed0.json (held-out check)

Fits, per sex, the printed model form to the calculator's shown 10-year risks on the scale log(-log(1 - risk)),
where it is linear: intercept (log(-log S10) - mean) + coefficients x terms. Rows whose shown risk is below 1% or
above 90% are left out (one decimal carries too little there). Prints the fitted coefficients next to the printed
ones, and how many shown values each set reproduces after rounding to one decimal. Then a linear program asks
whether any coefficients within half a printed unit (0.005) of the printed ones, with a free intercept, put every
patient's linear predictor inside the interval its shown value allows (shown +- 0.05 points).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog

HERE = Path(__file__).parent
SEEDS = [int(s) for s in sys.argv[1:]]  # empty: every seed
TERMS = {  # term: (men, women) printed coefficient; None = not in that sex's equation
    "ln_age": (31.97, 24.87), "ln_sbp_treated": (27.39, 20.71), "ln_sbp_untreated": (26.15, 19.98),
    "ln_tc": (0.62, 0.06), "ln_hdl": (-0.69, -0.22), "ln_waist": (-0.71, 1.48), "smoker": (3.96, 0.49),
    "diabetes": (0.36, 0.57), "north": (0.48, 0.54), "urban": (-0.16, None), "family_history": (6.22, None),
    "ln_age_ln_sbp_treated": (-6.02, -4.53), "ln_age_ln_sbp_untreated": (-5.73, -4.36),
    "ln_age_smoker": (-0.94, None), "ln_age_family_history": (-1.53, None),
}
PRINTED = {1: (140.68, 0.97), 2: (117.26, 0.99)}  # mean (coefficient x value), baseline survival


def terms(d):
    la, ls, t = np.log(d.age), np.log(d.sbp), d.bp_treated
    return pd.DataFrame({
        "ln_age": la, "ln_sbp_treated": ls * t, "ln_sbp_untreated": ls * (1 - t), "ln_tc": np.log(d.tc),
        "ln_hdl": np.log(d.hdl), "ln_waist": np.log(d.waist), "smoker": d.smoker, "diabetes": d.diabetes,
        "north": d.north, "urban": d.urban, "family_history": d.family_history,
        "ln_age_ln_sbp_treated": la * ls * t, "ln_age_ln_sbp_untreated": la * ls * (1 - t),
        "ln_age_smoker": la * d.smoker, "ln_age_family_history": la * d.family_history})


def main():
    d = pd.read_csv(HERE / "fixtures" / "calculator_reference.csv")
    d = d[d.seed.isin(SEEDS)] if SEEDS else d
    fits = {}
    for sex, label in ((1, "men"), (2, "women")):
        g = d[d.sex == sex]
        names = [k for k, v in TERMS.items() if v[sex - 1] is not None]
        X = terms(g)[names].to_numpy(float)
        printed = np.array([TERMS[k][sex - 1] for k in names])
        mean, s10 = PRINTED[sex]
        shown = g.risk_shown.to_numpy(float)

        def risk(b, c):
            return 100 * (1 - np.exp(-np.exp(c + X @ b)))

        p_risk = risk(printed, np.log(-np.log(s10)) - mean)
        ok = (shown >= 1) & (shown <= 90)
        y = np.log(-np.log(1 - shown[ok] / 100))
        A = np.column_stack([np.ones(ok.sum()), X[ok]])
        sol, *_ = np.linalg.lstsq(A, y, rcond=None)
        f_risk = risk(sol[1:], sol[0])
        print(f"\n{label}: {len(g)} patients, {ok.sum()} used in the fit")
        print(f"{'term':26s} {'printed':>8s} {'fitted':>10s}")
        for k, pv, fv in zip(names, printed, sol[1:]):
            print(f"{k:26s} {pv:8.2f} {fv:10.4f}")
        print(f"{'intercept':26s} {np.log(-np.log(s10)) - mean:8.2f} {sol[0]:10.4f}")
        for nm, r in (("printed", p_risk), ("fitted", f_risk)):
            eq = np.isclose(np.round(r, 1), shown)
            print(f"{nm}: {eq.sum()}/{len(g)} shown values equal; max abs diff {np.abs(r - shown).max():.2f} points")
        print("coefficients rounding to the printed ones fit every shown value:", feasible(X, shown, printed, 0.005))
        print("... with no bounds on the coefficients:", feasible(X, shown, printed, None))
        b, t = max_margin(X[ok], shown[ok], printed)
        m_risk = risk(b[1:], b[0])
        print(f"max-margin point (margin {t:.2e} on the cll scale): "
              f"{np.isclose(np.round(m_risk, 1), shown).sum()}/{len(g)} shown values equal")
        fits[label] = dict(zip(["intercept", *names], b))
    out = HERE / "fixtures" / ("recovered_coefficients" + "".join(f"_seed{s}" for s in SEEDS) + ".json")
    out.write_text(json.dumps(fits, indent=1) + "\n")
    print("\n->", out)


def max_margin(X, shown, centre, half=0.005):
    """Point inside the feasible region (coefficients within half of printed) farthest from its walls:
    maximise t subject to lo + t <= c + X b <= hi - t."""
    cll = lambda r: np.log(-np.log(1 - np.clip(r, 1e-9, 99.999) / 100))
    lo, hi = cll(np.maximum(shown - 0.05, 0)), cll(shown + 0.05)
    A = np.column_stack([np.ones(len(X)), X])
    one = np.ones((len(X), 1))
    A_ub = np.vstack([np.hstack([A, one]), np.hstack([-A, one])])
    bounds = [(None, None)] + [(c - half, c + half) for c in centre] + [(0, None)]
    res = linprog(np.r_[np.zeros(A.shape[1]), -1], A_ub=A_ub, b_ub=np.concatenate([hi, -lo]), bounds=bounds,
                  method="highs")
    assert res.status == 0, res.message
    return res.x[:-1], res.x[-1]


def feasible(X, shown, centre, half):
    """lo <= c + X b <= hi for every patient, with |b - centre| <= half (None: unbounded)."""
    cll = lambda r: np.log(-np.log(1 - np.clip(r, 1e-9, 99.999) / 100))
    lo, hi = cll(np.maximum(shown - 0.05, 0)), cll(shown + 0.05)
    A = np.column_stack([np.ones(len(X)), X])
    bounds = [(None, None)] + [(None, None) if half is None else (c - half, c + half) for c in centre]
    res = linprog(np.zeros(A.shape[1]), A_ub=np.vstack([A, -A]), b_ub=np.concatenate([hi, -lo]), bounds=bounds,
                  method="highs")
    return res.status == 0


if __name__ == "__main__":
    sys.exit(main())
