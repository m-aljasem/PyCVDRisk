"""Build reference/who_ish/REPORT.md. Run from repo root: python reference/who_ish/report.py"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import charts

from cvd_risk.who_ish import REGIONS, who_ish


def call(keys):
    """who_ish at chart cells (region, diabetes, male, smoker, age, sbp, chol); chol 0 = no-chol chart."""
    k = np.array([key[1:] for key in keys], dtype=float)
    reg = [key[0] for key in keys]
    lab = k[:, 5] > 0
    out = np.full(len(keys), np.nan)
    for m, tc in ((lab, k[:, 5]), (~lab, None)):
        out[m] = who_ish(k[m, 3], np.where(k[m, 1] == 1, "male", "female"), k[m, 2], k[m, 4], k[m, 0], np.array(reg)[m],
                         total_chol=None if tc is None else tc[m])
    return out


def whoish_r(age, male, smoker, sbp, dm, chl, region):
    """Transliteration of whoishRisk's WHO_ISH_Risk (source/whoishRisk/WHOISHRisk.R): the same
    ifelse chains, then the lookup in its table. chl 0 selects the no-cholesterol chart."""
    age = np.where((age > 17) & (age < 50), 40, age)
    age = np.where((age >= 50) & (age < 60), 50, age)
    age = np.where((age >= 60) & (age < 70), 60, age)
    age = np.where(age >= 70, 70, age)
    chl = np.where((chl > 0) & (chl < 4.5), 4, chl)
    chl = np.where((chl >= 4.5) & (chl < 5.5), 5, chl)
    chl = np.where((chl >= 5.5) & (chl < 6.5), 6, chl)
    chl = np.where((chl >= 6.5) & (chl < 7.5), 7, chl)
    chl = np.where(chl >= 7.5, 8, chl)
    sbp = np.where((sbp > 0) & (sbp < 140), 120, sbp)
    sbp = np.where((sbp >= 140) & (sbp < 160), 140, sbp)
    sbp = np.where((sbp >= 160) & (sbp < 180), 160, sbp)
    sbp = np.where(sbp >= 180, 180, sbp)
    t = charts.whoishrisk()
    return np.array([t.get((r, int(d), int(m), int(s), int(a), int(b), int(c)), np.nan)
                     for r, d, m, s, a, b, c in zip(region, dm, male, smoker, age, sbp, chl)])


def patients(n=20000, seed=2007):
    rng = np.random.default_rng(seed)
    age = rng.integers(40, 80, n) + rng.choice([0, 0.5, 0.99], n)
    sbp = np.where(rng.random(n) < 0.3, rng.choice([139.99, 140, 159.99, 160, 179.99, 180], n),
                   np.round(rng.uniform(90, 220, n), 1))
    chl = np.where(rng.random(n) < 0.3, rng.choice([4.49, 4.5, 5.49, 5.5, 6.49, 6.5, 7.49, 7.5], n),
                   np.round(rng.uniform(2.5, 10, n), 2))
    chl = np.where(rng.random(n) < 0.25, 0, chl)
    return (age, rng.integers(0, 2, n), rng.integers(0, 2, n), sbp, rng.integers(0, 2, n), chl,
            rng.choice(REGIONS, n))


def monotonic_breaks(t):
    """Cells where a higher risk factor gives a lower band: (factor, region, cell, band, band one step up)."""
    out = []
    steps = {"diabetes": (0, 1), "smoker": (2, 1), "age": (3, 10), "sbp": (4, 20), "cholesterol": (5, 1)}
    for (r, *k), v in sorted(t.items()):
        for name, (i, d) in steps.items():
            if name == "cholesterol" and k[5] == 0:
                continue
            up = list(k)
            up[i] += d
            w = t.get((r, *up))
            if w is not None and w < v:
                out.append((name, r, tuple(k), v, w))
    return out


def fmt_key(k):
    d, m, s, a, b, c = k
    return (f"{'diabetes' if d else 'no diabetes'}, {'men' if m else 'women'}, "
            f"{'smoker' if s else 'non-smoker'}, age {a}, SBP {b}, " + (f"chol {c}" if c else "no chol"))


def main():
    printed, worst = charts.official()
    keys = sorted(printed)
    want = np.array([printed[k] for k in keys])
    got = call(keys)
    pkg_ok = int((got == want).sum())
    wr = charts.whoishrisk()
    wr_bad = [k for k in keys if wr[k] != printed[k]]

    every = sorted(wr)
    ours = call(every)
    diff = [(k, wr[k], int(o)) for k, o in zip(every, ours) if o != wr[k]]

    age, male, sm, sbp, dm, chl, reg = patients()
    lab = chl > 0
    p = np.full(len(age), np.nan)
    sex = np.where(male == 1, "male", "female")
    p[lab] = who_ish(age[lab], sex[lab], sm[lab], sbp[lab], dm[lab], reg[lab], total_chol=chl[lab])
    p[~lab] = who_ish(age[~lab], sex[~lab], sm[~lab], sbp[~lab], dm[~lab], reg[~lab])
    r = whoish_r(age, male, sm, sbp, dm, chl, reg)
    corrected = {(k[0], *k[1:]) for k, _, _ in diff}
    a_bin = np.where(age < 50, 40, np.floor(age / 10) * 10)
    hit = np.array([(g, int(d), int(m), int(s), int(a), int(b), int(c)) in corrected for g, d, m, s, a, b, c in zip(
        reg, dm, male, sm, a_bin, np.select([sbp < 140, sbp < 160, sbp < 180], [120, 140, 160], 180),
        np.where(chl == 0, 0, np.clip(np.floor(chl + 0.5), 4, 8)))])
    agree = int((p == r).sum())
    other = int(((p != r) & ~hit).sum())

    breaks = monotonic_breaks(wr)
    assert not any(r in ("sear_d", "emr_b") for _, r, *_ in breaks)  # the text says none is printed

    L = [
        "# WHO/ISH risk prediction charts (WHO 2007): verification report", "",
        f"Generated {dt.date.today()} by `reference/who_ish/report.py`.", "",
        "## 1. Source", "",
        "WHO, *Prevention of cardiovascular disease: guidelines for assessment and management of",
        "cardiovascular risk* (2007), saved as `source/who2007_guideline.pdf`. There is no equation:",
        "Annex 5 says the charts tabulate mean 10-year risks from a simulated cohort of 1,000,000 people",
        "per age and sex group and subregion, built from Comparative Risk Assessment data. The",
        "guideline prints two sample charts: Annex 3 (SEAR-D, with cholesterol) and Annex 4 (EMR-B,",
        "without cholesterol). The charts for the other 12 subregions were published separately",
        "(WHO/ISH pocket guidelines). We could not download them because the WHO IRIS server timed out.",
        "Their cell values come from whoishRisk 1.0 (Collins et al., R, GPL-3), whose table",
        "(`source/whoishRisk/WHO_ISH_Scores.csv`) covers all 14 subregions. `make_table.py` builds the",
        "package table from it and replaces every printed cell with the value read from the guideline.", "",
        "Cell selection: the guideline's age rule (\"if age is 50-59 select 50, 60-69 select 60 etc\",",
        "p. 17), applied from 40 to 79. The guideline gives no rule for SBP or cholesterol; we use",
        "whoishRisk's (SBP <140 -> 120, 140-159 -> 140, ...; cholesterol <4.5 -> 4, 4.5-5.4 -> 5, ...).", "",
        "## 2. The guideline's charts (evidence level A)", "",
        "Each cell of Annexes 3 and 4 was read from a ghostscript render (200 dpi) by matching the colour",
        f"at the cell centre to the five key colours. The worst distance to the nearest key colour is {worst:.1f}",
        "(on the 0-441 RGB scale); the key colours are at least 90 apart.", "",
        "| Comparison | Cells | Agree |", "|---|---:|---:|",
        f"| `who_ish` vs guideline (SEAR-D 640 + EMR-B 128) | {len(keys)} | {pkg_ok} |",
        f"| whoishRisk vs guideline | {len(keys)} | {len(keys) - len(wr_bad)} |", "",
        "whoishRisk differs from the printed chart in these cells (all in SEAR-D):", "",
        "| Cell | whoishRisk | Guideline (used) |", "|---|---|---|",
    ]
    L += [f"| {k[0].upper()}: {fmt_key(k[1:])} | {charts.BANDS[wr[k]]} | {charts.BANDS[printed[k]]} |"
          for k in wr_bad]
    L += [
        "", "## 3. whoishRisk (evidence level B)", "",
        f"Every one of the {len(every)} cells (14 subregions x 768) was looked up through `who_ish`.",
        f"{len(every) - len(diff)} agree with whoishRisk's table. The {len(diff)} that differ are the",
        "guideline corrections in section 2.", "",
        "The lookup logic was then checked against a line-by-line transliteration of whoishRisk's",
        "R function (`WHO_ISH_Risk`, its `ifelse` binning chains and table match). The R code itself was not run.",
        f"The test set has {len(age)} random patients aged 40-79 (seed 2007); 30% have SBP and cholesterol on or just",
        f"below a cut point, and 25% use the no-cholesterol chart: {agree}/{len(age)} agree,",
        f"{len(age) - agree - other} differ only through the corrected cells, and {other} differ otherwise.", "",
        "## 4. Internal consistency of the table", "",
        "A higher risk factor should not give a lower band. Across the 14 subregions, these cells break",
        "that rule. None is in a printed chart, so they cannot be checked against the source:", "",
        "| Factor | Subregion | Cell | Band | Band one step up |", "|---|---|---|---|---|",
    ]
    L += [f"| {f} | {r.upper()} | {fmt_key(k)} | {charts.BANDS[v]} | {charts.BANDS[w]} |"
          for f, r, k, v, w in breaks]
    L += [
        "", "They are kept as published by whoishRisk. The simulated means need not be monotone, so these",
        "may be in the official charts too. Checking them needs the 12 regional charts.", "",
        "## 5. Limitations", "",
        "- 12 of 14 subregions rest on whoishRisk (level B) until the official regional charts are obtained.",
        "- The SBP and cholesterol cut points are whoishRisk's convention; the guideline states none.",
    ]
    (HERE / "REPORT.md").write_text("\n".join(L) + "\n")
    print(f"guideline {pkg_ok}/{len(keys)}; whoishRisk vs guideline {len(keys) - len(wr_bad)}; "
          f"table diffs {len(diff)}; patients {agree}/{len(age)} other={other}; breaks {len(breaks)}")
    assert pkg_ok == len(keys) and other == 0


if __name__ == "__main__":
    main()
