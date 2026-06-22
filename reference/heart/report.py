"""Build reference/heart/REPORT.md. Run from repo root: python reference/heart/report.py"""

from __future__ import annotations

import csv
import datetime as dt
from pathlib import Path

import numpy as np

from cvd_risk import heart

HERE = Path(__file__).parent
FX = HERE / "fixtures"


def load():
    p = list(csv.DictReader((FX / "patients.csv").open()))
    r = list(csv.DictReader((FX / "reference.csv").open()))
    col = lambda k: np.array([float(x[k]) for x in p])
    ours = heart(col("history"), col("ecg"), col("age"), col("risk_factors"), col("atherosclerosis"),
                 col("troponin_ratio"))
    ref = np.array([float(x["score"]) if x["score"] else np.nan for x in r])
    return p, ours, ref


# One row per rule in Melki & Jernberg 2013 (Table 1 and Methods): inputs -> expected points total
# with the other four items held at 0 (history 0, ECG 0, age 30, no risk factors, troponin ratio 0).
RULES = [
    ("history slightly suspicious", {"history": 0}, 0), ("history moderately suspicious", {"history": 1}, 1),
    ("history highly suspicious", {"history": 2}, 2), ("ECG normal", {"ecg": 0}, 0),
    ("ECG non-specific repolarisation", {"ecg": 1}, 1), ("ECG significant ST depression", {"ecg": 2}, 2),
    ("age 44 (<45)", {"age": 44}, 0), ("age 45", {"age": 45}, 1), ("age 64", {"age": 64}, 1),
    ("age 65", {"age": 65}, 2), ("1 risk factor", {"risk_factors": 1}, 1), ("2 risk factors", {"risk_factors": 2}, 1),
    ("3 risk factors", {"risk_factors": 3}, 2), ("atherosclerosis, no risk factors", {"atherosclerosis": 1}, 2),
    ("troponin below the limit", {"troponin_ratio": 0.99}, 0), ("troponin at the limit", {"troponin_ratio": 1}, 1),
    ("troponin 3x the limit", {"troponin_ratio": 3}, 1), ("troponin above 3x", {"troponin_ratio": 3.01}, 2),
]
BASE = {"history": 0, "ecg": 0, "age": 30, "risk_factors": 0, "atherosclerosis": 0, "troponin_ratio": 0}


# The developers' score card: (row, inputs inside the band, card points); three rows per item.
CARD = [
    ("history slightly / moderately / highly suspicious", [{"history": h} for h in (0, 1, 2)]),
    ("ECG normal / non-specific / significant ST-deviation", [{"ecg": e} for e in (0, 1, 2)]),
    ("age <=45 / 45-65 / >=65", [{"age": a} for a in (30, 55, 80)]),
    ("risk factors none / 1-2 / >=3", [{"risk_factors": n} for n in (0, 2, 4)]),
    ("troponin <=1x / 1-3x / >=3x", [{"troponin_ratio": t} for t in (0.5, 2, 6)]),
]
SRC = "https://web.archive.org/web/"
FILES = [
    ("heartscore_table.pdf", SRC + "20151123105354/http://www.heartscore.nl:80/resources/table.pdf",
     "4597706d6aa1ebd8a0d0dc69e578952cd6d2cb765109d58bab91cc59f75ae056"),
    ("heartscore_flyer.pdf", SRC + "20170629205528/http://www.heartscore.nl/resources/flyer.pdf",
     "c67c9f4829614affaaf8aa7222fdc86a145ade2fc3772a703c3b6c4e125e267e"),
    ("heartscore_score_page_2010.html", SRC + "20101026082052/http://www.heartscore.nl:80/score/",
     "ec1748e86537ef4035b429d8a42ff5d5116cfbe28c106b6f7ba004ac3d45cefd"),
    ("heartscore_home.html", "https://www.heartscore.nl/",
     "25d8efc552a559437d14a1504e4ccab7d754db0133f578674cdbb35df28ed8cf"),
    ("six2008_nhj.pdf", SRC + "20160404220740/http://heartscore.nl/resources/nhj_2008_06_191.pdf",
     "ae52131d55ddcd6098093d3df6e658c4b24c2d7de2a556c960b85c527dc57053"),
    ("backus2010_multicenter.pdf", SRC + "20160405025502/http://heartscore.nl/resources/multicenterheart-2010.pdf",
     "93a383538882dc44d3f18a11f185f6f81b4a559728bff6f07269e69ac72f33a4"),
]


def card_checks():
    """Card rows, all 3^5 row combinations, and the three cut points the developers print inconsistently."""
    card_ok = sum(float(heart(**{**BASE, **kw})) == pts for _, rows in CARD for pts, kw in enumerate(rows))
    grid = np.array(np.meshgrid(*[range(3)] * 5, indexing="ij")).reshape(5, -1)
    kw = {k: np.array([list(r[i].values())[0] for i in g]) for (_, r), g in zip(CARD, grid)
          for k in r[0]}
    got = heart(kw["history"], kw["ecg"], kw["age"], kw["risk_factors"], 0, kw["troponin_ratio"])
    ath = heart(kw["history"], kw["ecg"], kw["age"], np.where(grid[3] == 2, 0, kw["risk_factors"]),
                (grid[3] == 2).astype(int), kw["troponin_ratio"])
    want = grid.sum(axis=0)
    combo_ok = int(((got == want) & (ath == want)).sum())
    bnd = [("age 45", float(heart(**{**BASE, "age": 45})), 0, "overlapping (<=45 0, 45-65 1)"),
           ("troponin 1.0x", float(heart(**{**BASE, "troponin_ratio": 1})), 0, "overlapping (<=1x 0, 1-3x 1)"),
           ("troponin 3.0x", float(heart(**{**BASE, "troponin_ratio": 3})), 2, "overlapping (1-3x 1, >=3x 2)")]
    return card_ok, grid.shape[1], combo_ok, bnd


def main():
    p, ours, ref = load()
    same = (ours == ref) | (np.isnan(ours) & np.isnan(ref))
    rules = [(name, float(heart(**{**BASE, **kw})), want) for name, kw, want in RULES]
    rules_ok = sum(got == want for _, got, want in rules)
    version = (FX / "VERSION").read_text().splitlines()
    L = [
        "# HEART score: verification report", "",
        f"Generated by `reference/heart/report.py`.", "",
        "## 1. Source (evidence level C: the published rules; the developers' score card)", "",
        "The developers (Six, Backus, Kelder) publish the score at www.heartscore.nl. The site has no",
        "calculator of its own (checked live and in Wayback captures 2010-2026): it links to MDCalc",
        "and QxMD, third-party tools. It does publish an official score card, saved in `source/` (section 4).",
        "The rules implemented are those of the developers' validation paper (Backus 2010, Methods; also",
        "`source/backus2010_multicenter.pdf`, archived from heartscore.nl), repeated by Melki & Jernberg",
        "(Crit Pathw Cardiol 2013, Methods; `source/melki2013.pdf`): age \"younger than 45\" 0, \"45 to 65\" 1,",
        "\"65 years or older\" 2; troponin \"below the threshold\" 0, \"between 1 and 3 times\" 1, \"higher than",
        "3 times\" 2. The printed tables (\"<=45 / 45-65 / >=65\", \"<=normal limit / 1-3x / >=3x\") overlap at",
        "the cut points; the Methods resolve them.", "",
        "| Rule | Points | Expected |", "|---|---:|---:|",
    ]
    L += [f"| {n} | {g:g} | {w} |" for n, g, w in rules]
    L += [
        "", f"{rules_ok}/{len(rules)} rules give the published points.", "",
        "## 2. RiskScorescvd (evidence level B)", "",
        *[f"- {v}" for v in version], "",
        "RiskScorescvd grades history from a count of typical symptoms (0-1, 2-4, 5-6) and troponin from",
        "hs-troponin I against fixed sex-specific limits (34 and 16 ng/L). `make_patients.py` therefore",
        "draws each patient in both forms: our history grade and a symptom count in the matching range,",
        "and our troponin ratio equal to its hs-troponin I divided by its limit. 30% of troponin values",
        "and 20% of ages sit on or next to a cut point.", "",
        f"{len(p)} random patients (seed 2008): {int(same.sum())}/{len(p)} identical scores.", "",
        "## 3. Differences by design", "",
        "- Ages are not restricted to whole years: 64.5 scores 1 here, while RiskScorescvd's `45 <= Age <= 64`",
        "  gives NA.",
        "- RiskScorescvd counts five risk factors; Melki also counts obesity and angina pectoris. Here the",
        "  caller passes the count.",
        "- RiskScorescvd scores an ECG given as all-zero flags as 1 point; here the ECG is the clinician's",
        "  0-2 grade.",
    ]
    card_ok, n_combo, combo_ok, bnd = card_checks()
    L += [
        "", "## 4. The developers' score card (heartscore.nl)", "",
        "| File | URL | sha256 |", "|---|---|---|",
        *[f"| `source/{f}` | {u} | `{h}` |" for f, u, h in FILES], "",
        "The card (`heartscore_table.pdf`, 2010-2015; `heartscore_flyer.pdf`, 2017-2026) and the 2010-2016",
        "/score/ page print the same 15 rows as Backus 2010 Table 1, so they add provenance rather than an",
        "independent check. ECG row 2 reads \"significant ST-deviation\" (papers: ST depression) and row 1 adds",
        "\"LBBB / PM\"; both are part of the clinician's ECG grade, an input here. The card's risk factors are",
        "hypercholesterolaemia, hypertension, diabetes, smoking, family history and obesity (BMI >30);",
        "Melki also counts angina pectoris. The caller passes the count.", "",
        f"- Card rows, one value inside each band: {card_ok}/{3 * len(CARD)} give the card's points.",
        f"- Every combination of the five rows (3^5 = {n_combo}), plus atherosclerosis in place of 3 risk",
        f"  factors: {combo_ok}/{n_combo} totals equal the sum of the card's points.",
        "- Risk groups on the site: 0-3 low (discharge), 4-6 intermediate, 7-10 high. The docstring quotes the",
        "  same bands; RiskScorescvd's `classify` uses <=3, 4-6, >6.", "",
        "Cut points: the developers' own material disagrees with itself at three values.", "",
        "| Value | Here (Backus 2010 Methods) | `heartscore_table.pdf` | flyer, /score/ page |",
        "|---|---:|---:|---|",
        *[f"| {n} | {g:g} | {t} | {f} |" for n, g, t, f in bnd], "",
        "Classified as an ambiguity in the developers' material, not as our error: both papers by the",
        "developers state the rules used here, and the current flyer prints overlapping bands. Six 2008",
        "(the pilot study, `source/six2008_nhj.pdf`) used 1-2x / >2x for troponin and misprints the top age",
        "band as \"<=65\"; the 3x cut has been used since Backus 2010 and is on every card.",
    ]
    (HERE / "REPORT.md").write_text("\n".join(L) + "\n")
    print(f"rules {rules_ok}/{len(rules)}; RiskScorescvd {int(same.sum())}/{len(p)}")
    print(f"card rows {card_ok}/{3 * len(CARD)}; combinations {combo_ok}/{n_combo}")
    assert rules_ok == len(rules) and same.all() and card_ok == 3 * len(CARD) and combo_ok == n_combo


if __name__ == "__main__":
    main()
