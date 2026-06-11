"""Recover the JALS Table 6 constants (5-year risk at score 0, per outcome and age group).

Methods: 5-year risk = 1 - exp(-(5 x 1-year rate at total score 0) x 1.0718^score), 1.0718 = 2^0.1. Table 6 prints
it to 2 decimals for scores 0, 5, ..., 60 by age group (AF models). Each column fits one constant c (risk =
1 - exp(-c 2^(score/10))); c is the middle of the range that matches most of the column's 13 printed values. Writes
fixtures/table6_constants.json and prints how many printed 5- and 10-year values the constants give back. The same
for the online supplement's Table 5 (the ACC/AHA blood-pressure variant): fixtures/supp_table5_constants.json.
"""

import csv
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
AGES = ["40-49", "50-59", "60-69", "70-79", "80-89"]
OUTCOMES = ["stroke", "ami", "composite", "cvd"]


def risk5(c, score):
    return 1 - np.exp(-c * 2 ** (np.asarray(score) / 10))


def risk10(c, c_next, score):
    return 1 - (1 - risk5(c, score)) * (1 - risk5(c_next, score))


def main():
    for name in ("table6", "supp_table5"):
        print(name)
        fit(name)


def fit(name):
    rows = list(csv.DictReader(open(HERE / f"fixtures/{name}.csv")))
    const = {}
    for o in OUTCOMES:
        const[o] = []
        for ag in AGES:
            col = [r for r in rows if r["outcome"] == o and r["age"] == ag]
            s = np.array([int(r["score"]) for r in col])
            v = np.array([float(r["risk5"]) for r in col])
            cl = lambda p: np.log(-np.log(1 - p / 100))  # noqa: E731
            lo, hi = cl(np.maximum(v - 0.005, 1e-6)) - s * np.log(2) / 10, cl(v + 0.005) - s * np.log(2) / 10
            grid = np.exp(np.linspace(hi.min() - 0.02, lo.max() + 0.02, 20001))
            hits = np.array([(np.abs(np.round(100 * risk5(c, s), 2) - v) < 1e-9).sum() for c in grid])
            best = np.flatnonzero(hits == hits.max())
            const[o].append(float("%.6g" % grid[best[len(best) // 2]]))  # middle of the best-matching range
    json.dump(const, open(HERE / f"fixtures/{name}_constants.json", "w"), indent=1)
    for o in OUTCOMES:
        n5 = n10 = 0
        m5 = m10 = 0.0
        for r in (r for r in rows if r["outcome"] == o):
            i = AGES.index(r["age"])
            c, cn = const[o][i], const[o][min(i + 1, 4)]  # 80-89: the last 5 years use the same column
            d5 = abs(round(100 * float(risk5(c, int(r["score"]))), 2) - float(r["risk5"]))
            d10 = abs(round(100 * float(risk10(c, cn, int(r["score"]))), 2) - float(r["risk10"]))
            n5 += d5 < 1e-9
            n10 += d10 < 1e-9
            m5, m10 = max(m5, d5), max(m10, d10)
        print(f"{o:9s} c = {const[o]}  5-year {n5}/65 (max {m5:.2f})  10-year {n10}/65 (max {m10:.2f})")


if __name__ == "__main__":
    main()
