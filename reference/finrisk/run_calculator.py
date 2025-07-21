"""Query the official FINRISK calculator (Duodecim, forms.ebmedscloud.org/en-GB/questionnaire/finriski).

The page posts its answers to /api/analysis/predict and draws the server's answer (static read of the
page's JS: chunk 0qw1eimuiy70~.js, module 39375). Field ids, from the page's FHIR questionnaire 872 v4634:
3775 sex (5286 male, 5287 female), 3776 age 30-74, 3777 smoker, 3778 total cholesterol mmol/L
(1 decimal), 3779 HDL mmol/L (2 decimals), 3780 SBP, 3781 diabetes, 3782 parent MI, 3783 parent stroke.
The page shows CHD and stroke risk (absoluteRisk, fractions) and combined = 1 - (1 - CHD)(1 - stroke).

Sends N seeded patients, one request per 0.5 s, and appends inputs + the raw responses to
fixtures/calculator_reference.jsonl; rerunning skips patients already saved (the server sometimes stalls:
press Ctrl+C and run again). Run by the owner from repo root: python reference/finrisk/run_calculator.py
"""

import json
import time
import urllib.request
from pathlib import Path

import numpy as np

URL = "https://forms.ebmedscloud.org/api/analysis/predict"
OUT = Path(__file__).parent / "fixtures" / "calculator_reference.jsonl"
N = 300


def patients(n=N, seed=2016):
    g = np.random.default_rng(seed)
    for i in range(n):
        yield dict(id=i, sex=("male", "female")[i % 2], age=int(g.integers(30, 65)), smoker=bool(g.random() < 0.3),
                   total_chol=round(float(g.uniform(3.0, 9.0)), 1), hdl_chol=round(float(g.uniform(0.6, 2.5)), 2),
                   sbp=int(g.integers(100, 201)), diabetes=bool(g.random() < 0.15),
                   parent_mi=bool(g.random() < 0.25), parent_stroke=bool(g.random() < 0.2))


def answers(p):
    return {"3775": "5286" if p["sex"] == "male" else "5287", "3776": p["age"], "3777": p["smoker"],
            "3778": p["total_chol"], "3779": p["hdl_chol"], "3780": p["sbp"], "3781": p["diabetes"],
            "3782": p["parent_mi"], "3783": p["parent_stroke"]}


def post(body):
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json",
                                                                    "User-Agent": "PyCVDRisk-validation"})
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.load(r)


def main():
    OUT.parent.mkdir(exist_ok=True)
    done = {json.loads(ln)["id"] for ln in OUT.read_text().splitlines()} if OUT.exists() else set()
    with OUT.open("a") as f:
        for p in patients():
            if p["id"] in done:
                continue
            resp = post(answers(p))
            f.write(json.dumps({**p, "response": resp}) + "\n")
            f.flush()
            print(f"{p['id'] + 1}/{N}", end="\r", flush=True)
            time.sleep(0.5)
    print(f"{OUT}: {N} rows")


if __name__ == "__main__":
    main()
