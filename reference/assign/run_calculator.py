"""Run patients through the official ASSIGN v2.0 calculator (Right Decision Service, NHS Scotland).

https://www.rightdecisions.scot.nhs.uk/assign-v20/assign-cardiovascular-risk-score-calculator/
The page is a Vue app that computes in the browser (static read of /calculators/assign/dist/js/app.js,
method executeAlgorithm; copied to source/rightdecisions_executeAlgorithm.js). Each patient is typed into
the real form (SIMD entered directly, not by postcode), "Calculate score" is pressed and the displayed
"ASSIGN Score" is saved. The page rounds the score to 1 decimal below 10 and to a whole number from 10.

Drives the `browse` headless-Chrome CLI (set BROWSE to its path). Appends to
fixtures/calculator_reference.jsonl; rerunning skips patients already saved.
Run from repo root: python reference/assign/run_calculator.py
"""

import json
import os
import re
import subprocess
import time
from pathlib import Path

import numpy as np

URL = "https://www.rightdecisions.scot.nhs.uk/assign-v20/assign-cardiovascular-risk-score-calculator/"
BROWSE = os.environ.get("BROWSE", "browse")
OUT = Path(__file__).parent / "fixtures" / "calculator_reference.jsonl"
N = 200


def patients(n=N, seed=2007):
    g = np.random.default_rng(seed)
    for i in range(n):
        smoker = bool(g.random() < 0.35)
        tc = round(float(g.uniform(3.0, 9.0)), 1)
        yield dict(id=i, sex=("male", "female")[i % 2], age=int(g.integers(30, 75)),
                   total_chol=tc, hdl_chol=round(float(g.uniform(0.6, min(2.8, tc - 0.5))), 1),
                   sbp=int(g.integers(95, 211)), diabetes=int(g.random() < 0.15), family_history=int(g.random() < 0.4),
                   cigarettes_per_day=int(g.integers(1, 41)) if smoker else 0, simd_score=round(float(g.uniform(1, 89)), 1))


def b(*args):
    r = subprocess.run([BROWSE, *map(str, args)], capture_output=True, text=True, timeout=60)
    if r.returncode:
        raise RuntimeError(f"browse {args}: {r.stderr or r.stdout}")
    return r.stdout.strip()


def load():
    for attempt in range(5):  # the site or the network occasionally drops a page load
        try:
            return b("goto", URL)
        except RuntimeError:
            if attempt == 4:
                raise
            time.sleep(10)


def run(p):
    load()
    b("wait", "#calculate-score")
    b("fill", "#ageInput", p["age"])
    b("click", f"label[for={p['sex']}]")
    b("click", "label[for=simd]")
    b("fill", "#simd-conditional-field", p["simd_score"])
    b("click", "label[for=familyYes]" if p["family_history"] else "label[for=familyNo]")
    b("click", "label[for=diabetesYes]" if p["diabetes"] else "label[for=diabetesNo]")
    if p["cigarettes_per_day"]:
        b("click", "label[for=smokerYes]")
        b("fill", "#cigarettes-per-day-conditional-field", p["cigarettes_per_day"])
    else:
        b("click", "label[for=smokerNo]")
    b("fill", "#bloodPressureInput", p["sbp"])
    b("fill", "#totalCholesterolInput", p["total_chol"])
    b("fill", "#hdlCholesterolInput", p["hdl_chol"])
    b("click", "#calculate-score")
    b("wait", "--networkidle")
    m = re.search(r"ASSIGN Score:\s*([\d.]+)", b("text"))
    if not m:
        raise RuntimeError(f"no score shown for patient {p['id']}")
    return float(m.group(1))


def main():
    OUT.parent.mkdir(exist_ok=True)
    done = {json.loads(ln)["id"] for ln in OUT.read_text().splitlines()} if OUT.exists() else set()
    with OUT.open("a") as f:
        for p in patients():
            if p["id"] in done:
                continue
            p["shown"] = run(p)
            f.write(json.dumps(p) + "\n")
            f.flush()
            print(p["id"], p["shown"], flush=True)


if __name__ == "__main__":
    main()
