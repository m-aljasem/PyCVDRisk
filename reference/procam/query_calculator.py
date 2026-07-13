"""Query the Assmann-Stiftung's PROCAM-Tests (www.assmann-stiftung.de/procam-tests/, frame /PROCAM21/index_neu.html,
saved in source/procam21/) for seeded synthetic patients and write fixtures/calculator_{coronary,stroke}.csv.

    python -I reference/procam/query_calculator.py [n_coronary] [n_stroke]

The page computes nothing itself: each slider change sends a GET to /PROCAM/api/healthcheck.php (PROCAM-
Gesundheitstest: men and women 20-75, coronary) or /PROCAM/api/strokecheck.php (Schlaganfalltest, 35-65) and shows
the JSON's "risk" (10-year %, coronary with two decimals, stroke with one). Inputs are whole numbers within the
page's slider ranges (age 20-75 / 35-65, SBP 100-225 mmHg, LDL 75-250, HDL 25-75, TG 50-400 mg/dL). One GET per
patient, then a 1 s pause. The page's "save result" call (services.herzalter-bestimmen.de/api/calculate) stores a
record, so it is not used.
"""

import csv
import json
import random
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

BASE = "https://www.assmann-stiftung.de/PROCAM/api/"
FIX = Path(__file__).parent / "fixtures"
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124 Safari/537.36"


def coronary(rnd):
    return dict(age=rnd.randint(20, 75), gender=rnd.choice((1, 2)), diabetes=int(rnd.random() < 0.2),
                smoking=int(rnd.random() < 0.35), anamnesis=int(rnd.random() < 0.25), systrr=rnd.randint(100, 225),
                ldl=rnd.randint(75, 250), hdl=rnd.randint(25, 75), triglyceride=rnd.randint(50, 400))


def stroke(rnd):
    return dict(age=rnd.randint(35, 65), gender=rnd.choice((1, 2)), diabetes=int(rnd.random() < 0.3),
                smoking=int(rnd.random() < 0.4), systrr=rnd.randint(100, 225))


def get(endpoint, params):
    url = BASE + endpoint + "?" + urllib.parse.urlencode(params)
    for attempt in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=20) as r:
                return json.loads(r.read().decode("utf-8-sig"))
        except Exception as e:  # noqa: BLE001 - retry transient network errors, then give up on this patient
            print(f"retry {attempt + 1}: {e}", file=sys.stderr)
            time.sleep(5 * (attempt + 1))
    return None


def run(kind, n, seed):
    rnd, endpoint, make = random.Random(seed), *{
        "coronary": ("healthcheck.php", coronary), "stroke": ("strokecheck.php", stroke)}[kind]
    out = FIX / f"calculator_{kind}.csv"
    rows = []
    for i in range(n):
        p = make(rnd)
        extra = {"units": "mg/dL", "language": "en-us"} if kind == "coronary" else {"language": "en-us"}
        d = get(endpoint, {**p, **extra})
        rows.append({"id": i + 1, **p, "risk": None if d is None else d.get("risk"),
                     "relative_risk": None if d is None else d.get("relativeRisk"),
                     "adjusted": None if d is None else d.get("adjusted"),
                     "heart_age": None if d is None else d.get("heartAge", "")})
        time.sleep(1.0)
    FIX.mkdir(exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{kind}: {sum(r['risk'] is not None for r in rows)}/{n} -> {out}")


if __name__ == "__main__":
    run("coronary", int(sys.argv[1]) if len(sys.argv) > 1 else 500, 2007)
    run("stroke", int(sys.argv[2]) if len(sys.argv) > 2 else 400, 2008)
